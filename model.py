"""Modelo predictivo de recompra a 90 días (validación fuera de tiempo).

Para cada fecha de corte se construyen variables con el historial anterior y se predice
si el cliente comprará en los 90 días siguientes. Entrenamiento: 6 cortes de 2010-2011.
Prueba: corte 2011-09-10 (ventana 2011-09-10 a 2011-12-09), posterior a todo el entrenamiento.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42
HORIZONTE = 90
CORTES_ENTRENAMIENTO = ["2010-03-12", "2010-06-10", "2010-09-08", "2010-12-07", "2011-03-08", "2011-06-06"]
CORTE_PRUEBA = "2011-09-10"

fac = pd.read_csv("facturas.csv", parse_dates=["fecha"])
dev = pd.read_csv("devoluciones.csv", parse_dates=["fecha"])
os.makedirs("resultados", exist_ok=True)
os.makedirs("graficos", exist_ok=True)

VARIABLES = ["recencia_dias", "frecuencia", "monetario", "ticket_promedio", "antiguedad_dias",
             "dias_entre_compras", "productos_por_factura", "unidades_por_factura",
             "facturas_90d", "facturas_180d", "ingreso_90d", "devoluciones", "es_uk"]


def construir(corte):
    corte = pd.Timestamp(corte)
    h = fac[fac.fecha <= corte]
    g = h.groupby("cliente")
    X = pd.DataFrame({
        "recencia_dias": (corte - g.fecha.max()).dt.days,
        "frecuencia": g.factura.nunique(),
        "monetario": g.ingreso.sum(),
        "antiguedad_dias": (corte - g.fecha.min()).dt.days,
        "productos_por_factura": g.productos.mean(),
        "unidades_por_factura": g.unidades.mean(),
        "es_uk": g.pais.agg(lambda s: int(s.iloc[-1] == "United Kingdom")),
    })
    X["ticket_promedio"] = X.monetario / X.frecuencia
    X["dias_entre_compras"] = np.where(X.frecuencia > 1, X.antiguedad_dias / (X.frecuencia - 1), X.antiguedad_dias)
    for d in (90, 180):
        r = h[h.fecha > corte - pd.Timedelta(days=d)]
        X[f"facturas_{d}d"] = r.groupby("cliente").factura.nunique()
    X["ingreso_90d"] = h[h.fecha > corte - pd.Timedelta(days=90)].groupby("cliente").ingreso.sum()
    X["devoluciones"] = dev[dev.fecha <= corte].groupby("cliente").factura.nunique()
    X = X.fillna({"facturas_90d": 0, "facturas_180d": 0, "ingreso_90d": 0, "devoluciones": 0})
    fut = fac[(fac.fecha > corte) & (fac.fecha <= corte + pd.Timedelta(days=HORIZONTE))]
    X["compro"] = X.index.isin(fut.cliente.unique()).astype(int)
    X["ingreso_futuro"] = fut.groupby("cliente").ingreso.sum().reindex(X.index).fillna(0)
    X["corte"] = str(corte.date())
    return X


train = pd.concat([construir(c) for c in CORTES_ENTRENAMIENTO])
test = construir(CORTE_PRUEBA)
print(f"entrenamiento: {len(train):,} filas, tasa de recompra {train.compro.mean():.1%}")
print(f"prueba: {len(test):,} clientes, tasa de recompra {test.compro.mean():.1%}")
train.groupby("corte").compro.agg(["size", "mean"]).round(3).to_csv("resultados/tasa_recompra_por_corte.csv")

log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
logistica = make_pipeline(
    ColumnTransformer([("log", log, ["recencia_dias", "frecuencia", "monetario", "ticket_promedio",
                                       "antiguedad_dias", "dias_entre_compras", "facturas_90d",
                                       "facturas_180d", "ingreso_90d", "devoluciones",
                                       "productos_por_factura", "unidades_por_factura"])],
                      remainder="passthrough"),
    StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, random_state=SEED))
gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                     l2_regularization=1.0, random_state=SEED)

Xtr, ytr = train[VARIABLES], train.compro
Xte, yte = test[VARIABLES], test.compro
modelos = {"Regresión logística": logistica, "Gradient boosting": gbm}
scores = {"Base: menor recencia": -Xte.recencia_dias.values}
for n, m in modelos.items():
    m.fit(Xtr, ytr)
    scores[n] = m.predict_proba(Xte)[:, 1]


def lift_top(score, y, frac):
    k = int(len(y) * frac)
    idx = np.argsort(-score)[:k]
    return y.values[idx].mean() / y.mean(), y.values[idx].mean(), y.values[idx].sum() / y.sum()


filas = []
for n, s in scores.items():
    l10, p10, c10 = lift_top(s, yte, 0.10)
    l20, p20, c20 = lift_top(s, yte, 0.20)
    filas.append({"modelo": n, "auc": roc_auc_score(yte, s), "auc_pr": average_precision_score(yte, s),
                  "lift_top10": l10, "precision_top10": p10, "lift_top20": l20,
                  "precision_top20": p20, "compradores_capturados_top20": c20})
met = pd.DataFrame(filas).round(3)
met.to_csv("resultados/metricas.csv", index=False)
print(met.to_string(index=False))

# Valor capturado: ingreso real de la ventana de prueba en el 20 % mejor puntuado
mejor = "Gradient boosting"
k = int(len(test) * 0.2)
top = np.argsort(-scores[mejor])[:k]
print("ingreso futuro capturado top20:", round(test.ingreso_futuro.values[top].sum() / test.ingreso_futuro.sum(), 3))

# Importancia por permutación (conjunto de prueba, AUC)
pi = permutation_importance(modelos[mejor], Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=VARIABLES).sort_values(ascending=False)
imp.round(4).to_csv("resultados/importancia_variables.csv", header=["caida_auc"])
print(imp.round(4).head(8))

# Segmentos por probabilidad
test = test.assign(prob=scores[mejor])
test["decil"] = pd.qcut(test.prob.rank(method="first"), 10, labels=False) + 1
dec = test.groupby("decil").agg(clientes=("compro", "size"), prob_media=("prob", "mean"), tasa_real=("compro", "mean"),
                                ingreso_futuro=("ingreso_futuro", "sum")).round(3)
dec.to_csv("resultados/deciles.csv")
print(dec)
test[["prob", "compro"]].to_csv("resultados/predicciones_prueba.csv")

# Gráficos
AZUL, GRIS = "#2A6F97", "#5B6B7B"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for (n, s), c in zip(scores.items(), ["#9AA8B5", "#A9C7DB", AZUL]):
    fpr, tpr, _ = roc_curve(yte, s)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC {roc_auc_score(yte, s):.2f})")
ax.plot([0, 1], [0, 1], ls="--", color="#CCCCCC")
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curva ROC en el período de prueba", loc="left", color=GRIS); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("graficos/roc.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(dec.index, 100 * dec.tasa_real, color=AZUL)
ax.axhline(100 * yte.mean(), color=GRIS, ls="--", lw=1)
ax.text(10.4, 100 * yte.mean() + 1.5, f"promedio {100 * yte.mean():.0f} %", ha="right", color=GRIS, fontsize=9)
ax.set_xticks(range(1, 11)); ax.set_xlabel("Decil de probabilidad predicha (1 = menor, 10 = mayor)")
ax.set_ylabel("% que sí compró"); ax.set_title("Tasa real de recompra por decil", loc="left", color=GRIS)
fig.tight_layout(); fig.savefig("graficos/deciles.png", dpi=130); plt.close(fig)

top8 = imp.head(8)[::-1]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(top8.index, top8.values, color=AZUL)
ax.set_xlabel("Caída del AUC al permutar la variable"); ax.set_title("Variables más importantes", loc="left", color=GRIS)
fig.tight_layout(); fig.savefig("graficos/importancia.png", dpi=130); plt.close(fig)
