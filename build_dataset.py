"""Limpia Online Retail II y deja dos tablas a nivel de factura: facturas.csv y devoluciones.csv."""
import sys
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "online_retail_II.xlsx"
df = pd.concat(pd.read_excel(SRC, sheet_name=None).values(), ignore_index=True)
df.columns = ["factura", "codigo", "descripcion", "cantidad", "fecha", "precio", "cliente", "pais"]
df["factura"] = df["factura"].astype(str)
df["codigo"] = df["codigo"].astype(str)
df = df.drop_duplicates()
df = df[df["codigo"].str.match(r"^\d{5}")]          # solo productos reales
df = df[df["cliente"].notna()]                       # el modelo necesita cliente identificado
df["cliente"] = df["cliente"].astype(int)
df["ingreso"] = (df["cantidad"] * df["precio"]).round(2)

dev = df[df["factura"].str.startswith("C")]
dev = dev.groupby(["cliente", "factura"]).agg(fecha=("fecha", "min"), ingreso=("ingreso", "sum")).reset_index()
dev.to_csv("devoluciones.csv", index=False)

ven = df[~df["factura"].str.startswith("C") & (df["cantidad"] > 0) & (df["precio"] > 0)]
fac = ven.groupby(["cliente", "factura"]).agg(
    fecha=("fecha", "min"), ingreso=("ingreso", "sum"),
    unidades=("cantidad", "sum"), productos=("codigo", "nunique"), pais=("pais", "first")).reset_index()
fac["ingreso"] = fac["ingreso"].round(2)
fac.to_csv("facturas.csv", index=False)
print(f"facturas: {len(fac):,} | clientes: {fac.cliente.nunique():,} | devoluciones: {len(dev):,}")
print(f"ingreso total con cliente: {fac.ingreso.sum():,.2f}")
