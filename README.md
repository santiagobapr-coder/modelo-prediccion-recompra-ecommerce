# Modelo predictivo de recompra a 90 días (eCommerce)

Modelo que estima, para cada cliente de una tienda online, **la probabilidad de que vuelva a comprar en los próximos 90 días**, para priorizar campañas de retención y reactivación. Hecho con Python (pandas, scikit-learn) y datos públicos, con validación fuera de tiempo para evitar fugas de información.

## Problema y enfoque

- **Pregunta:** ¿qué clientes comprarán en los próximos 90 días y cuáles están en riesgo de no volver?
- **Etiqueta:** 1 si el cliente tiene al menos una factura en los 90 días posteriores a la fecha de corte.
- **Variables (13), calculadas solo con el historial anterior al corte:** recencia, frecuencia, monto total, ticket promedio, antigüedad, días promedio entre compras, productos y unidades por factura, facturas e ingresos de los últimos 90 y 180 días, devoluciones y país (Reino Unido o no).
- **Validación fuera de tiempo:** el entrenamiento usa 6 fechas de corte entre marzo de 2010 y junio de 2011 (21.729 filas). La prueba usa el corte del 10 de septiembre de 2011, con ventana hasta el 9 de diciembre de 2011 (5.256 clientes), posterior a todo el entrenamiento.
- **Modelos:** un baseline simple (ordenar por menor recencia), regresión logística y gradient boosting (`HistGradientBoostingClassifier`).

## Datos

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI, licencia CC BY 4.0): transacciones reales entre diciembre de 2009 y diciembre de 2011. Tras la limpieza (duplicados, códigos que no son productos, devoluciones separadas, solo clientes identificados) quedan 36.594 facturas de 5.852 clientes.

## Cómo reproducirlo

```bash
pip install -r requirements.txt
# Descargar online_retail_II.xlsx desde el enlace de arriba (dentro del .zip) y dejarlo en esta carpeta
python build_dataset.py online_retail_II.xlsx   # genera facturas.csv y devoluciones.csv
python model.py                                 # entrena, evalúa y genera resultados/ y graficos/
```

## Resultados (conjunto de prueba, 5.256 clientes; 43,5 % compró en la ventana)

| Modelo | AUC | AUC-PR | Precisión en el 20 % mejor puntuado | Compradores capturados en ese 20 % |
|---|---|---|---|---|
| Baseline (menor recencia) | 0,762 | 0,685 | 75,1 % | 34,5 % |
| Regresión logística | 0,792 | 0,768 | 84,7 % | 38,9 % |
| Gradient boosting | **0,800** | **0,776** | **85,0 %** | **39,0 %** |

- El modelo mejora al baseline de recencia (AUC de 0,76 a 0,80), pero la recencia por sí sola ya es un predictor fuerte; la ganancia viene de combinarla con el ritmo de compra y el monto.
- Del 10 % de clientes con mayor probabilidad, el 93,1 % compró; del 10 % con menor probabilidad, solo el 8,6 %.
- El 20 % mejor puntuado concentra el **69,9 % del ingreso** que se generó en la ventana de prueba.
- **Variables más importantes** (caída del AUC al permutarlas): recencia (0,038), días entre compras (0,022), monto total (0,008), facturas de los últimos 180 días (0,007) y frecuencia (0,005).

![Curva ROC](graficos/roc.png)
![Recompra por decil](graficos/deciles.png)
![Importancia de variables](graficos/importancia.png)

## Cómo se usaría en una decisión de negocio

1. **No gastar incentivos en el decil 10.** Ya compra el 93 % de las veces: un descuento ahí probablemente solo reduzca margen (el dataset no trae información de promociones para comprobarlo).
2. **Concentrar campañas de retención en los deciles medios (4 a 7),** donde la recompra real va del 32 % al 52 % y un mensaje a tiempo puede inclinar la decisión.
3. **Campañas de reactivación de bajo costo en los deciles 1 a 3** (probabilidad real entre 9 % y 21 %), con un tope de inversión por cliente.
4. **Recalcular el ranking cada mes** con la fecha de corte actualizada.

Estas acciones son una propuesta basada en los datos del dataset público; no se probaron con una campaña real.

## Limitaciones

- La tasa de recompra cambia mucho entre cortes (32 % a 60 %) por la estacionalidad. La prueba incluye el pico de noviembre, y por eso el modelo **subestima** las probabilidades absolutas (por ejemplo, 87 % predicho vs 93 % real en el decil 10). Sirve para **ordenar** clientes, no como probabilidad calibrada.
- Se evalúa una sola ventana de prueba, y los mismos clientes aparecen en varios cortes de entrenamiento.
- Los hiperparámetros se fijaron a mano, sin búsqueda exhaustiva.
- El dataset no trae canal de adquisición, costos ni margen, por lo que no se estiman ROAS ni atribución.

## Archivos

- `build_dataset.py`: limpieza y tablas a nivel de factura.
- `model.py`: construcción de variables, entrenamiento, evaluación y gráficos.
- `graficos/`: imágenes usadas en este README.
