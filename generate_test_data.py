"""
SupplyPredict — generate_test_data.py
======================================
Genera dos archivos CSV para validar el pipeline de reentrenamiento:

  1. PROBANDO_PIPELINE_SINTETICO.csv
     → Datos sintéticos con pico de 300% para los top 10 productos.
       Al subirlo y reentrenar, el modelo debe reflejar la tendencia alcista.

  2. ROLLBACK_ESTADO_ACTUAL.csv
     → Copia del df_supply_clean.csv original para restaurar el estado.

USO:
  python generate_test_data.py

SALIDA en carpeta data/:
  data/PROBANDO_PIPELINE_SINTETICO.csv     (para subir en la pestaña Carga de datos)
  data/df_supply_clean_SINTETICO.csv       (reemplaza df_supply_clean.csv para que train.py lo lea)
  data/ROLLBACK_ESTADO_ACTUAL.csv          (backup del original — restaurar después de la prueba)
  data/INSTRUCCIONES_TEST.txt              (guía paso a paso)
"""

import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
ORIGINAL_CSV   = DATA_DIR / "df_supply_clean.csv"
SPIKE_FACTOR   = 3.0     # 300% del promedio = pico muy visible
N_TOP          = 10       # Top 10 productos
SPIKE_WEEKS    = 2        # 2 semanas de datos sintéticos
SEED           = 42

np.random.seed(SEED)

# ─── Archivos de salida ──────────────────────────────────────────────
OUT_SINTETICO_UPLOAD  = DATA_DIR / "PROBANDO_PIPELINE_SINTETICO.csv"
OUT_SINTETICO_SUPPLY  = DATA_DIR / "df_supply_clean_SINTETICO.csv"
OUT_ROLLBACK          = DATA_DIR / "ROLLBACK_ESTADO_ACTUAL.csv"
OUT_INSTRUCCIONES     = DATA_DIR / "INSTRUCCIONES_TEST.txt"


def main():
    print("=" * 65)
    print("  SupplyPredict — Generador de datos de prueba")
    print("=" * 65)

    # ──────────────────────────────────────────────────────────────────
    # 1. Leer datos originales
    # ──────────────────────────────────────────────────────────────────
    if not ORIGINAL_CSV.exists():
        print(f"[ERROR] No se encontró {ORIGINAL_CSV}")
        print("        Ejecuta desde la raíz del repo supplypredict-api")
        return

    df = pd.read_csv(ORIGINAL_CSV, parse_dates=["ds"])
    print(f"\n[original] {df.shape[0]:,} filas | {df['ID_Producto'].nunique()} productos")
    print(f"[original] Rango: {df['ds'].min().date()} → {df['ds'].max().date()}")
    last_date = df["ds"].max()

    # ──────────────────────────────────────────────────────────────────
    # 2. Identificar top 10 productos por ventas totales
    # ──────────────────────────────────────────────────────────────────
    top_products = (
        df.groupby("ID_Producto")["y"]
        .sum()
        .nlargest(N_TOP)
        .index.tolist()
    )

    print(f"\n[top10] Productos seleccionados para pico de {SPIKE_FACTOR*100:.0f}%:")
    product_stats = {}
    for pid in top_products:
        sub = df[df["ID_Producto"] == pid]
        stats = {
            "avg_y":        sub["y"].mean(),
            "std_y":        sub["y"].std(),
            "last_stock":   sub["Stock_Disponible"].iloc[-1],
            "last_lt":      sub["Lead_Time_Dias"].iloc[-1],
            "last_compra":  sub["Cantidad_Comprada"].iloc[-1],
        }
        product_stats[pid] = stats
        print(f"  {pid:<30} avg={stats['avg_y']:>12,.0f}  pico={stats['avg_y']*SPIKE_FACTOR:>12,.0f}")

    # ──────────────────────────────────────────────────────────────────
    # 3. Generar filas sintéticas (formato df_supply_clean.csv)
    # ──────────────────────────────────────────────────────────────────
    #
    # Lógica del pico:
    #   - 2 semanas después de last_date (secuenciales)
    #   - Demanda = avg * SPIKE_FACTOR + ruido normal (±10%)
    #   - Stock = last_stock (sin cambio, para que el modelo lo detecte
    #     como desabastecimiento por demanda alta)
    #   - Cantidad_Comprada = 0 (sin compras, escenario pesimista)
    #   - Lead_Time_Dias = último valor registrado
    #
    # ¿Por qué 300%?
    #   LightGBM usa lag_1w y lag_2w como features principales.
    #   Un pico de 300% en las últimas 2 semanas crea lags que son 3x
    #   más altos que la media histórica. Esto fuerza al modelo a
    #   proyectar una tendencia alcista en las siguientes 8 semanas,
    #   porque los rolling_mean_4w y expanding_mean se ven arrastrados
    #   hacia arriba por estos 2 puntos extremos.
    # ──────────────────────────────────────────────────────────────────

    synthetic_rows = []
    for week_offset in range(1, SPIKE_WEEKS + 1):
        week_date = last_date + timedelta(weeks=week_offset)

        for pid in top_products:
            stats = product_stats[pid]
            # Demanda inflada con ruido ±10%
            spike_demand = stats["avg_y"] * SPIKE_FACTOR * (1 + np.random.normal(0, 0.10))
            spike_demand = max(0, round(spike_demand, 2))

            synthetic_rows.append({
                "ID_Producto":      pid,
                "ds":               week_date,
                "y":                spike_demand,
                "Stock_Disponible": stats["last_stock"],
                "Stock_Minimo":     0.0,
                "Cantidad_Comprada": 0.0,
                "Lead_Time_Dias":   stats["last_lt"],
            })

    df_synthetic = pd.DataFrame(synthetic_rows)
    print(f"\n[sintético] {len(df_synthetic)} filas generadas ({N_TOP} productos × {SPIKE_WEEKS} semanas)")
    print(f"[sintético] Fechas: {df_synthetic['ds'].min().date()} → {df_synthetic['ds'].max().date()}")

    # ──────────────────────────────────────────────────────────────────
    # 4. Crear df_supply_clean_SINTETICO.csv (original + sintético)
    # ──────────────────────────────────────────────────────────────────
    df_combined = pd.concat([df, df_synthetic], ignore_index=True)
    df_combined = (
        df_combined
        .drop_duplicates(subset=["ID_Producto", "ds"], keep="last")
        .sort_values(["ID_Producto", "ds"])
        .reset_index(drop=True)
    )
    df_combined.to_csv(OUT_SINTETICO_SUPPLY, index=False)
    print(f"\n[save] {OUT_SINTETICO_SUPPLY}")
    print(f"       {df_combined.shape[0]:,} filas ({df.shape[0]:,} originales + {len(df_synthetic)} sintéticas)")

    # ──────────────────────────────────────────────────────────────────
    # 5. Crear PROBANDO_PIPELINE_SINTETICO.csv (formato upload-sales)
    # ──────────────────────────────────────────────────────────────────
    # Este es el CSV que el usuario sube via la pestaña "Carga de datos"
    # Formato: CODIGO_PRODUCTO, DESCRIPCION_PRODUCTO, CANTIDAD_TOTAL

    upload_rows = []
    for pid in top_products:
        stats = product_stats[pid]
        total_spike = stats["avg_y"] * SPIKE_FACTOR * SPIKE_WEEKS
        upload_rows.append({
            "CODIGO_PRODUCTO":      pid,
            "DESCRIPCION_PRODUCTO": f"[TEST] Producto {pid} — pico {SPIKE_FACTOR*100:.0f}%",
            "CANTIDAD TOTAL":       round(total_spike, 0),
        })

    df_upload = pd.DataFrame(upload_rows)
    df_upload.to_csv(OUT_SINTETICO_UPLOAD, index=False, encoding="utf-8-sig")
    print(f"[save] {OUT_SINTETICO_UPLOAD}")
    print(f"       {len(df_upload)} productos con ventas infladas al {SPIKE_FACTOR*100:.0f}%")

    # ──────────────────────────────────────────────────────────────────
    # 6. Crear ROLLBACK_ESTADO_ACTUAL.csv (backup del original)
    # ──────────────────────────────────────────────────────────────────
    shutil.copy2(ORIGINAL_CSV, OUT_ROLLBACK)
    print(f"[save] {OUT_ROLLBACK}")
    print(f"       Copia exacta de {ORIGINAL_CSV} ({df.shape[0]:,} filas)")

    # ──────────────────────────────────────────────────────────────────
    # 7. Escribir instrucciones de uso
    # ──────────────────────────────────────────────────────────────────
    instructions = f"""
╔══════════════════════════════════════════════════════════════════╗
║  INSTRUCCIONES DE PRUEBA DEL PIPELINE DE REENTRENAMIENTO       ║
║  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}                                   ║
╚══════════════════════════════════════════════════════════════════╝

ARCHIVOS GENERADOS:
  1. PROBANDO_PIPELINE_SINTETICO.csv  → Para subir en la pestaña "Carga de datos"
  2. df_supply_clean_SINTETICO.csv    → Reemplaza df_supply_clean.csv para reentrenar
  3. ROLLBACK_ESTADO_ACTUAL.csv       → Backup del original (restaurar después)

═══════════════════════════════════════════════════════════════════
  PASO 1: HACER BACKUP (ANTES DE CUALQUIER COSA)
═══════════════════════════════════════════════════════════════════

  El archivo ROLLBACK_ESTADO_ACTUAL.csv ya fue generado como backup.
  TAMBIÉN descarga el df_supply_clean.csv actual del repositorio:

    git clone https://github.com/0242324-cpu/supplypredict-api.git /tmp/backup
    cp /tmp/backup/data/df_supply_clean.csv ./df_supply_clean_BACKUP_FISICO.csv

  Esto es tu "punto de restauración" real.

═══════════════════════════════════════════════════════════════════
  PASO 2: PROBAR EL PIPELINE SINTÉTICO
═══════════════════════════════════════════════════════════════════

  Opción A — Vía Frontend (recomendado):
  ────────────────────────────────────────
  1. Primero, reemplaza el histórico para que train.py vea los datos nuevos:

     cp data/df_supply_clean.csv data/df_supply_clean_BACKUP.csv
     cp data/df_supply_clean_SINTETICO.csv data/df_supply_clean.csv

  2. Abre https://supplypredict-web.vercel.app → Pestaña "Carga de datos"
  3. Sube PROBANDO_PIPELINE_SINTETICO.csv
  4. Activa el toggle "Reentrenar modelo con estos datos"
  5. Click "🧠 Subir y reentrenar"
  6. Espera ~2.5 minutos (verás el spinner)
  7. Revisa el resultado en el panel

  Opción B — Manual:
  ────────────────────
  1. cp data/df_supply_clean_SINTETICO.csv data/df_supply_clean.csv
  2. python train.py
  3. Revisa data/model_metrics.csv

═══════════════════════════════════════════════════════════════════
  PASO 3: QUÉ OBSERVAR EN EL DASHBOARD
═══════════════════════════════════════════════════════════════════

  Después del reentrenamiento, abre el Dashboard y busca estos cambios:

  A) En la página de MÉTRICAS:
     - La MAPE mediana puede SUBIR (porque el pico artificial es ruido)
     - Los productos top 10 deben mostrar un MAPE diferente al actual
     - Puede haber más productos en Grado D (el pico distorsiona)

  B) En la página de DETALLE de cualquiera de estos productos:
     - {chr(10).join(f'     • {pid}' for pid in top_products)}

     El gráfico de Forecast debe mostrar VALORES MÁS ALTOS que antes,
     porque el modelo aprendió las 2 semanas de ventas al 300%.

  C) En la página DASHBOARD:
     - Las alertas pueden cambiar porque el reorder_point se recalcula
       con la nueva demanda más alta.
     - Más productos en CRÍTICO (stock no alcanza para demanda 3x).

═══════════════════════════════════════════════════════════════════
  PASO 4: ROLLBACK (RESTAURAR ESTADO ORIGINAL)
═══════════════════════════════════════════════════════════════════

  Para volver al estado original:

  1. Restaurar datos:
     cp data/ROLLBACK_ESTADO_ACTUAL.csv data/df_supply_clean.csv

  2. Reentrenar con datos originales:
     python train.py

  3. Push resultados:
     git add data/predictions.pkl data/model_metrics.csv data/reorder_points.csv data/df_supply_clean.csv
     git commit -m "rollback: restaurar datos originales post-prueba"
     git push origin main

  Resultado esperado tras rollback:
     MAPE mediana:  ~41.8%
     Grado A:       ~76
     Grado D:       ~91

═══════════════════════════════════════════════════════════════════
  DATOS DEL PICO SINTÉTICO
═══════════════════════════════════════════════════════════════════

  Factor de inflación:    {SPIKE_FACTOR*100:.0f}% del promedio histórico
  Semanas agregadas:      {SPIKE_WEEKS}
  Productos afectados:    {N_TOP}
  Fechas nuevas:          {(last_date + timedelta(weeks=1)).date()} → {(last_date + timedelta(weeks=SPIKE_WEEKS)).date()}
  Filas sintéticas:       {len(df_synthetic)}

  ¿POR QUÉ EL MODELO LO DETECTA?
  LightGBM v2.0 usa lag_1w y lag_2w como features principales.
  Un pico de 300% en las últimas 2 semanas crea lags que son 3x más
  altos que la media histórica. Los rolling_mean (4w, 8w) y
  expanding_mean se arrastran hacia arriba por estos puntos extremos,
  forzando al modelo a proyectar demanda más alta en el forecast.

═══════════════════════════════════════════════════════════════════
  ⚠ IMPORTANTE
═══════════════════════════════════════════════════════════════════

  Estos datos son SINTÉTICOS para prueba. NO representan ventas reales.
  SIEMPRE haz rollback después de probar.
  El archivo ROLLBACK_ESTADO_ACTUAL.csv es tu seguro de vida.
"""

    with open(OUT_INSTRUCCIONES, "w", encoding="utf-8") as f:
        f.write(instructions)
    print(f"[save] {OUT_INSTRUCCIONES}")

    # ──────────────────────────────────────────────────────────────────
    # 8. Resumen final
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN")
    print("=" * 65)
    print(f"  Archivos generados en {DATA_DIR}/:")
    print(f"    ✓ PROBANDO_PIPELINE_SINTETICO.csv   (upload frontend)")
    print(f"    ✓ df_supply_clean_SINTETICO.csv      (histórico + picos)")
    print(f"    ✓ ROLLBACK_ESTADO_ACTUAL.csv          (backup original)")
    print(f"    ✓ INSTRUCCIONES_TEST.txt              (guía paso a paso)")
    print(f"\n  Productos con pico de {SPIKE_FACTOR*100:.0f}%:")
    for pid in top_products:
        avg = product_stats[pid]["avg_y"]
        print(f"    {pid:<30}  avg: {avg:>12,.0f} → spike: {avg*SPIKE_FACTOR:>12,.0f}")
    print("=" * 65)
    print("  Lee INSTRUCCIONES_TEST.txt para los pasos completos.")
    print("  ⚠ HAZ ROLLBACK DESPUÉS DE PROBAR.")


if __name__ == "__main__":
    main()
