"""
SupplyPredict - Prophet Training Script
Genera predictions.pkl y reorder_points.csv en el formato
exacto que espera main.py (supplypredict-api).

Output:
  data/predictions.pkl     → {product_id: {dates, yhat, yhat_lower, yhat_upper}}
  data/reorder_points.csv  → product_id, current_stock, reorder_point, ...
"""
import pandas as pd
import numpy as np
import pickle
import json
import time
import sys
import os
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

DATA_FILE  = os.getenv("DATA_FILE",  "data/df_supply_clean.csv")
PRED_FILE  = os.getenv("PRED_FILE",  "data/predictions.pkl")
RP_FILE    = os.getenv("RP_FILE",    "data/reorder_points.csv")
LOG_FILE   = os.getenv("LOG_FILE",   "data/training_log.jsonl")

FORECAST_WEEKS = 30   # semanas hacia adelante
MIN_ROWS       = 20   # mínimo de filas por producto para entrenar

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
log(f"Cargando {DATA_FILE}...")
df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
log(f"  {len(df):,} filas | {df['ID_Producto'].nunique()} productos")

products   = df["ID_Producto"].unique()
total      = len(products)
predictions = {}
reorder_rows = []
skipped    = []

start = time.time()

# ── 2. Entrenar Prophet por producto ─────────────────────────────────────────
for i, pid in enumerate(products, 1):
    pdata = df[df["ID_Producto"] == pid][["ds", "y"]].copy().sort_values("ds")

    if len(pdata) < MIN_ROWS:
        skipped.append(pid)
        log(f"  [{i}/{total}] SKIP {pid} ({len(pdata)} filas)")
        continue

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
        )
        m.fit(pdata)

        future   = m.make_future_dataframe(periods=FORECAST_WEEKS, freq="W")
        forecast = m.predict(future)
        fut_only = forecast.tail(FORECAST_WEEKS)

        # Formato exacto que espera main.py
        predictions[pid] = {
            "dates":      fut_only["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "yhat":       fut_only["yhat"].clip(lower=0).round(2).tolist(),
            "yhat_lower": fut_only["yhat_lower"].clip(lower=0).round(2).tolist(),
            "yhat_upper": fut_only["yhat_upper"].clip(lower=0).round(2).tolist(),
        }

        # ── Reorder point ─────────────────────────────────────────────────────
        prod_full = df[df["ID_Producto"] == pid]
        current_stock = float(prod_full["Stock_Disponible"].iloc[-1])
        lead_time     = float(prod_full["Lead_Time_Dias"].mean())
        avg_daily     = float(pdata["y"].mean())
        std_daily     = float(pdata["y"].std()) if len(pdata) > 1 else 0.0
        safety_stock  = 1.5 * std_daily

        reorder_point = (avg_daily * lead_time) + safety_stock

        # Cobertura en días
        days_coverage = (current_stock / avg_daily) if avg_daily > 0 else (
            9999 if current_stock > 0 else 0
        )
        days_coverage = max(days_coverage, 0)

        # Status
        if current_stock < 0 or days_coverage < lead_time:
            status = "CRÍTICO"
        elif days_coverage < lead_time * 1.5:
            status = "URGENTE"
        else:
            status = "NORMAL"

        # Cantidad recomendada de compra (30 días cobertura)
        qty_recommended = max(reorder_point - current_stock + avg_daily * 30, 0)

        # forecast próximas 4 semanas (~30 días)
        forecast_next_30d = sum(predictions[pid]["yhat"][:4])

        reorder_rows.append({
            "product_id":      pid,
            "current_stock":   round(current_stock, 0),
            "reorder_point":   round(reorder_point, 0),
            "lead_time_days":  round(lead_time, 1),
            "avg_daily_sales": round(avg_daily, 2),
            "safety_stock":    round(safety_stock, 0),
            "days_coverage":   round(days_coverage, 1),
            "status":          status,
            "qty_recommended": round(qty_recommended, 0),
            "forecast_next_30d": round(forecast_next_30d, 0),
        })

        elapsed = time.time() - start
        log(f"  [{i}/{total}] OK {pid} | status={status} | {elapsed:.0f}s")

        # Escribir log en tiempo real
        with open(LOG_FILE, "a") as lf:
            lf.write(json.dumps({
                "product_id": pid, "status": status,
                "current_stock": current_stock, "days_coverage": round(days_coverage,1),
                "lead_time": round(lead_time,1), "ts": datetime.now().isoformat()
            }) + "\n")

    except Exception as e:
        log(f"  [{i}/{total}] ERROR {pid}: {e}")

# ── 3. Guardar outputs ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

with open(PRED_FILE, "wb") as f:
    pickle.dump(predictions, f)
log(f"✓ predictions.pkl guardado ({len(predictions)} productos)")

df_rp = pd.DataFrame(reorder_rows)
df_rp.to_csv(RP_FILE, index=False)
log(f"✓ reorder_points.csv guardado ({len(df_rp)} filas)")

# ── 4. Resumen ────────────────────────────────────────────────────────────────
elapsed_total = time.time() - start
n_crit  = sum(1 for r in reorder_rows if r["status"] == "CRÍTICO")
n_urg   = sum(1 for r in reorder_rows if r["status"] == "URGENTE")
n_norm  = sum(1 for r in reorder_rows if r["status"] == "NORMAL")

log("=" * 55)
log(f"ENTRENAMIENTO COMPLETADO en {elapsed_total:.0f}s")
log(f"  Productos entrenados : {len(predictions)}/{total}")
log(f"  Saltados (<{MIN_ROWS} filas): {len(skipped)}")
log(f"  CRÍTICO : {n_crit}")
log(f"  URGENTE : {n_urg}")
log(f"  NORMAL  : {n_norm}")
log("=" * 55)

if __name__ == "__main__":
    sys.exit(0)
