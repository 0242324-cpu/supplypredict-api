"""
SupplyPredict - LightGBM Training Script
Reemplaza Prophet con Gradient Boosting para mejor performance.

Output (formato idéntico a main.py):
  data/predictions.pkl  → {product_id: {dates, yhat, yhat_lower, yhat_upper}}
  data/reorder_points.csv → product_id, current_stock, reorder_point, ...
  data/model_metrics.csv  → product_id, n_obs, mae, rmse, mape, ...
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
import time
import sys
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE   = os.getenv("DATA_FILE",   "data/df_supply_clean.csv")
PRED_FILE   = os.getenv("PRED_FILE",   "data/predictions.pkl")
RP_FILE     = os.getenv("RP_FILE",     "data/reorder_points.csv")
METRICS_FILE= os.getenv("METRICS_FILE","data/model_metrics.csv")
LOG_FILE    = os.getenv("LOG_FILE",    "data/training_log.jsonl")

FORECAST_WEEKS = 8      # semanas hacia adelante (balance entre precisión y visualización)
MIN_ROWS       = 30     # mínimo de filas por producto
TEST_WEEKS     = 8      # semanas reservadas para test
SAFETY_FACTOR  = 1.5    # factor de seguridad para safety stock (z ≈ 1.5 → ~93%)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def create_features(df_prod):
    """
    Crea features temporales, lag y rolling para un producto.
    Recibe DataFrame con columnas: ds, y, Stock_Disponible, Cantidad_Comprada, Lead_Time_Dias
    """
    df = df_prod.sort_values("ds").reset_index(drop=True)

    # ── Lag features (en unidades de observación = semanas) ──
    df["lag_1"]  = df["y"].shift(1)
    df["lag_2"]  = df["y"].shift(2)
    df["lag_4"]  = df["y"].shift(4)
    df["lag_8"]  = df["y"].shift(8)
    df["lag_13"] = df["y"].shift(13)   # ~trimestre
    df["lag_26"] = df["y"].shift(26)   # ~semestre
    df["lag_52"] = df["y"].shift(52)   # ~año

    # ── Rolling statistics ──
    df["roll_mean_4"]  = df["y"].shift(1).rolling(4,  min_periods=2).mean()
    df["roll_std_4"]   = df["y"].shift(1).rolling(4,  min_periods=2).std()
    df["roll_mean_8"]  = df["y"].shift(1).rolling(8,  min_periods=4).mean()
    df["roll_mean_12"] = df["y"].shift(1).rolling(12, min_periods=6).mean()
    df["roll_mean_26"] = df["y"].shift(1).rolling(26, min_periods=8).mean()

    # ── Trend features ──
    df["diff_1"] = df["y"].diff(1)
    df["diff_4"] = df["y"].diff(4)

    # ── Ratio features ──
    rm4 = df["roll_mean_4"]
    rm12 = df["roll_mean_12"]
    df["ratio_4_12"] = np.where(rm12 > 0, rm4 / rm12, 1.0)

    # ── Calendar features ──
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"]        = df["ds"].dt.month
    df["quarter"]      = df["ds"].dt.quarter

    # Fourier terms for yearly seasonality (period=52 weeks)
    woy = df["week_of_year"].astype(float)
    df["sin_52"]  = np.sin(2 * np.pi * woy / 52)
    df["cos_52"]  = np.cos(2 * np.pi * woy / 52)
    df["sin_26"]  = np.sin(2 * np.pi * woy / 26)
    df["cos_26"]  = np.cos(2 * np.pi * woy / 26)

    # ── External features (regresores) ──
    df["stock_disponible"]  = df["Stock_Disponible"]
    df["qty_comprada_lag1"] = df["Cantidad_Comprada"].shift(1)
    df["qty_comprada_lag2"] = df["Cantidad_Comprada"].shift(2)
    df["lead_time"]         = df["Lead_Time_Dias"]

    # ── Volatility (CV rolling) ──
    rm = df["y"].shift(1).rolling(12, min_periods=6).mean()
    rs = df["y"].shift(1).rolling(12, min_periods=6).std()
    df["cv_12"] = np.where(rm > 0, rs / rm, 0)

    return df


FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52",
    "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_mean_12", "roll_mean_26",
    "diff_1", "diff_4", "ratio_4_12",
    "week_of_year", "month", "quarter",
    "sin_52", "cos_52", "sin_26", "cos_26",
    "stock_disponible", "qty_comprada_lag1", "qty_comprada_lag2", "lead_time",
    "cv_12",
]


# ── LightGBM Hyperparameters ─────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":       "regression",
    "metric":          "mae",
    "boosting_type":   "gbdt",
    "n_estimators":    400,
    "learning_rate":   0.05,
    "max_depth":       7,
    "num_leaves":      31,
    "min_data_in_leaf": 10,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}


# ── Recursive Multi-Step Forecast ─────────────────────────────────────────────

def recursive_forecast(model, last_known, df_prod, n_steps, features):
    """
    Genera pronóstico recursivo: predice semana t+1, usa como lag para t+2, etc.
    last_known: última fila conocida con features
    df_prod: DataFrame completo del producto (para promedios)
    """
    preds = []
    current = last_known.copy()

    # Valores promedio para features que no podemos calcular dinámicamente
    avg_stock = df_prod["Stock_Disponible"].iloc[-5:].mean()
    avg_qty   = df_prod["Cantidad_Comprada"].iloc[-5:].mean()
    lt_val    = current.get("lead_time", df_prod["Lead_Time_Dias"].iloc[-1])
    last_date = df_prod["ds"].max()

    for step in range(n_steps):
        # Predecir
        X_step = pd.DataFrame([current])[features]
        pred = max(model.predict(X_step)[0], 0)
        preds.append(pred)

        # Actualizar lags para el siguiente paso
        current["lag_52"] = current.get("lag_26", pred)  # simplificado
        current["lag_26"] = current.get("lag_13", pred)
        current["lag_13"] = current.get("lag_8", pred)
        current["lag_8"]  = current.get("lag_4", pred)
        current["lag_4"]  = current.get("lag_2", pred)
        current["lag_2"]  = current.get("lag_1", pred)
        current["lag_1"]  = pred

        # Rolling stats update (aproximado)
        recent = preds[-4:] if len(preds) >= 4 else preds
        current["roll_mean_4"]  = np.mean(recent)
        current["roll_std_4"]   = np.std(recent) if len(recent) > 1 else 0
        current["roll_mean_8"]  = current.get("roll_mean_8", np.mean(recent)) * 0.8 + pred * 0.2
        current["roll_mean_12"] = current.get("roll_mean_12", np.mean(recent)) * 0.85 + pred * 0.15
        current["roll_mean_26"] = current.get("roll_mean_26", np.mean(recent)) * 0.9 + pred * 0.1

        # Diff & ratio
        current["diff_1"] = pred - preds[-2] if len(preds) >= 2 else 0
        current["diff_4"] = pred - (preds[-4] if len(preds) >= 4 else pred)
        rm4  = current["roll_mean_4"]
        rm12 = current["roll_mean_12"]
        current["ratio_4_12"] = rm4 / rm12 if rm12 > 0 else 1.0

        # Calendar (avanzar semana)
        future_date = last_date + timedelta(weeks=step + 1)
        woy = future_date.isocalendar()[1]
        current["week_of_year"] = woy
        current["month"]   = future_date.month
        current["quarter"] = (future_date.month - 1) // 3 + 1
        current["sin_52"]  = np.sin(2 * np.pi * woy / 52)
        current["cos_52"]  = np.cos(2 * np.pi * woy / 52)
        current["sin_26"]  = np.sin(2 * np.pi * woy / 26)
        current["cos_26"]  = np.cos(2 * np.pi * woy / 26)

        # External (mantener últimos conocidos)
        current["stock_disponible"]  = avg_stock
        current["qty_comprada_lag1"] = avg_qty
        current["qty_comprada_lag2"] = avg_qty
        current["lead_time"]         = lt_val

        # CV (mantener último)
        # current["cv_12"] ya está

    return preds


# ── MAPE seguro ───────────────────────────────────────────────────────────────

def safe_mape(y_true, y_pred):
    """MAPE robusto que evita divisiones por cero."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return 999.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

log(f"Cargando {DATA_FILE}...")
df = pd.read_csv(DATA_FILE, parse_dates=["ds"])
log(f"  {len(df):,} filas | {df['ID_Producto'].nunique()} productos")

products = df["ID_Producto"].unique()
total = len(products)

predictions  = {}
reorder_rows = []
metrics_rows = []
skipped      = []
start        = time.time()

# Limpiar log
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# ── Entrenar LightGBM por producto ───────────────────────────────────────────

for i, pid in enumerate(products, 1):
    pdata = df[df["ID_Producto"] == pid].copy()

    if len(pdata) < MIN_ROWS:
        skipped.append(pid)
        log(f"  [{i}/{total}] SKIP {pid} ({len(pdata)} filas)")
        continue

    try:
        # ── Feature engineering ───────────────────────────────────────────
        df_feat = create_features(pdata)
        df_feat = df_feat.dropna(subset=["lag_4", "roll_mean_4"])  # mínimo necesario

        if len(df_feat) < MIN_ROWS:
            skipped.append(pid)
            log(f"  [{i}/{total}] SKIP {pid} (post-features: {len(df_feat)} filas)")
            continue

        # ── Train / Test split (temporal) ─────────────────────────────────
        split_idx = max(len(df_feat) - TEST_WEEKS, int(len(df_feat) * 0.7))
        train_df = df_feat.iloc[:split_idx]
        test_df  = df_feat.iloc[split_idx:]

        X_train = train_df[FEATURES]
        y_train = train_df["y"]
        X_test  = test_df[FEATURES]
        y_test  = test_df["y"]

        # ── Entrenar modelo ───────────────────────────────────────────────
        model = lgb.LGBMRegressor(**LGB_PARAMS)

        eval_set = [(X_test, y_test)] if len(X_test) >= 3 else None
        callbacks = [lgb.early_stopping(30, verbose=False)] if eval_set else None

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        # ── Evaluar en test ───────────────────────────────────────────────
        y_pred_test = np.clip(model.predict(X_test), 0, None)
        mae  = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape = safe_mape(y_test.values, y_pred_test)

        avg_sales = float(pdata["y"].mean())
        mae_pct = (mae / avg_sales * 100) if avg_sales > 0 else 999

        # ── Forecast futuro ───────────────────────────────────────────────
        # Usar última fila con features como punto de partida
        last_row = df_feat.iloc[-1][FEATURES].to_dict()
        last_row["cv_12"] = float(df_feat["cv_12"].iloc[-1]) if not np.isnan(df_feat["cv_12"].iloc[-1]) else 0

        future_preds = recursive_forecast(
            model, last_row, pdata, FORECAST_WEEKS, FEATURES
        )

        # Generar fechas futuras
        last_date = pdata["ds"].max()
        future_dates = [(last_date + timedelta(weeks=w+1)).strftime("%Y-%m-%d")
                        for w in range(FORECAST_WEEKS)]

        # Intervalos de confianza (basados en residuos)
        residuals = y_test.values - y_pred_test
        std_residual = np.std(residuals) if len(residuals) > 1 else np.std(pdata["y"]) * 0.2

        yhat       = np.array(future_preds)
        yhat_lower = np.clip(yhat - 1.96 * std_residual, 0, None)
        yhat_upper = yhat + 1.96 * std_residual

        predictions[pid] = {
            "dates":      future_dates,
            "yhat":       yhat.round(2).tolist(),
            "yhat_lower": yhat_lower.round(2).tolist(),
            "yhat_upper": yhat_upper.round(2).tolist(),
        }

        # ── Reorder point ─────────────────────────────────────────────────
        current_stock = float(pdata["Stock_Disponible"].iloc[-1])

        # Lead time: probar ambos métodos, usar el que tenga más sentido
        lt_last = float(pdata["Lead_Time_Dias"].iloc[-1])
        lt_mean = float(pdata["Lead_Time_Dias"].mean())

        # Usar último valor si es razonable (>0), sino promedio
        lead_time = lt_last if lt_last > 0 else lt_mean
        # Guardar ambos para comparar después
        lead_time = max(lead_time, 1)  # mínimo 1 día

        avg_daily = avg_sales / 7  # convertir ventas semanales a diarias
        std_daily = float(pdata["y"].std()) / 7 if len(pdata) > 1 else 0

        safety_stock = SAFETY_FACTOR * std_daily * np.sqrt(lead_time)
        reorder_point = (avg_daily * lead_time) + safety_stock

        # Cobertura en días
        if avg_daily > 0:
            days_coverage = current_stock / avg_daily
        else:
            days_coverage = 9999 if current_stock > 0 else 0
        days_coverage = max(days_coverage, 0)

        # Status
        if current_stock < 0 or days_coverage < lead_time:
            status = "CRÍTICO"
        elif days_coverage < lead_time * 1.5:
            status = "URGENTE"
        else:
            status = "NORMAL"

        # Cantidad recomendada (30 días cobertura)
        qty_recommended = max(reorder_point - current_stock + avg_daily * 30, 0)

        # Forecast próximas ~4 semanas
        forecast_next_30d = sum(future_preds[:min(4, len(future_preds))])

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

        # ── Métricas ──────────────────────────────────────────────────────
        metrics_rows.append({
            "product_id": pid,
            "n_obs":      len(pdata),
            "mae":        round(mae, 2),
            "rmse":       round(rmse, 2),
            "mape":       round(mape, 2),
            "avg_sales":  round(avg_sales, 2),
            "mae_pct":    round(mae_pct, 2),
        })

        elapsed = time.time() - start
        grade = "A" if mape < 20 else "B" if mape < 50 else "C" if mape < 100 else "D"
        log(f"  [{i}/{total}] OK {pid} | MAPE={mape:.1f}% ({grade}) | {elapsed:.0f}s")

        # Log en tiempo real
        with open(LOG_FILE, "a") as lf:
            lf.write(json.dumps({
                "product_id": pid, "status": status, "mape": round(mape, 1),
                "grade": grade, "current_stock": current_stock,
                "days_coverage": round(days_coverage, 1),
                "lead_time": round(lead_time, 1),
                "ts": datetime.now().isoformat(),
            }) + "\n")

    except Exception as e:
        log(f"  [{i}/{total}] ERROR {pid}: {e}")
        import traceback
        traceback.print_exc()


# ── Guardar outputs ───────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

with open(PRED_FILE, "wb") as f:
    pickle.dump(predictions, f)
log(f"✓ predictions.pkl guardado ({len(predictions)} productos)")

df_rp = pd.DataFrame(reorder_rows)
df_rp.to_csv(RP_FILE, index=False)
log(f"✓ reorder_points.csv guardado ({len(df_rp)} filas)")

df_met = pd.DataFrame(metrics_rows)
df_met.to_csv(METRICS_FILE, index=False)
log(f"✓ model_metrics.csv guardado ({len(df_met)} filas)")


# ── Resumen ───────────────────────────────────────────────────────────────────

elapsed_total = time.time() - start
n_crit = sum(1 for r in reorder_rows if r["status"] == "CRÍTICO")
n_urg  = sum(1 for r in reorder_rows if r["status"] == "URGENTE")
n_norm = sum(1 for r in reorder_rows if r["status"] == "NORMAL")

mapes = [m["mape"] for m in metrics_rows if m["mape"] < 500]
grades = {"A": 0, "B": 0, "C": 0, "D": 0}
for m in metrics_rows:
    g = "A" if m["mape"] < 20 else "B" if m["mape"] < 50 else "C" if m["mape"] < 100 else "D"
    grades[g] += 1

log("=" * 60)
log(f"  ENTRENAMIENTO LightGBM COMPLETADO en {elapsed_total:.0f}s")
log(f"  Modelo            : LightGBM (Gradient Boosting)")
log(f"  Features           : {len(FEATURES)}")
log(f"  Productos entrenados: {len(predictions)}/{total}")
log(f"  Saltados (<{MIN_ROWS} filas) : {len(skipped)}")
log(f"")
log(f"  MÉTRICAS:")
log(f"    MAPE mediana     : {sorted(mapes)[len(mapes)//2]:.1f}%" if mapes else "    Sin métricas")
log(f"    MAPE promedio    : {np.mean(mapes):.1f}%" if mapes else "")
log(f"    Grade A (<20%)   : {grades['A']}")
log(f"    Grade B (<50%)   : {grades['B']}")
log(f"    Grade C (<100%)  : {grades['C']}")
log(f"    Grade D (≥100%)  : {grades['D']}")
log(f"")
log(f"  STATUS:")
log(f"    CRÍTICO          : {n_crit}")
log(f"    URGENTE          : {n_urg}")
log(f"    NORMAL           : {n_norm}")
log("=" * 60)

if __name__ == "__main__":
    sys.exit(0)
