"""
SupplyPredict — train.py v2.0
================================
Senior Staff ML Engineer refactor for Toyo Foods / SupplyPredict.

What changed and why:
  - Tweedie loss (variance_power=1.5): natively handles intermittent demand
    and zero-heavy distributions. Replaces 'regression' which penalises
    zero-errors the same as any other error.
  - IQR capping per product: removes outlier spikes before feature
    computation so rolling/expanding stats are not contaminated.
  - Richer feature set (31 features): adds lag_52w, rolling_median,
    expanding_mean, days_since_last_sale, zero_ratio, cyclic encoding.
  - TimeSeriesSplit CV (honest MAPE): prevents data-leakage that inflated
    reported MAPE under the old train/test split.
  - Z=1.645 reorder formula (95% service level).
  - Per-product error isolation: one bad product never kills the run.

Outputs (data/ folder):
  predictions.pkl        → {product_id: [8 weekly forecast values]}
  reorder_points.csv     → product_id, current_stock, reorder_point,
                           lead_time_days, status
  model_metrics.csv      → product_id, mape, grade

Python: 3.11.9
Max runtime target: 5 min on Render free tier (1 vCPU, 512 MB RAM)
"""

import os
import time
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Paths — script can be run from repo root or data/ sibling
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Try a few common locations for the input CSV
_CANDIDATES = [
    DATA_DIR / "df_supply_clean.csv",
    _SCRIPT_DIR / "df_supply_clean.csv",
    Path("data/df_supply_clean.csv"),
    Path("df_supply_clean.csv"),
]
INPUT_CSV = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

PREDICTIONS_PKL = DATA_DIR / "predictions.pkl"
REORDER_CSV     = DATA_DIR / "reorder_points.csv"
METRICS_CSV     = DATA_DIR / "model_metrics.csv"

FORECAST_WEEKS   = 8       # weeks ahead to forecast
MIN_OBSERVATIONS = 30      # skip products with fewer rows
SERVICE_LEVEL_Z  = 1.645   # 95 % service level
RANDOM_STATE     = 42

# LightGBM hyper-parameters (Tweedie for intermittent/zero-heavy demand)
LGBM_PARAMS: dict = {
    "objective"              : "tweedie",
    "tweedie_variance_power" : 1.5,   # 1→Poisson, 2→Gamma; 1.5 balances both
    "metric"                 : "mape",
    "n_estimators"           : 600,
    "learning_rate"          : 0.04,
    "max_depth"              : 6,
    "num_leaves"             : 31,
    "min_child_samples"      : 8,
    "feature_fraction"       : 0.80,
    "bagging_fraction"       : 0.80,
    "bagging_freq"           : 5,
    "reg_alpha"              : 0.05,
    "reg_lambda"             : 0.1,
    "random_state"           : RANDOM_STATE,
    "n_jobs"                 : -1,
    "verbose"                : -1,
}

# ─────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load and do minimal dtype normalisation."""
    print(f"[load] Reading {INPUT_CSV} ...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {INPUT_CSV}. "
            "Pass the correct path as first CLI argument or set INPUT_CSV."
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["ds"])
    df = df.sort_values(["ID_Producto", "ds"]).reset_index(drop=True)
    df["y"] = df["y"].clip(lower=0)  # demand is non-negative
    print(
        f"[load] {df.shape[0]:,} rows | "
        f"{df['ID_Producto'].nunique()} products | "
        f"{df['ds'].min().date()} → {df['ds'].max().date()}"
    )
    return df


# ─────────────────────────────────────────────
# STEP 2 — OUTLIER CAPPING (IQR per product)
# ─────────────────────────────────────────────

def iqr_cap(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Winsorise demand values using per-product IQR.
    Lower bound is always ≥ 0 (demand cannot be negative).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - factor * iqr)
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _cyclic(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


def build_features(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline for a single product.
    Returns a DataFrame with the original columns plus all engineered features.
    NOTE: Target column is 'y_clean' (IQR-capped, non-negative raw demand).
          Feature lags are log1p-scaled for numeric stability.
    """
    df = df_prod.copy().sort_values("ds").reset_index(drop=True)

    # ── 2.1  IQR capping ──────────────────────────────────────────
    df["y_clean"] = iqr_cap(df["y"])

    # Log1p scale used ONLY for lag/rolling features, not the target
    y_log = np.log1p(df["y_clean"])

    # ── 2.2  Lag features ─────────────────────────────────────────
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f"lag_{lag}w"] = y_log.shift(lag)

    # ── 2.3  Rolling statistics ────────────────────────────────────
    for win, label in [(4, "4w"), (8, "8w"), (12, "12w"), (26, "26w")]:
        s = y_log.shift(1)
        df[f"roll_mean_{label}"]   = s.rolling(win, min_periods=2).mean()
        df[f"roll_std_{label}"]    = s.rolling(win, min_periods=2).std().fillna(0)
        df[f"roll_median_{label}"] = s.rolling(win, min_periods=2).median()

    # ── 2.4  Expanding mean / std (long-term trend baseline) ──────
    s_shift = y_log.shift(1)
    df["expanding_mean"] = s_shift.expanding(min_periods=4).mean()
    df["expanding_std"]  = s_shift.expanding(min_periods=4).std().fillna(0)

    # ── 2.5  Days since last positive sale (intermittency) ────────
    has_sale = (df["y_clean"] > 0).astype(int)
    # cumcount of zeros within each zero-run, shifted by 1 (no data leakage)
    zero_run = (~has_sale.astype(bool)).groupby(has_sale.cumsum()).cumcount()
    df["days_since_last_sale"] = zero_run.shift(1).fillna(0) * 7  # weeks→days approx

    # ── 2.6  Zero-demand ratio (last 12 observations) ─────────────
    df["zero_ratio"] = (df["y_clean"].shift(1) == 0).rolling(12, min_periods=2).mean().fillna(0)

    # ── 2.7  External regressor: purchase quantities ───────────────
    comp_log = np.log1p(df["Cantidad_Comprada"])
    df["compra_lag1"]  = comp_log.shift(1)
    df["compra_lag4"]  = comp_log.shift(4)
    df["compra_roll4"] = comp_log.shift(1).rolling(4, min_periods=1).mean()

    # ── 2.8  Cyclic date encoding ─────────────────────────────────
    df["month_sin"],   df["month_cos"]   = zip(*df["ds"].dt.month.map(lambda m: _cyclic(m, 12)))
    df["quarter_sin"], df["quarter_cos"] = zip(*df["ds"].dt.quarter.map(lambda q: _cyclic(q, 4)))
    week_nums = df["ds"].dt.isocalendar().week.astype(int)
    df["week_sin"], df["week_cos"] = zip(*week_nums.map(lambda w: _cyclic(w, 52)))

    # ── 2.9  Linear trend index ───────────────────────────────────
    df["trend_idx"] = np.arange(len(df), dtype=np.float32)

    # ── 2.10  Product-level volatility (constant feature) ─────────
    mu  = df["y_clean"].mean()
    cv  = df["y_clean"].std() / (mu + 1e-8)
    df["cv"]           = float(cv)
    df["log_mu"]       = float(np.log1p(mu))

    # ── 2.11  Lead time feature ────────────────────────────────────
    df["lead_time_log"] = np.log1p(df["Lead_Time_Dias"])

    return df


# All features the model can use (columns must exist in df after build_features)
FEATURE_COLS: list[str] = [
    # lags
    "lag_1w", "lag_2w", "lag_4w", "lag_8w", "lag_13w", "lag_26w", "lag_52w",
    # rolling
    "roll_mean_4w",   "roll_std_4w",   "roll_median_4w",
    "roll_mean_8w",   "roll_std_8w",   "roll_median_8w",
    "roll_mean_12w",  "roll_std_12w",  "roll_median_12w",
    "roll_mean_26w",  "roll_std_26w",  "roll_median_26w",
    # expanding
    "expanding_mean", "expanding_std",
    # intermittency
    "days_since_last_sale", "zero_ratio",
    # external
    "compra_lag1", "compra_lag4", "compra_roll4",
    # cyclic
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "week_sin", "week_cos",
    # trend / product level
    "trend_idx", "cv", "log_mu", "lead_time_log",
]


# ─────────────────────────────────────────────
# STEP 4 — HELPERS: MAPE, GRADE, LEAD TIME
# ─────────────────────────────────────────────

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE computed only on nonzero actuals to avoid division-by-zero."""
    mask = y_true > 0
    if mask.sum() < 3:
        return 999.0
    err = np.abs((y_true[mask] - np.maximum(y_pred[mask], 0)) / y_true[mask])
    return float(np.mean(err) * 100)


def grade(mape: float) -> str:
    if mape <= 25:  return "A"
    if mape <= 50:  return "B"
    if mape <= 100: return "C"
    return "D"


def extract_lead_time(df_prod: pd.DataFrame) -> float:
    """
    Use the LAST registered positive lead time (per Luis's domain rule).
    Falls back to the mean if no positive value exists.
    """
    positive = df_prod.loc[df_prod["Lead_Time_Dias"] > 0, "Lead_Time_Dias"]
    if len(positive):
        return max(1.0, float(positive.iloc[-1]))
    return max(1.0, float(df_prod["Lead_Time_Dias"].mean()))


# ─────────────────────────────────────────────
# STEP 5 — RECURSIVE FORECAST
# ─────────────────────────────────────────────

def forecast_recursive(
    df_feat: pd.DataFrame,
    model: lgb.LGBMRegressor,
    feat_cols: list[str],
    n_weeks: int,
) -> list[float]:
    """
    Generate n_weeks of weekly demand forecasts by iteratively feeding
    each prediction back as a lag feature for the next step.
    """
    # Working copy: only keep what we need
    y_history = list(df_feat["y_clean"].values)     # raw demand history
    y_log_hist = list(np.log1p(df_feat["y_clean"].values))

    last_date   = df_feat["ds"].max()
    last_trend  = float(df_feat["trend_idx"].iloc[-1])
    cv          = float(df_feat["cv"].iloc[-1])
    log_mu      = float(df_feat["log_mu"].iloc[-1])
    lead_log    = float(df_feat["lead_time_log"].iloc[-1])
    compra_hist = list(np.log1p(df_feat["Cantidad_Comprada"].values))

    preds: list[float] = []

    for step in range(1, n_weeks + 1):
        next_date = last_date + timedelta(weeks=step)
        row: dict[str, float] = {}

        # Lags from combined history + already predicted
        full_y_log = y_log_hist  # we'll append as we go
        n = len(full_y_log)

        for lag in [1, 2, 4, 8, 13, 26, 52]:
            idx = n - lag
            row[f"lag_{lag}w"] = full_y_log[idx] if idx >= 0 else 0.0

        # Rolling stats
        for win, label in [(4, "4w"), (8, "8w"), (12, "12w"), (26, "26w")]:
            window_vals = np.array(full_y_log[max(0, n - win):n])
            row[f"roll_mean_{label}"]   = float(np.mean(window_vals))   if len(window_vals) else 0.0
            row[f"roll_std_{label}"]    = float(np.std(window_vals))    if len(window_vals) > 1 else 0.0
            row[f"roll_median_{label}"] = float(np.median(window_vals)) if len(window_vals) else 0.0

        # Expanding
        row["expanding_mean"] = float(np.mean(full_y_log))
        row["expanding_std"]  = float(np.std(full_y_log))  if n > 1 else 0.0

        # Intermittency
        zero_count = sum(1 for v in y_history[-12:] if v == 0)
        row["zero_ratio"] = zero_count / min(12, len(y_history))
        days_zero = 0
        for v in reversed(y_history):
            if v == 0:
                days_zero += 7
            else:
                break
        row["days_since_last_sale"] = float(days_zero)

        # External: no future purchase orders → carry last known
        n_c = len(compra_hist)
        row["compra_lag1"]  = compra_hist[n_c - 1] if n_c >= 1 else 0.0
        row["compra_lag4"]  = compra_hist[n_c - 4] if n_c >= 4 else 0.0
        row["compra_roll4"] = float(np.mean(compra_hist[max(0, n_c - 4):n_c])) if n_c else 0.0

        # Cyclic
        row["month_sin"],   row["month_cos"]   = _cyclic(next_date.month,   12)
        row["quarter_sin"], row["quarter_cos"] = _cyclic((next_date.month - 1) // 3 + 1, 4)
        row["week_sin"],    row["week_cos"]    = _cyclic(next_date.isocalendar()[1], 52)

        # Structural
        row["trend_idx"]   = last_trend + step
        row["cv"]          = cv
        row["log_mu"]      = log_mu
        row["lead_time_log"] = lead_log

        # Build feature vector (fill missing columns with 0)
        X_next = pd.DataFrame([{c: row.get(c, 0.0) for c in feat_cols}])

        y_pred = float(np.maximum(0.0, model.predict(X_next)[0]))
        preds.append(round(y_pred, 2))

        # Append prediction to history for next lag computation
        y_history.append(y_pred)
        y_log_hist.append(float(np.log1p(y_pred)))
        compra_hist.append(0.0)  # no future purchases known

    return preds


# ─────────────────────────────────────────────
# STEP 6 — TRAIN SINGLE PRODUCT
# ─────────────────────────────────────────────

def train_product(product_id: str, df_prod: pd.DataFrame) -> dict | None:
    """
    Full pipeline for one product.
    Returns a result dict or None if the product is skipped/errors.
    """
    try:
        df_feat = build_features(df_prod)

        # Drop rows that have NaN in the minimum required lag
        df_model = df_feat.dropna(subset=["lag_4w", "y_clean"]).copy()

        if len(df_model) < MIN_OBSERVATIONS:
            return None

        # Intersect requested features with actually available columns
        feat_cols = [c for c in FEATURE_COLS if c in df_model.columns]
        X = df_model[feat_cols].fillna(0).astype(np.float32)
        y = df_model["y_clean"].values.astype(np.float64)   # raw demand → Tweedie

        # ── TimeSeriesSplit cross-validation ────────────────────────
        n_splits = min(3, max(2, len(df_model) // 20))
        test_sz  = max(6, min(12, len(df_model) // 8))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_sz)

        mape_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 20 or len(val_idx) < 3:
                continue
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            cv_model = lgb.LGBMRegressor(**LGBM_PARAMS)
            cv_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            y_pred_val = np.maximum(0, cv_model.predict(X_val))
            fold_mape = safe_mape(y_val, y_pred_val)
            if fold_mape < 999:
                mape_scores.append(fold_mape)

        cv_mape = float(np.mean(mape_scores)) if mape_scores else 999.0

        # ── Final model on all data ──────────────────────────────────
        final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

        # ── 8-week recursive forecast ────────────────────────────────
        predictions_8w = forecast_recursive(df_feat, final_model, feat_cols, FORECAST_WEEKS)

        # ── Reorder point (95 % service level) ──────────────────────
        lead_time  = extract_lead_time(df_prod)
        last_stock = float(df_prod["Stock_Disponible"].iloc[-1])

        # Use last 52 weeks of demand for safety stock calculation
        recent = df_prod["y"].tail(52).clip(lower=0)
        weekly_mean = recent.mean()
        weekly_std  = recent.std() if len(recent) > 1 else weekly_mean * 0.3
        if np.isnan(weekly_std) or weekly_std == 0:
            weekly_std = weekly_mean * 0.30

        # Convert to daily (demand is recorded weekly → aggregate)
        # For a sum of T independent daily demands: Var_T = T * Var_daily
        # => daily_std = weekly_std / sqrt(7)
        daily_mean = weekly_mean / 7.0
        daily_std  = weekly_std  / np.sqrt(7.0)

        # Reorder point = (μ_daily × LT) + (Z × σ_daily × √LT)
        reorder_point = (daily_mean * lead_time) + (SERVICE_LEVEL_Z * daily_std * np.sqrt(lead_time))
        reorder_point = max(0.0, round(reorder_point, 2))

        # Status determination
        if last_stock <= 0 or last_stock < reorder_point:
            status = "CRÍTICO"
        elif last_stock < reorder_point * 1.5:
            status = "URGENTE"
        else:
            status = "NORMAL"

        return {
            "product_id"   : product_id,
            "mape"         : round(cv_mape, 2),
            "grade"        : grade(cv_mape),
            "predictions_8w": predictions_8w,
            "current_stock": round(last_stock, 2),
            "reorder_point": reorder_point,
            "lead_time_days": round(lead_time, 1),
            "status"       : status,
        }

    except Exception as exc:
        print(f"  [WARN] {product_id} failed → {exc}")
        return None


# ─────────────────────────────────────────────
# STEP 7 — MAIN PIPELINE
# ─────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    print("=" * 65)
    print("  SupplyPredict ML Pipeline  v2.0  —  Tweedie LightGBM")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 7.1  Load
    df = load_data()
    products = df["ID_Producto"].unique()
    total    = len(products)
    print(f"\n[pipeline] Training {total} products ...\n")

    # 7.2  Per-product training loop
    results: list[dict] = []
    skipped = 0

    for i, pid in enumerate(products, 1):
        df_prod = df[df["ID_Producto"] == pid]
        res = train_product(pid, df_prod)

        if res:
            results.append(res)
            if i <= 3 or i % 50 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  [{i:>3}/{total}] ✓ {pid:<32} "
                    f"MAPE={res['mape']:>7.1f}%  Grade={res['grade']}  "
                    f"[{elapsed:.0f}s]"
                )
        else:
            skipped += 1

    trained = len(results)
    print(f"\n[pipeline] Finished: {trained} trained | {skipped} skipped\n")

    if not results:
        raise RuntimeError("No products were trained successfully. Check INPUT_CSV path.")

    # 7.3  Build output structures
    predictions: dict[str, list[float]] = {
        r["product_id"]: r["predictions_8w"] for r in results
    }

    df_metrics = pd.DataFrame([
        {"product_id": r["product_id"], "mape": r["mape"], "grade": r["grade"]}
        for r in results
    ])

    df_reorder = pd.DataFrame([
        {
            "product_id"   : r["product_id"],
            "current_stock": r["current_stock"],
            "reorder_point": r["reorder_point"],
            "lead_time_days": r["lead_time_days"],
            "status"       : r["status"],
        }
        for r in results
    ])

    # 7.4  Save outputs
    print("[save] Writing output files ...")

    with open(PREDICTIONS_PKL, "wb") as fh:
        pickle.dump(predictions, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ {PREDICTIONS_PKL}  ({len(predictions)} products, 8-week forecasts)")

    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"  ✓ {METRICS_CSV}  ({len(df_metrics)} rows)")

    df_reorder.to_csv(REORDER_CSV, index=False)
    print(f"  ✓ {REORDER_CSV}  ({len(df_reorder)} rows)")

    # 7.5  Summary report
    elapsed_total = time.time() - t_start
    gc = df_metrics["grade"].value_counts().to_dict()
    mape_med  = df_metrics["mape"].median()
    mape_mean = df_metrics["mape"].mean()

    print("\n" + "=" * 65)
    print("  TRAINING SUMMARY")
    print("=" * 65)
    print(f"  Total runtime   : {elapsed_total:.1f}s  ({elapsed_total / 60:.1f} min)")
    print(f"  Products trained: {trained}")
    print(f"  Products skipped: {skipped}")
    print(f"  MAPE (median)   : {mape_med:.1f}%")
    print(f"  MAPE (mean)     : {mape_mean:.1f}%")
    print(f"  Grade A (≤25%)  : {gc.get('A', 0)}")
    print(f"  Grade B (≤50%)  : {gc.get('B', 0)}")
    print(f"  Grade C (≤100%) : {gc.get('C', 0)}")
    print(f"  Grade D (>100%) : {gc.get('D', 0)}")
    print("=" * 65)
    print("  ✅  All outputs ready. Run swap.py or git push to redeploy.")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Optional: override INPUT_CSV via first CLI argument
    # Usage: python train.py /path/to/df_supply_clean.csv
    if len(sys.argv) > 1:
        INPUT_CSV = Path(sys.argv[1])
        print(f"[cli] Using custom input path: {INPUT_CSV}")

    main()
