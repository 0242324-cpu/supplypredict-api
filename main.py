"""
SupplyPredict API - main.py
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, pickle, pandas as pd, numpy as np
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ── Cargar datos (una vez al importar el módulo) ─────────────────────
def _load_data():
    rp_path   = os.getenv("REORDER_POINTS_PATH", "data/reorder_points.csv")
    pred_path = os.getenv("PREDICTIONS_PATH",    "data/predictions.pkl")
    try:
        df   = pd.read_csv(rp_path)
        with open(pred_path, "rb") as f:
            preds = pickle.load(f)
        print(f"✓ Datos cargados: {len(df)} productos, {len(preds)} forecasts")
        return df, preds
    except Exception as e:
        print(f"✗ Error cargando datos: {e}")
        return None, None

_reorder_df, _predictions = _load_data()

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="SupplyPredict API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ──────────────────────────────────────────────────────────
def get_df():
    if _reorder_df is None:
        raise HTTPException(503, "Datos no disponibles")
    return _reorder_df.copy()

def row_to_dict(row):
    return {
        "product_id":        row["product_id"],
        "current_stock":     float(row["current_stock"]),
        "reorder_point":     float(row["reorder_point"]),
        "lead_time_days":    float(row["lead_time_days"]),
        "avg_daily_sales":   float(row["avg_daily_sales"]),
        "safety_stock":      float(row["safety_stock"]),
        "days_coverage":     float(row["days_coverage"]),
        "status":            row["status"],
        "qty_recommended":   float(row["qty_recommended"]),
        "forecast_next_30d": float(row["forecast_next_30d"]) if pd.notna(row.get("forecast_next_30d")) else None,
    }

# ── ENDPOINTS ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "ready":            _reorder_df is not None,
        "products_loaded":  len(_reorder_df) if _reorder_df is not None else 0,
        "forecasts_loaded": len(_predictions) if _predictions is not None else 0,
    }

@app.get("/dashboard")
def dashboard():
    df = get_df()
    critical = df[df["status"] == "CRÍTICO"]
    urgent   = df[df["status"] == "URGENTE"]
    top_alerts = (df.sort_values(["current_stock", "days_coverage"])
                    .head(10).apply(row_to_dict, axis=1).tolist())
    return {
        "total_products":  len(df),
        "critical_alerts": len(critical),
        "urgent_alerts":   len(urgent),
        "normal_count":    len(df) - len(critical) - len(urgent),
        "pct_at_risk":     round((len(critical)+len(urgent))/len(df)*100, 1),
        "top_alerts":      top_alerts,
    }

@app.get("/products")
def products(
    page:   int            = Query(1, ge=1),
    limit:  int            = Query(10, ge=1, le=100),
    status: Optional[str]  = None,
    search: Optional[str]  = None,
):
    df = get_df()
    if status:
        df = df[df["status"] == status.upper()]
    if search:
        df = df[df["product_id"].str.contains(search, case=False, na=False)]
    df = df.sort_values(["current_stock", "days_coverage"])
    total  = len(df)
    offset = (page - 1) * limit
    page_df = df.iloc[offset: offset + limit]
    return {
        "products":    page_df.apply(row_to_dict, axis=1).tolist(),
        "total":       total,
        "page":        page,
        "per_page":    limit,
        "total_pages": max(1, -(-total // limit)),
    }

@app.get("/product/{product_id}")
def product_detail(product_id: str):
    df  = get_df()
    row = df[df["product_id"] == product_id]
    if row.empty:
        raise HTTPException(404, f"Producto '{product_id}' no encontrado")
    product = row_to_dict(row.iloc[0])
    # Forecast
    fc = (_predictions or {}).get(product_id)
    forecast_data = None
    if fc:
        forecast_data = {
            "dates":      fc["dates"],
            "yhat":       [round(v,0) for v in fc["yhat"]],
            "yhat_lower": [round(v,0) for v in fc["yhat_lower"]],
            "yhat_upper": [round(v,0) for v in fc["yhat_upper"]],
        }
    # Recomendación
    rec = None
    if product["current_stock"] < 0 or product["days_coverage"] < product["lead_time_days"]:
        rec = {
            "action":       "COMPRAR HOY",
            "reason":       f"Stock {'negativo' if product['current_stock']<0 else 'insuficiente'} — lead time {product['lead_time_days']:.0f}d",
            "qty_suggested": product["qty_recommended"],
            "urgency":      "CRÍTICO" if product["current_stock"] < 0 else "URGENTE",
        }
    return {"product": product, "forecast": forecast_data, "recommendation": rec}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
