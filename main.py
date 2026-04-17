"""
SupplyPredict API - main.py
Sin dependencias pesadas - CSV puro + pickle estándar
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, pickle, csv, math
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Cargar datos ─────────────────────────────────────────────────────
def _load_csv(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"✓ CSV cargado: {len(rows)} filas de {path}")
    except Exception as e:
        print(f"✗ Error CSV: {e}")
    return rows

def _load_pkl(path: str):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ PKL cargado: {len(data)} entradas de {path}")
        return data
    except Exception as e:
        print(f"✗ Error PKL: {e}")
        return {}

def _float(v, default=0.0):
    try: return float(v) if v not in (None, '', 'nan', 'None') else default
    except: return default

def _int(v, default=0):
    try: return int(float(v)) if v not in (None, '', 'nan', 'None') else default
    except: return default

# Cargar al iniciar
RO_PATH   = os.getenv("REORDER_POINTS_PATH", "data/reorder_points.csv")
PRED_PATH = os.getenv("PREDICTIONS_PATH",    "data/predictions.pkl")

_raw_rows    = _load_csv(RO_PATH)
_predictions = _load_pkl(PRED_PATH)

# Índice por product_id
_products_index = {r["product_id"]: r for r in _raw_rows}

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="SupplyPredict API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ──────────────────────────────────────────────────────────
def row_to_dict(r: dict) -> dict:
    return {
        "product_id":        r["product_id"],
        "current_stock":     _float(r.get("current_stock")),
        "reorder_point":     _float(r.get("reorder_point")),
        "lead_time_days":    _float(r.get("lead_time_days")),
        "avg_daily_sales":   _float(r.get("avg_daily_sales")),
        "safety_stock":      _float(r.get("safety_stock")),
        "days_coverage":     _float(r.get("days_coverage")),
        "status":            r.get("status", "CRÍTICO"),
        "qty_recommended":   _float(r.get("qty_recommended")),
        "forecast_next_30d": _float(r.get("forecast_next_30d")) if r.get("forecast_next_30d") else None,
    }

def get_rows(status=None, search=None):
    rows = _raw_rows
    if status:
        rows = [r for r in rows if r.get("status","").upper() == status.upper()]
    if search:
        rows = [r for r in rows if search.lower() in r.get("product_id","").lower()]
    return sorted(rows, key=lambda r: _float(r.get("current_stock"), 0))

# ── ENDPOINTS ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "ready":            len(_raw_rows) > 0,
        "products_loaded":  len(_raw_rows),
        "forecasts_loaded": len(_predictions),
    }

@app.get("/dashboard")
def dashboard():
    if not _raw_rows:
        raise HTTPException(503, "Datos no disponibles")
    critical = [r for r in _raw_rows if r.get("status") == "CRÍTICO"]
    urgent   = [r for r in _raw_rows if r.get("status") == "URGENTE"]
    top      = sorted(_raw_rows, key=lambda r: _float(r.get("current_stock"), 0))[:10]
    return {
        "total_products":  len(_raw_rows),
        "critical_alerts": len(critical),
        "urgent_alerts":   len(urgent),
        "normal_count":    len(_raw_rows) - len(critical) - len(urgent),
        "pct_at_risk":     round((len(critical)+len(urgent)) / len(_raw_rows) * 100, 1),
        "top_alerts":      [row_to_dict(r) for r in top],
    }

@app.get("/products")
def products(
    page:   int           = Query(1, ge=1),
    limit:  int           = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    search: Optional[str] = None,
):
    if not _raw_rows:
        raise HTTPException(503, "Datos no disponibles")
    rows   = get_rows(status=status, search=search)
    total  = len(rows)
    offset = (page - 1) * limit
    page_rows = rows[offset: offset + limit]
    return {
        "products":    [row_to_dict(r) for r in page_rows],
        "total":       total,
        "page":        page,
        "per_page":    limit,
        "total_pages": max(1, math.ceil(total / limit)),
    }

@app.get("/product/{product_id}")
def product_detail(product_id: str):
    r = _products_index.get(product_id)
    if not r:
        raise HTTPException(404, f"Producto '{product_id}' no encontrado")
    product = row_to_dict(r)

    # Forecast
    fc = _predictions.get(product_id)
    forecast_data = None
    if fc:
        forecast_data = {
            "dates":      fc.get("dates", []),
            "yhat":       [round(v, 0) for v in fc.get("yhat", [])],
            "yhat_lower": [round(v, 0) for v in fc.get("yhat_lower", [])],
            "yhat_upper": [round(v, 0) for v in fc.get("yhat_upper", [])],
        }

    # Recomendación
    rec = None
    cs  = product["current_stock"]
    dc  = product["days_coverage"]
    lt  = product["lead_time_days"]
    if cs < 0 or dc < lt:
        rec = {
            "action":        "COMPRAR HOY",
            "reason":        f"Stock {'negativo' if cs < 0 else 'insuficiente'} — lead time {lt:.0f}d",
            "qty_suggested": product["qty_recommended"],
            "urgency":       "CRÍTICO" if cs < 0 else "URGENTE",
        }

    return {"product": product, "forecast": forecast_data, "recommendation": rec}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
