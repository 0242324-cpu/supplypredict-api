"""
SupplyPredict API - main.py
Soporta datos básicos y datos enriquecidos (catálogo + multi-almacén + órdenes)
Para hacer swap: reemplazar archivos en data/ y reiniciar el servicio
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, pickle, csv, math
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Configuración de paths ────────────────────────────────────────────
# Modo enriquecido si existe reorder_points_full.csv, si no usa el básico
FULL_PATH = os.getenv("FULL_DATA_PATH",    "data/reorder_points_full.csv")
BASE_PATH = os.getenv("REORDER_POINTS_PATH","data/reorder_points.csv")
PRED_PATH = os.getenv("PREDICTIONS_PATH",   "data/predictions.pkl")

def _load_csv(path):
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(row)
        print(f"✓ {path}: {len(rows)} filas")
    except Exception as e:
        print(f"✗ {path}: {e}")
    return rows

def _load_pkl(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ {path}: {len(data)} entradas")
        return data
    except Exception as e:
        print(f"✗ {path}: {e}")
        return {}

def _float(v, default=0.0):
    try: return float(v) if v not in (None,'','nan','None','NaN') else default
    except: return default

def _str(v, default=''):
    return v if v and v not in ('nan','None','NaN') else default

# ── Cargar datos al iniciar ───────────────────────────────────────────
# Intenta cargar el CSV enriquecido primero
if os.path.exists(FULL_PATH):
    _raw_rows   = _load_csv(FULL_PATH)
    _enriched   = True
    print("Modo: ENRIQUECIDO (catálogo + multi-almacén + órdenes)")
else:
    _raw_rows   = _load_csv(BASE_PATH)
    _enriched   = False
    print("Modo: BÁSICO (solo reorder_points)")

_predictions    = _load_pkl(PRED_PATH)
_products_index = {r["product_id"]: r for r in _raw_rows}

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="SupplyPredict API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: fila → dict ───────────────────────────────────────────────
def row_to_dict(r):
    status = _str(r.get('status_consolidado') or r.get('status'), 'CRÍTICO')
    stock  = _float(r.get('stock_consolidado') or r.get('current_stock'))
    return {
        "product_id":          r["product_id"],
        # Enriquecidos (None si no hay catálogo)
        "nombre":              _str(r.get('nombre'), r["product_id"]),
        "categoria":           _str(r.get('categoria'), 'Sin categoría'),
        "unidad":              _str(r.get('unidad'), ''),
        "proveedor":           _str(r.get('proveedor') or r.get('proveedor_principal'), ''),
        # Stock
        "current_stock":       _float(r.get('current_stock')),
        "stock_consolidado":   stock,
        "reorder_point":       _float(r.get('reorder_point')),
        "lead_time_days":      _float(r.get('lead_time_days')),
        "avg_daily_sales":     _float(r.get('avg_daily_sales')),
        "safety_stock":        _float(r.get('safety_stock')),
        "days_coverage":       _float(r.get('days_coverage')),
        "status":              status,
        "qty_recommended":     _float(r.get('qty_recommended')),
        "forecast_next_30d":   _float(r.get('forecast_next_30d')) if r.get('forecast_next_30d') else None,
        # Orden abierta
        "orden_abierta":       bool(_str(r.get('orden_id'))),
        "orden_id":            _str(r.get('orden_id')),
        "orden_cantidad":      _float(r.get('orden_cantidad')) if r.get('orden_cantidad') else None,
        "orden_fecha_llegada": _str(r.get('orden_fecha_llegada')),
        "orden_proveedor":     _str(r.get('orden_proveedor')),
        # Meta
        "enriched":            _enriched,
    }

def get_sorted(status=None, search=None, categoria=None):
    rows = _raw_rows
    if status:
        key = 'status_consolidado' if _enriched else 'status'
        rows = [r for r in rows if r.get(key,'').upper() == status.upper()]
    if search:
        rows = [r for r in rows
                if search.lower() in r.get("product_id","").lower()
                or search.lower() in r.get("nombre","").lower()]
    if categoria:
        rows = [r for r in rows if r.get("categoria","").lower() == categoria.lower()]

    def sort_key(r):
        stock = _float(r.get('stock_consolidado') or r.get('current_stock'))
        dc    = _float(r.get('days_coverage'), 999)
        return (stock, dc)

    return sorted(rows, key=sort_key)

# ── ENDPOINTS ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "ready":            len(_raw_rows) > 0,
        "products_loaded":  len(_raw_rows),
        "forecasts_loaded": len(_predictions),
        "mode":             "enriched" if _enriched else "basic",
    }

@app.get("/dashboard")
def dashboard():
    if not _raw_rows: raise HTTPException(503, "Datos no disponibles")
    key = 'status_consolidado' if _enriched else 'status'
    critical      = [r for r in _raw_rows if r.get(key) == 'CRÍTICO']
    urgent        = [r for r in _raw_rows if r.get(key) == 'URGENTE']
    orden_abierta = [r for r in _raw_rows if r.get(key) == 'ORDEN_ABIERTA']
    top           = get_sorted()[:10]

    # Categorías disponibles
    categorias = sorted(set(r.get('categoria','') for r in _raw_rows if r.get('categoria')))

    return {
        "total_products":    len(_raw_rows),
        "critical_alerts":   len(critical),
        "urgent_alerts":     len(urgent),
        "orden_abierta":     len(orden_abierta),
        "normal_count":      len(_raw_rows) - len(critical) - len(urgent) - len(orden_abierta),
        "pct_at_risk":       round((len(critical)+len(urgent)) / len(_raw_rows) * 100, 1),
        "top_alerts":        [row_to_dict(r) for r in top],
        "categorias":        categorias,
        "mode":              "enriched" if _enriched else "basic",
    }

@app.get("/products")
def products(
    page:      int           = Query(1, ge=1),
    limit:     int           = Query(10, ge=1, le=100),
    status:    Optional[str] = None,
    search:    Optional[str] = None,
    categoria: Optional[str] = None,
):
    if not _raw_rows: raise HTTPException(503, "Datos no disponibles")
    rows   = get_sorted(status=status, search=search, categoria=categoria)
    total  = len(rows)
    offset = (page - 1) * limit
    return {
        "products":    [row_to_dict(r) for r in rows[offset:offset+limit]],
        "total":       total,
        "page":        page,
        "per_page":    limit,
        "total_pages": max(1, math.ceil(total / limit)),
    }

@app.get("/product/{product_id}")
def product_detail(product_id: str):
    r = _products_index.get(product_id)
    if not r: raise HTTPException(404, f"Producto '{product_id}' no encontrado")
    product = row_to_dict(r)

    fc = (_predictions or {}).get(product_id)
    forecast_data = None
    if fc:
        forecast_data = {
            "dates":      fc.get("dates", []),
            "yhat":       [round(v,0) for v in fc.get("yhat",[])],
            "yhat_lower": [round(v,0) for v in fc.get("yhat_lower",[])],
            "yhat_upper": [round(v,0) for v in fc.get("yhat_upper",[])],
        }

    stock = product["stock_consolidado"]
    dc    = product["days_coverage"]
    lt    = product["lead_time_days"]
    rec   = None
    if not product["orden_abierta"] and (stock < 0 or dc < lt):
        rec = {
            "action":        "COMPRAR HOY",
            "reason":        f"Stock {'negativo' if stock<0 else 'insuficiente'} — lead time {lt:.0f}d",
            "qty_suggested": product["qty_recommended"],
            "urgency":       "CRÍTICO" if stock < 0 else "URGENTE",
        }
    elif product["orden_abierta"]:
        rec = {
            "action":        "ORDEN EN PROCESO",
            "reason":        f"Orden {product['orden_id']} — llega {product['orden_fecha_llegada']}",
            "qty_suggested": product["orden_cantidad"],
            "urgency":       "ORDEN_ABIERTA",
        }

    return {"product": product, "forecast": forecast_data, "recommendation": rec}

@app.get("/categorias")
def categorias():
    cats = sorted(set(r.get('categoria','') for r in _raw_rows if r.get('categoria')))
    return {"categorias": cats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
