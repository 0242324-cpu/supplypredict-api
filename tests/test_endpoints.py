"""
tests/test_endpoints.py
Correr con: pytest tests/ -v
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["REORDER_POINTS_PATH"] = os.path.join(ROOT, "data", "reorder_points.csv")
os.environ["PREDICTIONS_PATH"]    = os.path.join(ROOT, "data", "predictions.pkl")

sys.path.insert(0, ROOT)
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


# ── /health ──────────────────────────────────────────────────────────
def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200

def test_health_has_status_ok():
    r = client.get("/health")
    assert r.json()["status"] == "ok"

def test_health_has_products_loaded():
    r = client.get("/health")
    data = r.json()
    assert "products_loaded" in data
    assert data["products_loaded"] > 0


# ── /dashboard ───────────────────────────────────────────────────────
def test_dashboard_returns_200():
    r = client.get("/dashboard")
    assert r.status_code == 200

def test_dashboard_has_required_fields():
    r = client.get("/dashboard")
    data = r.json()
    for field in ["total_products", "critical_alerts", "urgent_alerts", "top_alerts"]:
        assert field in data, f"Campo faltante: {field}"

def test_dashboard_top_alerts_is_list():
    r = client.get("/dashboard")
    assert isinstance(r.json()["top_alerts"], list)

def test_dashboard_counts_are_positive():
    r = client.get("/dashboard")
    data = r.json()
    assert data["total_products"] > 0
    assert data["critical_alerts"] >= 0


# ── /products ─────────────────────────────────────────────────────────
def test_products_returns_200():
    r = client.get("/products")
    assert r.status_code == 200

def test_products_has_required_fields():
    r = client.get("/products")
    data = r.json()
    for field in ["products", "total", "page", "per_page", "total_pages"]:
        assert field in data, f"Campo faltante: {field}"

def test_products_pagination_default():
    r = client.get("/products")
    data = r.json()
    assert data["page"] == 1
    assert data["per_page"] == 10
    assert len(data["products"]) <= 10

def test_products_pagination_page2():
    r = client.get("/products?page=2&limit=5")
    assert r.status_code == 200
    data = r.json()
    assert data["page"] == 2
    assert data["per_page"] == 5

def test_products_filter_by_status():
    r = client.get("/products?status=CRÍTICO")
    assert r.status_code == 200

def test_products_search():
    r = client.get("/products?search=SE-ARR")
    assert r.status_code == 200

def test_products_each_has_product_id():
    r = client.get("/products")
    for p in r.json()["products"]:
        assert "product_id" in p
        assert "current_stock" in p
        assert "status" in p


# ── /product/{id} ────────────────────────────────────────────────────
def test_product_detail_valid():
    pid = client.get("/products").json()["products"][0]["product_id"]
    r = client.get(f"/product/{pid}")
    assert r.status_code == 200

def test_product_detail_has_fields():
    pid = client.get("/products").json()["products"][0]["product_id"]
    data = client.get(f"/product/{pid}").json()
    assert "product" in data
    assert "forecast" in data
    assert "recommendation" in data

def test_product_detail_not_found():
    r = client.get("/product/PRODUCTO-INEXISTENTE-999")
    assert r.status_code == 404

def test_product_detail_forecast_structure():
    pid = client.get("/products").json()["products"][0]["product_id"]
    data = client.get(f"/product/{pid}").json()
    fc = data.get("forecast")
    if fc:
        assert "dates" in fc
        assert "yhat" in fc
        assert len(fc["dates"]) == len(fc["yhat"])
