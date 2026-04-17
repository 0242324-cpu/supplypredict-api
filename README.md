# SupplyPredict API

FastAPI backend para predicción de desabastecimiento — Toyo Foods.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado de la API |
| GET | `/dashboard` | Stats globales + top 10 alertas |
| GET | `/products` | Lista paginada con filtros |
| GET | `/product/{id}` | Detalle + forecast + recomendación |

## Setup local

```bash
pip install -r requirements.txt
cp .env.example .env
# Copiar predictions.pkl y reorder_points.csv a data/
python main.py
```

## Deploy

Conectar repo a Render.com — auto-deploy en cada push.

**Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
