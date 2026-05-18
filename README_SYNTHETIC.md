# Dataset Sintético — SupplyPredict (Proyecto Académico)

**Archivo**: `df_supply_synthetic.csv`
**Generado**: 2026-05-17
**Propósito**: Proyecto escolar — demostrar un sistema de supply chain bien gestionado.

---

## ¿Por qué un dataset sintético?

Los datos reales de Toyo Foods reflejan **una operación con problemas crónicos de inventario**:
- **63% de los registros con stock negativo** (desabasto persistente)
- **0% del tiempo en estado "Normal"** según la lógica de alertas del sistema
- El sistema, al aplicarse a estos datos, dispara alertas críticas en casi todos los productos siempre, lo que vuelve el dashboard inútil para demostrar el valor del modelo

Para fines del proyecto escolar, generamos un dataset sintético que conserva la **demanda real** pero recrea el inventario con una política de gestión profesional.

---

## ¿Qué se conserva del original?

| Columna | Tratamiento |
|---|---|
| `ID_Producto` | Idéntico (495 productos) |
| `ds` | Idéntico (rango 2021-12-27 → 2026-03-30, semanal, lunes) |
| **`y`** (demanda) | **Idéntico — no se toca** |
| `Lead_Time_Dias` | Idéntico (último valor registrado por producto) |
| `Stock_Disponible` | **Regenerado** con simulación (s, Q) |
| `Stock_Minimo` | **Regenerado** = reorder point calculado |
| `Cantidad_Comprada` | **Regenerado** = arribos según política |

**Importante**: `y` (la señal que el modelo predice) NO cambia. Esto garantiza que el modelo de forecast tiene exactamente la misma dificultad/calidad con el dataset real o sintético.

---

## Política de inventario simulada

Sistema clásico **(s, Q)** con periodo de revisión continua:

```
service_level     = 99.9%  →  z = 3.0
demanda_p80       = percentile 80 de y histórica (conservador, no media)
lead_time_weeks   = Lead_Time_Dias / 7  (clamped 1-12)

safety_stock      = z × σ(y) × √(lead_time_weeks)
reorder_point (s) = (demanda_p80 × lead_time_weeks + safety_stock) × 1.3
order_qty (Q)     = max(media(y) × 10, demanda_p80 × 6)
stock_inicial     = s × 2.5
```

### Dinámica semana a semana:
1. Si llegan arribos pendientes esta semana → `stock += qty_pedido`
2. Restar demanda → `stock = max(0, stock - y_t)`
3. Si `stock < s` y no hay orden pendiente → lanzar orden, llega en `lead_time_weeks` semanas
4. Registrar stock final

El stock fluctúa naturalmente entre `s` y `s + Q` formando un patrón "diente de sierra" típico de sistemas (s, Q).

---

## Validación del dataset sintético

### Estructura de inventario

| Métrica | Real (T.I.) | Sintético |
|---|---|---|
| Stock negativo | 63.0% ⚠️ | **0.0%** ✅ |
| Stock = 0 | 1.2% | **1.1%** (rara vez) |
| Stock > 0 | 35.8% | **98.9%** |
| Estado Crítico | 64.2% ⚠️ | **1.1%** ✅ |
| Estado Urgente (bajo reorder) | 35.8% | 39.5% (natural en ciclo) |
| Estado Normal | **0.0%** | **59.4%** ✅ |

### Performance del modelo (LightGBM, W3: train 2022-2024, test 2025)

| | Real | Sintético |
|---|---|---|
| MAPE mediana | 38.9% | 39.3% |
| % productos <50% MAPE | 63.7% | 62.6% |
| % productos <100% MAPE | 79.5% | 77.4% |

**Diferencia: <1pp**. El cambio del stock NO afecta el forecast de demanda (como debe ser, porque `y` no cambió).

---

## Limitaciones honestas (para reportar en el proyecto)

1. **El stock no refleja eventos operativos reales**: huelgas de proveedores, picos de demanda no anticipados, cambios de mix, etc. La simulación asume un proveedor que entrega siempre puntual (±5% en cantidad).

2. **Política única para todos los productos**: en producción real, cada categoría tendría parámetros distintos (perecederos vs no perecederos, ABC analysis, etc).

3. **Q1 2026 sigue siendo más difícil de predecir** que 2025 — eso es un problema de la demanda real (88% de productos cambian de comportamiento, 49 productos sin data en Q1 2026), no del inventario. El stock sintético en Q1 2026 está sano, pero el forecast tiene MAPE alto porque la `y` real es atípica.

4. **No es data para usar en producción de Toyo Foods**. Es exclusivamente para fines pedagógicos y demostración del sistema.

---

## Cómo usar el dataset

### Para entrenar el modelo:
```python
df = pd.read_csv('df_supply_synthetic.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Recomendación: train 2022-2024 (W3)
train = df[df['ds'] <= '2024-12-31']
test  = df[df['ds'] >  '2024-12-31']
```

### Para deploy:
Reemplazar `df_supply_clean.csv` por `df_supply_synthetic.csv` en el pipeline de `swap.py`. El resto del backend funciona sin cambios.

### Para presentación:
- Mostrar las gráficas `06_real_vs_synthetic.png` para demostrar el cambio
- Mostrar la distribución de status (1% crítico vs 64% antes)
- Mostrar el dashboard en producción con alertas significativas (no todo en rojo)

---

## Parámetros del generador (para reproducir o ajustar)

En `generate_synthetic.py`:
```python
Z_SERVICE_LEVEL    = 3.0     # 99.9% service level
ORDER_COVERAGE_WEEKS = 10    # Q cubre 10 semanas
INITIAL_STOCK_MULT = 2.5     # stock inicial = 2.5 × s
NOISE_PCT          = 0.05    # ±5% ruido en arribos
DEMAND_PERCENTILE  = 0.80    # usar p80 en política
REORDER_BUFFER     = 1.3     # margen extra sobre reorder point
```

**Semilla aleatoria**: `np.random.default_rng(42)` — reproducible.

---

## Estado del dataset

- **Filas**: 88,723
- **Productos**: 495
- **Rango**: 2021-12-27 → 2026-03-30 (222 semanas)
- **Granularidad**: semanal (lunes)
- **Tamaño**: 9.4 MB en memoria, ~7 MB CSV en disco

---

**Status**: Listo para usar en proyecto académico.
