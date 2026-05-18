"""
Generador de dataset SINTÉTICO para SupplyPredict (proyecto escolar).

Estrategia:
- Conservar la DEMANDA real (columna y) — es la señal que el modelo predice.
- Regenerar Stock_Disponible, Stock_Minimo y Cantidad_Comprada con una
  simulación de política de inventario (s, Q) bien gestionada:
    * Service level objetivo: 98% (z = 2.054)
    * Safety stock = z × σ_demanda × √(lead_time_semanas)
    * Reorder point (s) = μ_demanda × lead_time_semanas + safety_stock
    * Order quantity (Q) ≈ cobertura de 6 semanas
    * Stock inicial = s × 2
- Resultado: stock fluctúa naturalmente entre s y s + Q, rara vez bajo s,
  nunca negativo. Cantidad_Comprada refleja órdenes recibidas esa semana.
- Lead_Time_Dias se conserva.

OUTPUT: df_supply_synthetic.csv (mismo esquema que el original).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Reproducibilidad
RNG = np.random.default_rng(42)

# Parámetros de la política (calibrados para stock casi siempre positivo)
Z_SERVICE_LEVEL = 3.0     # ~99.9% service level
ORDER_COVERAGE_WEEKS = 10 # cantidad de orden cubre ~10 semanas de demanda
INITIAL_STOCK_MULT = 2.5  # stock inicial = reorder_point × 2.5
NOISE_PCT = 0.05          # ±5% de ruido en arribos
DEMAND_PERCENTILE = 0.80  # usar p80 de demanda en vez de mean (más conservador)
REORDER_BUFFER = 1.3      # multiplier extra de seguridad sobre reorder point


def simulate_product(df_p):
    """
    Para un producto, simula stock, compras y reorder point.
    Mantiene 'y' (demanda) intacta.
    """
    d = df_p.sort_values('ds').reset_index(drop=True).copy()

    # Estadísticas base de demanda
    y_mean = d['y'].mean()
    y_p80 = d['y'].quantile(DEMAND_PERCENTILE)  # demanda percentile 80 (conservador)
    y_std = d['y'].std() if len(d) > 1 else y_mean * 0.2
    if pd.isna(y_std) or y_std == 0:
        y_std = max(y_mean * 0.1, 1.0)
    if y_mean <= 0:
        # producto con demanda ~0 → política trivial
        d['Stock_Disponible'] = 100.0
        d['Stock_Minimo'] = 10.0
        d['Cantidad_Comprada'] = 0.0
        return d

    # Lead time en semanas (clamping razonable: 1 a 12 semanas)
    lead_time_days = d['Lead_Time_Dias'].iloc[-1]
    if pd.isna(lead_time_days) or lead_time_days <= 0:
        lead_time_days = 14  # default 2 semanas
    lead_time_weeks = max(1, min(int(round(lead_time_days / 7)), 12))

    # Política (s, Q) conservadora: usa p80 de demanda + buffer
    safety_stock = Z_SERVICE_LEVEL * y_std * np.sqrt(lead_time_weeks)
    reorder_point = (y_p80 * lead_time_weeks + safety_stock) * REORDER_BUFFER
    # Order qty asegura cubrir picos de demanda en ciclo
    order_qty = max(y_mean * ORDER_COVERAGE_WEEKS, y_p80 * 6)
    initial_stock = reorder_point * INITIAL_STOCK_MULT

    # Simulación semana por semana
    n = len(d)
    stock = np.zeros(n)
    arrivals = np.zeros(n)  # Cantidad_Comprada por semana
    pending_orders = []     # lista de (semana_arribo, cantidad)

    current_stock = initial_stock

    for t in range(n):
        # 1) Arribos pendientes esta semana
        still_pending = []
        for arrival_week, qty in pending_orders:
            if arrival_week == t:
                current_stock += qty
                arrivals[t] += qty
            else:
                still_pending.append((arrival_week, qty))
        pending_orders = still_pending

        # 2) Restar demanda (clip a 0 — no se vende lo que no hay)
        demand = d['y'].iloc[t]
        current_stock = max(0.0, current_stock - demand)

        # 3) Política (s, Q): si stock < s y no hay nada pendiente, ordenar
        if current_stock < reorder_point and len(pending_orders) == 0:
            # Ruido en la cantidad para realismo (proveedor entrega ±5%)
            noisy_qty = order_qty * (1 + RNG.uniform(-NOISE_PCT, NOISE_PCT))
            arrival_t = min(t + lead_time_weeks, n - 1)
            pending_orders.append((arrival_t, noisy_qty))

        # 4) Registrar
        stock[t] = current_stock

    d['Stock_Disponible'] = stock
    d['Stock_Minimo'] = reorder_point
    d['Cantidad_Comprada'] = arrivals

    return d


def main():
    print("=" * 70)
    print("GENERANDO DATASET SINTÉTICO")
    print("=" * 70)

    df = pd.read_csv('/mnt/project/df_supply_clean.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    print(f"\nInput: {df.shape[0]} registros, {df['ID_Producto'].nunique()} productos")
    print(f"Stock negativo (real): {(df['Stock_Disponible'] < 0).sum()} registros "
          f"({(df['Stock_Disponible'] < 0).mean()*100:.1f}%)")
    print(f"Stock cero (real): {(df['Stock_Disponible'] == 0).sum()} registros "
          f"({(df['Stock_Disponible'] == 0).mean()*100:.1f}%)")

    parts = []
    products = df['ID_Producto'].unique()
    for i, p in enumerate(products, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(products)}] productos procesados")
        sim = simulate_product(df[df['ID_Producto'] == p])
        parts.append(sim)

    df_syn = pd.concat(parts, ignore_index=True).sort_values(['ID_Producto', 'ds'])
    df_syn = df_syn[['ID_Producto', 'ds', 'y', 'Stock_Disponible',
                     'Stock_Minimo', 'Cantidad_Comprada', 'Lead_Time_Dias']]

    # Validaciones
    n_neg = (df_syn['Stock_Disponible'] < 0).sum()
    n_zero = (df_syn['Stock_Disponible'] == 0).sum()
    n_below_reorder = (df_syn['Stock_Disponible'] < df_syn['Stock_Minimo']).sum()

    print(f"\n--- Validación dataset sintético ---")
    print(f"  Registros: {len(df_syn)}")
    print(f"  Productos: {df_syn['ID_Producto'].nunique()}")
    print(f"  Stock negativo: {n_neg} ({n_neg/len(df_syn)*100:.2f}%)  ← debe ser 0")
    print(f"  Stock = 0: {n_zero} ({n_zero/len(df_syn)*100:.2f}%)  ← debería ser muy raro")
    print(f"  Stock bajo reorder: {n_below_reorder} ({n_below_reorder/len(df_syn)*100:.1f}%)  ← natural ~3-15%")
    print(f"  Total Cantidad_Comprada: {df_syn['Cantidad_Comprada'].sum():,.0f}")
    print(f"  Demanda total (y): {df_syn['y'].sum():,.0f}")
    print(f"  Ratio compras/demanda: {df_syn['Cantidad_Comprada'].sum() / df_syn['y'].sum():.2f}")

    # Comparativa real vs sintético
    print(f"\n--- Real vs Sintético ---")
    print(f"  Real | Stock <0: {(df['Stock_Disponible']<0).mean()*100:.1f}%, "
          f"Stock=0: {(df['Stock_Disponible']==0).mean()*100:.1f}%, "
          f"Stock>0: {(df['Stock_Disponible']>0).mean()*100:.1f}%")
    print(f"  Syn  | Stock <0: {n_neg/len(df_syn)*100:.1f}%, "
          f"Stock=0: {n_zero/len(df_syn)*100:.1f}%, "
          f"Stock>0: {(df_syn['Stock_Disponible']>0).mean()*100:.1f}%")

    out_path = '/home/claude/data-output/df_supply_synthetic.csv'
    df_syn.to_csv(out_path, index=False)
    print(f"\n✓ Guardado: {out_path}")
    print(f"  Tamaño: {df_syn.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB en memoria")


if __name__ == '__main__':
    main()
