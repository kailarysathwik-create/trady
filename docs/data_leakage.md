# Data Leakage Prevention & Data Tiers in TRADY

## 1. The Core Principle: Absolute Causality

In quantitative research and mathematical finance, **lookahead bias (data leakage)** is the most dangerous and deceptive error. It occurs whenever information from the future ($t + \Delta t$) is inadvertently used to make a decision, compute a feature, calculate a normalization parameter, or execute a simulated trade at time $t$.

A model suffering from lookahead bias will appear exceptionally profitable in historical backtests, but will experience catastrophic failure in forward paper trading and live reality because the future is unknown at decision time.

The TRADY Data Engine establishes architectural firewalls to guarantee **strict causality**:

1. **Chronological Monotonicity**: All datasets must be sorted strictly by timestamp in ascending order.
2. **Immutable History**: Historical data slices must never be silently edited, interpolated, or rewritten post-hoc.
3. **No Future Normalization**: Normalization and summary statistics must never be computed across future periods.

---

## 2. The Four Distinct Data Tiers

To prevent data contamination and leakage, TRADY enforces four distinct, decoupled tiers of data:

```
[ Tier 1: RAW DATA ]
       │
       │ (Read-only Ingestion, Validation & UTC Alignment)
       ▼
[ Tier 2: CLEAN DATA (INTERIM / PROCESSED) ]
       │
       │ (Causal Lagged Feature Transformers)
       ▼
[ Tier 3: FEATURE DATA ]
       │
       │ (Strict Temporal In-Sample Split / Walk-Forward Cross Validation)
       ▼
[ Tier 4: EXPERIMENT DATA ]
```

### Tier 1: RAW DATA (`data/raw/`)
* **Definition**: Unaltered, external source records directly received from providers or local fixtures (CSV, JSON, APIs).
* **Mutabilty**: **IMMUTABLE**. Never edited in-place, cleaned, or overwritten.
* **Integrity**: Every raw dataset has an accompanying provenance record containing retrieval timestamp, provider name, source hash, and raw schema.
* **Access**: The research models and backtest engines NEVER read directly from `data/raw/`.

### Tier 2: CLEAN DATA (`data/interim/` & `data/processed/`)
* **Definition**: Validated, canonicalized, timezone-normalized (UTC), and chronologically ordered OHLCV records stored in high-performance Parquet format.
* **Mutabilty**: Derived deterministically from Raw Data.
* **Invariants Enforced**:
  - Positive prices ($O, H, L, C > 0$).
  - High and low mathematical bounds ($H \ge O, C, L$ and $L \le O, C, H$).
  - Non-negative volume ($V \ge 0$).
  - Zero duplicate timestamps for any symbol.
* **Analytical Queries**: Queried via DuckDB for slicing, filtering, and aggregated statistics.

### Tier 3: FEATURE DATA (`data/features/`)
* **Definition**: Causal mathematical transformations (returns, moving averages, volatility estimators, normalized signals) derived from Clean Data.
* **Leakage Rules**:
  - **Strict Lagging**: Any feature at bar $t$ must only use information available at or before bar $t$.
  - **No Global Scaling**: Rolling z-scores or min-max normalization must use expanding or rolling backward-looking windows ($[t-k, t]$). Computing mean or standard deviation across the full dataset period $[0, T]$ is **strictly forbidden**.
  - **Target Separation**: Target returns ($y_t = \ln(P_{t+1} / P_t)$) must be strictly separated from feature matrices ($X_t$).

### Tier 4: EXPERIMENT DATA (`reports/` & `experiments/`)
* **Definition**: Immutable snapshots of model training runs, hyperparameters, out-of-sample predictions, simulated order records, and evaluation metrics.
* **Auditability**: Every experiment records the exact git commit, dataset hash, random seed, and configuration snapshot.
* **Walk-Forward Invariance**: Walk-forward cross-validation splits must strictly honor time. Shuffling or random K-Fold cross-validation on time-series data is an architectural violation.

---

## 3. Storage Architecture: Parquet & DuckDB

- **Parquet** provides columnar, compressed, typed, and disk-efficient persistence for processed datasets, preventing schema drift.
- **DuckDB** provides zero-overhead, in-process analytical SQL queries directly on Parquet files without requiring persistent server daemons or memory duplication.
