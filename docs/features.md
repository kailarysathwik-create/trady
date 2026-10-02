# TRADY Feature Engine: Architecture & Mathematical Specifications

## 1. Overview & Purpose

The **TRADY Feature Engine** (`src/trady/features/`) transforms validated historical OHLCV data into numerical features for exploratory quantitative research and future machine-learning models.

TRADY is an educational quantitative research platform. It does **NOT** execute real-money transactions, does **NOT** connect to live brokers, and does **NOT** promise guaranteed investment returns.

---

## 2. Fundamental Law: Strict Temporal Causality

In quantitative modeling, lookahead bias (peeking into future data) is fatal. The Feature Engine enforces the following invariant:

$$\forall \text{ Feature } F, \quad F(t) = f(X_{\tau \le t})$$

> [!IMPORTANT]
> **Zero Future Leakage:** Every feature computed at timestamp $T$ must depend exclusively on observations recorded at or before $T$.
>
> 1. **Index Alignment:** Rolling windows of length $k$ ending at $t$ only read slice $[t - k + 1 \dots t]$.
> 2. **Multi-Symbol Partitioning:** In multi-asset universes, feature calculation is always executed on strictly partitioned single-symbol time series. Rolling windows never span across ticker boundaries.
> 3. **Warmup NaNs:** Any operation requiring $k$ historical periods produces explicit `NaN` (or null) for all indices $t < k - 1$.
> 4. **No Future Normalization:** Standard deviations, moving averages, and min/max ranges use only past rolling windows—never whole-dataset global statistics.

---

## 3. Mathematical Definitions of Feature Groups

### A. Price Returns (`PRICE`)

| Feature | Formula | Lookback ($k$) | Description |
|---|---|---|---|
| Simple Return | $R_{t,k} = \frac{P_t - P_{t-k}}{P_{t-k}}$ | $k \ge 1$ | Percentage change in close price over $k$ bars. |
| Log Return | $r_{t,k} = \ln\left(\frac{P_t}{P_{t-k}}\right)$ | $k \ge 1$ | Continuously compounded return over $k$ bars. |
| Multi-Period Returns | $\{R_{t,1}, R_{t,3}, R_{t,5}, R_{t,10}, R_{t,21}\}$ | $1, 3, 5, 10, 21$ | Multiple horizon price momentum profiles. |

---

### B. Momentum & Trend Relationships (`MOMENTUM`)

| Feature | Formula | Lookback ($k$) | Description |
|---|---|---|---|
| Rolling Momentum | $M_{t,k} = P_t - P_{t-k}$ | $k \ge 1$ | Raw price difference over $k$ periods. |
| Rate of Change (ROC) | $\text{ROC}_{t,k} = \frac{P_t - P_{t-k}}{P_{t-k}} \times 100$ | $k \ge 1$ | Percentage momentum velocity. |
| Price to SMA Ratio | $\text{DistSMA}_{t,k} = \frac{P_t - \text{SMA}_k(P)_t}{\text{SMA}_k(P)_t}$ | $k \ge 1$ | Percentage deviation of price from its $k$-period moving average. |
| SMA Ratio (Fast/Slow) | $\text{SMASpread}_t = \frac{\text{SMA}_{\text{fast}, t}}{\text{SMA}_{\text{slow}, t}} - 1$ | $\text{slow} > \text{fast}$ | Trend alignment between short and long moving averages. |

$$\text{SMA}_{t,k} = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$

---

### C. Volatility & Dispersion (`VOLATILITY`)

| Feature | Formula | Lookback ($k$) | Description |
|---|---|---|---|
| Rolling Return Std | $\sigma_{t,k} = \sqrt{\frac{1}{k-1} \sum_{i=0}^{k-1} (r_{t-i} - \bar{r}_t)^2}$ | $k \ge 2$ | Sample standard deviation of 1-period returns. |
| Realized Volatility | $\text{RV}_{t,k} = \sigma_{t,k} \times \sqrt{252}$ | $k \ge 2$ | Annualized realized volatility (assuming 252 daily bars/yr). |
| True Range (TR) | $\text{TR}_t = \max(H_t - L_t, \|H_t - C_{t-1}\|, \|L_t - C_{t-1}\|)$ | $1$ | High-low volatility adjusted for overnight gaps. |
| Average True Range | $\text{ATR}_{t,k} = \frac{1}{k} \sum_{i=0}^{k-1} \text{TR}_{t-i}$ | $k \ge 1$ | Moving average of True Range. |
| Normalized ATR | $\text{NATR}_{t,k} = \frac{\text{ATR}_{t,k}}{C_t} \times 100$ | $k \ge 1$ | ATR expressed as a percentage of close price. |

---

### D. Volume Dynamics (`VOLUME`)

| Feature | Formula | Lookback ($k$) | Description |
|---|---|---|---|
| Volume Change | $\Delta V_{t,k} = \frac{V_t - V_{t-k}}{V_{t-k}}$ | $k \ge 1$ | Percentage change in trading volume over $k$ bars. |
| Volume-to-SMA Ratio | $\text{VolRatio}_{t,k} = \frac{V_t}{\text{SMA}_k(V)_t}$ | $k \ge 1$ | Relative volume activity vs. historical rolling average. |
| Volume Z-Score | $Z_{V,t,k} = \frac{V_t - \mu_{V,t,k}}{\sigma_{V,t,k} + \epsilon}$ | $k \ge 2$ | Standardized volume anomaly detection. |

---

### E. Price Structure & Range Position (`PRICE_STRUCTURE`)

| Feature | Formula | Lookback ($k$) | Description |
|---|---|---|---|
| Distance from High | $\text{DistHigh}_{t,k} = \frac{P_t - \max_{0 \le i < k} H_{t-i}}{\max_{0 \le i < k} H_{t-i}} \le 0$ | $k \ge 1$ | Drawdown relative to $k$-period rolling peak. |
| Distance from Low | $\text{DistLow}_{t,k} = \frac{P_t - \min_{0 \le i < k} L_{t-i}}{\min_{0 \le i < k} L_{t-i}} \ge 0$ | $k \ge 1$ | Premium relative to $k$-period rolling trough. |
| Normalized Range Pos | $\text{PosRange}_{t,k} = \frac{P_t - \min_{k} L}{\max_k H - \min_k L} \in [0, 1]$ | $k \ge 1$ | Relative position within the rolling high-low channel. |

---

## 4. Feature Registry & Metadata

Every feature registers a [`FeatureMetadata`](file:///s:/Trady/src/trady/features/registry.py#L11-L45) descriptor:
```python
FeatureMetadata(
    name="vol_std_20d",
    description="Rolling standard deviation of 1-period returns over 20 bars",
    feature_group="VOLATILITY",
    required_columns=("close",),
    lookback_period=20,
    min_observations=21,
    is_point_in_time=True,
)
```

CLI commands:
- `trady features list`
- `trady features compute <dataset>`
- `trady features inspect <dataset>`

---

## 5. Storage Tier Separation

```
data/
├── raw/        <- Immutable source data (JSON/CSV)
├── interim/    <- Unvalidated work-in-progress data
├── processed/  <- Validated, canonical OHLCV Parquet
└── features/   <- Causal numerical feature Parquet + .meta.json
```
Features are stored strictly separately from raw and processed market data.
