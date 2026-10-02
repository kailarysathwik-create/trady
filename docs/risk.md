# TRADY Risk Engine

## 1. Overview & Educational Research Scope

The TRADY Risk Engine serves as an independent, deterministic guardian layer sitting strictly between quantitative machine-learning models / strategy rules and simulated portfolio positions.

### Critical Research Guardrails
- **Educational & Research Focus Only**: TRADY is strictly an educational research and paper-trading platform. It does not provide investment advice, does not claim guaranteed trading profits, and does not execute real-money transactions.
- **Zero Bypass Invariant**: No model prediction, rule, or signal may generate a simulated order that bypasses the Risk Engine. Every order must be explicitly audited and approved.
- **Total Transparency**: The Risk Engine never hides adjustments. Every proposed order is categorized as **ACCEPTED**, **REDUCED**, or **REJECTED** with an immutable audit record and explicit human-readable justification.

---

## 2. Risk Engine Architecture & Pipeline

```
Machine Learning Predictions P(T)
               │
               ▼
Strategy Rule Generation (Bar T Close)
               │
               ▼ [Proposed Order]
┌────────────────────────────────────────────────────────┐
│                   TRADY RISK ENGINE                    │
│                                                        │
│  1. Drawdown Circuit Breaker Check                     │
│  2. Sizing Rule Calculation (Fractional / ATR / Vol)   │
│  3. Maximum Position Exposure Cap ($ limit)            │
│  4. Maximum Position Concentration Cap (% equity)     │
│  5. Maximum Portfolio Gross Exposure Cap (% equity)   │
│  6. Available Cash & Fee Buffer Check                  │
│                                                        │
│  Output:                                               │
│    - RiskDecision (ACCEPTED / REDUCED / REJECTED)      │
│    - Approved Order (or None if rejected)              │
└────────────────────────────────────────────────────────┘
               │
               ▼ [Approved Order + Audit Record]
Simulated Execution Queue (Bar T+1 Open)
```

---

## 3. Configurable Research Constraints

### A. Maximum Position Exposure (Dollar Cap)
- **Parameter**: `max_position_exposure` (float, e.g., $25,000.00).
- **Rule**: Total dollar value of any single position cannot exceed the absolute cap:
  $$V_{\text{proj}} = (Q_{\text{current}} + Q_{\text{order}}) \times P \le \text{max\_position\_exposure}$$
- **Action**: If projected value exceeds the limit, the order quantity is reduced to fill the remaining capacity. If remaining capacity is zero or negative, the order is **REJECTED**.

### B. Maximum Position Concentration (% of Portfolio Equity)
- **Parameter**: `max_position_concentration` (float, e.g., 0.20 for 20%).
- **Rule**: Single asset exposure cannot exceed a configured fraction of total mark-to-market equity:
  $$\frac{V_{\text{proj}}}{E_t} \le \text{max\_position\_concentration}$$
- **Action**: If an asset's concentration breaches the limit, order size is scaled down (**REDUCED**).

### C. Maximum Portfolio Gross Exposure (Aggregate Leverage Cap)
- **Parameter**: `max_portfolio_exposure` (float, e.g., 1.0 for 100% unleveraged).
- **Rule**: Aggregate gross exposure across all positions combined cannot exceed the portfolio limit:
  $$\frac{\sum |V_i| + (Q_{\text{order}} \times P)}{E_t} \le \text{max\_portfolio\_exposure}$$
- **Action**: Any order that would push gross exposure above the threshold is reduced to the available capacity or **REJECTED**.

### D. Maximum Simulated Drawdown Circuit Breaker
- **Parameter**: `max_drawdown_limit` (float, e.g., 0.15 for 15%).
- **Rule**: If the portfolio drawdown from its high-water mark breaches the limit:
  $$\text{DD}_t = \frac{\text{HWM}_t - E_t}{\text{HWM}_t} \ge \text{max\_drawdown\_limit}$$
- **Action**:
  - **Risk-Increasing Orders (BUY)**: Strictly **REJECTED**. Trading is halted to protect capital.
  - **Risk-Decreasing Orders (SELL)**: **ACCEPTED**. Exiting or reducing positions to de-risk is always permitted.

### E. Available Cash & Fee Buffer
- **Parameter**: `cash_buffer_pct` (float, e.g., 0.01 for 1%).
- **Rule**: Simulated purchases cannot consume 100% of liquid cash, reserving a buffer for commissions and slippage.

---

## 4. Position Sizing Rules

The Risk Engine supports 4 distinct position sizing modes:

1. **Fractional Equity Sizing (`fractional_equity`)**:
   $$Q = \frac{E_t \times \text{target\_position\_pct}}{P}$$
2. **Fixed Cash Sizing (`fixed_cash`)**:
   $$Q = \frac{\min(\text{fixed\_cash\_amount}, \text{cash} \times (1 - \text{buffer}))}{P}$$
3. **ATR-Based Fixed Risk Sizing (`atr_risk`)**:
   Sizes position inversely proportional to recent price range (ATR):
   $$Q = \frac{E_t \times \text{target\_risk\_pct}}{\text{ATR} \times \text{atr\_multiplier}}$$
4. **Volatility Target Sizing (`volatility_target`)**:
   Scales position size by inverse asset volatility to target constant portfolio risk contribution:
   $$Q = \frac{E_t \times \text{target\_volatility}}{\sigma_{\text{ann}} \times P}$$

---

## 5. Audit Trail & Decision Records

Every evaluated order produces an immutable `RiskDecision` audit record:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `decision_id` | `str` | Unique identifier (e.g. `risk_a1b2c3d4`) |
| `timestamp` | `datetime` | Evaluation bar timestamp |
| `symbol` | `str` | Asset ticker symbol |
| `action` | `str` | `BUY` or `SELL` |
| `status` | `str` | `ACCEPTED`, `REDUCED`, or `REJECTED` |
| `requested_quantity` | `float` | Proposed quantity from strategy |
| `approved_quantity` | `float` | Final quantity approved by risk engine |
| `requested_value` | `float` | Dollar value of proposed order |
| `approved_value` | `float` | Dollar value of approved order |
| `price` | `float` | Evaluation mark price |
| `reason` | `str` | Transparent human-readable explanation |
| `constraint_triggered` | `str` | Name of binding constraint or `None` |
| `current_equity` | `float` | Portfolio equity at evaluation |
| `current_drawdown` | `float` | Current drawdown magnitude |
| `portfolio_exposure_pct` | `float` | Gross exposure ratio at evaluation |
| `position_concentration_pct` | `float` | Single asset concentration at evaluation |

All decision records are saved to `risk_decisions.parquet` and rendered in the interactive `report.html` report.

---

## 6. Python API Example

```python
from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.risk.config import RiskConfig
from trady.risk.engine import RiskEngine

# 1. Configure strict risk constraints
risk_config = RiskConfig(
    max_position_exposure=25_000.0,  # Max $25k in single asset
    max_position_concentration=0.20,  # Max 20% equity concentration
    max_portfolio_exposure=1.0,  # Max 100% gross leverage
    max_drawdown_limit=0.10,  # 10% drawdown circuit breaker
    sizing_method="atr_risk",  # Volatility-based sizing
    target_risk_pct=0.01,  # 1% equity risk per trade
    atr_multiplier=2.0,  # 2x ATR stop distance
)

# 2. Instantiate Risk Engine and Backtester
risk_engine = RiskEngine(config=risk_config)
engine = BacktestEngine(
    config=BacktestConfig(initial_cash=100_000.0),
    risk_engine=risk_engine,
)

# 3. Run simulation
result = engine.run(data=df, strategy=strategy)

# 4. Inspect risk decisions
decisions_df = result.to_risk_decisions_dataframe()
print(decisions_df[["symbol", "status", "approved_quantity", "reason"]])
```
