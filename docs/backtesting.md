# TRADY Historical Backtesting Engine

## 1. Overview & Educational Scope

The TRADY Historical Backtesting Engine provides a rigorous, modular simulation environment to evaluate quantitative trading hypotheses driven by statistical and machine-learning model predictions.

### Mandatory Research Guardrails
- **Educational & Analytical Focus Only**: TRADY is strictly an educational research and paper-trading platform. It does not provide financial advice, does not claim guaranteed trading profits, and does not execute real-money transactions.
- **No Overfitting**: Backtesting is designed for robust evaluation and hypothesis invalidation, not for curve-fitting or hyperparameter mining.
- **Independence**: The backtesting engine consumes historical OHLCV data alongside model predictions from the Modeling Engine. It does not dictate model architecture.

---

## 2. Anti-Lookahead Execution Architecture

Real-world algorithmic trading faces severe execution realities that naive backtests frequently ignore. TRADY enforces strict causal isolation to prevent simulation leakage:

```
Timeline:
Bar T:
  [OPEN ─── HIGH ─── LOW ─── CLOSE]
                              └─── Features at or before T
                              └─── Model inference P(T) generated
                              └─── Strategy generates Order(T+1)
Bar T+1:
  [OPEN ─── HIGH ─── LOW ─── CLOSE]
     └─── Order(T+1) executes at OPEN + Slippage
     └─── Position updated in PortfolioTracker
     └─── Mark-to-market at Bar T+1 CLOSE
```

### Invariants Enforced:
1. **Next-Bar Execution**:
   Signals generated using Bar $T$ close are strictly executed at Bar $T+1$ `open`. Same-bar execution (executing on the same bar close used to compute signals) is strictly forbidden.
2. **Physical Price Range Clamping**:
   - Buy fill price: $\min(P_{\text{open}} \times (1 + \text{slippage}), P_{\text{high}})$.
   - Sell fill price: $\max(P_{\text{open}} \times (1 - \text{slippage}), P_{\text{low}})$.
   Execution can never occur outside the high-low boundary of the bar.
3. **Bar Volume Liquidity Caps**:
   A simulated order cannot exceed `max_volume_pct` (default: 10%) of the bar's actual traded volume, preventing impossible size assumptions in thin markets.
4. **Frictions & Transaction Costs**:
   Every simulated trade incurs proportional commission (bps), fixed ticket fees ($), and bid/ask slippage (bps).

---

## 3. Mathematical Metric Definitions

The backtester calculates 12 core performance and risk metrics:

| Metric | Mathematical Formula | Description |
| :--- | :--- | :--- |
| **Total Return** | $\frac{E_{\text{final}} - E_{\text{initial}}}{E_{\text{initial}}}$ | Cumulative percentage change in portfolio equity. |
| **Annualized Return (CAGR)** | $\left(\frac{E_{\text{final}}}{E_{\text{initial}}}\right)^{\frac{252}{N}} - 1$ | Geometric compound annual growth rate over $N$ trading bars. |
| **Volatility** | $\sigma(r) \times \sqrt{252}$ | Annualized standard deviation of bar-to-bar equity returns. |
| **Maximum Drawdown** | $\max_{t} \left(\frac{\text{HWM}_t - E_t}{\text{HWM}_t}\right)$ | Maximum peak-to-trough decline in portfolio equity. |
| **Sharpe Ratio** | $\frac{\text{CAGR} - r_f}{\text{Volatility}}$ | Risk-adjusted return relative to risk-free benchmark $r_f$. |
| **Sortino Ratio** | $\frac{\text{CAGR} - r_f}{\sigma_{\text{down}} \times \sqrt{252}}$ | Risk-adjusted return penalizing only downside volatility ($r_t < 0$). |
| **Win Rate** | $\frac{N_{\text{wins}}}{N_{\text{trades}}}$ | Proportion of completed roundtrip trades with net profit $> 0$. |
| **Average Win** | $\frac{1}{N_{\text{wins}}} \sum \text{NetPnL}_{\text{wins}}$ | Average dollar gain of winning roundtrip trades. |
| **Average Loss** | $\frac{1}{N_{\text{losses}}} \sum \text{NetPnL}_{\text{losses}}$ | Average dollar loss of losing roundtrip trades. |
| **Profit Factor** | $\frac{\sum \text{GrossWins}}{\sum \|\text{GrossLosses}\|}$ | Ratio of total gross profit to total gross loss. |
| **Turnover** | $\frac{\sum \text{TradedValue}}{2 \times \bar{E}}$ | Portfolio turnover ratio measuring trading intensity. |
| **Trade Count** | $N_{\text{trades}}$ | Total number of closed roundtrip trades. |

---

## 4. Chronological Walk-Forward Evaluation

To rigorously test whether model-driven strategies persist across distinct market regimes without look-ahead bias, TRADY provides chronological walk-forward cross-validation:

```
Fold 0: [ Train Window 0 ] -> [ Test Window 0 (OOS) ]
Fold 1: [ Train Window 1 (Expanded/Rolled) ] -> [ Test Window 1 (OOS) ]
Fold 2: [ Train Window 2 (Expanded/Rolled) ] -> [ Test Window 2 (OOS) ]
Overall Out-Of-Sample: [ Test 0 ] + [ Test 1 ] + [ Test 2 ]
```

1. **Expanding Window**: Train start is anchored at $t=0$, training size grows with each fold.
2. **Rolling Window**: Train window moves forward maintaining a constant lookback duration.
3. **Out-of-Sample Stitching**: Out-of-sample test equity curves and roundtrip trades are combined into an un-snooped aggregate track record.

---

## 5. Experiment Artifacts

Every backtest generates 5 standardized research artifacts saved in `reports/backtests/<run_id>/`:

1. `metrics.json`: Standardized JSON containing all 12 core performance metrics, trade statistics, and portfolio diagnostic metrics.
2. `equity_curve.parquet`: High-resolution bar-by-bar portfolio snapshots (`timestamp`, `cash`, `positions_value`, `portfolio_equity`, `daily_return`, `drawdown`, `gross_exposure`, `net_exposure`).
3. `trades.parquet`: Complete roundtrip trade audit log (`trade_id`, `symbol`, `entry_time`, `exit_time`, `quantity`, `entry_price`, `exit_price`, `gross_pnl`, `net_pnl`, `total_fees`, `holding_bars`, `exit_reason`).
4. `configuration.json`: Full execution parameters, risk limits, model details, and run metadata.
5. `report.html`: Self-contained interactive HTML scorecard with styled metrics cards, parameter breakdowns, and trade execution tables.

---

## 6. CLI Usage

```bash
# 1. Run a backtest using a trained machine-learning model
trady backtest run data/processed/BTC_USDT_1d.parquet \
  --model models/rf_model.joblib \
  --entry-threshold 0.55 \
  --exit-threshold 0.45 \
  --holding-bars 10 \
  --stop-loss 0.05 \
  --take-profit 0.10

# 2. Inspect an existing backtest artifact directory
trady backtest inspect reports/backtests/run_20261002_143000

# 3. Execute chronological walk-forward validation
trady backtest walk-forward data/processed/modeling_dataset.parquet \
  --target target_binary_up_5d \
  --model-type logistic_regression \
  --folds 3 \
  --window-type expanding
```
