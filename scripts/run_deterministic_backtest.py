"""Deterministic end-to-end sample backtest experiment for TRADY.

Generates reproducible market data, features, targets, fits a research model,
simulates a disciplined strategy preventing look-ahead bias, and exports:
- metrics.json
- equity_curve.parquet
- trades.parquet
- configuration.json
- report.html
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.backtesting.reporting import save_backtest_artifacts
from trady.backtesting.strategy import ModelDrivenStrategy
from trady.features.pipeline import FeaturePipeline
from trady.models.linear import TradyLogisticRegression
from trady.targets.pipeline import TargetPipeline


def main() -> None:
    print("=" * 70)
    print(" TRADY DETERMINISTIC RESEARCH BACKTEST EXPERIMENT")
    print("=" * 70)

    # 1. Generate deterministic synthetic OHLCV time series (200 bars)
    np.random.seed(42)
    n_bars = 200
    base_time = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

    p = 100.0
    opens, highs, lows, closes, vols, timestamps = [], [], [], [], [], []

    for i in range(n_bars):
        ret = np.random.normal(0.001, 0.015)
        # Periodic trend regime
        if (i // 40) % 2 == 1:
            ret += 0.005  # Bullish impulse
        else:
            ret -= 0.002  # Consolidation

        o = p
        c = p * (1.0 + ret)
        h = max(o, c) * (1.0 + abs(np.random.normal(0.002, 0.003)))
        low = min(o, c) * (1.0 - abs(np.random.normal(0.002, 0.003)))
        v = float(np.random.uniform(50_000, 200_000))

        opens.append(o)
        highs.append(h)
        lows.append(low)
        closes.append(c)
        vols.append(v)
        timestamps.append(base_time + timedelta(days=i))
        p = c

    raw_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SPY"] * n_bars,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )
    market_table = pa.Table.from_pandas(raw_df)

    # 2. Compute Features
    print("\n[1/4] Computing technical and momentum features...")
    feat_pipe = FeaturePipeline()
    feat_table, feat_report = feat_pipe.compute(market_table)
    print(f"  [OK] Generated {feat_table.num_columns} columns.")

    # 3. Compute Forward Targets
    print("\n[2/4] Computing forward return targets...")
    tgt_pipe = TargetPipeline()
    modeling_ds, tgt_report = tgt_pipe.compute(
        market_data=market_table,
        features_data=feat_table,
    )
    target_name = "target_binary_up_5d"
    print(f"  [OK] Forward target selected: {target_name}")

    # 4. Train Research Model on in-sample partition (first 120 bars)
    print("\n[3/4] Training chronological research model...")
    ds_df = modeling_ds.table.to_pandas()
    train_df = ds_df.iloc[:120].dropna(
        subset=list(modeling_ds.feature_names) + [target_name]
    )
    test_df = ds_df.iloc[120:].copy().reset_index(drop=True)

    x_train = train_df[list(modeling_ds.feature_names)].to_numpy()
    y_train = train_df[target_name].to_numpy()

    model = TradyLogisticRegression(random_seed=42)
    model.fit(
        X=x_train,
        y=y_train,
        feature_names=modeling_ds.feature_names,
        target_name=target_name,
    )

    # Predict out-of-sample
    x_test = test_df[list(modeling_ds.feature_names)].to_numpy()
    x_test = np.nan_to_num(x_test, nan=0.0)
    pred_probs = model.predict_proba(x_test)[:, 1]

    # 5. Simulate Out-of-Sample Backtest
    print("\n[4/4] Running anti-lookahead historical backtest simulation...")
    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_bps=5.0,
        fixed_fee_per_order=1.0,
        slippage_bps=5.0,
        position_size_type="fractional_equity",
        position_size_value=0.95,
        holding_period_bars=10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
    )

    strategy = ModelDrivenStrategy(
        config=config,
        entry_threshold=0.52,
        exit_threshold=0.48,
        is_classification=True,
    )

    engine = BacktestEngine(config=config)
    result = engine.run(
        data=test_df,
        strategy=strategy,
        predictions=pred_probs,
        close_positions_at_end=True,
    )

    # 6. Save Artifacts
    output_dir = Path("reports") / "sample_backtest"
    artifacts = save_backtest_artifacts(
        result=result,
        output_dir=output_dir,
        extra_metadata={
            "experiment_name": "deterministic_sample_backtest",
            "model_type": "logistic_regression",
            "target": target_name,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
    )

    m = result.metrics
    print("\n" + "=" * 70)
    print(" BACKTEST EVALUATION SCORECARD:")
    print("=" * 70)
    print(f" Initial Capital:     ${m.initial_cash:,.2f}")
    print(f" Final Equity:        ${m.final_equity:,.2f}")
    print(f" Total Return:        {m.total_return * 100.0:+.2f}%")
    print(f" Annualized Return:   {m.annualized_return * 100.0:+.2f}%")
    print(f" Annual Volatility:   {m.volatility * 100.0:.2f}%")
    print(f" Sharpe Ratio:        {m.sharpe_ratio:.2f}")
    print(f" Sortino Ratio:       {m.sortino_ratio:.2f}")
    print(f" Max Drawdown:        {m.maximum_drawdown * 100.0:.2f}%")
    print(f" Trade Count:         {m.trade_count}")
    print(f" Win Rate:            {m.win_rate * 100.0:.1f}%")
    print(f" Profit Factor:       {m.profit_factor:.2f}")
    print(f" Portfolio Turnover:  {m.turnover:.2f}x")
    print(f" Total Fees Paid:     ${m.total_fees:,.2f}")
    print("-" * 70)
    print(" GENERATED EXPERIMENT ARTIFACTS:")
    for name, path in artifacts.items():
        print(f"  * {name:22s}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
