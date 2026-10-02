"""Chronological walk-forward cross-validation and evaluation engine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine, BacktestResult
from trady.backtesting.metrics import BacktestMetrics, calculate_metrics
from trady.backtesting.portfolio import EquityPoint, TradeRecord
from trady.backtesting.strategy import BaseStrategy, ModelDrivenStrategy
from trady.models.base import TradyModel
from trady.models.registry import create_model


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for chronological walk-forward evaluation."""

    n_folds: int = 3
    train_size_bars: int | None = None
    test_size_bars: int | None = None
    window_type: str = "expanding"  # "expanding" or "rolling"
    min_train_bars: int = 50
    backtest_config: BacktestConfig = BacktestConfig()

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1.")
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError("window_type must be 'expanding' or 'rolling'.")
        if self.min_train_bars < 10:
            raise ValueError("min_train_bars must be at least 10.")


@dataclass
class WalkForwardFoldResult:
    """Results from a single out-of-sample walk-forward fold."""

    fold_idx: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_bars: int
    test_bars: int
    metrics: BacktestMetrics
    result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample walk-forward results across all folds."""

    config: WalkForwardConfig
    folds: list[WalkForwardFoldResult]
    combined_equity_curve: list[EquityPoint]
    combined_trades: list[TradeRecord]
    overall_metrics: BacktestMetrics

    def summary_table(self) -> pd.DataFrame:
        """Construct comparison table across all out-of-sample folds."""
        rows = []
        for f in self.folds:
            m = f.metrics
            rows.append(
                {
                    "fold": f.fold_idx,
                    "train_bars": f.train_bars,
                    "test_bars": f.test_bars,
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                    "total_return": m.total_return,
                    "annualized_return": m.annualized_return,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown": m.maximum_drawdown,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "trade_count": m.trade_count,
                }
            )
        # Add overall summary row
        ov = self.overall_metrics
        rows.append(
            {
                "fold": "OVERALL",
                "train_bars": sum(f.train_bars for f in self.folds),
                "test_bars": sum(f.test_bars for f in self.folds),
                "test_start": str(self.folds[0].test_start) if self.folds else "",
                "test_end": str(self.folds[-1].test_end) if self.folds else "",
                "total_return": ov.total_return,
                "annualized_return": ov.annualized_return,
                "sharpe_ratio": ov.sharpe_ratio,
                "max_drawdown": ov.maximum_drawdown,
                "win_rate": ov.win_rate,
                "profit_factor": ov.profit_factor,
                "trade_count": ov.trade_count,
            }
        )
        return pd.DataFrame(rows)


class WalkForwardEvaluator:
    """Executes chronological out-of-sample walk-forward evaluations."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def generate_windows(
        self,
        total_bars: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (train_start, train_end, test_start, test_end) bar indices."""
        n_folds = self.config.n_folds
        min_train = self.config.min_train_bars

        if total_bars <= min_train:
            raise ValueError(
                f"total_bars ({total_bars}) must exceed min_train_bars ({min_train})."
            )

        if self.config.test_size_bars is not None:
            test_size = self.config.test_size_bars
        else:
            avail_for_test = total_bars - min_train
            test_size = max(10, avail_for_test // n_folds)

        if self.config.train_size_bars is not None:
            train_size = self.config.train_size_bars
        else:
            train_size = min_train

        windows: list[tuple[int, int, int, int]] = []
        for k in range(n_folds):
            test_end = total_bars - (n_folds - 1 - k) * test_size
            test_start = test_end - test_size
            if test_start < min_train:
                continue

            if self.config.window_type == "expanding":
                train_start = 0
                train_end = test_start
            else:  # rolling
                train_end = test_start
                train_start = max(0, train_end - train_size)

            if (train_end - train_start) >= min_train and (test_end - test_start) > 0:
                windows.append((train_start, train_end, test_start, test_end))

        if not windows:
            raise ValueError(
                f"Could not generate walk-forward windows for total_bars={total_bars}, "
                f"min_train={min_train}, n_folds={n_folds}."
            )

        return windows

    def evaluate(
        self,
        data: pd.DataFrame,
        feature_names: list[str] | tuple[str, ...],
        target_name: str,
        model_type: str = "logistic_regression",
        task_type: str = "classification",
        model_params: dict[str, Any] | None = None,
        strategy_factory: Callable[..., BaseStrategy] | None = None,
    ) -> WalkForwardResult:
        """Run full walk-forward train/test/simulate sequence."""
        total_bars = len(data)
        windows = self.generate_windows(total_bars)
        fold_results: list[WalkForwardFoldResult] = []

        all_oos_equity_points: list[EquityPoint] = []
        all_oos_trades: list[TradeRecord] = []
        current_equity = self.config.backtest_config.initial_cash

        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            train_df = data.iloc[tr_start:tr_end]
            test_df = data.iloc[te_start:te_end].copy().reset_index(drop=True)

            # Drop NaNs from training data
            train_clean = train_df.dropna(subset=list(feature_names) + [target_name])
            if len(train_clean) < 10:
                continue

            x_train = train_clean[list(feature_names)].to_numpy(dtype=np.float64)
            y_train = train_clean[target_name].to_numpy()

            # Train model
            model: TradyModel = create_model(
                model_type=model_type,
                task_type=task_type,
                parameters=model_params,
                random_seed=self.config.backtest_config.random_seed + fold_idx,
            )
            model.fit(
                X=x_train,
                y=y_train,
                feature_names=tuple(feature_names),
                target_name=target_name,
            )

            # Generate OOS test predictions
            x_test = test_df[list(feature_names)].to_numpy(dtype=np.float64)
            # If any NaN in test features, fill with 0.0 or forward fill for safety
            if np.isnan(x_test).any():
                x_test = np.nan_to_num(x_test, nan=0.0)

            if task_type == "classification":
                pred_test = model.predict_proba(x_test)[:, 1]
            else:
                pred_test = model.predict(x_test)

            # Instantiate strategy
            if strategy_factory is not None:
                strategy = strategy_factory(self.config.backtest_config)
            else:
                strategy = ModelDrivenStrategy(
                    config=self.config.backtest_config,
                    entry_threshold=0.5 if task_type == "classification" else 0.0,
                    exit_threshold=0.5 if task_type == "classification" else 0.0,
                    is_classification=(task_type == "classification"),
                )

            # Run backtest on test slice with current equity base
            fold_bt_config = BacktestConfig(
                initial_cash=current_equity,
                commission_bps=self.config.backtest_config.commission_bps,
                fixed_fee_per_order=self.config.backtest_config.fixed_fee_per_order,
                slippage_bps=self.config.backtest_config.slippage_bps,
                max_volume_pct=self.config.backtest_config.max_volume_pct,
                position_size_type=self.config.backtest_config.position_size_type,
                position_size_value=self.config.backtest_config.position_size_value,
                max_gross_leverage=self.config.backtest_config.max_gross_leverage,
                allow_shorting=self.config.backtest_config.allow_shorting,
                holding_period_bars=self.config.backtest_config.holding_period_bars,
                stop_loss_pct=self.config.backtest_config.stop_loss_pct,
                take_profit_pct=self.config.backtest_config.take_profit_pct,
                risk_free_rate=self.config.backtest_config.risk_free_rate,
                periods_per_year=self.config.backtest_config.periods_per_year,
                random_seed=self.config.backtest_config.random_seed,
            )

            engine = BacktestEngine(config=fold_bt_config)
            sim_res = engine.run(
                data=test_df,
                strategy=strategy,
                predictions=pred_test,
                close_positions_at_end=True,
            )

            fold_res = WalkForwardFoldResult(
                fold_idx=fold_idx,
                train_start=train_df.iloc[0]["timestamp"],
                train_end=train_df.iloc[-1]["timestamp"],
                test_start=test_df.iloc[0]["timestamp"],
                test_end=test_df.iloc[-1]["timestamp"],
                train_bars=len(train_df),
                test_bars=len(test_df),
                metrics=sim_res.metrics,
                result=sim_res,
            )
            fold_results.append(fold_res)

            # Update continuous equity state for next fold
            if sim_res.equity_curve:
                current_equity = sim_res.equity_curve[-1].portfolio_equity
                all_oos_equity_points.extend(sim_res.equity_curve)
            all_oos_trades.extend(sim_res.trades)

        # Calculate overall metrics
        overall_metrics = calculate_metrics(
            equity_curve=all_oos_equity_points,
            trades=all_oos_trades,
            initial_cash=self.config.backtest_config.initial_cash,
            periods_per_year=self.config.backtest_config.periods_per_year,
            risk_free_rate=self.config.backtest_config.risk_free_rate,
        )

        return WalkForwardResult(
            config=self.config,
            folds=fold_results,
            combined_equity_curve=all_oos_equity_points,
            combined_trades=all_oos_trades,
            overall_metrics=overall_metrics,
        )
