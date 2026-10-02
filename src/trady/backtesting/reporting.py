"""Reporting and artifact generation for TRADY backtest experiments."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trady.backtesting.engine import BacktestResult


def generate_html_report(
    result: BacktestResult,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a clean, self-contained HTML research report."""
    m = result.metrics
    cfg = result.config
    extra = extra_metadata or {}
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    def pct(val: float | None) -> str:
        if val is None:
            return "N/A"
        return f"{val * 100.0:+.2f}%"

    def num(val: float | None, dec: int = 2) -> str:
        if val is None:
            return "N/A"
        return f"{val:.{dec}f}"

    def curr(val: float | None) -> str:
        if val is None:
            return "N/A"
        return f"${val:,.2f}"

    trades_rows = ""
    for t in result.trades[:100]:
        if t.net_pnl > 0:
            pnl_class = "profit"
        elif t.net_pnl < 0:
            pnl_class = "loss"
        else:
            pnl_class = "neutral"
        trades_rows += f"""
        <tr>
            <td>{t.trade_id}</td>
            <td>{t.symbol}</td>
            <td>{t.entry_time}</td>
            <td>{t.exit_time}</td>
            <td>{t.side}</td>
            <td>{t.quantity:.4f}</td>
            <td>${t.entry_price:.2f}</td>
            <td>${t.exit_price:.2f}</td>
            <td class="{pnl_class}">${t.net_pnl:+.2f}</td>
            <td class="{pnl_class}">{t.pnl_pct * 100.0:+.2f}%</td>
            <td>${t.total_fees:.2f}</td>
            <td>{t.holding_bars}</td>
            <td><span class="badge">{t.exit_reason}</span></td>
        </tr>
        """

    tot_ret_cls = "profit" if m.total_return >= 0 else "loss"
    ann_ret_cls = "profit" if m.annualized_return >= 0 else "loss"
    stop_loss_str = pct(cfg.stop_loss_pct) if cfg.stop_loss_pct else "None"
    take_profit_str = pct(cfg.take_profit_pct) if cfg.take_profit_pct else "None"
    ticket_fee_str = f"{cfg.commission_bps} bps + ${cfg.fixed_fee_per_order:.2f} ticket"
    pos_size_str = f"{cfg.position_size_type} ({cfg.position_size_value})"

    if not result.trades:
        trades_section = (
            '<p style="color:var(--text-secondary);">'
            "No trades were executed during this simulation.</p>"
        )
    else:
        trades_section = f"""
        <table>
            <thead>
                <tr>
                    <th>Trade ID</th>
                    <th>Symbol</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry Px</th>
                    <th>Exit Px</th>
                    <th>Net PnL</th>
                    <th>Return</th>
                    <th>Fees</th>
                    <th>Bars</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {trades_rows}
            </tbody>
        </table>
        """

    extra_section = ""
    if extra:
        extra_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in extra.items()
        )
        extra_section = f"""
        <h2>Execution Context</h2>
        <table>
            <tr><th>Attribute</th><th>Value</th></tr>
            {extra_rows}
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRADY Research Backtest Report</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --profit: #3fb950;
            --loss: #f85149;
            --badge-bg: #21262d;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        h1 {{
            margin: 0;
            color: #ffffff;
            font-size: 26px;
        }}
        .meta-subtitle {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
        }}
        .metric-title {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 600;
            margin-top: 5px;
            color: #ffffff;
        }}
        .profit {{ color: var(--profit); }}
        .loss {{ color: var(--loss); }}
        .neutral {{ color: var(--text-secondary); }}
        h2 {{
            font-size: 18px;
            color: #ffffff;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 25px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--badge-bg);
            color: var(--text-secondary);
            font-weight: 600;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .badge {{
            background: var(--badge-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            border: 1px solid var(--border-color);
        }}
        .disclaimer {{
            background-color: var(--card-bg);
            border-left: 4px solid var(--accent);
            padding: 12px 16px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 40px;
            border-radius: 0 6px 6px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TRADY &bull; Quantitative Research Backtest Report</h1>
            <div class="meta-subtitle">
                Generated at: {generated_at} |
                Initial Capital: {curr(m.initial_cash)} |
                Final Equity: {curr(m.final_equity)} |
                Evaluation Engine: Anti-Lookahead v1.0
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="metric-title">Total Return</div>
                <div class="metric-value {tot_ret_cls}">{pct(m.total_return)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Annualized Return</div>
                <div class="metric-value {ann_ret_cls}">{pct(m.annualized_return)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value">{num(m.sharpe_ratio)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Sortino Ratio</div>
                <div class="metric-value">{num(m.sortino_ratio)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value loss">{pct(m.maximum_drawdown)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value">{pct(m.win_rate)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Profit Factor</div>
                <div class="metric-value">{num(m.profit_factor)}</div>
            </div>
            <div class="card">
                <div class="metric-title">Completed Trades</div>
                <div class="metric-value">{m.trade_count}</div>
            </div>
        </div>

        <h2>Simulation Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
            <tr>
                <td>Initial Cash</td><td>{curr(cfg.initial_cash)}</td>
                <td>Commission</td><td>{ticket_fee_str}</td>
            </tr>
            <tr>
                <td>Slippage</td><td>{cfg.slippage_bps} bps</td>
                <td>Max Bar Volume Pct</td><td>{cfg.max_volume_pct * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Position Sizing</td><td>{pos_size_str}</td>
                <td>Max Gross Leverage</td><td>{cfg.max_gross_leverage:.1f}x</td>
            </tr>
            <tr>
                <td>Holding Period Bars</td><td>{cfg.holding_period_bars or "None"}</td>
                <td>Risk-Free Benchmark</td><td>{cfg.risk_free_rate * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Stop Loss Pct</td><td>{stop_loss_str}</td>
                <td>Take Profit Pct</td><td>{take_profit_str}</td>
            </tr>
        </table>

        <h2>Performance Diagnostics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
            <tr>
                <td>Annualized Volatility</td><td>{pct(m.volatility)}</td>
                <td>Portfolio Turnover</td><td>{num(m.turnover, 3)}x</td>
            </tr>
            <tr>
                <td>Total Fees Paid</td><td>{curr(m.total_fees)}</td>
                <td>Net Trade Profit</td><td>{curr(m.net_profit)}</td>
            </tr>
            <tr>
                <td>Winning Trades</td><td>{m.win_count}</td>
                <td>Losing Trades</td><td>{m.loss_count}</td>
            </tr>
            <tr>
                <td>Average Win</td><td>{curr(m.average_win)}</td>
                <td>Average Loss</td><td>{curr(m.average_loss)}</td>
            </tr>
            <tr>
                <td>Gross Profit</td><td>{curr(m.gross_profit)}</td>
                <td>Gross Loss</td><td>{curr(m.gross_loss)}</td>
            </tr>
            <tr>
                <td>Market Exposure Time</td><td>{pct(m.exposure_time_pct)}</td>
                <td>Equity Observations</td><td>{len(result.equity_curve)} bars</td>
            </tr>
        </table>

        {extra_section}

        <h2>Trade Execution Log (Roundtrips)</h2>
        {trades_section}

        <div class="disclaimer">
            <strong>TRADY Educational Research Notice:</strong>
            This backtest simulation is strictly for quantitative hypothesis
            testing and statistical research. It does not guarantee historical
            repeatability or future profitability. Past performance in backtests
            is subject to regime changes and market uncertainties. TRADY does
            not execute live trading orders or manage capital.
        </div>
    </div>
</body>
</html>
"""
    return html


def save_backtest_artifacts(
    result: BacktestResult,
    output_dir: Path | str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save all 5 required backtest experiment artifacts to disk.

    Artifacts:
    1. metrics.json
    2. equity_curve.parquet
    3. trades.parquet
    4. configuration.json
    5. report.html

    Args:
        result: BacktestResult instance.
        output_dir: Destination folder path.
        extra_metadata: Additional context to store in configuration.json.

    Returns:
        Dictionary mapping artifact name to saved file Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # 1. metrics.json
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics.to_dict(), f, indent=4)
    artifacts["metrics.json"] = metrics_path

    # 2. equity_curve.parquet
    equity_df = result.to_equity_dataframe()
    equity_path = out_path / "equity_curve.parquet"
    equity_df.to_parquet(equity_path, index=False)
    artifacts["equity_curve.parquet"] = equity_path

    # 3. trades.parquet
    trades_df = result.to_trades_dataframe()
    trades_path = out_path / "trades.parquet"
    trades_df.to_parquet(trades_path, index=False)
    artifacts["trades.parquet"] = trades_path

    # 4. configuration.json
    config_dict = {
        "backtest_config": result.config.to_dict(),
        "extra_metadata": extra_metadata or {},
        "saved_at": datetime.now(UTC).isoformat(),
    }
    config_path = out_path / "configuration.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    artifacts["configuration.json"] = config_path

    # 5. report.html
    html_content = generate_html_report(result, extra_metadata=extra_metadata)
    report_path = out_path / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    artifacts["report.html"] = report_path

    return artifacts
