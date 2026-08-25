"""Runner-level holdout split plugin: in-sample vs out-of-sample.

Divides the backtest data range at ``split_fraction`` of its span and runs
the normal execution path once per window. The merged :class:`BacktestResult`
promotes out-of-sample metrics to the top-level objective fields (so
optimizers validate on unseen data) and stores per-window metrics as
first-class columns (``in_sample_*`` / ``out_of_sample_*``).
"""

import pandas as pd

from ..core.job import BacktestResult, JobStatus
from .base import RunnerPlugin, Window

IN_SAMPLE = "in_sample"
OUT_OF_SAMPLE = "out_of_sample"

_WINDOW_LABELS = {
    IN_SAMPLE: "In-Sample",
    OUT_OF_SAMPLE: "Out-of-Sample",
}


class TrainValSplit(RunnerPlugin):
    name = "train_val_split"

    def __init__(self, split_fraction: float) -> None:
        if not 0.0 < float(split_fraction) < 1.0:
            raise ValueError(
                f"train_val_split fraction must be in (0, 1); got {split_fraction}"
            )
        self.split_fraction = float(split_fraction)

    def split_timestamp(self, first_ts: pd.Timestamp, last_ts: pd.Timestamp):
        """Timestamp dividing in-sample from out-of-sample windows."""
        return first_ts + (last_ts - first_ts) * self.split_fraction

    def expand(
        self, cfg, df: pd.DataFrame | None
    ) -> dict[str, Window]:
        """Derive IS/OOS windows from the (already date-filtered) data.

        The boundary bar belongs to OOS only; IS ends one bar interval
        earlier. When a bar frame is supplied, each OOS slice preloads
        ``cfg.warmup_bars`` before its trading start so indicators warm up
        while orders stay gated via ``active_from``.
        """
        if df is None:
            # L2 / loader-fetched mode: explicit bounds are required.
            if not cfg.end:
                raise ValueError(
                    "train_val_split requires an explicit 'end' date"
                )
            range_start = pd.Timestamp(cfg.start, tz="UTC")
            range_end = pd.Timestamp(cfg.end, tz="UTC")
            slices: dict[str, pd.DataFrame | None] = {
                IN_SAMPLE: None,
                OUT_OF_SAMPLE: None,
            }
        else:
            ts = df["timestamp"]
            if len(df) < 2:
                raise ValueError(
                    f"Not enough bars for train/val split ({len(df)} rows)."
                )
            range_start = ts.iloc[0]
            range_end = ts.iloc[-1]

        from ..utils import interval_delta  # deferred: utils pulls strategies

        bar_delta = interval_delta(cfg.interval)
        split_ts = self.split_timestamp(range_start, range_end)
        is_end = split_ts - bar_delta

        if df is not None:
            warmup = max(getattr(cfg, "warmup_bars", 0) or 0, 0)
            oos_load_from = (
                split_ts - warmup * bar_delta if warmup else split_ts
            )
            oos_load_from = max(oos_load_from, range_start)
            slices = {
                IN_SAMPLE: df[
                    (ts >= range_start) & (ts <= is_end)
                ].reset_index(drop=True),
                OUT_OF_SAMPLE: df[
                    (ts >= oos_load_from) & (ts <= range_end)
                ].reset_index(drop=True),
            }

        return {
            IN_SAMPLE: Window(
                _WINDOW_LABELS[IN_SAMPLE], range_start, is_end, slices[IN_SAMPLE]
            ),
            OUT_OF_SAMPLE: Window(
                _WINDOW_LABELS[OUT_OF_SAMPLE],
                split_ts,
                range_end,
                slices[OUT_OF_SAMPLE],
            ),
        }

    def combine(
        self,
        job_id: str,
        results: dict[str, BacktestResult],
        windows: dict[str, Window],
    ) -> BacktestResult:
        """Merge per-window results; OOS metrics become the top-level ones."""
        is_res = results.get(IN_SAMPLE)
        oos_res = results.get(OUT_OF_SAMPLE)
        if oos_res is None or oos_res.status != JobStatus.DONE:
            fail = oos_res or is_res
            return BacktestResult(
                job_id=job_id,
                status=fail.status if fail else JobStatus.FAILED,
                error=fail.error if fail else "Missing window result",
            )

        return BacktestResult(
            job_id=job_id,
            status=JobStatus.DONE,
            sharpe_ratio=oos_res.sharpe_ratio,
            num_trades=oos_res.num_trades,
            pnl=oos_res.pnl,
            sqn=oos_res.sqn,
            stats=oos_res.stats,
            positions=oos_res.positions,
            fills=oos_res.fills,
            error=None,
            duration_seconds=(is_res.duration_seconds if is_res else 0.0)
            + oos_res.duration_seconds,
            funding_pnl=oos_res.funding_pnl,
            # Per-window metrics as first-class columns
            in_sample_sharpe_ratio=is_res.sharpe_ratio if is_res else None,
            in_sample_num_trades=is_res.num_trades if is_res else None,
            in_sample_pnl=is_res.pnl if is_res else None,
            in_sample_sqn=is_res.sqn if is_res else None,
            in_sample_funding_pnl=is_res.funding_pnl if is_res else None,
            in_sample_duration_seconds=is_res.duration_seconds if is_res else None,
            out_of_sample_sharpe_ratio=oos_res.sharpe_ratio,
            out_of_sample_num_trades=oos_res.num_trades,
            out_of_sample_pnl=oos_res.pnl,
            out_of_sample_sqn=oos_res.sqn,
            out_of_sample_funding_pnl=oos_res.funding_pnl,
            out_of_sample_duration_seconds=oos_res.duration_seconds,
        )

    def summarize(self, results: dict[str, BacktestResult]) -> None:
        is_res = results.get(IN_SAMPLE)
        oos_res = results.get(OUT_OF_SAMPLE)
        if is_res is None or oos_res is None:
            return
        print("\n========== TRAIN/VAL SPLIT SUMMARY ==========")
        print(f"{'metric':<16} {'in-sample':>14} {'out-of-sample':>15}")
        for name, is_v, oos_v in (
            ("Sharpe", is_res.sharpe_ratio, oos_res.sharpe_ratio),
            ("Trades", is_res.num_trades, oos_res.num_trades),
            ("PnL", is_res.pnl, oos_res.pnl),
            ("SQN", is_res.sqn, oos_res.sqn),
        ):
            from ..core.runner import _fmt_metric

            print(f"{name:<16} {_fmt_metric(is_v):>14} {_fmt_metric(oos_v):>15}")
