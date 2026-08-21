"""Runner-level holdout split plugin: in-sample vs out-of-sample.

Divides the backtest data range at ``split_fraction`` of its span and runs
the normal execution path once per window. The merged :class:`BacktestResult`
carries per-window metrics under ``splits`` and promotes the out-of-sample
metrics to the top-level objective fields, so optimizers validate on unseen
data.
"""

import pandas as pd

from ..core.job import BacktestResult, JobStatus

IN_SAMPLE = "in_sample"
OUT_OF_SAMPLE = "out_of_sample"

_WINDOW_LABELS = {
    IN_SAMPLE: "In-Sample",
    OUT_OF_SAMPLE: "Out-of-Sample",
}


class TrainValSplit:
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

    def combine(
        self,
        job_id: str,
        results: dict[str, BacktestResult],
        windows: dict[str, tuple],
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

        splits = {}
        for key, res in results.items():
            start, end = windows.get(key, (None, None))
            splits[key] = {
                "label": _WINDOW_LABELS.get(key, key),
                "start": str(start) if start else None,
                "end": str(end) if end else None,
                "sharpe_ratio": res.sharpe_ratio,
                "num_trades": res.num_trades,
                "pnl": res.pnl,
                "sqn": res.sqn,
                "duration_seconds": res.duration_seconds,
            }

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
            splits=splits,
        )
