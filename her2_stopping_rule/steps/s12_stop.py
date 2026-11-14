"""Step 12: the stopping rule.

The repository's thesis, written as code. Walk down the ranking and stop where
the next design's predicted gain falls below the spread between two independent
scoring sources. Past that point the ranking is reading its own noise, and
adding designs is not selection but decoration.

The threshold is measured, not chosen: it is the observed disagreement between
the two folding models on these designs. When step 7 has not run there is no
measured spread, and the rule says so rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


@dataclass(frozen=True)
class StopDecision:
    """Where the ranking stopped being informative.

    Attributes:
        threshold: The spread the gain had to beat, in score units.
        n_kept: Designs above the stopping point.
        n_available: Designs that were available to keep.
        basis: How the threshold was derived.
        stopped_at: Identifier of the first design below the line, if any.
    """

    threshold: float
    n_kept: int
    n_available: int
    basis: str
    stopped_at: str | None


def decide(df: pl.DataFrame, cfg: Config) -> StopDecision:
    """Find the point where predicted gain falls below measured noise.

    Args:
        df: Live designs, ranked, carrying ``p_high`` and ``model_spread``.
        cfg: Pipeline configuration, for the submission cap.

    Returns:
        The stopping decision.
    """
    ranked = df.sort("rank", nulls_last=True)
    spreads = ranked["model_spread"].drop_nulls()

    if spreads.len() == 0:
        return StopDecision(
            threshold=float("nan"),
            n_kept=min(ranked.height, cfg.n_submit),
            n_available=ranked.height,
            basis=(
                "no measured spread: step 7 did not run, so the cap is the "
                "submission limit rather than a stopping rule"
            ),
            stopped_at=None,
        )

    median = spreads.median()
    threshold = float(median) if isinstance(median, (int, float)) else float("nan")
    scores = ranked["p_high"].fill_null(0.0).to_list()
    ids = ranked["design_id"].to_list()

    n_kept = len(scores)
    stopped_at = None
    for i in range(1, len(scores)):
        if abs(scores[i - 1] - scores[i]) < threshold:
            n_kept, stopped_at = i, ids[i]
            break

    return StopDecision(
        threshold=threshold,
        n_kept=min(n_kept, cfg.n_submit),
        n_available=ranked.height,
        basis="median disagreement between the two folding models on these designs",
        stopped_at=stopped_at,
    )


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Apply the stopping rule and record its threshold.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with designs past the stopping point marked
        ``dropped_by = "stop"``.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    decision = decide(live, cfg)
    keep = (
        live.sort("rank", nulls_last=True).head(decision.n_kept)["design_id"].to_list()
    )
    out = tbl.drop(df, pl.col("design_id").is_in(keep).not_(), "stop")

    if funnel is not None:
        with funnel.step(
            "12_stop",
            rows_in=live.height,
            thresholds={
                "spread_threshold": decision.threshold,
                "basis": decision.basis,
            },
        ) as rec:
            rec.rows_out = tbl.alive(out).height
            rec.notes = f"kept {decision.n_kept} of {decision.n_available}; " + (
                f"stopped at {decision.stopped_at}"
                if decision.stopped_at
                else "no design fell below the threshold"
            )
    return out
