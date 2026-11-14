"""Step 4a: the prior-art filter.

This is the only step before submission that rejects designs outright, and it is
the reason the pipeline must be computational rather than hand-curated. Five
hand-picked designs were all rejected by abguard; ten random ones passed half
the time. Plausible-looking mutations collide with the FACS screen precisely
because the screen explored plausible space too.

The filter runs locally and exactly so abguard only ever sees survivors, which
keeps a 900-second CLI call off the hot path. See
:mod:`her2_stopping_rule.novelty` for how 10^14 comparisons become tractable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..config import Config
from ..data import load_blacklist
from ..novelty import FAR, scan

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Reject designs too close to any provided CDR3.

    Args:
        df: The design table, carrying ``cdr3_imgt`` on live rows.
        cfg: Pipeline configuration, for ``min_dist_to_provided``.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``min_dist_to_provided`` and ``nearest_provided_id``
        filled in, and ``dropped_by = "novelty"`` on designs that fail.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    blacklist = load_blacklist(cfg)

    # Scan one edit-distance group at a time. Distance from trastuzumab bounds
    # the distance between any two sequences, so a group only needs references
    # that could possibly be near it. For 3-mutation candidates that is 37,042
    # of 525,042 references: the difference between minutes and an hour.
    loops = live["cdr3_imgt"].to_list()
    groups = live["n_mutations"].to_list()

    min_dist = np.full(len(loops), FAR, dtype=np.int32)
    nearest: list[str | None] = [None] * len(loops)
    n_compared = 0
    refs_used: dict[int, int] = {}

    for k in sorted({g for g in groups if g is not None}):
        rows = [i for i, g in enumerate(groups) if g == k]
        mask = blacklist.relevant_to(int(k), cfg.min_dist_to_provided)
        refs = blacklist.windows[mask]
        ref_ids = [blacklist.ids[i] for i in np.flatnonzero(mask).tolist()]
        refs_used[int(k)] = len(ref_ids)
        if refs.shape[0] == 0:
            continue
        # Compare on the ten-residue window: S1, R2 and Y13 never vary.
        windows = [loops[i][2:12] for i in rows]
        block = np.frombuffer("".join(windows).encode(), dtype=np.uint8).reshape(-1, 10)
        scanned = scan(block, refs)
        n_compared += scanned.n_compared
        for local, row in enumerate(rows):
            min_dist[row] = int(scanned.min_distance[local])
            idx = int(scanned.nearest_index[local])
            nearest[row] = ref_ids[idx] if idx >= 0 else None

    eligible = min_dist >= cfg.min_dist_to_provided
    distances = [None if d >= FAR else int(d) for d in min_dist.tolist()]

    annotations = pl.DataFrame(
        {
            "design_id": live["design_id"],
            "_min_dist": pl.Series(distances, dtype=pl.Int32),
            "_nearest": pl.Series(nearest, dtype=pl.Utf8),
            "_eligible": pl.Series(eligible.tolist(), dtype=pl.Boolean),
        }
    )
    merged = df.join(annotations, on="design_id", how="left").with_columns(
        pl.coalesce(pl.col("_min_dist"), pl.col("min_dist_to_provided")).alias(
            "min_dist_to_provided"
        ),
        pl.coalesce(pl.col("_nearest"), pl.col("nearest_provided_id")).alias(
            "nearest_provided_id"
        ),
    )
    merged = tbl.drop(merged, pl.col("_eligible").not_().fill_null(False), "novelty")
    merged = merged.drop("_min_dist", "_nearest", "_eligible")

    if funnel is not None:
        with funnel.step(
            "04a_novelty",
            rows_in=live.height,
            thresholds={
                "min_dist_to_provided": cfg.min_dist_to_provided,
                "blacklist_size": len(blacklist),
                "references_scanned_per_group": refs_used,
                "blacklist_skipped_other_length": blacklist.n_skipped,
            },
        ) as rec:
            rec.rows_out = tbl.alive(merged).height
            saved = live.height * len(blacklist) - n_compared
            rec.notes = (
                f"{n_compared:,} pairs compared; "
                f"pigeonhole skipped {saved:,} provably distant pairs"
            )
    return merged
