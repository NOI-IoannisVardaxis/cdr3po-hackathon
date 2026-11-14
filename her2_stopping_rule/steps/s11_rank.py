"""Step 11: rank on all three judged axes, then diversify.

Ranking by predicted affinity alone would fill the slate with one loop family.
Ten designs from one family share one systematic error, and the tier system
punishes that harshly: ten designs that each improve one property lose to two
that improve all three. So designs are clustered by CDR3 and the slate takes the
best from each cluster, which keeps a wrong hypothesis from costing everything.

Clustering uses Hamming distance on the 13-mer by default. Embeddings (ESMC or
SaProt via JapanFold, ESM-2 via Nebius) can replace the metric, but at this
sequence length they add a network dependency for marginal gain.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ..config import Config
from ..sequences import encode

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def cluster_loops(loops: list[str], n_clusters: int) -> np.ndarray:
    """Group CDR3s into families by hierarchical clustering.

    Args:
        loops: IMGT CDR3 sequences.
        n_clusters: Maximum number of families to cut the tree into.

    Returns:
        A 1-based cluster label per loop.

    Examples:
        >>> labels = cluster_loops(["SRWGGDGFYAMDY", "SRWGGDGFYAMDA",
        ...                         "SRAAADHFTAMDY"], 2)
        >>> len(labels)
        3
        >>> int(labels[0]) == int(labels[1])
        True
        >>> int(labels[0]) != int(labels[2])
        True
    """
    if len(loops) <= 1:
        return np.ones(len(loops), dtype=int)
    arr = encode(loops).astype(np.int16)
    distances = pdist(arr, metric="hamming") * arr.shape[1]
    tree = linkage(distances, method="average")
    return fcluster(tree, t=min(n_clusters, len(loops)), criterion="maxclust")


def beats_baseline_threshold(cfg: Config) -> float:
    """Surrogate score above which a design is counted as beating the baseline.

    Read from the calibration step 5a writes, which connects surrogate scores
    to measured KDs. Falls back to a stringent constant when that file is
    absent, never to 0.5: at 0.5 the surrogate keeps 88% of measured variants
    and captures the base rate rather than an improvement.

    Args:
        cfg: Pipeline configuration.

    Returns:
        The threshold.
    """
    path = cfg.results_dir / "affinity_calibration.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text())["threshold"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return 0.99


def predicted_tier(cfg: Config) -> pl.Expr:
    """Count how many judged axes a design is predicted to beat the baseline on.

    This mirrors how the competition actually scores: teams rank by number of
    S-tier designs, then A-tier, then B-tier. Ten designs that each improve one
    property lose to two that improve all three, so the count must lead the
    ordering and no weighted blend can express that.

    There are no free parameters here. Each axis contributes exactly one point
    when the evidence says it beats trastuzumab, and nothing when the evidence
    is missing: an unmeasured axis is not an improved one.

    Args:
        cfg: Pipeline configuration, for the axis thresholds.

    Returns:
        An integer expression from 0 to 3.
    """
    affinity = (
        (pl.col("p_high") >= beats_baseline_threshold(cfg))
        .fill_null(False)
        .cast(pl.Int32)
    )
    epitope = (
        (pl.col("epitope_recovery") >= cfg.min_epitope_recovery)
        .fill_null(False)
        .cast(pl.Int32)
    )
    # Thermostability. The engineered disulfide was the original evidence and
    # cannot be submitted -- abguard reads an inter-chain pair as two unpaired
    # cysteines -- so that term is null for every design and left alone would
    # cap every tier at 2 of 3, making S-tier arithmetically unreachable.
    #
    # The replacement is ESM-IF1's inverse-folding likelihood: how well a
    # sequence fits the backbone it is supposed to fold into. That is a
    # recognised stability proxy and emphatically not a melting temperature.
    # The bar is trastuzumab's own score, from the same model on the same
    # backbone in the same run, so a design counts only by beating the
    # molecule it is trying to improve.
    stability = (pl.col("esm_if_vs_baseline") > 0).fill_null(False).cast(pl.Int32)
    disulfide = pl.col("disulfide_formed").fill_null(False).cast(pl.Int32)
    stability = (stability + disulfide).clip(0, 1)
    return affinity + epitope + stability


def composite_score(df: pl.DataFrame, cfg: Config) -> pl.Expr:
    """Order designs by predicted tier, breaking ties with affinity.

    Tier dominates: a design beating two axes outranks any design beating one,
    whatever their affinities. Within a tier, the calibrated predictor decides,
    and model disagreement acts as a small penalty so a design the two folding
    models argue about loses to an equivalent one they agree on.

    Args:
        df: Live designs.
        cfg: Pipeline configuration.

    Returns:
        A polars expression producing the ordering score.
    """
    tie_break = pl.col("p_high").fill_null(0.0) - pl.col("model_spread").fill_null(0.0)
    return predicted_tier(cfg).cast(pl.Float64) + tie_break.clip(0.0, 0.999)


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Score, cluster and order the surviving designs.

    Args:
        df: The design table.
        cfg: Pipeline configuration, for ``n_clusters``.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``cluster``, ``rank`` and ``reason`` filled in.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    # The two tracks are validated independently and the prize pool is split
    # evenly between them, so each is ranked on its own merits. Interleaving
    # into the one submitted list happens in step 13.
    ranked_parts: list[pl.DataFrame] = []
    for track in live["track_predicted"].unique(maintain_order=True).to_list():
        part = live.filter(pl.col("track_predicted") == track)
        labels = cluster_loops(part["cdr3_imgt"].to_list(), cfg.n_clusters)
        ranked_parts.append(
            part.with_columns(
                pl.Series("cluster", labels, dtype=pl.Int32),
                composite_score(part, cfg).alias("_score"),
                predicted_tier(cfg).alias("_tier"),
            )
            .sort("_score", descending=True)
            .with_columns(
                pl.int_range(pl.len()).over("cluster").alias("_within_cluster")
            )
            .sort(["_within_cluster", "_score"], descending=[False, True])
            .with_row_index("_rank", offset=1)
        )
    scored = pl.concat(ranked_parts, how="vertical")

    reasons = scored.select(
        "design_id",
        pl.col("cluster"),
        pl.col("_rank").cast(pl.Int32).alias("_new_rank"),
        pl.format(
            "tier {}/3; CDR3 {} ({} mutations), P(high)={}, {} package, cluster {}",
            pl.col("_tier"),
            pl.col("mutations"),
            pl.col("n_mutations"),
            pl.col("p_high").round(3),
            pl.col("stability_package"),
            pl.col("cluster"),
        ).alias("_reason"),
    )

    out = (
        df.join(reasons, on="design_id", how="left")
        .with_columns(
            pl.coalesce(pl.col("cluster_right"), pl.col("cluster")).alias("cluster"),
            pl.coalesce(pl.col("_new_rank"), pl.col("rank")).alias("rank"),
            pl.coalesce(pl.col("_reason"), pl.col("reason")).alias("reason"),
        )
        .drop([c for c in ("cluster_right", "_new_rank", "_reason") if c])
    )

    if funnel is not None:
        with funnel.step(
            "11_rank", rows_in=live.height, thresholds={"n_clusters": cfg.n_clusters}
        ) as rec:
            rec.rows_out = live.height
            rec.notes = f"{len(set(labels.tolist()))} loop families represented"
    return out
