"""Step 8: set the confidence cutoff from measured non-binders.

The differentiator, and the most defensible slide in the pitch. Everyone will
filter on ipTM. Almost nobody will be able to say what their threshold's
false-positive rate is, because almost nobody will calibrate it against pairs
that were measured and found not to bind.

We were handed the proof that confidence is unreliable here: of 454 VHH-antigen
pairs with no measurable binding by SPR, 21 still score AlphaFold3 ipTM at or
above 0.7. Picking 0.8 because it is a round number admits known failures. This
step picks the cutoff that achieves a stated false-positive rate on those
measured negatives instead.

The calibration data is tried smallest-first: a 150 KB score table before a
1.8 GB archive of per-replica metrics.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Calibration sources, cheapest first. Each is a key under the bucket's
#: ``data/`` prefix and the local path it is cached at.
SOURCES: tuple[tuple[str, str], ...] = (
    (
        "data/zenodo-18390239-ab-ag-complexes/iptm_confidence_scores.zip",
        "data/raw/calibration/iptm_confidence_scores.zip",
    ),
    (
        "data/sigma2-ab-ag-champloo/chai1_all_per_replica_confidence_metrics.zip",
        "data/raw/calibration/chai1_all_per_replica_confidence_metrics.zip",
    ),
)

#: Column names that plausibly hold an ipTM score, in preference order.
_IPTM_NAMES = ("iptm", "ipTM", "iptm_score", "interface_ptm")

#: Column names that plausibly hold a binary binding label.
_LABEL_NAMES = ("binds", "is_binder", "cognate", "label", "binding")


@dataclass(frozen=True)
class Calibration:
    """An ipTM cutoff with a stated error rate.

    Attributes:
        cutoff: The ipTM threshold.
        target_fpr: False-positive rate the cutoff was chosen to achieve.
        achieved_fpr: Rate actually achieved on the measured negatives.
        n_negatives: Measured non-binders used.
        n_positives: Measured binders used, when the source has them.
        source: Which file supplied the data.
        note: Why the cutoff is what it is, for the record.
    """

    cutoff: float
    target_fpr: float
    achieved_fpr: float
    n_negatives: int
    n_positives: int
    source: str
    note: str


def _find_column(frame: pl.DataFrame, names: tuple[str, ...]) -> str | None:
    """Locate the first column matching any of several candidate names.

    Args:
        frame: The table to search.
        names: Candidate column names, in preference order.

    Returns:
        The matching column name, or None.

    Examples:
        >>> import polars as pl
        >>> _find_column(pl.DataFrame({"iptm": [0.5]}), ("iptm", "ipTM"))
        'iptm'
        >>> _find_column(pl.DataFrame({"x": [1]}), ("iptm",)) is None
        True
    """
    lowered = {c.lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_scores(cfg: Config) -> tuple[pl.DataFrame | None, str]:
    """Load whichever calibration table is available locally.

    Args:
        cfg: Pipeline configuration, for the repository root.

    Returns:
        The table and the source it came from, or ``(None, "")`` if neither
        file has been synced.
    """
    for _, local in SOURCES:
        path = cfg.repo_root / local
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".csv")]
            if not members:
                continue
            with zf.open(members[0]) as fh:
                return pl.read_csv(io.BytesIO(fh.read())), local
    return None, ""


def choose_cutoff(
    scores: np.ndarray, labels: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Pick the lowest ipTM achieving at most the target false-positive rate.

    Args:
        scores: ipTM values.
        labels: 1 for measured binders, 0 for measured non-binders.
        target_fpr: Maximum tolerated fraction of non-binders admitted.

    A design is admitted when its ipTM is at or above the cutoff, so a cutoff
    placed exactly on a negative's score admits that negative. Taking a plain
    quantile therefore overshoots the budget on ties, and at a target of zero it
    never reaches zero. Instead the cutoff is set just above the highest
    negative that the budget allows through, which makes the achieved rate at
    most the target by construction.

    Args:
        scores: ipTM values.
        labels: 1 for measured binders, 0 for measured non-binders.
        target_fpr: Maximum tolerated fraction of non-binders admitted.

    Returns:
        The cutoff and the false-positive rate it actually achieves, which is
        always less than or equal to ``target_fpr``.

    Examples:
        One of three measured non-binders may pass at a 34% budget:

        >>> s = np.array([0.9, 0.8, 0.4, 0.3])
        >>> y = np.array([1, 0, 0, 0])
        >>> cutoff, fpr = choose_cutoff(s, y, 0.34)
        >>> round(fpr, 3)
        0.333
        >>> bool(0.4 < cutoff <= 0.8)
        True

        At a zero budget no negative gets through, and the cutoff clears the
        highest of them:

        >>> cutoff, fpr = choose_cutoff(s, y, 0.0)
        >>> fpr
        0.0
        >>> bool(cutoff > 0.8)
        True

        With no measured negatives there is nothing to calibrate against:

        >>> import math
        >>> math.isnan(choose_cutoff(s, np.array([1, 1, 1, 1]), 0.05)[0])
        True
    """
    negatives = scores[labels == 0]
    if negatives.size == 0:
        return float("nan"), float("nan")

    descending = np.sort(negatives)[::-1]
    allowed = int(np.floor(target_fpr * descending.size))
    if allowed >= descending.size:
        cutoff = float(-np.inf)
    else:
        cutoff = float(np.nextafter(descending[allowed], np.inf))
    achieved = float((negatives >= cutoff).mean())
    return cutoff, achieved


def calibrate(cfg: Config) -> Calibration | None:
    """Derive the ipTM cutoff from measured non-binders.

    Args:
        cfg: Pipeline configuration, for ``iptm_fpr``.

    Returns:
        The calibration, or None when no calibration data is present locally.
    """
    frame, source = load_scores(cfg)
    if frame is None:
        return None

    score_col = _find_column(frame, _IPTM_NAMES)
    label_col = _find_column(frame, _LABEL_NAMES)
    if score_col is None or label_col is None:
        return None

    clean = frame.select(
        pl.col(score_col).cast(pl.Float64, strict=False).alias("score"),
        pl.col(label_col).cast(pl.Int64, strict=False).alias("label"),
    ).drop_nulls()

    scores = clean["score"].to_numpy()
    labels = clean["label"].to_numpy()
    cutoff, achieved = choose_cutoff(scores, labels, cfg.iptm_fpr)

    return Calibration(
        cutoff=cutoff,
        target_fpr=cfg.iptm_fpr,
        achieved_fpr=achieved,
        n_negatives=int((labels == 0).sum()),
        n_positives=int((labels == 1).sum()),
        source=source,
        note=(
            "cutoff is the ipTM quantile that admits at most the target fraction "
            "of pairs measured by SPR as non-binding, not a round number"
        ),
    )


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Apply the calibrated ipTM cutoff, if calibration data is present.

    When no calibration data has been synced the step records that fact and
    drops nothing, rather than silently falling back to an uncalibrated guess.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with low-confidence designs marked ``dropped_by = "calibrate"``.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    result = calibrate(cfg)
    out = df
    if result is not None and not np.isnan(result.cutoff):
        (cfg.results_dir / "calibration.json").write_text(
            json.dumps(result.__dict__, indent=2, default=str)
        )
        out = tbl.drop(
            df,
            (pl.col("iptm_primary") < result.cutoff).fill_null(False),
            "calibrate",
        )

    if funnel is not None:
        with funnel.step(
            "08_calibrate",
            rows_in=live.height,
            thresholds=(
                {
                    "iptm_cutoff": result.cutoff,
                    "target_fpr": result.target_fpr,
                    "achieved_fpr": result.achieved_fpr,
                    "source": result.source,
                }
                if result
                else {"iptm_cutoff": None}
            ),
        ) as rec:
            rec.rows_out = tbl.alive(out).height
            rec.notes = (
                f"cutoff {result.cutoff:.3f} at {result.achieved_fpr:.1%} FPR on "
                f"{result.n_negatives} measured non-binders ({result.source})"
                if result
                else (
                    "no calibration data synced; nothing dropped. Fetch with: "
                    "aws s3 cp s3://apheris-oslo-hackathon/"
                    f"{SOURCES[0][0]} {SOURCES[0][1]}"
                )
            )
    return out
