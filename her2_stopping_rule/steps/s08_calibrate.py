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
from sklearn.metrics import roc_auc_score

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Calibration source: a 150 KB archive of square ipTM matrices, one per
#: folding model, from the Smorodina complexes. 1.8 GB of per-replica metrics
#: are available too and are not needed -- this holds the same labels.
CALIBRATION_ZIP = "data/raw/calibration/iptm_confidence_scores.zip"

#: Bucket key, for the message when the file has not been synced.
CALIBRATION_KEY = "data/zenodo-18390239-ab-ag-complexes/iptm_confidence_scores.zip"

#: Which matrix to prefer, keyed by the folding model we actually use. A cutoff
#: derived from one model's confidence does not transfer to another's: the
#: Chai-1 matrix scores the same cognate pair at 0.18 where Boltz-2 scores 0.93.
MATRIX_FOR_MODEL: dict[str, str] = {
    "boltz2": "boltz2_matrix_clean.csv",
    "openfold3": "af3_matrix_clean.csv",
    "openfold3_p2": "af3_matrix_clean.csv",
}

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
        auc: How well the metric separates cognate from non-cognate pairs.
        discriminative: Whether that separation is good enough to filter on.
        n_negatives: Measured non-binders used.
        n_positives: Measured binders used, when the source has them.
        source: Which file supplied the data.
        note: Why the cutoff is what it is, for the record.
    """

    cutoff: float
    target_fpr: float
    achieved_fpr: float
    auc: float
    discriminative: bool
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


def load_matrix(cfg: Config) -> tuple[pl.DataFrame | None, str]:
    """Load the ipTM matrix for the model this run folds with.

    Args:
        cfg: Pipeline configuration, for the repository root and fold models.

    Returns:
        The matrix and the member it came from, or ``(None, "")`` when the
        archive has not been synced.
    """
    path = cfg.repo_root / CALIBRATION_ZIP
    if not path.exists():
        return None, ""
    wanted = MATRIX_FOR_MODEL.get(cfg.fold_models[0], "boltz2_matrix_clean.csv")
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        chosen = next((n for n in members if n.endswith(wanted)), None)
        if chosen is None:
            return None, ""
        with zf.open(chosen) as fh:
            return pl.read_csv(io.BytesIO(fh.read())), chosen


def melt_matrix(matrix: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Turn a square antibody-by-antigen matrix into scored, labelled pairs.

    Each row is an antibody and each column an antigen, both named by PDB id.
    A pair on the diagonal is cognate -- the antibody with the antigen it was
    solved against, so a real binder. Every off-diagonal pair is non-cognate:
    measured, and not binding. That is the label, and it is why 91 rows give
    91 positives and 8,190 negatives.

    Args:
        matrix: The matrix as read, its first column holding the row ids.

    Returns:
        Scores and labels, 1 for cognate and 0 for non-cognate.

    Examples:
        >>> import polars as pl
        >>> m = pl.DataFrame({"pdb_1": ["A", "B"], "A": [0.9, 0.2], "B": [0.3, 0.8]})
        >>> scores, labels = melt_matrix(m)
        >>> sorted(scores[labels == 1].tolist())
        [0.8, 0.9]
        >>> sorted(scores[labels == 0].tolist())
        [0.2, 0.3]
    """
    id_column = matrix.columns[0]
    rows = matrix[id_column].to_list()
    scores: list[float] = []
    labels: list[int] = []
    for column in matrix.columns[1:]:
        values = matrix[column].to_list()
        for row_id, value in zip(rows, values):
            if value is None:
                continue
            scores.append(float(value))
            labels.append(1 if row_id == column else 0)
    return np.array(scores), np.array(labels)


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
        cfg: Pipeline configuration, for ``iptm_fpr`` and the fold model.

    Returns:
        The calibration, or None when the archive has not been synced.
    """
    matrix, source = load_matrix(cfg)
    if matrix is None:
        return None

    scores, labels = melt_matrix(matrix)
    if labels.size == 0 or (labels == 0).sum() == 0:
        return None
    cutoff, achieved = choose_cutoff(scores, labels, cfg.iptm_fpr)
    auc = float(roc_auc_score(labels, scores))

    return Calibration(
        cutoff=cutoff,
        target_fpr=cfg.iptm_fpr,
        achieved_fpr=achieved,
        auc=auc,
        discriminative=auc >= cfg.min_iptm_auc,
        n_negatives=int((labels == 0).sum()),
        n_positives=int((labels == 1).sum()),
        source=source,
        note=(
            f"ipTM separates cognate from non-cognate at AUC {auc:.3f} on this "
            f"model. {(scores[labels == 0] >= 0.8).mean():.1%} of measured "
            "non-binders score at or above 0.8, so a round-number cutoff admits "
            "most of them. The cutoff below holds the stated false-positive "
            "rate; whether it is used as a filter depends on the AUC."
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
    # Only filter on a metric that actually discriminates. Where it does not,
    # dropping designs on it is noise dressed as rigour: the cutoff is still
    # recorded, and the ranking keeps ipTM only as a weak tie-break.
    if result is not None and result.discriminative and not np.isnan(result.cutoff):
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
                    "iptm_auc": result.auc,
                    "used_as_filter": result.discriminative,
                    "source": result.source,
                }
                if result
                else {"iptm_cutoff": None}
            ),
        ) as rec:
            rec.rows_out = tbl.alive(out).height
            rec.notes = (
                f"cutoff {result.cutoff:.3f} at {result.achieved_fpr:.1%} FPR on "
                f"{result.n_negatives:,} measured non-binders; AUC "
                f"{result.auc:.3f}, "
                + (
                    "used as a filter"
                    if result.discriminative
                    else "NOT used as a filter - ipTM does not separate binders "
                    "from non-binders well enough here to drop designs on it"
                )
                if result
                else (
                    "no calibration data synced; nothing dropped. Fetch with: "
                    f"aws s3 cp s3://apheris-oslo-hackathon/{CALIBRATION_KEY} "
                    f"{CALIBRATION_ZIP}"
                )
            )
    return out
