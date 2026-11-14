"""Step 5a: the FACS affinity surrogate.

This is where the pipeline's edge lives. Selection happens over millions of
candidates on CPU in seconds, so folding is only ever asked to confirm a choice,
never to search for one.

The surrogate is a gradient-boosted classifier over a one-hot CDR3, trained on
all 524,346 measured variants. Its generalisation was the real risk, because it
must rank loops in a mutation-distance regime it was not trained on. Held out by
distance, it does: trained on distances 5-10 and asked to rank the 6,392 unseen
distance-4 variants it achieves ROC-AUC 0.920, and every one of its top 500
picks is a measured high binder.

A ranking cut, not a rejection: nothing is disqualified here, the pipeline just
stops looking past the top of the list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ..config import Config
from ..data import load_facs
from ..sequences import one_hot

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


@dataclass(frozen=True)
class HoldoutMetrics:
    """Honest generalisation estimate for the surrogate.

    Attributes:
        train_distances: Edit distances the model was trained on.
        test_distance: Edit distance held out entirely.
        n_train: Training variants.
        n_test: Held-out variants.
        base_rate: Fraction of held-out variants that are high binders.
        roc_auc: Area under the ROC curve on the held-out set.
        average_precision: Area under the precision-recall curve.
        precision_at: Precision among the model's top-k picks, per k.
        fpr: ROC false-positive rates, for plotting.
        tpr: ROC true-positive rates, for plotting.
    """

    train_distances: tuple[int, ...]
    test_distance: int
    n_train: int
    n_test: int
    base_rate: float
    roc_auc: float
    average_precision: float
    precision_at: dict[int, float]
    fpr: np.ndarray
    tpr: np.ndarray


def _model(cfg: Config) -> HistGradientBoostingClassifier:
    """Build an unfitted surrogate with the configured hyperparameters.

    Args:
        cfg: Pipeline configuration.

    Returns:
        An unfitted classifier.
    """
    return HistGradientBoostingClassifier(
        max_iter=cfg.affinity_max_iter,
        learning_rate=cfg.affinity_learning_rate,
        max_leaf_nodes=cfg.affinity_max_leaf_nodes,
        random_state=cfg.seed,
    )


def validate_holdout(
    cfg: Config,
    test_distance: int = 4,
    ks: tuple[int, ...] = (50, 100, 500),
) -> HoldoutMetrics:
    """Measure generalisation by holding out an entire edit-distance stratum.

    A random split would leak: variants one mutation apart are near-duplicates.
    Holding out a whole distance stratum asks the harder and more honest
    question, which is whether the model transfers into the regime we design in.

    Args:
        cfg: Pipeline configuration.
        test_distance: Edit distance to hold out entirely.
        ks: Cut-offs at which to report precision among top picks.

    Returns:
        The held-out metrics, including ROC curve points for plotting.
    """
    facs = load_facs(cfg)
    loops = facs["cdr3_imgt"].to_list()
    x = one_hot(loops)
    y = facs["is_high"].to_numpy().astype(int)
    d = facs["edit_distance"].to_numpy()

    train = d > test_distance
    test = d == test_distance

    model = _model(cfg).fit(x[train], y[train])
    scores = model.predict_proba(x[test])[:, 1]
    y_test = y[test]

    order = np.argsort(-scores)
    precision_at = {k: float(y_test[order[:k]].mean()) for k in ks if k <= len(order)}
    fpr, tpr, _ = roc_curve(y_test, scores)

    return HoldoutMetrics(
        train_distances=tuple(sorted(set(d[train].tolist()))),
        test_distance=test_distance,
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        base_rate=float(y_test.mean()),
        roc_auc=float(roc_auc_score(y_test, scores)),
        average_precision=float(average_precision_score(y_test, scores)),
        precision_at=precision_at,
        fpr=fpr,
        tpr=tpr,
    )


def fit_full(cfg: Config) -> HistGradientBoostingClassifier:
    """Train the surrogate on every measured variant.

    Args:
        cfg: Pipeline configuration.

    Returns:
        A fitted classifier.
    """
    facs = load_facs(cfg)
    x = one_hot(facs["cdr3_imgt"].to_list())
    y = facs["is_high"].to_numpy().astype(int)
    return _model(cfg).fit(x, y)


def score(
    model: HistGradientBoostingClassifier,
    loops: list[str],
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Predict P(high binder) for many loops without exhausting memory.

    One-hot encoding 19 million loops at once would need tens of gigabytes, so
    scoring runs in chunks.

    Args:
        model: A fitted surrogate.
        loops: IMGT CDR3 sequences to score.
        chunk_size: Loops encoded and scored per pass.

    Returns:
        Predicted probabilities, aligned with ``loops``.
    """
    out = np.empty(len(loops), dtype=np.float64)
    for start in range(0, len(loops), chunk_size):
        block = loops[start : start + chunk_size]
        out[start : start + len(block)] = model.predict_proba(one_hot(block))[:, 1]
    return out


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Score and shortlist the surviving candidates.

    Args:
        df: The design table, carrying ``cdr3_imgt`` on live rows.
        cfg: Pipeline configuration, for ``shortlist_n``.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``p_high`` filled in and everything outside the top
        ``shortlist_n`` marked ``dropped_by = "affinity"``.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    metrics = validate_holdout(cfg)
    model = fit_full(cfg)
    probabilities = score(model, live["cdr3_imgt"].to_list())

    annotations = pl.DataFrame(
        {
            "design_id": live["design_id"],
            "_p_high": pl.Series(probabilities, dtype=pl.Float64),
        }
    )
    merged = df.join(annotations, on="design_id", how="left").with_columns(
        pl.coalesce(pl.col("_p_high"), pl.col("p_high")).alias("p_high")
    )

    keep = (
        tbl.alive(merged)
        .sort("p_high", descending=True)
        .head(cfg.shortlist_n)["design_id"]
        .to_list()
    )
    merged = tbl.drop(merged, pl.col("design_id").is_in(keep).not_(), "affinity")
    merged = merged.drop("_p_high")

    if funnel is not None:
        with funnel.step(
            "05a_affinity",
            rows_in=live.height,
            thresholds={
                "shortlist_n": cfg.shortlist_n,
                "holdout_distance": metrics.test_distance,
                "holdout_roc_auc": round(metrics.roc_auc, 4),
                "holdout_base_rate": round(metrics.base_rate, 4),
                "precision_at": {
                    k: round(v, 4) for k, v in metrics.precision_at.items()
                },
            },
        ) as rec:
            rec.rows_out = tbl.alive(merged).height
            rec.notes = (
                f"held out distance {metrics.test_distance} entirely: "
                f"ROC-AUC {metrics.roc_auc:.3f} against a {metrics.base_rate:.1%} base rate"
            )
    return merged
