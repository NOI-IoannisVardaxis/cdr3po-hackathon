"""Step 7: co-fold every construct against HER2 with two independent models.

The point is not the score but the disagreement. Two models asked the same
question give two answers, and the gap between them is the noise floor of the
whole ranking. Step 12 compares predicted gains against that gap, so this step
is where the stopping rule's threshold is actually measured.

Both models take a template, which gives a binder a fairer check than folding
the target unguided. ESMFold2 is available but is not used here for that reason.

A caution carried over from the smoke test: ipTM above 0.95 is not evidence of
binding. The Smorodina non-binders score that well too, which is why step 8
calibrates the cutoff on measured negatives rather than trusting a round number.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config
from ..foundry import TEMPLATE_CAPABLE, FoundryError, available
from ..foundry import run as foundry_run

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def write_fasta(df: pl.DataFrame, path: Path) -> Path:
    """Write constructs as a FASTA of paired chains.

    Args:
        df: Live designs carrying ``design_id``, ``heavy_sequence`` and
            ``light_sequence``.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in df.iter_rows(named=True):
        lines.append(f">{row['design_id']}|heavy")
        lines.append(row["heavy_sequence"])
        lines.append(f">{row['design_id']}|light")
        lines.append(row["light_sequence"])
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_refold_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Read ``refold_metrics.csv`` into per-design scores.

    Args:
        path: The metrics CSV written by ``refold-score``.

    Returns:
        A mapping from design id to its metrics. Empty if the file is absent.
    """
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        key = str(row.get("query_id") or row.get("design") or "")
        if not key:
            continue
        out[key] = {
            name: float(row[name])
            for name in ("iptm", "min_ipae", "binder_plddt", "ptm", "complex_plddt")
            if row.get(name) is not None
        }
    return out


def fold_with(
    model: str,
    df: pl.DataFrame,
    cfg: Config,
    workdir: Path,
) -> dict[str, dict[str, float]]:
    """Fold every construct with one model and return its scores.

    Args:
        model: Key from :data:`~her2_stopping_rule.foundry.IMAGES`.
        df: Live designs to fold.
        cfg: Pipeline configuration.
        workdir: Directory mounted into the container.

    Returns:
        Per-design metrics, empty when the model is unavailable or fails.

    Raises:
        ValueError: If the model cannot take a template.
    """
    if model not in TEMPLATE_CAPABLE:
        raise ValueError(
            f"{model} folds the target unguided, which is not a fair check for a "
            "binder; use one of {sorted(TEMPLATE_CAPABLE)}"
        )
    if not available(model):
        return {}

    out_dir = workdir / "fold" / model
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(df, out_dir / "designs.fasta")

    try:
        foundry_run(
            model,
            [
                "python",
                "-m",
                "apheris_boltz.cli",
                "predict",
                "--input",
                str((out_dir / "designs.fasta").relative_to(workdir)),
                "--output",
                str(out_dir.relative_to(workdir)),
                "--write-detailed-scores",
                "true",
                "--num-diffusion-samples",
                "1",
                "--seeds",
                str(cfg.seed),
                "--query-failures",
                "skip",
            ],
            workdir=workdir,
        )
    except (FoundryError, OSError):
        return {}
    return parse_refold_metrics(out_dir / "refold_metrics.csv")


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Fold every live construct with both configured models.

    Adds no filter of its own. It records ipTM, min interface PAE, binder pLDDT
    and, most importantly, the spread between the two models.

    Args:
        df: The design table.
        cfg: Pipeline configuration, for ``fold_models`` and ``n_finalists``.
        funnel: Optional funnel for counts.

    Returns:
        The table with fold metrics filled in on folded designs.
    """
    from .. import table as tbl

    live = tbl.alive(df).head(cfg.n_finalists)
    if live.height == 0:
        return df

    workdir = cfg.results_dir / "foundry"
    workdir.mkdir(parents=True, exist_ok=True)

    primary, secondary = cfg.fold_models[0], cfg.fold_models[1]
    scores_primary = fold_with(primary, live, cfg, workdir)
    scores_secondary = fold_with(secondary, live, cfg, workdir)

    (workdir / "fold_scores.json").write_text(
        json.dumps({primary: scores_primary, secondary: scores_secondary}, indent=2)
    )

    ids = live["design_id"].to_list()
    annotations = pl.DataFrame(
        {
            "design_id": ids,
            "_iptm_primary": [scores_primary.get(i, {}).get("iptm") for i in ids],
            "_iptm_secondary": [scores_secondary.get(i, {}).get("iptm") for i in ids],
            "_min_ipae": [scores_primary.get(i, {}).get("min_ipae") for i in ids],
            "_binder_plddt": [
                scores_primary.get(i, {}).get("binder_plddt") for i in ids
            ],
        }
    ).with_columns(
        (pl.col("_iptm_primary") - pl.col("_iptm_secondary")).abs().alias("_spread")
    )

    merged = (
        df.join(annotations, on="design_id", how="left")
        .with_columns(
            pl.coalesce("_iptm_primary", "iptm_primary").alias("iptm_primary"),
            pl.coalesce("_iptm_secondary", "iptm_secondary").alias("iptm_secondary"),
            pl.coalesce("_min_ipae", "min_ipae").alias("min_ipae"),
            pl.coalesce("_binder_plddt", "binder_plddt").alias("binder_plddt"),
            pl.coalesce("_spread", "model_spread").alias("model_spread"),
        )
        .drop(
            "_iptm_primary", "_iptm_secondary", "_min_ipae", "_binder_plddt", "_spread"
        )
    )

    if funnel is not None:
        with funnel.step(
            "07_fold",
            rows_in=live.height,
            thresholds={"models": list(cfg.fold_models), "seed": cfg.seed},
        ) as rec:
            rec.rows_out = live.height
            got = len(scores_primary), len(scores_secondary)
            spreads = merged["model_spread"].drop_nulls()
            median = spreads.median() if spreads.len() else None
            rec.notes = f"{primary} scored {got[0]}, {secondary} scored {got[1]}; " + (
                f"median two-model spread {float(median):.4f}"
                if isinstance(median, (int, float))
                else "no spread measured, both models unavailable"
            )
    return merged
