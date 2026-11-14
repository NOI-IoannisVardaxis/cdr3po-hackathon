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

from .. import queries
from ..config import Config
from ..foundry import CLI_MODULE, TEMPLATE_CAPABLE, FoundryError, available
from ..foundry import run as foundry_run

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


#: Identifier for the trastuzumab control folded alongside the designs.
BASELINE_ID = "baseline_trastuzumab"


def baseline_row() -> pl.DataFrame:
    """Build the trastuzumab control as a one-row design table.

    Folded through exactly the same path as the designs, so its scores are
    comparable: a baseline measured a different way is not a baseline.

    Returns:
        A single row carrying the native sequences.

    Examples:
        >>> row = baseline_row()
        >>> row.height
        1
        >>> row["design_id"][0]
        'baseline_trastuzumab'
        >>> bool(row["is_baseline"][0])
        True
    """
    from ..config import CDR3_IMGT, TRASTUZUMAB_VH, TRASTUZUMAB_VL

    return pl.DataFrame(
        {
            "design_id": [BASELINE_ID],
            "cdr3_imgt": [CDR3_IMGT],
            "heavy_sequence": [TRASTUZUMAB_VH],
            "light_sequence": [TRASTUZUMAB_VL],
            "kind": ["paired"],
            "is_baseline": [True],
            "track_predicted": ["baseline"],
        }
    )


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


#: Apheris score names, mapped to the short names the table uses.
SCORE_KEYS = {
    "interface_predicted_tm_score": "iptm",
    "predicted_tm_score": "ptm",
    "average_predicted_local_distance_difference_test": "binder_plddt",
    "global_predicted_distance_error": "min_ipae",
    "model_ranking_score": "ranking",
}


def fold_label(model: str, use_template: bool) -> str:
    """Directory name a fold run writes into.

    Shared with step 9, which has to find the structures again. Hard-coding
    the model name there is wrong whenever templates are off, and silently
    finds nothing.

    Args:
        model: Key from :data:`~her2_stopping_rule.foundry.IMAGES`.
        use_template: Whether the run was given the 1N8Z template.

    Returns:
        The directory name under ``foundry/fold/``.

    Examples:
        >>> fold_label("boltz2", True)
        'boltz2'
        >>> fold_label("boltz2", False)
        'boltz2_notemplate'
    """
    return model if use_template else f"{model}_notemplate"


def parse_scores(out_dir: Path) -> dict[str, dict[str, float]]:
    """Read per-design scores out of a Foundry predict run.

    ``predict`` writes one ``*_scores.json`` per design per seed, under
    ``<design_id>/seed_<n>/``. It does not write ``refold_metrics.csv`` -- that
    comes from ``refold-score``, which scores RFD4 designs against their own
    backbones and is not what runs here.

    Args:
        out_dir: The directory passed to ``--output``.

    Returns:
        Per-design metrics under short names. Empty if nothing was written.
    """
    scores: dict[str, dict[str, float]] = {}
    if not out_dir.exists():
        return scores
    for path in sorted(out_dir.glob("*/seed_*/*_scores.json")):
        design_id = path.parent.parent.name
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        values = {
            short: float(payload[long])
            for long, short in SCORE_KEYS.items()
            if isinstance(payload.get(long), (int, float))
        }
        if values:
            # Several seeds would each write a file; keep the first.
            scores.setdefault(design_id, values)
    return scores


def fold_with(
    model: str,
    df: pl.DataFrame,
    cfg: Config,
    workdir: Path,
    use_template: bool = True,
) -> dict[str, dict[str, float]]:
    """Fold every construct with one model and return its scores.

    Args:
        model: Key from :data:`~her2_stopping_rule.foundry.IMAGES`.
        df: Live designs to fold.
        cfg: Pipeline configuration.
        workdir: Directory mounted into the container.
        use_template: Give the model the 1N8Z template. Without it the model is
            not told where trastuzumab binds, so recovering that epitope is
            evidence rather than an echo of the input.

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

    label = fold_label(model, use_template)
    out_dir = workdir / "fold" / label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # The model refuses a non-empty --output, so the query file lives beside
    # the output directory rather than inside it.
    inputs_dir = workdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    query_path = queries.write(
        [
            (r["design_id"], r["heavy_sequence"], r["light_sequence"])
            for r in df.iter_rows(named=True)
        ],
        cfg.path("structure"),
        inputs_dir / f"{label}.json",
    )

    template = workdir / "template.cif"
    args = [
        "python",
        "-m",
        CLI_MODULE[model],
        "predict",
        "--input",
        str(query_path.relative_to(workdir)),
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
    ]
    if use_template and template.exists():
        args += ["--template-paths", str(template.relative_to(workdir))]

    try:
        foundry_run(model, args, workdir=workdir)
    except (FoundryError, OSError):
        return {}
    return parse_scores(out_dir)


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
    shutil.copyfile(cfg.path("structure"), workdir / "template.cif")

    # Fold trastuzumab itself alongside the designs, through exactly the same
    # path. Step 9 needs to know how much of the native epitope this model
    # reproduces on the molecule the epitope was defined from: a threshold
    # stricter than that would be asking designs to beat the measurement.
    to_fold = pl.concat(
        [tbl.conform(live), tbl.conform(baseline_row())], how="vertical"
    )

    primary, secondary = cfg.fold_models[0], cfg.fold_models[1]
    templated = not cfg.fold_without_template
    scores_primary = fold_with(primary, to_fold, cfg, workdir, use_template=templated)
    scores_secondary = fold_with(
        secondary, to_fold, cfg, workdir, use_template=templated
    )
    # Recorded separately so step 9 can say whether the epitope was found
    # unprompted; identical to the primary pass when templates are off.
    scores_no_template = scores_primary if cfg.fold_without_template else {}

    (workdir / "fold_scores.json").write_text(
        json.dumps(
            {
                primary: scores_primary,
                secondary: scores_secondary,
                f"{primary}_no_template": scores_no_template,
            },
            indent=2,
        )
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
    )

    # The spread is only a noise floor if both models can predict the complex
    # the epitope was defined from. Record it only when the secondary model
    # clears that bar on the baseline control; otherwise leave it null, and
    # the stopping rule will say it has nothing to measure against rather
    # than inventing a threshold from another model's failure.
    base_primary = scores_primary.get(BASELINE_ID, {}).get("iptm")
    base_secondary = scores_secondary.get(BASELINE_ID, {}).get("iptm")
    spread_usable = (
        base_secondary is not None and base_secondary >= cfg.min_baseline_iptm
    )
    annotations = annotations.with_columns(
        (pl.col("_iptm_primary") - pl.col("_iptm_secondary")).abs().alias("_spread")
        if spread_usable
        else pl.lit(None, dtype=pl.Float64).alias("_spread")
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
            thresholds={
                "models": list(cfg.fold_models),
                "seed": cfg.seed,
                "fold_without_template": cfg.fold_without_template,
                "baseline_control": BASELINE_ID,
                "baseline_iptm_primary": base_primary,
                "baseline_iptm_secondary": base_secondary,
                "min_baseline_iptm": cfg.min_baseline_iptm,
                "spread_usable": spread_usable,
            },
        ) as rec:
            rec.rows_out = live.height
            got = len(scores_primary), len(scores_secondary)
            base = BASELINE_ID in scores_primary
            spreads = merged["model_spread"].drop_nulls()
            median = spreads.median() if spreads.len() else None
            rec.notes = (
                f"{primary} scored {got[0]}, {secondary} scored {got[1]}; "
                + (
                    f"median two-model spread {float(median):.4f}"
                    if isinstance(median, (int, float))
                    else "no spread measured, both models unavailable"
                )
                + (
                    f"; baseline control folded: {base}"
                    f"; no-template control: {len(scores_no_template)} scored"
                )
            )
    return merged
