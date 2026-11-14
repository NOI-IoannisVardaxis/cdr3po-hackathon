"""Step 10b: the thermostability axis, by inverse-folding likelihood.

Without this step no design can reach S-tier. The axis used to read
``disulfide_formed``, and the engineered disulfide cannot be submitted at all
-- abguard's per-chain cysteine parity check reads an inter-chain pair as two
unpaired cysteines -- so that term was null for every design and the maximum
achievable tier was 2 of 3, arithmetically.

The replacement is **ESM-IF1**, hosted on Nebius: given a backbone, it reports
how likely each residue is at each position. The mean log-likelihood of the
sequence a design actually carries measures how well that sequence fits the
fold it is supposed to adopt, which is a recognised proxy for stability.

**It is not a melting temperature, and must never be presented as one.** The
organisers' own guide rates computational substitutes for Tm as "moderate".
What this produces is a rank, and the bar is trastuzumab's own score from the
same model on the same backbone in the same run -- so a design counts only by
beating the molecule it is trying to improve, never by clearing a number
somebody chose.

Designs are scored on their **folded** structures from step 7, with the
antigen stripped: ESM-IF1 scores the antibody against its own fold, and
leaving HER2 in would ask a different question.
"""

from __future__ import annotations

import glob
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import polars as pl

from ..config import TRASTUZUMAB_VH, Config
from .s07_fold import BASELINE_ID, fold_label
from .s09_epitope import find_refold

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Chains kept for scoring: the Fv only.
FV_CHAINS = ("A", "B")


def write_bundle(df: pl.DataFrame, cfg: Config, out: Path) -> tuple[Path, int]:
    """Collect each design's folded Fv into a tar for the hosted model.

    Args:
        df: Live designs.
        cfg: Pipeline configuration.
        out: Directory to stage PDBs in; the tar is written beside it.

    Returns:
        The tar path and how many structures went into it.
    """
    label = fold_label(cfg.fold_models[0], not cfg.fold_without_template)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for design_id in [*df["design_id"].to_list(), BASELINE_ID]:
        path = find_refold(cfg.results_dir, design_id, label)
        if path is None:
            continue
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        for chain in [c.name for c in st[0]]:
            if chain not in FV_CHAINS:
                st[0].remove_chain(chain)
        st.setup_entities()
        st.write_pdb(str(out / f"{design_id}.pdb"))
        written += 1

    tar_path = out.parent / "esmif.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(out, arcname="pdbs")
    return tar_path, written


def read_scores(results_dir: Path, df: pl.DataFrame) -> dict[str, float]:
    """Read per-design log-likelihoods out of a completed ESM-IF1 run.

    The platform names its output files by artifact id, not by design, so each
    one is matched back by reconstructing the heavy chain it scored.

    Args:
        results_dir: Directory the run was fetched into.
        df: Live designs, to match sequences against.

    Returns:
        Mean log-likelihood per design id, including the baseline.
    """
    by_sequence = {
        row["heavy_sequence"]: row["design_id"] for row in df.iter_rows(named=True)
    }
    by_sequence[TRASTUZUMAB_VH] = BASELINE_ID

    scores: dict[str, float] = {}
    for csv_path in sorted(glob.glob(str(results_dir / "*.csv"))):
        table = pl.read_csv(csv_path)
        if "pdb_chain" not in table.columns or "pdb_res" not in table.columns:
            continue
        heavy = "".join(table.filter(pl.col("pdb_chain") == "B")["pdb_res"].to_list())
        design_id = by_sequence.get(heavy)
        if design_id is None:
            continue
        # The likelihood of the residue actually present, averaged over the Fv.
        likelihoods = [
            row[row["pdb_res"]]
            for row in table.iter_rows(named=True)
            if row["pdb_res"] in table.columns
        ]
        if likelihoods:
            scores[design_id] = sum(likelihoods) / len(likelihoods)
    return scores


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Score stability against the baseline and record it.

    Flags rather than drops: a design that does not beat trastuzumab's score
    simply does not earn the stability point.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``esm_if_score`` and ``esm_if_vs_baseline`` filled in.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    fetched = cfg.results_dir / "esmif_out"
    scores = read_scores(fetched, live) if fetched.exists() else {}
    baseline = scores.get(BASELINE_ID)

    if not scores or baseline is None:
        if funnel is not None:
            with funnel.step("10b_stability", rows_in=live.height) as rec:
                rec.rows_out = live.height
                rec.notes = (
                    "no ESM-IF1 scores and no baseline to compare against, so no "
                    "design can earn the stability point and every tier is "
                    "capped at 2 of 3"
                )
        return df

    ids = live["design_id"].to_list()
    annotations = pl.DataFrame(
        {
            "design_id": ids,
            "_esm": pl.Series([scores.get(i) for i in ids], dtype=pl.Float64),
        }
    ).with_columns((pl.col("_esm") - baseline).alias("_delta"))

    merged = (
        df.join(annotations, on="design_id", how="left")
        .with_columns(
            pl.coalesce("_esm", "esm_if_score").alias("esm_if_score"),
            pl.coalesce("_delta", "esm_if_vs_baseline").alias("esm_if_vs_baseline"),
        )
        .drop("_esm", "_delta")
    )

    beating = int(tbl.alive(merged).filter(pl.col("esm_if_vs_baseline") > 0).height)
    if funnel is not None:
        with funnel.step(
            "10b_stability",
            rows_in=live.height,
            thresholds={
                "model": "esm-if1",
                "baseline_log_likelihood": baseline,
                "bar": "beat the baseline's own score, same model same run",
                "is_melting_temperature": False,
            },
        ) as rec:
            rec.rows_out = live.height
            rec.notes = (
                f"{len(scores) - 1} designs scored; {beating} beat the baseline "
                f"log-likelihood of {baseline:.4f}. A rank, not a Tm."
            )
    return merged
