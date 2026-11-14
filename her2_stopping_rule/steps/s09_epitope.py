"""Step 9: confirm the epitope survived, and that the disulfide formed.

Two checks that no tool on the VM performs. ``refold-score`` reports interface
PAE, binder pLDDT and RMSD to design; abguard checks cysteine pairing by
sequence. Neither looks at where the paratope ended up or whether an engineered
disulfide actually closed.

**Epitope.** A rewritten CDR3 can bind HER2 tightly through a shifted paratope
and still fail Adaptyv's competitive binding assay, which is one of the three
judged axes. So each refold is compared against the native 1N8Z contact set and
designs that lost too much of it are dropped. This is the evidence-based version
of freezing the contact residues up front, which the FACS data showed would cost
affinity: contact-intact designs are 67.5% high binders against 81.9% for those
that mutate a contact.

**Disulfide.** Cysteines present but far apart mean the bond did not form and
the stability rationale for that design is void. Measured as an SG-SG distance,
with the design flagged rather than silently credited.

Verdicts are always reported with ``contact_cutoff_a``, because the contact set
is cutoff-dependent and sits on a cliff.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config
from ..structure import contact_map, epitope_recovery, load, sg_distance

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Where step 7 leaves refolded structures.
FOLD_SUBDIR = "foundry/fold"


def find_refold(results_dir: Path, design_id: str, model: str) -> Path | None:
    """Locate a refolded structure for one design.

    Args:
        results_dir: The run's results directory.
        design_id: The design to find.
        model: Which model's output to search.

    Returns:
        The structure path, or None when the refold is absent.
    """
    root = results_dir / FOLD_SUBDIR / model
    if not root.exists():
        return None
    safe = re.escape(design_id)
    for path in root.rglob("*.cif"):
        if re.search(safe, path.name):
            return path
    return None


def parse_disulfide_site(site: str | None) -> tuple[int, int] | None:
    """Read a disulfide site label back into residue numbers.

    Args:
        site: A label of the form ``"H106-L34"``, or None.

    Returns:
        The heavy and light residue numbers, or None.

    Examples:
        >>> parse_disulfide_site("H106-L34")
        (106, 34)
        >>> parse_disulfide_site(None) is None
        True
        >>> parse_disulfide_site("none") is None
        True
    """
    if not site:
        return None
    match = re.fullmatch(r"H(\d+)-L(\d+)", site)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def derive_threshold(baseline_recovery: float | None, cfg: Config) -> tuple[float, str]:
    """Set the epitope bar from what the model manages on trastuzumab itself.

    A fixed threshold is a guess about a model's accuracy. If the folding model
    reproduces only 80% of trastuzumab's own native contacts, demanding 75% of
    a design is nearly free; if it reproduces 99%, 75% is far too lenient. So
    the bar is a fraction of the baseline's own self-recovery, measured in the
    same run by the same model.

    Args:
        baseline_recovery: Fraction of its own native epitope the folded
            trastuzumab control recovered, or None if it was not folded.
        cfg: Pipeline configuration.

    Returns:
        The threshold and a one-line statement of where it came from.

    Examples:
        >>> from ..config import Config
        >>> cfg = Config()
        >>> t, why = derive_threshold(0.80, cfg)
        >>> round(t, 3)
        0.72
        >>> "baseline" in why
        True
        >>> t2, why2 = derive_threshold(None, cfg)
        >>> t2 == cfg.min_epitope_recovery
        True
        >>> "fallback" in why2
        True
    """
    if baseline_recovery is None:
        return (
            cfg.min_epitope_recovery,
            (
                f"fallback constant {cfg.min_epitope_recovery}: the trastuzumab "
                "control was not folded, so there is nothing to calibrate against"
            ),
        )
    threshold = baseline_recovery * cfg.epitope_recovery_fraction
    return (
        threshold,
        (
            f"{cfg.epitope_recovery_fraction:.0%} of the baseline's own recovery "
            f"({baseline_recovery:.3f}), measured by the same model in this run"
        ),
    )


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Measure epitope recovery and disulfide formation on every refold.

    Args:
        df: The design table.
        cfg: Pipeline configuration, for ``contact_cutoff_a`` and the recovery
            threshold.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``epitope_recovery``, ``disulfide_sg_distance`` and
        ``disulfide_formed`` filled in, and designs that lost the epitope marked
        ``dropped_by = "epitope"``.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    native = contact_map(load(cfg.path("structure")), cfg.contact_cutoff_a)
    from .s07_fold import fold_label

    # Ask step 7 where it wrote, rather than assuming the model name: with
    # templates off the directory is <model>_notemplate, and looking in the
    # wrong one finds nothing and reports it as "no refolds".
    primary = fold_label(cfg.fold_models[0], not cfg.fold_without_template)

    # What does this model manage on the molecule the epitope was defined from?
    from .s07_fold import BASELINE_ID

    baseline_recovery: float | None = None
    baseline_path = find_refold(cfg.results_dir, BASELINE_ID, primary)
    if baseline_path is not None:
        baseline_recovery = epitope_recovery(
            contact_map(load(baseline_path), cfg.contact_cutoff_a), native
        )
    threshold, threshold_basis = derive_threshold(baseline_recovery, cfg)

    recoveries: list[float | None] = []
    recoveries_nt: list[float | None] = []
    sg_distances: list[float | None] = []
    formed: list[bool | None] = []
    n_found = 0

    for row in live.iter_rows(named=True):
        path = find_refold(cfg.results_dir, row["design_id"], primary)
        if path is None:
            recoveries.append(None)
            recoveries_nt.append(None)
            sg_distances.append(None)
            formed.append(None)
            continue
        n_found += 1
        st = load(path)
        design_contacts = contact_map(st, cfg.contact_cutoff_a)
        recoveries.append(epitope_recovery(design_contacts, native))

        # The templated fold was shown where trastuzumab binds, so recovering
        # that epitope is partly an echo of the input. The unprompted fold is
        # the stronger evidence, recorded beside it rather than instead of it.
        nt_path = find_refold(cfg.results_dir, row["design_id"], primary)
        recoveries_nt.append(
            epitope_recovery(contact_map(load(nt_path), cfg.contact_cutoff_a), native)
            if nt_path is not None
            else None
        )

        site = parse_disulfide_site(row.get("disulfide_site"))
        if site is None:
            sg_distances.append(None)
            formed.append(None)
        else:
            distance = sg_distance(st, site[0], site[1])
            sg_distances.append(distance)
            formed.append(distance is not None and distance <= cfg.disulfide_sg_max)

    annotations = pl.DataFrame(
        {
            "design_id": live["design_id"],
            "_recovery": pl.Series(recoveries, dtype=pl.Float64),
            "_recovery_nt": pl.Series(recoveries_nt, dtype=pl.Float64),
            "_sg": pl.Series(sg_distances, dtype=pl.Float64),
            "_formed": pl.Series(formed, dtype=pl.Boolean),
        }
    )
    merged = (
        df.join(annotations, on="design_id", how="left")
        .with_columns(
            pl.coalesce("_recovery", "epitope_recovery").alias("epitope_recovery"),
            pl.coalesce("_recovery_nt", "epitope_recovery_no_template").alias(
                "epitope_recovery_no_template"
            ),
            pl.coalesce("_sg", "disulfide_sg_distance").alias("disulfide_sg_distance"),
            pl.coalesce("_formed", "disulfide_formed").alias("disulfide_formed"),
            pl.lit(cfg.contact_cutoff_a).alias("contact_cutoff_a"),
        )
        .drop("_recovery", "_recovery_nt", "_sg", "_formed")
    )

    # Only drop designs we actually measured. A missing refold is not evidence
    # of a lost epitope.
    merged = tbl.drop(
        merged,
        pl.col("epitope_recovery").is_not_null()
        & (pl.col("epitope_recovery") < threshold),
        "epitope",
    )

    if funnel is not None:
        with funnel.step(
            "09_epitope",
            rows_in=live.height,
            thresholds={
                "contact_cutoff_a": cfg.contact_cutoff_a,
                "epitope_threshold": threshold,
                "threshold_basis": threshold_basis,
                "baseline_self_recovery": baseline_recovery,
                "disulfide_sg_max": cfg.disulfide_sg_max,
                "native_epitope_residues": len(native.her2_residues),
            },
        ) as rec:
            rec.rows_out = tbl.alive(merged).height
            n_formed = sum(1 for f in formed if f)
            rec.notes = (
                f"{n_found}/{live.height} refolds found; "
                f"{n_formed} disulfides confirmed closed; "
                f"native epitope {len(native.her2_residues)} HER2 residues "
                f"at {cfg.contact_cutoff_a} A; threshold {threshold:.3f} - "
                f"{threshold_basis}"
            )
    return merged
