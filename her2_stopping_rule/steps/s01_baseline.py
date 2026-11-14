"""Step 1: the trastuzumab baseline.

Everything downstream is a comparison against this one row, so it is computed
first and written to disk on its own. Without it no claim about a design is
checkable.

Most of it was already in the data: row 1 of the SPR table is trastuzumab
itself, CDR3 ``WGGDGFYAMD``, KD 0.152 nM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from ..config import BASELINE_KD_NM, CDR3_IMGT, Config
from ..data import load_spr
from ..structure import contact_map, disulfide_sites, load

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Cutoffs recorded so the sensitivity of the contact set is on the record.
SENSITIVITY_CUTOFFS = (4.0, 4.5, 5.0, 6.0)


@dataclass(frozen=True)
class Baseline:
    """Trastuzumab's reference numbers.

    Attributes:
        cdr3_imgt: The native IMGT CDR3.
        kd_nm: Measured affinity from the SPR table.
        contact_cutoff_a: Cutoff used for the operative contact map.
        contact_positions: IMGT positions touching HER2 at that cutoff.
        contacts_per_imgt: Atom-pair counts per IMGT position.
        n_her2_residues: Size of the native epitope.
        contact_sensitivity: Contact positions at each probed cutoff, so the
            choice of cutoff is visible rather than implicit.
        disulfide_candidates: Interface pairs able to carry a disulfide.
    """

    cdr3_imgt: str
    kd_nm: float
    contact_cutoff_a: float
    contact_positions: tuple[int, ...]
    contacts_per_imgt: dict[int, int]
    n_her2_residues: int
    contact_sensitivity: dict[str, list[int]]
    disulfide_candidates: list[dict[str, Any]]


def compute(cfg: Config) -> Baseline:
    """Measure every reference quantity from the data and the structure.

    Args:
        cfg: Pipeline configuration.

    Returns:
        The baseline record.

    Raises:
        FileNotFoundError: If the SPR table or the structure is missing.
    """
    spr = load_spr(cfg)
    native = spr.filter(pl.col("cdr3_imgt") == CDR3_IMGT)
    kd = float(native["kd_nM"][0]) if native.height else BASELINE_KD_NM

    st = load(cfg.path("structure"))
    operative = contact_map(st, cfg.contact_cutoff_a)
    sensitivity = {
        f"{c:.1f}": list(contact_map(st, c).contact_positions)
        for c in SENSITIVITY_CUTOFFS
    }
    sites = disulfide_sites(st, cfg.disulfide_cb_min, cfg.disulfide_cb_max)

    return Baseline(
        cdr3_imgt=CDR3_IMGT,
        kd_nm=kd,
        contact_cutoff_a=cfg.contact_cutoff_a,
        contact_positions=operative.contact_positions,
        contacts_per_imgt=operative.contacts_per_imgt,
        n_her2_residues=len(operative.her2_residues),
        contact_sensitivity=sensitivity,
        disulfide_candidates=[asdict(s) for s in sites],
    )


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Compute the baseline and write it to ``results/baseline.json``.

    Args:
        df: The design table, passed through unchanged.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table, untouched. This step produces a reference, not designs.
    """
    baseline = compute(cfg)
    path = cfg.results_dir / "baseline.json"
    path.write_text(json.dumps(asdict(baseline), indent=2, default=str))

    if funnel is not None:
        with funnel.step(
            "01_baseline",
            rows_in=0,
            thresholds={"contact_cutoff_a": cfg.contact_cutoff_a},
        ) as rec:
            rec.rows_out = 0
            best = (
                baseline.disulfide_candidates[0]
                if baseline.disulfide_candidates
                else None
            )
            rec.notes = (
                f"KD {baseline.kd_nm} nM; contacts at IMGT "
                f"{list(baseline.contact_positions)} at {cfg.contact_cutoff_a} A; "
                + (
                    f"best disulfide site {best['heavy_residue']}{best['heavy_seqid']}(H)-"
                    f"{best['light_residue']}{best['light_seqid']}(L) "
                    f"at {best['cb_distance']:.2f} A"
                    if best
                    else "no viable disulfide site"
                )
            )
    return df
