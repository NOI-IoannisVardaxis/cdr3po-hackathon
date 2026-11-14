"""Step 6c: the de novo branch done properly, with the CDRs held fixed.

Step 6b reaches the de novo track by redesigning the framework with
ProteinMPNN and then grafting trastuzumab's CDRs back in. That works, and it
costs something measurable: ProteinMPNN chooses framework residues
*conditioned on the CDRs it also chose*, so overwriting 62 of 227 positions
afterwards leaves a chimera neither model endorses. The delivered de novo
designs scored a mean of -0.052 against the baseline on inverse-folding
likelihood, where the optimisation designs scored -0.021. The graft is the
difference.

AntiFold -- the antibody-aware model behind the ESM-IF1 endpoint -- takes a
``regions`` argument, so the framework can be redesigned while the CDRs are
**held fixed during design** rather than restored after it. The framework is
then chosen to fit the loops it will actually sit under.

Measured on the first run: 144 sequences, identity 0.568 to 0.797, and 32
below the 0.70 the de novo track requires.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import CDR3_IMGT, TRASTUZUMAB_VH, TRASTUZUMAB_VL, Config
from ..sequences import cdr3_identity, hamming, mutations_vs_native
from .s06b_denovo import DE_NOVO_IDENTITY, identity_to_trastuzumab

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: AntiFold writes the light chain first, then the heavy, separated by a slash.
_RECORD = re.compile(r"^>(?P<header>.*)$", re.MULTILINE)


def parse_designs(results_dir: Path) -> list[tuple[str, str]]:
    """Read designed chain pairs out of an AntiFold run.

    The first record in each file echoes the input sequence; the rest are
    samples. Chains come back as ``light/heavy``.

    Args:
        results_dir: Directory the run was fetched into.

    Returns:
        ``(heavy, light)`` pairs, excluding the echoed inputs.
    """
    designs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(str(results_dir / "*.txt"))):
        text = Path(path).read_text()
        if not text.startswith(">"):
            continue  # a manifest, not a FASTA
        for record in text.split(">")[2:]:  # [0] is empty, [1] is the input
            lines = record.splitlines()
            sequence = "".join(lines[1:]).replace(" ", "")
            if "/" not in sequence:
                continue
            light, heavy = sequence.split("/", 1)
            if len(heavy) == len(TRASTUZUMAB_VH) and len(light) == len(TRASTUZUMAB_VL):
                designs.append((heavy, light))
    return designs


def cdr3_of(heavy: str) -> str | None:
    """Extract the IMGT CDR3 from a designed heavy chain.

    Args:
        heavy: A 120-residue heavy chain.

    Returns:
        The 13-residue CDR3, or None if the flanking motifs are absent.

    Examples:
        >>> cdr3_of(TRASTUZUMAB_VH)
        'SRWGGDGFYAMDY'
        >>> cdr3_of("NOTANANTIBODY") is None
        True
    """
    start = heavy.find("YYC")
    end = heavy.find("WGQG", max(start, 0))
    if start < 0 or end < 0 or end - start - 3 != len(CDR3_IMGT):
        return None
    return heavy[start + 3 : end]


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Add AntiFold-designed de novo constructs to the table.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with qualifying de novo designs appended.
    """
    from .. import table as tbl

    source = cfg.results_dir / "antifold_out"
    if not source.exists():
        if funnel is not None:
            with funnel.step("06c_antifold", rows_in=0) as rec:
                rec.rows_out = 0
                rec.notes = f"no AntiFold output at {source}; branch skipped"
        return df

    parsed = parse_designs(source)
    rows: list[dict[str, object]] = []
    n_above = 0
    seen: set[tuple[str, str]] = set()

    for heavy, light in parsed:
        identity = identity_to_trastuzumab(heavy, light)
        if identity >= DE_NOVO_IDENTITY:
            n_above += 1
            continue
        loop = cdr3_of(heavy)
        if loop is None or (heavy, light) in seen:
            continue
        seen.add((heavy, light))
        rows.append(
            {
                "design_id": f"af_{loop}_{len(rows):03d}",
                "track_predicted": "de_novo",
                "cdr3_imgt": loop,
                "mutations": mutations_vs_native(loop),
                "n_mutations": hamming(loop, CDR3_IMGT),
                "cdr3_identity": cdr3_identity(loop),
                "heavy_sequence": heavy,
                "light_sequence": light,
                "kind": "paired",
                "stability_package": "antifold_framework",
                "framework_mutations": f"antifold:{identity:.3f}",
            }
        )

    generated = pl.DataFrame(rows) if rows else tbl.empty()
    out = pl.concat([tbl.conform(df), tbl.conform(generated)], how="vertical")

    if funnel is not None:
        with funnel.step(
            "06c_antifold",
            rows_in=len(parsed),
            thresholds={
                "de_novo_identity": DE_NOVO_IDENTITY,
                "cdrs": "held fixed during design, not grafted after",
            },
        ) as rec:
            rec.rows_out = generated.height
            rec.notes = (
                f"{len(parsed)} designed sequences; {generated.height} unique and "
                f"under {DE_NOVO_IDENTITY} identity, {n_above} rejected above it"
            )
    return out
