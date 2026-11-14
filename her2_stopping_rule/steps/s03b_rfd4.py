"""Step 3b: generate candidates with RFD4-Proteina.

The de novo front end. Runs on the GPU, so it is launched first and works in
the background while steps 3a-6a run on CPU: free GPU time that costs nothing
we would otherwise use.

**A caveat that must not be buried.** The de novo track requires under 70%
sequence identity to any known antibody. The weights pulled on this VM
(``antibody-2026-10-30``) redesign CDRs on a *fixed* framework, and CDRs are
roughly a quarter of a VHH. Redesigning all of them leaves identity near 73%,
just above the line. NVIDIA's general ``release-2026-10-19`` weights, which
could design a novel backbone, are not pulled locally.

So this step does not promise de novo designs. It generates candidates and lets
abguard classify them, which is the only correct behaviour anyway: the track is
decided for us and a design is never reclassified. Whatever lands in de novo
fills ranks 13-20; whatever lands in optimisation joins that pool and must then
clear the stricter three-mutation prior-art rule.

The de novo track's own CDRH3 rule is weaker than the optimisation one: "at
least 2 CDRH3 mutations away from trastuzumab" against three. A redesigned loop
clears it trivially, but it is checked locally so nothing reaches abguard that
could have been rejected here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import CDR3_IMGT, TRASTUZUMAB_VL, Config
from ..foundry import FoundryError, available
from ..foundry import run as foundry_run
from ..sequences import cdr3_identity, hamming, mutations_vs_native

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: RFD4 weights to generate with. The HER2 weights redesign trastuzumab's own
#: CDRH3 on its own framework, so they produce optimisation-track designs; the
#: antibody weights are the closest thing available to a de novo generator.
DEFAULT_WEIGHTS = "rfd4_antibody"

#: Design input shipped with the organisers' docs, relative to the RFD4 folder.
DEFAULT_INPUT = "Data/12-her2/her2_cdrh3.yaml"


def extract_cdr3(sequence: str) -> str | None:
    """Pull the IMGT CDR3 out of a designed heavy chain.

    The loop is bounded by the conserved cysteine motif that opens it and the
    ``WGQG`` motif that closes framework 4.

    Args:
        sequence: A designed heavy chain.

    Returns:
        The 13-residue CDR3, or None when the motifs are absent.

    Examples:
        >>> from ..config import TRASTUZUMAB_VH
        >>> extract_cdr3(TRASTUZUMAB_VH)
        'SRWGGDGFYAMDY'
        >>> extract_cdr3("NOTANANTIBODY") is None
        True
    """
    start = sequence.find("YYC")
    end = sequence.find("WGQG", max(start, 0))
    if start < 0 or end < 0 or end <= start + 3:
        return None
    return sequence[start + 3 : end] or None


def parse_designs(output_dir: Path) -> list[tuple[str, str]]:
    """Read designed sequences out of an RFD4 run directory.

    Args:
        output_dir: The ``design`` output folder.

    Returns:
        ``(design_name, heavy_sequence)`` pairs, empty when nothing was written.
    """
    designs: list[tuple[str, str]] = []
    if not output_dir.exists():
        return designs
    for fasta in sorted(output_dir.rglob("*.fasta")):
        name: str | None = None
        chunks: list[str] = []
        for line in fasta.read_text().splitlines():
            if line.startswith(">"):
                if name and chunks:
                    designs.append((name, "".join(chunks)))
                name, chunks = line[1:].split("|")[0].strip(), []
            elif line.strip():
                chunks.append(line.strip())
        if name and chunks:
            designs.append((name, "".join(chunks)))
    return designs


def generate(cfg: Config, n_designs: int, weights: str = DEFAULT_WEIGHTS) -> Path:
    """Run RFD4-Proteina and return its output directory.

    Args:
        cfg: Pipeline configuration.
        n_designs: Replicates to generate.
        weights: Key from :data:`~her2_stopping_rule.foundry.IMAGES`.

    Returns:
        The directory RFD4 wrote designs into.

    Raises:
        FoundryError: If the container fails.
        FileNotFoundError: If the RFD4 docs folder has not been synced.
    """
    rfd4_root = (cfg.repo_root / ".." / ".." / "rfd4").resolve()
    if not (rfd4_root / DEFAULT_INPUT).exists():
        raise FileNotFoundError(
            f"{rfd4_root / DEFAULT_INPUT} missing. Sync it with:\n"
            "  aws s3 sync s3://apheris-oslo-hackathon/rfd4-proteina/docs/ ~/rfd4"
        )

    out = rfd4_root / "runs" / "denovo"
    if out.exists():
        shutil.rmtree(out)

    foundry_run(
        weights,
        [
            "module-run",
            "design",
            "--input",
            DEFAULT_INPUT,
            "--output",
            str(out.relative_to(rfd4_root)),
            "--num-replicates",
            str(n_designs),
            "--seed",
            str(cfg.seed),
            "--recycle-strategy",
            "high/front_loaded",
        ],
        workdir=rfd4_root,
    )
    return out


def run(
    df: pl.DataFrame,
    cfg: Config,
    funnel: Funnel | None = None,
    n_designs: int = 32,
) -> pl.DataFrame:
    """Generate candidates on the GPU and add them to the table.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.
        n_designs: Replicates to ask RFD4 for.

    Returns:
        The table with generated candidates appended. Unchanged if the image is
        unavailable or the container fails, since this branch is optional.
    """
    from .. import table as tbl

    if not available(DEFAULT_WEIGHTS):
        if funnel is not None:
            with funnel.step("03b_rfd4", rows_in=0) as rec:
                rec.rows_out = 0
                rec.notes = f"{DEFAULT_WEIGHTS} image not present; branch skipped"
        return df

    try:
        out_dir = generate(cfg, n_designs)
    except (FoundryError, FileNotFoundError, OSError) as exc:
        if funnel is not None:
            with funnel.step("03b_rfd4", rows_in=0) as rec:
                rec.rows_out = 0
                rec.notes = f"generation failed, branch skipped: {exc}"
        return df

    rows: list[dict[str, object]] = []
    n_parsed = n_rejected = 0
    for name, heavy in parse_designs(out_dir):
        n_parsed += 1
        loop = extract_cdr3(heavy)
        if loop is None or len(loop) != len(CDR3_IMGT):
            n_rejected += 1
            continue
        distance = hamming(loop, CDR3_IMGT)
        if distance < cfg.min_cdrh3_mutations_denovo:
            n_rejected += 1
            continue
        rows.append(
            {
                "design_id": f"dn_{name}",
                "track_predicted": "de_novo",
                "cdr3_imgt": loop,
                "mutations": mutations_vs_native(loop),
                "n_mutations": distance,
                "cdr3_identity": cdr3_identity(loop),
                "heavy_sequence": heavy,
                "light_sequence": TRASTUZUMAB_VL,
                "kind": "paired",
                "stability_package": "rfd4",
                "framework_mutations": "rfd4-designed",
            }
        )

    generated = pl.DataFrame(rows) if rows else tbl.empty()
    out = pl.concat([tbl.conform(df), tbl.conform(generated)], how="vertical")

    if funnel is not None:
        with funnel.step(
            "03b_rfd4",
            rows_in=0,
            thresholds={
                "weights": DEFAULT_WEIGHTS,
                "n_designs": n_designs,
                "min_cdrh3_mutations_denovo": cfg.min_cdrh3_mutations_denovo,
            },
        ) as rec:
            rec.rows_out = generated.height
            rec.notes = (
                f"parsed {n_parsed}, rejected {n_rejected} on CDR3 extraction or the "
                "two-mutation floor. Track is decided by abguard, not here: these "
                "weights redesign CDRs on a fixed framework, so some may classify "
                "as optimisation."
            )
    return out
