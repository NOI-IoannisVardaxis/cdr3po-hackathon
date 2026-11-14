"""Step 6b: the de novo branch, by framework redesign.

The two tracks differ only in **mutation scope**. Confine changes to the CDR3
and identity stays near 98%, so abguard assigns optimisation. Allow changes
outside it and identity falls; below 70% the same tool assigns de novo. Verified
by probe against the installed build:

======================  ========  ===================
Framework mutations     Identity  abguard track
======================  ========  ===================
10                      0.956     optimisation
60                      0.740     optimisation
**80**                  **0.648** **de novo**
100                     0.559     de novo
======================  ========  ===================

Random mutation at that scale destroys the fold, so the changes come from
**ProteinMPNN inverse folding**: given the trastuzumab Fv backbone it proposes
sequences that fold to the same shape. Recovery runs near 40-50%, which lands
roughly where the table says the track flips.

Two things make this stronger than designing a binder from scratch. The
paratope geometry is unchanged, so the epitope axis — a judged property, tested
by competitive binding — comes nearly free. And the CDR3 is inherited from the
optimisation branch, so it carries a loop the surrogate already ranked against
524,346 measured variants rather than one nobody has scored.

The same design is therefore kept in both framework treatments, which is also a
hedge: if the redesigned framework fails to express, its optimisation twin is
still in the slate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl

from ..config import TRASTUZUMAB_VH, TRASTUZUMAB_VL, Config
from ..nebius import NebiusUnavailableError, available, design_sequences
from ..sequences import hamming
from ..structure import HEAVY_CHAIN, LIGHT_CHAIN, load

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Trastuzumab's CDRs, located by sequence so the spans survive any renumbering.
#: These are kept from the parent design; everything else may be redesigned.
HEAVY_CDRS: Final[tuple[str, ...]] = ("DTYIH", "RIYPTNGYTRYADSVKG")
LIGHT_CDRS: Final[tuple[str, ...]] = ("RASQDVNTAVA", "SASFLYS", "QQHYTTPPT")

#: Identity below which abguard assigns the de novo track.
DE_NOVO_IDENTITY: Final[float] = 0.70

#: Sampling temperatures to try. Low stays near native and may not clear the
#: threshold; high diversifies and risks the fold. Both are tried and whatever
#: lands under the threshold with the CDRs intact is kept.
TEMPERATURES: Final[tuple[float, ...]] = (0.2, 0.3)


def protected_spans(sequence: str, motifs: tuple[str, ...]) -> set[int]:
    """Locate residues that must survive redesign.

    Args:
        sequence: The chain to search.
        motifs: Subsequences to protect, located by exact match.

    Returns:
        Zero-based indices covered by any motif found.

    Examples:
        >>> sorted(protected_spans("AAXYZBB", ("XYZ",)))
        [2, 3, 4]
        >>> protected_spans("AAAA", ("QQ",))
        set()
    """
    keep: set[int] = set()
    for motif in motifs:
        start = sequence.find(motif)
        if start >= 0:
            keep.update(range(start, start + len(motif)))
    return keep


def graft(parent: str, redesigned: str, protect: set[int]) -> str:
    """Take the redesigned framework but keep the parent's protected residues.

    Args:
        parent: The design's own chain, carrying its CDRs.
        redesigned: ProteinMPNN's proposal for the same backbone.
        protect: Indices to take from ``parent`` rather than ``redesigned``.

    Returns:
        The grafted chain, the same length as ``parent``.

    Raises:
        ValueError: If the two chains differ in length.

    Examples:
        >>> graft("ABCD", "WXYZ", {1, 2})
        'WBCZ'
        >>> graft("ABCD", "WXYZ", set())
        'WXYZ'
        >>> graft("ABC", "WX", set())
        Traceback (most recent call last):
            ...
        ValueError: chain lengths differ: 3 vs 2
    """
    if len(parent) != len(redesigned):
        raise ValueError(f"chain lengths differ: {len(parent)} vs {len(redesigned)}")
    return "".join(
        parent[i] if i in protect else redesigned[i] for i in range(len(parent))
    )


def identity_to_trastuzumab(heavy: str, light: str) -> float:
    """Fraction of the construct identical to trastuzumab.

    This is the quantity abguard reports as ``identity_to_trastuzumab`` and uses
    to assign the track.

    Args:
        heavy: The design's heavy chain.
        light: Its light chain.

    Returns:
        Identity as a fraction, or 0.0 if either chain is the wrong length.

    Examples:
        >>> identity_to_trastuzumab(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        1.0
        >>> identity_to_trastuzumab("TOOSHORT", TRASTUZUMAB_VL)
        0.0
    """
    if len(heavy) != len(TRASTUZUMAB_VH) or len(light) != len(TRASTUZUMAB_VL):
        return 0.0
    total = len(TRASTUZUMAB_VH) + len(TRASTUZUMAB_VL)
    differences = hamming(heavy, TRASTUZUMAB_VH) + hamming(light, TRASTUZUMAB_VL)
    return 1.0 - differences / total


def write_fv_pdb(cfg: Config, path: Path) -> Path:
    """Write the Fv backbone alone, as the PDB ProteinMPNN expects.

    The antigen chains are dropped: we are redesigning the antibody's own
    framework, and including HER2 would invite the model to redesign it too.

    Args:
        cfg: Pipeline configuration, for the structure path.
        path: Destination PDB.

    Returns:
        The path written.
    """
    st = load(cfg.path("structure"))
    model = st[0]
    for chain in [c.name for c in model]:
        if chain not in (HEAVY_CHAIN, LIGHT_CHAIN):
            model.remove_chain(chain)
    st.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(path))
    return path


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Add a de novo twin for each optimisation design.

    Args:
        df: The design table, carrying full constructs from step 6a.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with de novo designs appended. Unchanged when the platform is
        unreachable, since this branch is optional.
    """
    from .. import table as tbl

    parents = tbl.alive(df).filter(pl.col("heavy_sequence").is_not_null())
    if parents.height == 0:
        return df

    if not available():
        if funnel is not None:
            with funnel.step("06b_denovo", rows_in=parents.height) as rec:
                rec.rows_out = parents.height
                rec.notes = (
                    "NEBIUS_CREDS unset; de novo branch skipped. Export it to the "
                    "team credentials file to enable framework redesign."
                )
        return df

    pdb = write_fv_pdb(cfg, cfg.results_dir / "foundry" / "fv_backbone.pdb")
    proposals: list[dict[str, str]] = []
    notes: list[str] = []
    for temperature in TEMPERATURES:
        try:
            result = design_sequences(
                pdb,
                chains=[LIGHT_CHAIN, HEAVY_CHAIN],
                n_samples=max(cfg.n_finalists // len(TEMPERATURES), 4),
                sampling_temp=temperature,
                seed=cfg.seed,
            )
            proposals.extend(result.sequences)
            notes.append(f"T={temperature}: {len(result.sequences)} proposals")
        except NebiusUnavailableError as exc:
            notes.append(f"T={temperature}: failed ({exc})")

    heavy_protect = protected_spans(TRASTUZUMAB_VH, HEAVY_CDRS)
    light_protect = protected_spans(TRASTUZUMAB_VL, LIGHT_CDRS)
    # The CDR3 is the parent's own designed loop and is always protected.
    cdr3_start = TRASTUZUMAB_VH.index("SRWGGDGFYAMDY")
    heavy_protect |= set(range(cdr3_start, cdr3_start + 13))

    rows: list[dict[str, object]] = []
    n_above = 0
    for parent in parents.iter_rows(named=True):
        for i, proposal in enumerate(proposals):
            redesigned_h = proposal.get(HEAVY_CHAIN, "")
            redesigned_l = proposal.get(LIGHT_CHAIN, "")
            if len(redesigned_h) != len(TRASTUZUMAB_VH):
                continue
            if len(redesigned_l) != len(TRASTUZUMAB_VL):
                continue
            heavy = graft(parent["heavy_sequence"], redesigned_h, heavy_protect)
            light = graft(parent["light_sequence"], redesigned_l, light_protect)
            identity = identity_to_trastuzumab(heavy, light)
            if identity >= DE_NOVO_IDENTITY:
                n_above += 1
                continue
            rows.append(
                {
                    "design_id": f"dn_{parent['design_id']}_m{i}",
                    "track_predicted": "de_novo",
                    "cdr3_imgt": parent["cdr3_imgt"],
                    "mutations": parent["mutations"],
                    "n_mutations": parent["n_mutations"],
                    "cdr3_identity": parent["cdr3_identity"],
                    "min_dist_to_provided": parent["min_dist_to_provided"],
                    "nearest_provided_id": parent["nearest_provided_id"],
                    "p_high": parent["p_high"],
                    "heavy_sequence": heavy,
                    "light_sequence": light,
                    "kind": "paired",
                    "stability_package": "mpnn_framework",
                    "framework_mutations": f"proteinmpnn:{identity:.3f}",
                }
            )

    generated = pl.DataFrame(rows) if rows else tbl.empty()
    out = pl.concat([tbl.conform(df), tbl.conform(generated)], how="vertical")

    if funnel is not None:
        with funnel.step(
            "06b_denovo",
            rows_in=parents.height,
            thresholds={
                "de_novo_identity": DE_NOVO_IDENTITY,
                "temperatures": list(TEMPERATURES),
                "proteinmpnn_weights": "v_48_020 vanilla",
            },
        ) as rec:
            rec.rows_out = tbl.alive(out).height
            rec.notes = (
                f"{'; '.join(notes)}; {generated.height} de novo designs under "
                f"{DE_NOVO_IDENTITY} identity, {n_above} rejected above it. "
                "Track is confirmed by abguard, never asserted here."
            )
    return out
