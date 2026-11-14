"""Building the query file the Apheris folding models take.

The models do not take a FASTA of the antibody. They take a ``queries.json``
naming **every chain in the complex**, the antigen included, because the thing
being predicted is the interface. A query carrying only the heavy and light
chains folds an Fv alone: no interface, no ipTM, nothing for the stopping rule
to measure a spread on.

HER2 comes from the same cropped 1N8Z the rest of the pipeline uses, split into
chains C (489-580) and D (591-607) across its unresolved loop, each carrying
explicit ``residue_ids`` so the numbering gap survives into the prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import gemmi

from .structure import ANTIGEN_CHAINS, load

#: One-letter codes for the residues the models accept.
_THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def antigen_chains(
    structure_path: Path, with_msa: bool = False
) -> list[dict[str, object]]:
    """Read HER2's chains out of the cropped complex.

    Args:
        structure_path: The ``her2_fv_1n8z.cif`` the pipeline folds against.
        with_msa: Omit ``msa_path`` so the model matches these chains to an
            MSA directory by sequence. An explicit ``null`` is an opt-out that
            no ``--msa-dir`` can override, which is why it cannot simply be
            left in place.

    Returns:
        One entry per antigen chain, with its sequence and residue numbers.

    Raises:
        FileNotFoundError: If the structure is missing.
    """
    model = load(structure_path)[0]
    chains: list[dict[str, object]] = []
    for chain in model:
        if chain.name not in ANTIGEN_CHAINS:
            continue
        sequence: list[str] = []
        residue_ids: list[str] = []
        for residue in chain:
            code = _THREE_TO_ONE.get(residue.name)
            if code is None or residue.seqid.num is None:
                continue
            sequence.append(code)
            residue_ids.append(str(residue.seqid.num))
        entry: dict[str, object] = {
            "chain_id": chain.name,
            "sequence": "".join(sequence),
            "residue_ids": residue_ids,
        }
        if not with_msa:
            entry["msa_path"] = None
        chains.append(entry)
    return chains


def build(
    designs: list[tuple[str, str, str]],
    structure_path: Path,
    antigen_msa: bool = False,
) -> dict[str, object]:
    """Assemble the query document for a batch of designs.

    Args:
        designs: ``(design_id, heavy_sequence, light_sequence)`` per design.
        structure_path: Source of the antigen chains.

    Returns:
        The query document, ready to serialise.

    Raises:
        ValueError: If the structure yields no antigen chains, which would
            silently turn every prediction into an unbound Fv fold.
    """
    antigen = antigen_chains(structure_path, with_msa=antigen_msa)
    if not antigen:
        raise ValueError(
            f"no antigen chains found in {structure_path}; folding without the "
            "target gives no interface and no ipTM"
        )
    queries = [
        {
            "query_id": design_id,
            "chains": [
                {"chain_id": "A", "sequence": light, "msa_path": None},
                {"chain_id": "B", "sequence": heavy, "msa_path": None},
                *antigen,
            ],
        }
        for design_id, heavy, light in designs
    ]
    return {"queries": queries}


def write(
    designs: list[tuple[str, str, str]],
    structure_path: Path,
    path: Path,
    antigen_msa: bool = False,
) -> Path:
    """Write the query file.

    Args:
        designs: ``(design_id, heavy_sequence, light_sequence)`` per design.
        structure_path: Source of the antigen chains.
        path: Destination JSON.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(designs, structure_path, antigen_msa), indent=1))
    return path


def chain_count(document: dict[str, object]) -> int:
    """Count chains in the first query, as a sanity check on the complex.

    A four-chain query is the Fv plus both HER2 fragments. Two chains means the
    antigen was lost and the fold would measure nothing.

    Args:
        document: A query document from :func:`build`.

    Returns:
        The number of chains in the first query, or 0 if there are none.

    Examples:
        >>> chain_count({"queries": [{"chains": [1, 2, 3, 4]}]})
        4
        >>> chain_count({"queries": []})
        0
    """
    queries = document.get("queries") or []
    if not isinstance(queries, list) or not queries:
        return 0
    first = queries[0]
    chains = first.get("chains") if isinstance(first, dict) else None
    return len(chains) if isinstance(chains, list) else 0


def to_gemmi_entity_count(structure_path: Path) -> int:
    """Count polymer chains in the structure, for diagnostics.

    Args:
        structure_path: The structure to inspect.

    Returns:
        Number of chains gemmi reports.
    """
    return sum(1 for _ in load(structure_path)[0])


__all__ = [
    "antigen_chains",
    "build",
    "chain_count",
    "gemmi",
    "to_gemmi_entity_count",
    "write",
]
