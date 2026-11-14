"""Structural measurements on the trastuzumab-HER2 complex.

All of it runs on ``her2_fv_1n8z.cif``: PDB 1N8Z cropped to the Fv and HER2
domain IV, chain A the light chain, chain B the heavy chain, chains C and D the
antigen split at its unresolved loop.

Two measurements matter downstream, and neither is produced by any tool on the
VM. ``refold-score`` reports interface PAE, binder pLDDT and RMSD to design;
abguard checks cysteine pairing at the sequence level. Contact recovery and
disulfide geometry are ours to compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

#: Chain identifiers in the cropped structure.
LIGHT_CHAIN = "A"
HEAVY_CHAIN = "B"
ANTIGEN_CHAINS = ("C", "D")

#: Heavy-chain residue numbers spanning the CDR3 in this file's numbering.
CDR3_RESIDUES = range(99, 109)

#: Offset from a heavy-chain residue number to its 1-based IMGT CDR3 position.
#: Residue 99 is IMGT position 3, so the offset is -96.
_IMGT_OFFSET = -96


@dataclass(frozen=True)
class ContactMap:
    """CDR3 contacts with HER2 at one distance cutoff.

    The contact set is cutoff-dependent and sits on a cliff: 4.0 A gives two
    positions, 4.5-5.0 A gives four, 6.0 A gives six. Any verdict derived from
    it must be reported with the cutoff that produced it.

    Attributes:
        cutoff_a: Heavy-atom distance defining a contact.
        contacts_per_imgt: Atom-pair count per 1-based IMGT CDR3 position.
        contact_positions: IMGT positions with at least one contact.
        her2_residues: HER2 residues contacted, as ``(chain, seqid)`` pairs.
    """

    cutoff_a: float
    contacts_per_imgt: dict[int, int]
    contact_positions: tuple[int, ...]
    her2_residues: frozenset[tuple[str, int]]


@dataclass(frozen=True)
class DisulfideSite:
    """A heavy-light residue pair that could carry an engineered disulfide.

    Attributes:
        heavy_seqid: Heavy-chain residue number.
        heavy_residue: Its native three-letter code.
        light_seqid: Light-chain residue number.
        light_residue: Its native three-letter code.
        cb_distance: C-beta to C-beta distance in angstroms.
        conservative: True when both natives are small residues, so the cysteine
            substitution does not delete interface packing.
    """

    heavy_seqid: int
    heavy_residue: str
    light_seqid: int
    light_residue: str
    cb_distance: float
    conservative: bool


#: Residues small enough that mutating them to cysteine costs little packing.
_SMALL = frozenset({"ALA", "GLY", "SER", "THR", "CYS", "VAL"})


def load(path: Path) -> gemmi.Structure:
    """Read a structure and strip hydrogens.

    Args:
        path: Path to a CIF or PDB file.

    Returns:
        The parsed structure.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing; sync rfd4-proteina/docs from the bucket"
        )
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    return st


def _coords(residue: gemmi.Residue) -> np.ndarray:
    """Stack a residue's atom coordinates.

    Args:
        residue: The residue.

    Returns:
        An ``(n_atoms, 3)`` array.
    """
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in residue])


def _beta_carbon(residue: gemmi.Residue) -> np.ndarray | None:
    """Locate a residue's C-beta, falling back to C-alpha for glycine.

    Args:
        residue: The residue.

    Returns:
        A length-3 coordinate array, or None if neither atom is present.
    """
    for name in ("CB", "CA"):
        atom = residue.find_atom(name, "*")
        if atom is not None:
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return None


def contact_map(st: gemmi.Structure, cutoff_a: float) -> ContactMap:
    """Measure which CDR3 positions touch HER2, at a given cutoff.

    Args:
        st: The Fv-antigen complex.
        cutoff_a: Heavy-atom distance defining a contact.

    Returns:
        The contact map at that cutoff.
    """
    model = st[0]
    antigen_atoms = []
    antigen_owner = []
    for chain in model:
        if chain.name not in ANTIGEN_CHAINS:
            continue
        for residue in chain:
            seqid = residue.seqid.num
            if seqid is None:
                continue
            for atom in residue:
                antigen_atoms.append([atom.pos.x, atom.pos.y, atom.pos.z])
                antigen_owner.append((chain.name, seqid))
    antigen = np.array(antigen_atoms)

    heavy = next(c for c in model if c.name == HEAVY_CHAIN)
    per_imgt: dict[int, int] = {}
    touched: set[tuple[str, int]] = set()

    for residue in heavy:
        seqid = residue.seqid.num
        if seqid is None or seqid not in CDR3_RESIDUES:
            continue
        imgt = seqid + _IMGT_OFFSET
        distances = np.linalg.norm(
            _coords(residue)[:, None, :] - antigen[None, :, :], axis=2
        )
        close = distances < cutoff_a
        per_imgt[imgt] = int(close.sum())
        for j in np.flatnonzero(close.any(axis=0)):
            touched.add(antigen_owner[j])

    return ContactMap(
        cutoff_a=cutoff_a,
        contacts_per_imgt=per_imgt,
        contact_positions=tuple(p for p, n in sorted(per_imgt.items()) if n > 0),
        her2_residues=frozenset(touched),
    )


def disulfide_sites(
    st: gemmi.Structure, cb_min: float, cb_max: float
) -> list[DisulfideSite]:
    """Find heavy-light pairs geometrically able to carry a disulfide.

    The textbook VH44-VL100 pair is marginal in this structure, so sites are
    found by scanning the interface rather than hard-coded. Geometry alone is
    not sufficient: a pair in the ideal window whose natives are bulky aromatics
    would buy a disulfide by deleting interface packing, which is a bad trade.
    That is what ``conservative`` records.

    Args:
        st: The Fv-antigen complex.
        cb_min: Lower bound on C-beta to C-beta distance.
        cb_max: Upper bound on the same.

    Returns:
        Candidate sites, nearest to the 3.8 A ideal first.
    """
    model = st[0]
    heavy = next(c for c in model if c.name == HEAVY_CHAIN)
    light = next(c for c in model if c.name == LIGHT_CHAIN)

    sites: list[DisulfideSite] = []
    for rh in heavy:
        ph = _beta_carbon(rh)
        heavy_seqid = rh.seqid.num
        if ph is None or heavy_seqid is None:
            continue
        for rl in light:
            pl_ = _beta_carbon(rl)
            light_seqid = rl.seqid.num
            if pl_ is None or light_seqid is None:
                continue
            distance = float(np.linalg.norm(ph - pl_))
            if not cb_min <= distance <= cb_max:
                continue
            sites.append(
                DisulfideSite(
                    heavy_seqid=heavy_seqid,
                    heavy_residue=rh.name,
                    light_seqid=light_seqid,
                    light_residue=rl.name,
                    cb_distance=distance,
                    conservative=rh.name in _SMALL and rl.name in _SMALL,
                )
            )
    sites.sort(key=lambda s: (not s.conservative, abs(s.cb_distance - 3.8)))
    return sites


def sg_distance(
    st: gemmi.Structure, heavy_seqid: int, light_seqid: int
) -> float | None:
    """Measure the SG-SG distance of an engineered disulfide in a refold.

    If the cysteines are present but far apart the disulfide did not form, and
    the stability rationale for that design is void.

    Args:
        st: A refolded structure.
        heavy_seqid: Heavy-chain residue number carrying a cysteine.
        light_seqid: Light-chain residue number carrying a cysteine.

    Returns:
        The distance in angstroms, or None if either SG atom is absent.
    """
    model = st[0]
    try:
        heavy = next(c for c in model if c.name == HEAVY_CHAIN)
        light = next(c for c in model if c.name == LIGHT_CHAIN)
    except StopIteration:
        return None

    def _sg(chain: gemmi.Chain, seqid: int) -> np.ndarray | None:
        for residue in chain:
            if residue.seqid.num == seqid:
                atom = residue.find_atom("SG", "*")
                if atom is None:
                    return None
                return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        return None

    a, b = _sg(heavy, heavy_seqid), _sg(light, light_seqid)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def epitope_recovery(design: ContactMap, native: ContactMap) -> float:
    """Fraction of the native HER2 epitope a design still contacts.

    Args:
        design: Contact map computed on the refolded design.
        native: Contact map computed on the trastuzumab baseline.

    Returns:
        A fraction between 0.0 and 1.0, or 1.0 when the native set is empty.

    Examples:
        >>> native = ContactMap(5.0, {}, (), frozenset({("C", 1), ("C", 2)}))
        >>> design = ContactMap(5.0, {}, (), frozenset({("C", 1)}))
        >>> epitope_recovery(design, native)
        0.5
    """
    if not native.her2_residues:
        return 1.0
    kept = design.her2_residues & native.her2_residues
    return len(kept) / len(native.her2_residues)
