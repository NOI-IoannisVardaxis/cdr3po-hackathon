"""Pure sequence operations on the IMGT CDR3 and the trastuzumab framework.

Every function here is total, deterministic and free of I/O, so each one carries
a doctest that doubles as its specification. These are the operations where an
off-by-one silently changes which designs are legal, so they are tested directly
rather than only through the steps that call them.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from .config import (
    AMINO_ACIDS,
    CDR3_IMGT,
    MUTABLE_IMGT,
    TRASTUZUMAB_VL,
    VH_PREFIX,
    VH_SUFFIX,
)


def hamming(a: str, b: str) -> int:
    """Count positions at which two equal-length sequences differ.

    Args:
        a: First sequence.
        b: Second sequence, the same length as ``a``.

    Returns:
        The number of differing positions.

    Raises:
        ValueError: If the sequences have different lengths.

    Examples:
        >>> hamming("SRWGGDGFYAMDY", "SRWGGDGFYAMDY")
        0
        >>> hamming("SRWGGDGFYAMDY", "SRAGGDGFYAMDY")
        1
        >>> hamming("SRWGGDGFYAMDY", "SRAGGDHFTAMDY")
        3
        >>> hamming("ABC", "AB")
        Traceback (most recent call last):
            ...
        ValueError: length mismatch: 3 != 2
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def cdr3_identity(design: str, native: str = CDR3_IMGT) -> float:
    """Fraction of CDR3 positions identical to the native loop.

    The task PDF states the optimisation rule as "under 70% CDRH3 identity",
    but abguard enforces a mutation count instead. This function exists so the
    identity can be recorded and the two readings compared.

    Args:
        design: The design's IMGT CDR3.
        native: The reference CDR3, trastuzumab's by default.

    Returns:
        Identity as a fraction between 0.0 and 1.0.

    Examples:
        >>> cdr3_identity("SRWGGDGFYAMDY")
        1.0
        >>> round(cdr3_identity("SRAGGDHFTAMDY"), 4)
        0.7692
        >>> round(cdr3_identity("SRAAADHFTAMDY"), 4)
        0.6154
    """
    return 1.0 - hamming(design, native) / len(native)


def mutations_vs_native(design: str, native: str = CDR3_IMGT) -> str:
    """Describe a design's CDR3 changes in IMGT numbering.

    Args:
        design: The design's IMGT CDR3.
        native: The reference CDR3, trastuzumab's by default.

    Returns:
        Comma-separated ``<native><position><design>`` terms in position order,
        using 1-based IMGT numbering. An empty string if the loops are identical.

    Examples:
        >>> mutations_vs_native("SRWGGDGFYAMDY")
        ''
        >>> mutations_vs_native("SRAGGDGFYAMDY")
        'W3A'
        >>> mutations_vs_native("SRAGGDHFTAMDY")
        'W3A,G7H,Y9T'
    """
    return ",".join(
        f"{n}{i + 1}{d}" for i, (n, d) in enumerate(zip(native, design)) if n != d
    )


def splice_heavy_chain(cdr3: str) -> str:
    """Build a full heavy chain by inserting a CDR3 into trastuzumab's framework.

    Args:
        cdr3: A 13-residue IMGT CDR3.

    Returns:
        The 120-residue VH sequence carrying that loop.

    Raises:
        ValueError: If the CDR3 is not 13 residues long.

    Examples:
        >>> vh = splice_heavy_chain("SRAGGDHFTAMDY")
        >>> len(vh)
        120
        >>> vh.startswith("EVQLVESGGG")
        True
        >>> "SRAGGDHFTAMDY" in vh
        True
        >>> splice_heavy_chain("SRWGG")
        Traceback (most recent call last):
            ...
        ValueError: CDR3 must be 13 residues, got 5
    """
    if len(cdr3) != len(CDR3_IMGT):
        raise ValueError(f"CDR3 must be 13 residues, got {len(cdr3)}")
    return VH_PREFIX + cdr3 + VH_SUFFIX


def total_length(heavy: str, light: str = TRASTUZUMAB_VL) -> int:
    """Combined length of a paired construct, as abguard measures it.

    Args:
        heavy: The heavy chain variable domain.
        light: The light chain variable domain.

    Returns:
        Total residue count, which must not exceed 250.

    Examples:
        >>> total_length(splice_heavy_chain("SRAGGDHFTAMDY"))
        227
    """
    return len(heavy) + len(light)


def has_unpaired_cysteine(cdr3: str) -> bool:
    """Check whether a CDR3 introduces an odd number of cysteines.

    This is a cheap pre-filter only. abguard performs the real check across the
    whole construct; this exists to discard obvious failures before they reach it.

    Args:
        cdr3: A 13-residue IMGT CDR3.

    Returns:
        True if the loop contains an odd number of cysteines.

    Examples:
        >>> has_unpaired_cysteine("SRWGGDGFYAMDY")
        False
        >>> has_unpaired_cysteine("SRCGGDGFYAMDY")
        True
        >>> has_unpaired_cysteine("SRCGGDCFYAMDY")
        False
    """
    return cdr3.count("C") % 2 == 1


def enumerate_cdr3(
    n_mutations: int,
    native: str = CDR3_IMGT,
    positions: tuple[int, ...] = MUTABLE_IMGT,
) -> Iterator[str]:
    """Yield every CDR3 with exactly ``n_mutations`` substitutions.

    The complete space, not a sample. Deliberately dumb: free cysteines and
    implausible loops are emitted too, and later steps remove them.

    Args:
        n_mutations: Number of positions to substitute.
        native: The loop to mutate away from.
        positions: 1-based IMGT positions allowed to vary.

    Yields:
        Each distinct CDR3 at exactly that mutation count.

    Raises:
        ValueError: If more mutations are requested than there are positions.

    Examples:
        >>> len(list(enumerate_cdr3(1, positions=(3,))))
        19
        >>> sorted(list(enumerate_cdr3(1, positions=(3,))))[:2]
        ['SRAGGDGFYAMDY', 'SRCGGDGFYAMDY']
        >>> all(hamming(c, CDR3_IMGT) == 2 for c in enumerate_cdr3(2, positions=(3, 4)))
        True
        >>> len(list(enumerate_cdr3(2, positions=(3, 4, 5))))
        1083
    """
    if n_mutations > len(positions):
        raise ValueError(
            f"cannot make {n_mutations} mutations across {len(positions)} positions"
        )
    idx = [p - 1 for p in positions]
    for chosen in itertools.combinations(idx, n_mutations):
        alternatives = [[a for a in AMINO_ACIDS if a != native[i]] for i in chosen]
        for combo in itertools.product(*alternatives):
            loop = list(native)
            for i, aa in zip(chosen, combo):
                loop[i] = aa
            yield "".join(loop)


def count_enumerated(n_mutations: int, n_positions: int = len(MUTABLE_IMGT)) -> int:
    """Size of the space :func:`enumerate_cdr3` will produce, without building it.

    Args:
        n_mutations: Number of positions substituted.
        n_positions: Number of positions allowed to vary.

    Returns:
        ``C(n_positions, n_mutations) * 19 ** n_mutations``.

    Examples:
        >>> count_enumerated(3)
        823080
        >>> count_enumerated(4)
        27367410
        >>> count_enumerated(1, 1)
        19
    """
    from math import comb

    return comb(n_positions, n_mutations) * 19**n_mutations


def encode(sequences: list[str]) -> np.ndarray:
    """Pack equal-length sequences into a 2-D uint8 array for vectorised work.

    Args:
        sequences: Sequences of identical length.

    Returns:
        An ``(n, length)`` array of ASCII codes.

    Examples:
        >>> arr = encode(["SRW", "SRA"])
        >>> arr.shape
        (2, 3)
        >>> int((arr[0] != arr[1]).sum())
        1
    """
    if not sequences:
        return np.empty((0, 0), dtype=np.uint8)
    width = len(sequences[0])
    return np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(-1, width)


def one_hot(sequences: list[str]) -> np.ndarray:
    """One-hot encode sequences over the fixed amino-acid alphabet.

    Args:
        sequences: Sequences of identical length.

    Returns:
        A float32 array of shape ``(n, length * 20)``.

    Examples:
        >>> x = one_hot(["SRWGGDGFYAMDY"])
        >>> x.shape
        (1, 260)
        >>> float(x.sum())
        13.0
    """
    arr = encode(sequences)
    n, width = arr.shape
    lookup = np.full(256, -1, dtype=np.int16)
    for i, aa in enumerate(AMINO_ACIDS):
        lookup[ord(aa)] = i
    codes = lookup[arr]
    out = np.zeros((n, width * 20), dtype=np.float32)
    rows = np.repeat(np.arange(n), width)
    cols = (np.tile(np.arange(width), n) * 20) + codes.ravel()
    valid = codes.ravel() >= 0
    out[rows[valid], cols[valid]] = 1.0
    return out
