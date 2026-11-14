"""Exact prior-art filtering at scale.

A design must differ in at least three positions from every one of the 525,063
provided CDR3s. Written naively that is 28 million candidates times 525 thousand
references times ten positions, around 10^14 byte comparisons, which does not
finish. This module makes it exact and fast with the pigeonhole principle.

Split the ten-residue window into three blocks. Two sequences within Hamming
distance 2 must agree exactly on at least one block, because two mismatches
cannot cover three blocks. So a candidate that shares no block key with any
reference is guaranteed to be at distance 3 or more and needs no comparison at
all. Only candidates that collide on some block are compared, and only against
the references in that block's bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

#: Block boundaries over the 10-residue window. Three blocks guarantee that any
#: pair within Hamming distance 2 agrees exactly on at least one of them.
BLOCKS: Final[tuple[tuple[int, ...], ...]] = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))

#: Distance reported for a candidate no reference came close to. Any value above
#: the threshold is equivalent; this one is obviously a sentinel.
FAR: Final[int] = 99


@dataclass(frozen=True)
class NoveltyResult:
    """Outcome of the prior-art scan.

    Attributes:
        min_distance: Per candidate, the Hamming distance to its nearest
            provided CDR3, or :data:`FAR` when no reference was within 2.
        nearest_index: Per candidate, the blacklist row index of that nearest
            reference, or -1 when none was close.
        n_compared: Candidate-reference pairs actually compared, for reporting
            how much work the pigeonhole saved.
    """

    min_distance: np.ndarray
    nearest_index: np.ndarray
    n_compared: int


def _block_keys(arr: np.ndarray, block: tuple[int, ...]) -> np.ndarray:
    """Pack a block's residues into one integer key per row.

    Args:
        arr: ``(n, 10)`` uint8 array of windows.
        block: Column indices forming the block.

    Returns:
        An ``(n,)`` int64 array of keys.

    Examples:
        >>> a = np.array([[65, 66, 67, 68, 69, 70, 71, 72, 73, 74]], dtype=np.uint8)
        >>> _block_keys(a, (0, 1)).shape
        (1,)
        >>> b = a.copy(); b[0, 0] = 90
        >>> bool(_block_keys(a, (0, 1))[0] != _block_keys(b, (0, 1))[0])
        True
    """
    key = np.zeros(arr.shape[0], dtype=np.int64)
    for i, col in enumerate(block):
        key |= arr[:, col].astype(np.int64) << (8 * i)
    return key


def _expand_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Flatten a set of contiguous ranges into one index array.

    Args:
        starts: Start offset of each range.
        counts: Length of each range, all strictly positive.

    Returns:
        The concatenated indices of every range.

    Examples:
        >>> _expand_ranges(np.array([0, 5]), np.array([3, 2])).tolist()
        [0, 1, 2, 5, 6]
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if len(starts) > 1:
        jump = np.cumsum(counts)[:-1]
        out[jump] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(out)


def scan(
    candidates: np.ndarray,
    references: np.ndarray,
    chunk_size: int = 500_000,
    max_pairs: int = 20_000_000,
) -> NoveltyResult:
    """Find each candidate's distance to its nearest provided CDR3.

    Exact, not approximate: the pigeonhole guarantee means skipped pairs are
    provably at distance 3 or more, so nothing within the threshold is missed.

    Args:
        candidates: ``(m, 10)`` uint8 array of candidate windows.
        references: ``(n, 10)`` uint8 array of provided windows.
        chunk_size: Candidates processed per pass, bounding peak memory.
        max_pairs: Pair budget per sub-batch, bounding peak memory again.

    Returns:
        A :class:`NoveltyResult` aligned with ``candidates``.

    Examples:
        >>> ref = np.frombuffer(b"WGGDGFYAMD", dtype=np.uint8).reshape(1, 10)
        >>> same = np.frombuffer(b"WGGDGFYAMD", dtype=np.uint8).reshape(1, 10)
        >>> scan(same, ref).min_distance.tolist()
        [0]
        >>> near = np.frombuffer(b"WGGDGFYAMA", dtype=np.uint8).reshape(1, 10)
        >>> scan(near, ref).min_distance.tolist()
        [1]
        >>> far = np.frombuffer(b"AAAAAAAAAA", dtype=np.uint8).reshape(1, 10)
        >>> int(scan(far, ref).min_distance[0]) >= 3
        True
    """
    m = candidates.shape[0]
    best = np.full(m, FAR, dtype=np.int16)
    nearest = np.full(m, -1, dtype=np.int64)
    compared = 0

    prepared = []
    for block in BLOCKS:
        keys = _block_keys(references, block)
        order = np.argsort(keys, kind="stable")
        prepared.append((keys[order], order))

    for start in range(0, m, chunk_size):
        chunk = candidates[start : start + chunk_size]
        for block, (ref_keys, ref_order) in zip(BLOCKS, prepared):
            ckeys = _block_keys(chunk, block)
            lo = np.searchsorted(ref_keys, ckeys, side="left")
            hi = np.searchsorted(ref_keys, ckeys, side="right")
            counts = (hi - lo).astype(np.int64)
            hits = np.flatnonzero(counts)
            if hits.size == 0:
                continue

            # Walk the hit list in sub-batches so the expanded pair arrays stay
            # inside the memory budget however dense a bucket turns out to be.
            cum = np.cumsum(counts[hits])
            cut = 0
            while cut < hits.size:
                upto = int(np.searchsorted(cum, cum[cut] + max_pairs, side="right"))
                upto = max(upto, cut + 1)
                sel = hits[cut:upto]
                sel_counts = counts[sel]
                ref_pos = _expand_ranges(lo[sel], sel_counts)
                cand_idx = np.repeat(sel, sel_counts)
                ref_idx = ref_order[ref_pos]

                dist = (chunk[cand_idx] != references[ref_idx]).sum(axis=1)
                compared += int(dist.size)

                global_idx = cand_idx + start
                improved = dist < best[global_idx]
                if improved.any():
                    gi = global_idx[improved]
                    # np.minimum.at keeps the smallest distance when a candidate
                    # appears several times in the same batch.
                    np.minimum.at(best, gi, dist[improved].astype(np.int16))
                    exact = best[gi] == dist[improved]
                    nearest[gi[exact]] = ref_idx[improved][exact]
                cut = upto

    return NoveltyResult(min_distance=best, nearest_index=nearest, n_compared=compared)


def is_eligible(result: NoveltyResult, min_distance: int) -> np.ndarray:
    """Boolean mask of candidates clearing the prior-art rule.

    Args:
        result: Output of :func:`scan`.
        min_distance: Required Hamming distance from every provided CDR3.

    Returns:
        A boolean array, True where the candidate is submittable.

    Examples:
        >>> r = NoveltyResult(np.array([2, 3, FAR]), np.array([0, 1, -1]), 0)
        >>> is_eligible(r, 3).tolist()
        [False, True, True]
    """
    return result.min_distance >= min_distance
