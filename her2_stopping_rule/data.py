"""Loaders for the organisers' trastuzumab tables.

Both provided tables form the prior-art blacklist. Using only the FACS table
passes designs abguard then rejects: two of six probe designs were rejected
against ``trastuzumab_her2_spr_all_768`` specifically. All 525,063 or nothing.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import CDR3_IMGT, Config


def _to_imgt(window: str) -> str:
    """Pad a 10-residue mutation window to the 13-residue IMGT CDR3.

    Every provided variant shares trastuzumab's ``SR`` prefix and ``Y`` suffix;
    only the ten middle residues were ever varied by the screen.

    Args:
        window: The 10-residue window.

    Returns:
        The full IMGT CDR3.

    Examples:
        >>> _to_imgt("WGGDGFYAMD")
        'SRWGGDGFYAMDY'
    """
    return CDR3_IMGT[:2] + window + CDR3_IMGT[-1]


def load_facs(cfg: Config) -> pl.DataFrame:
    """Read the 524,346-variant FACS binding screen.

    Args:
        cfg: Pipeline configuration, for the data path.

    Returns:
        Columns ``cdr3_imgt``, ``window``, ``binding_class``, ``edit_distance``,
        ``is_high`` and ``source``.

    Raises:
        FileNotFoundError: If the table has not been synced from the bucket.
    """
    path = cfg.path("facs")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:\n"
            "  aws s3 sync s3://apheris-oslo-hackathon/data/her2-trastuzumab/ "
            "data/raw/her2-trastuzumab/"
        )
    with zipfile.ZipFile(path) as zf:
        name = next(
            n for n in zf.namelist() if n.endswith(".csv") and "MACOSX" not in n
        )
        with zf.open(name) as fh:
            df = pl.read_csv(io.BytesIO(fh.read()))
    return df.select(
        pl.col("cdrh3").alias("window"),
        (pl.lit(CDR3_IMGT[:2]) + pl.col("cdrh3") + pl.lit(CDR3_IMGT[-1])).alias(
            "cdr3_imgt"
        ),
        pl.col("binding_class"),
        pl.col("edit_distance").cast(pl.Int32),
        (pl.col("binding_class") == "high").alias("is_high"),
        pl.lit("facs").alias("source"),
    )


def load_spr(cfg: Config) -> pl.DataFrame:
    """Read the 768-row SPR affinity table.

    Rows whose CDR3 is not 10 residues come from a different loop definition and
    cannot be compared by Hamming distance; they are kept but flagged, and
    abguard remains the arbiter for them.

    Args:
        cfg: Pipeline configuration, for the data path.

    Returns:
        Columns ``cdr3_imgt``, ``window``, ``kd_nM``, ``binding_result``,
        ``comparable`` and ``source``.

    Raises:
        FileNotFoundError: If the table has not been synced from the bucket.
    """
    path = cfg.path("spr")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; sync data/raw/her2-trastuzumab/")
    df = pl.read_csv(path)
    return df.select(
        pl.col("cdrh3").alias("window"),
        pl.when(pl.col("cdrh3").str.len_chars() == 10)
        .then(pl.lit(CDR3_IMGT[:2]) + pl.col("cdrh3") + pl.lit(CDR3_IMGT[-1]))
        .otherwise(None)
        .alias("cdr3_imgt"),
        pl.col("kd_nM"),
        pl.col("binding_result"),
        (pl.col("cdrh3").str.len_chars() == 10).alias("comparable"),
        pl.lit("spr").alias("source"),
    )


@dataclass(frozen=True)
class Blacklist:
    """Every provided CDR3 a design must differ from.

    Attributes:
        windows: ``(n, 10)`` uint8 array of mutation windows.
        ids: Human-readable identifier per row, for ``nearest_provided_id``.
        distances: Each reference's own edit distance from trastuzumab.
        n_skipped: Provided CDR3s of a different length, left to abguard.
    """

    windows: np.ndarray
    ids: list[str]
    distances: np.ndarray
    n_skipped: int

    def __len__(self) -> int:
        """Number of comparable CDR3s in the blacklist."""
        return len(self.ids)

    def relevant_to(self, n_mutations: int, min_distance: int) -> np.ndarray:
        """Select the only references that can possibly be too close.

        Distance from trastuzumab bounds distance between any two sequences:
        a candidate ``n_mutations`` from the native loop and a reference ``d``
        from it are at least ``|n_mutations - d|`` apart. So a reference far
        from trastuzumab cannot be near a candidate that is close to it, and
        needs no comparison at all.

        For a 3-mutation candidate needing distance 3, only references within
        5 of trastuzumab can collide: 37,042 of 525,042, a fourteen-fold cut.

        Args:
            n_mutations: The candidate's distance from trastuzumab.
            min_distance: The prior-art threshold being enforced.

        Returns:
            A boolean mask over the blacklist.

        Examples:
            >>> import numpy as np
            >>> bl = Blacklist(np.zeros((3, 10), np.uint8), ["a", "b", "c"],
            ...                np.array([1, 5, 9]), 0)
            >>> bl.relevant_to(3, 3).tolist()
            [True, True, False]
        """
        return np.abs(self.distances - n_mutations) <= min_distance - 1


def load_blacklist(cfg: Config) -> Blacklist:
    """Build the prior-art blacklist from both provided tables.

    Args:
        cfg: Pipeline configuration.

    Returns:
        The combined blacklist, deduplicated, including trastuzumab itself.
    """
    facs = load_facs(cfg)
    spr = load_spr(cfg)
    n_skipped = int((~spr["comparable"]).sum())

    windows = facs["window"].to_list()
    ids = [f"facs:{w}" for w in windows]

    spr_ok = spr.filter(pl.col("comparable"))
    spr_windows = spr_ok["window"].to_list()
    windows += spr_windows
    ids += [f"spr:{w}" for w in spr_windows]

    # Trastuzumab is its own reference; abguard reports it as source "trastuzumab".
    windows.append(CDR3_IMGT[2:12])
    ids.append("trastuzumab")

    seen: dict[str, str] = {}
    for w, i in zip(windows, ids):
        seen.setdefault(w, i)
    uniq_windows = list(seen)
    uniq_ids = [seen[w] for w in uniq_windows]

    arr = np.frombuffer("".join(uniq_windows).encode(), dtype=np.uint8).reshape(-1, 10)
    native = np.frombuffer(CDR3_IMGT[2:12].encode(), dtype=np.uint8)
    distances = (arr != native).sum(axis=1).astype(np.int32)
    return Blacklist(
        windows=arr, ids=uniq_ids, distances=distances, n_skipped=n_skipped
    )
