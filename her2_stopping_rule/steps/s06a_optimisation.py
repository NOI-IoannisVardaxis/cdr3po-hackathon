"""Step 6a: turn loops into molecules, and engineer the thermostability axis.

The novelty rules constrain only the CDR3. The framework is unconstrained, so
this is where thermostability can be optimised without touching novelty or the
epitope. It is also the axis nothing on offer predicts and most teams will
ignore, which is what separates an S-tier design from an A-tier one.

Two packages are built per loop, and the second is a deliberate control:

``disulfide`` -- **rejected by abguard, off by default**
    An engineered cysteine pair at the best interface site found by geometry
    (ALA106 heavy, ALA34 light, C-beta separation 4.11 A, both natives small
    enough that the substitution costs no packing). abguard refuses every one
    of them: each chain already carries two cysteines, adding one makes three,
    and its per-chain parity check reads an inter-chain disulfide as two
    unpaired cysteines. Measured, not assumed -- 48 of 48 rejected against 0
    of 48 native. The geometry work stands and the site is recorded; the
    construct simply cannot be submitted.

``native``
    The framework unchanged. This exists because the disulfide is a prediction,
    not a measurement: if it destabilises the construct or fails to form, the
    paired design without it is the one that survives. Submitting both means a
    wrong hypothesis costs half the slate rather than all of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ..config import TRASTUZUMAB_VL, Config
from ..sequences import splice_heavy_chain, total_length
from ..structure import DisulfideSite, disulfide_sites, load

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Maximum construct length abguard accepts.
MAX_TOTAL_LENGTH = 250


@dataclass(frozen=True)
class Package:
    """A framework variant applied to every shortlisted loop.

    Attributes:
        name: Identifier written to ``stability_package``.
        heavy_mutations: Heavy-chain substitutions as ``(seqid, residue)``.
        light_mutations: Light-chain substitutions in the same form.
        site: The disulfide site, when this package engineers one.
    """

    name: str
    heavy_mutations: tuple[tuple[int, str], ...] = ()
    light_mutations: tuple[tuple[int, str], ...] = ()
    site: DisulfideSite | None = None

    def describe(self) -> str:
        """Render the package's substitutions for the record.

        Returns:
            A comma-separated description, or ``"none"`` for the control.

        Examples:
            >>> Package("native").describe()
            'none'
            >>> Package("disulfide", ((106, "C"),), ((34, "C"),)).describe()
            'H106C,L34C'
        """
        parts = [f"H{n}{aa}" for n, aa in self.heavy_mutations]
        parts += [f"L{n}{aa}" for n, aa in self.light_mutations]
        return ",".join(parts) if parts else "none"


def apply_mutations(sequence: str, mutations: tuple[tuple[int, str], ...]) -> str:
    """Substitute residues by 1-based position.

    Args:
        sequence: The chain to modify.
        mutations: ``(position, residue)`` pairs, 1-based.

    Returns:
        The modified sequence.

    Raises:
        IndexError: If a position lies outside the sequence.

    Examples:
        >>> apply_mutations("AAAA", ((2, "C"),))
        'ACAA'
        >>> apply_mutations("AAAA", ((1, "C"), (4, "W")))
        'CAAW'
        >>> apply_mutations("AAAA", ((9, "C"),))
        Traceback (most recent call last):
            ...
        IndexError: position 9 outside a sequence of length 4
    """
    chars = list(sequence)
    for position, residue in mutations:
        if not 1 <= position <= len(chars):
            raise IndexError(
                f"position {position} outside a sequence of length {len(chars)}"
            )
        chars[position - 1] = residue
    return "".join(chars)


def build_packages(cfg: Config) -> list[Package]:
    """Choose the framework packages for this run.

    Args:
        cfg: Pipeline configuration, for the disulfide geometry window.

    Returns:
        Packages to apply, control first so it is never crowded out.
    """
    packages = [Package(name="native")]
    st = load(cfg.path("structure"))
    sites = disulfide_sites(st, cfg.disulfide_cb_min, cfg.disulfide_cb_max)
    if sites:
        best = sites[0]
        packages.append(
            Package(
                name="disulfide",
                heavy_mutations=((best.heavy_seqid, "C"),),
                light_mutations=((best.light_seqid, "C"),),
                site=best,
            )
        )
    return packages[: cfg.packages_per_loop]


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Expand shortlisted loops into full paired constructs.

    Each surviving loop becomes one row per package. The original loop rows are
    marked ``dropped_by = "stability"`` so the table keeps a complete record of
    what was expanded into what.

    Args:
        df: The design table, carrying ``cdr3_imgt`` and ``p_high`` on live rows.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with full constructs appended.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    n_loops = max(cfg.n_finalists // max(len(build_packages(cfg)), 1), 1)
    loops = live.sort("p_high", descending=True, nulls_last=True).head(n_loops)
    packages = build_packages(cfg)

    rows: list[dict[str, object]] = []
    for loop in loops.iter_rows(named=True):
        heavy_base = splice_heavy_chain(loop["cdr3_imgt"])
        for package in packages:
            heavy = apply_mutations(heavy_base, package.heavy_mutations)
            light = apply_mutations(TRASTUZUMAB_VL, package.light_mutations)
            if total_length(heavy, light) > MAX_TOTAL_LENGTH:
                continue
            rows.append(
                {
                    "design_id": f"{loop['design_id']}__{package.name}",
                    "track_predicted": "optimisation",
                    "cdr3_imgt": loop["cdr3_imgt"],
                    "mutations": loop["mutations"],
                    "n_mutations": loop["n_mutations"],
                    "cdr3_identity": loop["cdr3_identity"],
                    "min_dist_to_provided": loop["min_dist_to_provided"],
                    "nearest_provided_id": loop["nearest_provided_id"],
                    "p_high": loop["p_high"],
                    "heavy_sequence": heavy,
                    "light_sequence": light,
                    "kind": "paired",
                    "stability_package": package.name,
                    "framework_mutations": package.describe(),
                    "disulfide_site": (
                        f"H{package.site.heavy_seqid}-L{package.site.light_seqid}"
                        if package.site
                        else None
                    ),
                    "disulfide_cb_distance": (
                        package.site.cb_distance if package.site else None
                    ),
                }
            )

    constructs = pl.DataFrame(rows) if rows else tbl.empty()

    # Every loop that reached this step is now retired, whether or not it was
    # expanded. A loop is not a design: it has no heavy or light chain, so a
    # loop left alive would reach the submission with empty sequences, where
    # abguard reports them as a repeated heavy/light pair rather than as the
    # missing data they are.
    retired = tbl.drop(
        df,
        pl.col("design_id").is_in(live["design_id"].to_list()),
        "stability",
    )
    out = pl.concat([tbl.conform(retired), tbl.conform(constructs)], how="vertical")

    if funnel is not None:
        with funnel.step(
            "06a_optimisation",
            rows_in=live.height,
            thresholds={
                "packages": [p.name for p in packages],
                "disulfide_cb_window": [cfg.disulfide_cb_min, cfg.disulfide_cb_max],
                "max_total_length": MAX_TOTAL_LENGTH,
            },
        ) as rec:
            rec.rows_out = tbl.alive(out).height
            site = next((p.site for p in packages if p.site), None)
            rec.notes = f"{loops.height} loops x {len(packages)} packages; " + (
                f"disulfide at H{site.heavy_seqid}{site.heavy_residue}-"
                f"L{site.light_seqid}{site.light_residue}, "
                f"CB-CB {site.cb_distance:.2f} A, "
                f"conservative={site.conservative}"
                if site
                else "no viable disulfide site in the geometry window"
            )
    return out
