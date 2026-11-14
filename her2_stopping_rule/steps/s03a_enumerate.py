"""Step 3a: enumerate the optimisation design space.

Writes down every CDR3 with three or four substitutions across IMGT positions
3-12. Not sampled and not generated: the complete space, 28.2 million loops.

Why enumeration beats a generative model here. The provided HER2 weights produce
loops around edit distance 7-8, where 33.6% of measured variants are high
binders. At distance 3 that figure is 90.0% and at distance 4 it is 74.6%. The
space is small enough to write down in full, so sampling it is strictly worse
than reading all of it.

Why both distances. Distance 3 carries the best affinity prior but only 9.3% of
it clears prior art; distance 4 supplies volume at 50-70%. Mixing them also
hedges the open question of whether this abguard build is final, since a rule
change to a 4-mutation floor would otherwise invalidate the whole slate.

This step is deliberately dumb. It emits free cysteines and implausible loops,
and later steps remove them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ..config import CDR3_IMGT, Config
from ..sequences import (
    cdr3_identity,
    count_enumerated,
    enumerate_cdr3,
    has_unpaired_cysteine,
    mutations_vs_native,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def design_id(cdr3: str, n_mutations: int) -> str:
    """Build a stable identifier for an enumerated loop.

    The identifier is a pure function of the sequence, so re-running the step
    produces the same ids and the table can be joined across runs.

    Args:
        cdr3: The design's IMGT CDR3.
        n_mutations: Its distance from trastuzumab.

    Returns:
        An identifier of the form ``opt_d<distance>_<cdr3>``.

    Examples:
        >>> design_id("SRAGGDHFTAMDY", 3)
        'opt_d3_SRAGGDHFTAMDY'
    """
    return f"opt_d{n_mutations}_{cdr3}"


def build(cfg: Config, drop_unpaired_cys: bool = True) -> pl.DataFrame:
    """Enumerate every candidate loop at the configured edit distances.

    Args:
        cfg: Pipeline configuration, for ``edit_distances``.
        drop_unpaired_cys: Discard loops with an odd cysteine count up front.
            abguard would reject them anyway; skipping them early keeps the
            table smaller at no cost to correctness.

    Returns:
        A table with ``design_id``, ``cdr3_imgt``, ``mutations``,
        ``n_mutations``, ``cdr3_identity`` and ``track_predicted``.

    Examples:
        >>> from ..config import Config
        >>> import dataclasses
        >>> cfg = dataclasses.replace(Config(), edit_distances=(1,))
        >>> df = build(cfg)
        >>> df.height > 0
        True
        >>> set(df["n_mutations"].to_list())
        {1}
    """
    rows: list[dict[str, object]] = []
    for n_mut in cfg.edit_distances:
        for loop in enumerate_cdr3(n_mut):
            if drop_unpaired_cys and has_unpaired_cysteine(loop):
                continue
            rows.append(
                {
                    "design_id": design_id(loop, n_mut),
                    "cdr3_imgt": loop,
                    "mutations": mutations_vs_native(loop),
                    "n_mutations": n_mut,
                    "cdr3_identity": cdr3_identity(loop),
                    "track_predicted": "optimisation",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "design_id": pl.Utf8,
            "cdr3_imgt": pl.Utf8,
            "mutations": pl.Utf8,
            "n_mutations": pl.Int32,
            "cdr3_identity": pl.Float64,
            "track_predicted": pl.Utf8,
        },
    )


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Generate the optimisation candidate space.

    Args:
        df: Existing design table; enumerated rows are appended to it.
        cfg: Pipeline configuration.
        funnel: Optional :class:`~her2_stopping_rule.funnel.Funnel` for counts.

    Returns:
        The table with every enumerated candidate added.
    """
    from .. import table as tbl

    theoretical = sum(count_enumerated(d) for d in cfg.edit_distances)
    candidates = build(cfg)

    if funnel is not None:
        with funnel.step(
            "03a_enumerate",
            rows_in=0,
            thresholds={
                "edit_distances": list(cfg.edit_distances),
                "mutable_positions": "IMGT 3-12",
                "native_cdr3": CDR3_IMGT,
            },
        ) as rec:
            rec.rows_out = candidates.height
            rec.notes = (
                f"{theoretical:,} loops in the complete space; "
                f"{theoretical - candidates.height:,} removed for unpaired cysteine"
            )

    return pl.concat([tbl.conform(df), tbl.conform(candidates)], how="vertical")
