"""The design table: one row per design, columns only ever added.

Steps never delete rows. A design that fails a filter gets ``dropped_by`` set to
the step that rejected it and stays in the table. Three things follow from that
rule: the funnel counts for the pitch are a query rather than a reconstruction,
any step can be re-run without breaking the ones after it, and when something
looks wrong late in the day you can see exactly which step killed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import polars as pl

#: Columns every design carries, with the step that first writes each one.
SCHEMA: Final[dict[str, Any]] = {
    # identity, step 3
    "design_id": pl.Utf8,
    "track_predicted": pl.Utf8,
    "cdr3_imgt": pl.Utf8,
    "mutations": pl.Utf8,
    "n_mutations": pl.Int32,
    "cdr3_identity": pl.Float64,
    # step 4 novelty
    "min_dist_to_provided": pl.Int32,
    "nearest_provided_id": pl.Utf8,
    # step 5 affinity
    "p_high": pl.Float64,
    # step 6 stability
    "heavy_sequence": pl.Utf8,
    "light_sequence": pl.Utf8,
    "kind": pl.Utf8,
    "stability_package": pl.Utf8,
    "framework_mutations": pl.Utf8,
    "ddg_total": pl.Float64,
    "disulfide_site": pl.Utf8,
    "disulfide_cb_distance": pl.Float64,
    # step 7 fold
    "iptm_primary": pl.Float64,
    "iptm_secondary": pl.Float64,
    "min_ipae": pl.Float64,
    "binder_plddt": pl.Float64,
    "binder_plddt_vs_baseline": pl.Float64,
    "model_spread": pl.Float64,
    # step 9 epitope
    "epitope_recovery": pl.Float64,
    "epitope_recovery_no_template": pl.Float64,
    "is_baseline": pl.Boolean,
    "contact_cutoff_a": pl.Float64,
    "disulfide_sg_distance": pl.Float64,
    "disulfide_formed": pl.Boolean,
    "esm_if_score": pl.Float64,
    "esm_if_vs_baseline": pl.Float64,
    # step 10 express
    "express_tools_run": pl.Utf8,
    "liability_flags": pl.Utf8,
    # step 11-13 decide
    "cluster": pl.Int32,
    "track": pl.Utf8,
    "track_agrees": pl.Boolean,
    "rank": pl.Int32,
    "reason": pl.Utf8,
    "run_id": pl.Utf8,
    # bookkeeping
    "dropped_by": pl.Utf8,
}

TABLE_NAME: Final[str] = "designs.parquet"


def empty() -> pl.DataFrame:
    """Create an empty design table with the full schema.

    Returns:
        A zero-row DataFrame carrying every column in :data:`SCHEMA`.

    Examples:
        >>> df = empty()
        >>> df.height
        0
        >>> "dropped_by" in df.columns
        True
    """
    return pl.DataFrame(schema=SCHEMA)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Add any missing schema columns as nulls, in canonical order.

    Steps add only the columns they own; this fills in the rest so the table
    always round-trips through parquet with a stable shape.

    Args:
        df: A table that may be missing columns.

    Returns:
        The same rows with every schema column present, in schema order.

    Examples:
        >>> df = pl.DataFrame({"design_id": ["d1"], "cdr3_imgt": ["SRWGGDGFYAMDY"]})
        >>> out = conform(df)
        >>> out.columns == list(SCHEMA)
        True
        >>> out["p_high"].to_list()
        [None]
    """
    missing = [
        pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in SCHEMA.items()
        if name not in df.columns
    ]
    # Cast the columns that are present, too. A step that builds rows with
    # pl.DataFrame(...) and no schema gets polars' inferred types -- Int64
    # where this schema says Int32 -- and the mismatch only surfaces later,
    # as a vstack failure in whichever step happens to concatenate next.
    casts = [
        pl.col(name).cast(dtype, strict=False).alias(name)
        for name, dtype in SCHEMA.items()
        if name in df.columns
    ]
    extra = [c for c in df.columns if c not in SCHEMA]
    return df.with_columns(missing).with_columns(casts).select(list(SCHEMA) + extra)


def load(results_dir: Path) -> pl.DataFrame:
    """Read the design table, or an empty one if the run has not started.

    Args:
        results_dir: Directory holding ``designs.parquet``.

    Returns:
        The design table, conformed to the schema.
    """
    path = results_dir / TABLE_NAME
    if not path.exists():
        return empty()
    return conform(pl.read_parquet(path))


def save(df: pl.DataFrame, results_dir: Path) -> Path:
    """Write the design table to parquet.

    Args:
        df: The table to persist.
        results_dir: Destination directory, created if missing.

    Returns:
        The path written.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / TABLE_NAME
    conform(df).write_parquet(path)
    return path


def alive(df: pl.DataFrame) -> pl.DataFrame:
    """Select rows no step has dropped.

    Args:
        df: The design table.

    Returns:
        Only rows whose ``dropped_by`` is null.

    Examples:
        >>> df = pl.DataFrame({"design_id": ["a", "b"], "dropped_by": [None, "novelty"]})
        >>> alive(df)["design_id"].to_list()
        ['a']
    """
    return df.filter(pl.col("dropped_by").is_null())


def drop(df: pl.DataFrame, mask: pl.Expr, step: str) -> pl.DataFrame:
    """Mark rows as rejected without removing them.

    Only rows that are still alive can be dropped, so the first step to reject a
    design is the one recorded against it.

    Args:
        df: The design table.
        mask: Expression selecting rows to reject.
        step: Name of the rejecting step, written into ``dropped_by``.

    Returns:
        The table with ``dropped_by`` set on the newly rejected rows.

    Examples:
        >>> df = pl.DataFrame({"design_id": ["a", "b"], "p_high": [0.9, 0.1],
        ...                    "dropped_by": [None, None]})
        >>> out = drop(df, pl.col("p_high") < 0.5, "affinity")
        >>> out["dropped_by"].to_list()
        [None, 'affinity']
    """
    return df.with_columns(
        pl.when(pl.col("dropped_by").is_null() & mask)
        .then(pl.lit(step))
        .otherwise(pl.col("dropped_by"))
        .alias("dropped_by")
    )
