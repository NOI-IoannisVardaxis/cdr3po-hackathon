"""Step 13: build the deliverable and let abguard have the last word.

Three files come out of the pipeline, for three audiences:

``results/submission.csv``
    Twenty ranked rows in exactly the five columns abguard accepts. The only
    file anyone outside the team sees. ``team-validate`` must return compliant
    on this exact file, not on an earlier draft of it.

``results/shortlist.csv``
    Every design that reached scoring, with all scores and ``dropped_by``
    naming the step that rejected it. This answers "why design 7 and not
    design 40", which is what the judgment score is really asking.

``results/funnel.json``
    Counts and thresholds per step, accumulated while the pipeline ran.

Optimisation designs take the first ranks and de novo the remainder. Only the
top N of the list is tested and N depends on pitch placement, so that ordering
makes de novo a lottery ticket paid for with idle GPU rather than with any of
the four guaranteed slots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config
from . import s02_guard

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def order_by_track(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Interleave the two tracks so optimisation fills the tested ranks first.

    Args:
        df: Ranked live designs carrying ``track_predicted`` and ``rank``.
        cfg: Pipeline configuration, for the rank split.

    Returns:
        The designs reordered, optimisation first, truncated to ``n_submit``.

    Examples:
        >>> import polars as pl
        >>> from ..config import Config
        >>> df = pl.DataFrame({
        ...     "design_id": ["a", "b", "c"],
        ...     "track_predicted": ["de_novo", "optimisation", "optimisation"],
        ...     "rank": [1, 2, 3],
        ...     "heavy_sequence": ["EVQL"] * 3,
        ...     "light_sequence": ["DIQM"] * 3,
        ... })
        >>> order_by_track(df, Config())["design_id"].to_list()
        ['b', 'c', 'a']
    """
    from .. import table as tbl

    # Only real constructs can be submitted. A row without both chains is an
    # intermediate that some step failed to retire, and empty sequences reach
    # abguard as duplicates rather than as the missing data they are.
    submittable = tbl.conform(df).filter(
        pl.col("heavy_sequence").is_not_null()
        & pl.col("light_sequence").is_not_null()
        & (pl.col("heavy_sequence").str.len_chars() > 0)
    )
    optimisation = submittable.filter(pl.col("track_predicted") == "optimisation").sort(
        "rank"
    )
    de_novo = submittable.filter(pl.col("track_predicted") != "optimisation").sort(
        "rank"
    )
    keep_opt = optimisation.head(cfg.n_optimisation_ranks)
    remaining = cfg.n_submit - keep_opt.height
    filler = de_novo.head(max(remaining, 0))
    if filler.height < remaining:
        extra = optimisation.slice(keep_opt.height, remaining - filler.height)
        return pl.concat([keep_opt, filler, extra], how="vertical").head(cfg.n_submit)
    return pl.concat([keep_opt, filler], how="vertical").head(cfg.n_submit)


def write_outputs(df: pl.DataFrame, cfg: Config) -> dict[str, Path]:
    """Write the submission and the shortlist.

    Args:
        df: The full design table.
        cfg: Pipeline configuration.

    Returns:
        A mapping of output name to the path written.
    """
    from .. import table as tbl

    final = order_by_track(tbl.alive(df), cfg).with_row_index("final_rank", offset=1)

    submission = final.select(
        pl.col("design_id").alias("sequence_id"),
        pl.col("kind"),
        pl.col("heavy_sequence"),
        pl.col("light_sequence"),
        pl.col("reason").fill_null("").alias("design_description"),
    )
    submission_path = s02_guard.write_submission(
        submission, cfg.results_dir / "submission.csv"
    )

    shortlist_path = cfg.results_dir / "shortlist.csv"
    df.filter(
        pl.col("p_high").is_not_null() | pl.col("heavy_sequence").is_not_null()
    ).write_csv(shortlist_path)

    annotated_path = cfg.results_dir / "submission_annotated.csv"
    final.select(
        "final_rank",
        "design_id",
        "track_predicted",
        "track",
        "cdr3_imgt",
        "mutations",
        "n_mutations",
        "min_dist_to_provided",
        "p_high",
        "stability_package",
        "framework_mutations",
        "reason",
    ).write_csv(annotated_path)

    paths = {
        "submission": submission_path,
        "shortlist": shortlist_path,
        "annotated": annotated_path,
    }

    # One file per track, ranked on its own. The tracks are validated
    # independently and the prize pool is split evenly, so each deserves its
    # own ordered view even though only one list is submitted.
    for track, name in (("optimisation", "optimisation"), ("de_novo", "denovo")):
        part = tbl.alive(df).filter(pl.col("track_predicted") == track)
        if part.height == 0:
            continue
        path = cfg.results_dir / f"{name}.csv"
        part.sort("rank", nulls_last=True).select(
            "rank",
            "design_id",
            "cdr3_imgt",
            "mutations",
            "n_mutations",
            "min_dist_to_provided",
            "p_high",
            "cluster",
            "stability_package",
            "framework_mutations",
            "iptm_primary",
            "iptm_secondary",
            "model_spread",
            "epitope_recovery",
            "disulfide_formed",
            "heavy_sequence",
            "light_sequence",
            "reason",
        ).write_csv(path)
        paths[name] = path

    return paths


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Write the deliverables and validate the real submission file.

    Args:
        df: The full design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``track`` and ``track_agrees`` filled from abguard's
        verdict, never from our own prediction.

    Raises:
        RuntimeError: If abguard rejects the file that would be submitted.
    """
    from .. import table as tbl

    paths = write_outputs(df, cfg)
    final = order_by_track(tbl.alive(df), cfg)

    payload = final.select(
        pl.col("design_id").alias("sequence_id"),
        pl.col("kind"),
        pl.col("heavy_sequence"),
        pl.col("light_sequence"),
        pl.col("reason").fill_null("").alias("design_description"),
    )
    verdicts = s02_guard.validate(payload, cfg)
    rejected = [s for s, v in verdicts.items() if not v.compliant]

    out = df.with_columns(
        pl.when(pl.col("design_id").is_in(final["design_id"].to_list()))
        .then(pl.col("track_predicted"))
        .otherwise(pl.col("track"))
        .alias("track"),
        pl.when(pl.col("design_id").is_in(rejected))
        .then(pl.lit(False))
        .otherwise(pl.col("track_agrees"))
        .alias("track_agrees"),
    )

    if funnel is not None:
        with funnel.step(
            "13_submit",
            rows_in=final.height,
            thresholds={
                "n_submit": cfg.n_submit,
                "n_optimisation_ranks": cfg.n_optimisation_ranks,
            },
        ) as rec:
            rec.rows_out = final.height - len(rejected)
            rec.notes = f"wrote {paths['submission'].name}; " + (
                "abguard: compliant" if not rejected else f"abguard rejected {rejected}"
            )

    if rejected:
        raise RuntimeError(
            f"abguard rejected {len(rejected)} designs in the final submission: "
            f"{rejected}. The file at {paths['submission']} must not be submitted "
            "until these are removed or replaced."
        )
    return out
