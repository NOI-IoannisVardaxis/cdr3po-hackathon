"""Command line interface.

One subcommand per step, plus ``run`` to execute several in order. Every
invocation reads ``results/designs.parquet``, applies its steps and writes the
table back, so steps can be run one at a time, re-run, or skipped.

Examples:
    List the steps and what each one does::

        uv run her2 steps

    Run the optimisation path end to end::

        uv run her2 run --path optimisation

    Run one step on its own, to inspect what it did::

        uv run her2 step 04a_novelty
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from typing import Any, Final

import click
import polars as pl

from . import table as tbl
from .config import Config
from .funnel import Funnel
from .steps import (
    s01_baseline,
    s02_guard,
    s03a_enumerate,
    s03b_rfd4,
    s04a_novelty,
    s05a_affinity,
    s06a_optimisation,
    s06b_denovo,
    s06c_antifold,
    s07_fold,
    s08_calibrate,
    s09_epitope,
    s10_express,
    s10b_stability,
    s11_rank,
    s12_stop,
    s13_submit,
    s14_report,
)

StepFn = Callable[[pl.DataFrame, Config, Funnel | None], pl.DataFrame]

#: Every step, in execution order, with a one-line description.
STEPS: Final[dict[str, tuple[StepFn, str]]] = {
    "01_baseline": (s01_baseline.run, "trastuzumab KD, contact map, disulfide sites"),
    "03a_enumerate": (
        s03a_enumerate.run,
        "every CDR3 at the configured edit distances",
    ),
    "04a_novelty": (s04a_novelty.run, "reject designs too close to provided CDR3s"),
    "05a_affinity": (s05a_affinity.run, "FACS surrogate scores and shortlists"),
    "06a_optimisation": (
        s06a_optimisation.run,
        "OPTIMISATION track: native framework + designed CDR3",
    ),
    "06b_denovo": (
        s06b_denovo.run,
        "ProteinMPNN redesigns the framework into the de novo track",
    ),
    "03b_rfd4": (s03b_rfd4.run, "RFD4-Proteina generates the de novo branch (GPU)"),
    "06c_antifold": (
        s06c_antifold.run,
        "DE NOVO v2: AntiFold redesigns the framework with CDRs held fixed",
    ),
    "02_guard": (s02_guard.run, "abguard verdict on every live construct"),
    "07_fold": (
        s07_fold.run,
        "co-fold with two Apheris models, measure the spread (GPU)",
    ),
    "08_calibrate": (s08_calibrate.run, "ipTM cutoff from measured non-binders"),
    "09_epitope": (s09_epitope.run, "contact recovery and disulfide formation"),
    "10_express": (s10_express.run, "developability flags, never drops"),
    "10b_stability": (
        s10b_stability.run,
        "ESM-IF1 inverse-folding score vs the baseline (the stability axis)",
    ),
    "11_rank": (s11_rank.run, "predicted tier, cluster, order"),
    "12_stop": (s12_stop.run, "cut where gain falls below the model spread"),
    "13_submit": (s13_submit.run, "write submission.csv and validate it"),
    "14_report": (s14_report.run, "figures and report.html for the pitch"),
}

#: Named step sequences. ``full`` is everything; the single-track paths exist
#: for when the GPU is busy or a branch is not wanted.
PATHS: Final[dict[str, tuple[str, ...]]] = {
    "denovo": (
        "01_baseline",
        "03a_enumerate",
        "04a_novelty",
        "05a_affinity",
        "06a_optimisation",
        "06b_denovo",
        "02_guard",
        "07_fold",
        "11_rank",
        "12_stop",
        "13_submit",
    ),
    "full": (
        "01_baseline",
        "03a_enumerate",
        "04a_novelty",
        "05a_affinity",
        "06a_optimisation",
        "06b_denovo",
        "02_guard",
        "07_fold",
        "08_calibrate",
        "09_epitope",
        "10_express",
        "10b_stability",
        "11_rank",
        "12_stop",
        "13_submit",
        "14_report",
    ),
    "optimisation": (
        "01_baseline",
        "03a_enumerate",
        "04a_novelty",
        "05a_affinity",
        "06a_optimisation",
        "02_guard",
        "11_rank",
        "12_stop",
        "13_submit",
    ),
}


def _config(**overrides: Any) -> Config:
    """Build a config with command-line overrides applied.

    Args:
        **overrides: Field names and values; ``None`` values are ignored.

    Returns:
        The configuration for this invocation.
    """
    clean = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(Config(), **clean)


@click.group()
def main() -> None:
    """HER2 antibody design pipeline."""


@main.command("steps")
def list_steps() -> None:
    """List every step in execution order."""
    for name, (_, description) in STEPS.items():
        click.echo(f"  {name:16s} {description}")


@main.command("config")
def show_config() -> None:
    """Print the active configuration as JSON."""
    import json

    click.echo(json.dumps(Config().to_dict(), indent=2))


@main.command("step")
@click.argument("name")
@click.option("--seed", type=int, default=None, help="Override the random seed.")
@click.option(
    "--contact-cutoff-a",
    type=float,
    default=None,
    help="Contact distance in angstroms.",
)
@click.option(
    "--results-dir",
    default=None,
    help="Write outputs here instead of results/, leaving an earlier run intact.",
)
def run_step(
    name: str,
    seed: int | None,
    contact_cutoff_a: float | None,
    results_dir: str | None,
) -> None:
    """Run one step by name.

    Args:
        name: Step name, as printed by ``her2 steps``.
        seed: Optional seed override.
        contact_cutoff_a: Optional contact cutoff override.
        results_dir: Optional output directory, so a rerun can be written
            beside a finished one rather than over it.
    """
    if name not in STEPS:
        raise click.BadParameter(f"unknown step {name!r}; try: her2 steps")
    overrides: dict[str, object] = {
        "seed": seed,
        "contact_cutoff_a": contact_cutoff_a,
    }
    if results_dir:
        paths = dict(Config()._rel_paths)
        paths["results"] = results_dir
        overrides["_rel_paths"] = paths
    cfg = _config(**overrides)
    funnel = Funnel(cfg.results_dir, cfg.to_dict())
    df = tbl.load(cfg.results_dir)

    fn, _ = STEPS[name]
    df = fn(df, cfg, funnel)
    tbl.save(df, cfg.results_dir)
    _report(df, name)


@main.command("run")
@click.option(
    "--path",
    type=click.Choice(sorted(PATHS)),
    default="optimisation",
    show_default=True,
    help="Named step sequence to execute.",
)
@click.option("--fresh/--resume", default=True, help="Start from an empty table.")
@click.option("--seed", type=int, default=None, help="Override the random seed.")
@click.option(
    "--edit-distances",
    default=None,
    help="Comma-separated CDR3 mutation counts, e.g. '3,4'.",
)
@click.option(
    "--shortlist-n", type=int, default=None, help="Loops kept after affinity ranking."
)
def run_path(
    path: str,
    fresh: bool,
    seed: int | None,
    edit_distances: str | None,
    shortlist_n: int | None,
) -> None:
    """Run a named sequence of steps in order.

    Args:
        path: Which sequence to run.
        fresh: Discard any existing table first.
        seed: Optional seed override.
        edit_distances: Optional comma-separated edit distances.
        shortlist_n: Optional shortlist size override.
    """
    distances = (
        tuple(int(x) for x in edit_distances.split(",")) if edit_distances else None
    )
    cfg = _config(seed=seed, edit_distances=distances, shortlist_n=shortlist_n)
    funnel = Funnel(cfg.results_dir, cfg.to_dict())
    df = tbl.empty() if fresh else tbl.load(cfg.results_dir)

    for name in PATHS[path]:
        fn, description = STEPS[name]
        click.echo(f"[{name}] {description}")
        try:
            df = fn(df, cfg, funnel)
        except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not swallow
            tbl.save(df, cfg.results_dir)
            click.echo(f"  failed: {exc}", err=True)
            sys.exit(1)
        tbl.save(df, cfg.results_dir)
        click.echo(f"  alive: {tbl.alive(df).height:,}")

    _report(df, path)


def _report(df: pl.DataFrame, label: str) -> None:
    """Print a short summary after a step or path completes.

    Args:
        df: The design table.
        label: What just ran.
    """
    live = tbl.alive(df)
    click.echo(f"\n{label}: {live.height:,} alive of {df.height:,} total")
    dropped = (
        df.filter(pl.col("dropped_by").is_not_null())
        .group_by("dropped_by")
        .len()
        .sort("len", descending=True)
    )
    for row in dropped.iter_rows(named=True):
        click.echo(f"  dropped by {row['dropped_by']:14s} {row['len']:,}")


if __name__ == "__main__":
    main()
