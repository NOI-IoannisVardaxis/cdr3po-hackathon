"""Step 10: developability screening.

A design that does not express in Adaptyv's cell-free system scores zero on all
three axes, so this is insurance against total loss rather than a ranking
signal. It flags rather than drops, because every tool here is a proxy.

What is and is not available, checked rather than assumed:

* **Aggrescan 3D, SoluProt, NetSolP** are academic web services. Running them
  would send design sequences to third parties before submission, which is a
  disclosure decision, so they are off by default and gated behind
  ``allow_external_services``.
* **ProteinMPNN** is reachable through Nebius but serves ``v_48_020``, the
  vanilla checkpoint, with no weight selector. The soluble weights the Adaptyv
  write-up recommends are not on the platform. The vanilla score is still a
  useful sequence-structure compatibility signal; it is simply not the
  solubility-tuned variant.
* **abguard liability-report** is local, instant, and covers the LAP liabilities.

One piece of judgment worth stating in the pitch: that Adaptyv advice targets de
novo designs, which have never been expressed. An optimisation design is a
clinically validated framework carrying three or four mutations in one loop, so
its expression risk is real but structurally different.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config
from . import s02_guard

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel


def liability_report(df: pl.DataFrame, cfg: Config) -> dict[str, str]:
    """Run ``abguard liability-report`` over the live constructs.

    Args:
        df: Live designs carrying the submission columns.
        cfg: Pipeline configuration.

    Returns:
        A mapping from design id to its liability summary. Empty on failure,
        since this step never blocks the pipeline.
    """
    if df.height == 0:
        return {}
    payload = df.select(
        pl.col("design_id").alias("sequence_id"),
        pl.col("kind"),
        pl.col("heavy_sequence"),
        pl.col("light_sequence"),
        pl.col("reason").fill_null("").alias("design_description"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = s02_guard.write_submission(payload, Path(tmp) / "designs.csv")
        proc = subprocess.run(
            [str(cfg.path("abguard")), "liability-report", str(path)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    if proc.returncode != 0:
        return {}
    return {"__report__": proc.stdout.strip()}


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Flag developability risk without dropping anything.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table with ``express_tools_run`` recording which tools answered.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df

    tools_run: list[str] = []
    report = liability_report(live, cfg)
    if report:
        tools_run.append("abguard_liability")
        (cfg.results_dir / "liability_report.txt").write_text(report["__report__"])

    merged = df.with_columns(
        pl.when(pl.col("design_id").is_in(live["design_id"].to_list()))
        .then(pl.lit(",".join(tools_run) if tools_run else "none"))
        .otherwise(pl.col("express_tools_run"))
        .alias("express_tools_run")
    )

    if funnel is not None:
        with funnel.step(
            "10_express",
            rows_in=live.height,
            thresholds={
                "tools_attempted": ["abguard_liability"],
                "external_web_services": False,
                "proteinmpnn_weights": "v_48_020 vanilla; soluble weights unavailable",
            },
        ) as rec:
            rec.rows_out = live.height
            rec.notes = (
                f"ran: {', '.join(tools_run) if tools_run else 'nothing'}. "
                "Aggrescan 3D, SoluProt and NetSolP skipped: external web services, "
                "not enabled. ProteinMPNN soluble weights unavailable on Nebius."
            )
    return merged
