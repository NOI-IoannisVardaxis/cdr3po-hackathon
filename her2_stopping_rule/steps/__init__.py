"""Pipeline steps.

Every step exposes ``run(df, cfg, funnel) -> pl.DataFrame``. A step reads the
design table, adds the columns it owns, and returns it. Steps never delete rows;
they mark rejections with :func:`her2_stopping_rule.table.drop`.

Steps are numbered so the execution order is readable from the module names.
Suffixes ``a`` and ``b`` mark the two track front ends: ``a`` is optimisation,
``b`` is de novo. Everything from step 7 onward is shared.
"""

from __future__ import annotations

__all__ = [
    "s01_baseline",
    "s02_guard",
    "s03a_enumerate",
    "s03b_rfd4",
    "s04a_novelty",
    "s05a_affinity",
    "s06a_optimisation",
    "s07_fold",
    "s08_calibrate",
    "s09_epitope",
    "s10_express",
    "s11_rank",
    "s12_stop",
    "s13_submit",
    "s14_report",
]
