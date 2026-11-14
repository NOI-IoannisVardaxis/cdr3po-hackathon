"""Step 14: assemble the run into one report for the pitch.

Draws every figure whose data exists and writes ``results/report.html``: the
flow, the funnel, the thresholds and the twenty designs with their reasons.

A figure whose step did not run is skipped, and the report says which steps did
not run rather than leaving a gap the reader has to notice. The whole point of
the funnel record is that the numbers on the slide are the numbers that ran, and
that only holds if absences are visible too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from .. import plots
from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Measured from the FACS screen; the figure that sets the design distance.
DISTANCE_CURVE: tuple[tuple[int, float, float | None], ...] = (
    (1, 96.0, None),
    (2, 97.5, None),
    (3, 90.0, 9.3),
    (4, 74.6, 69.5),
    (5, 59.4, 91.7),
    (6, 43.9, 97.6),
    (7, 33.6, 99.1),
    (8, 26.8, None),
    (9, 18.9, None),
    (10, 15.8, None),
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning an empty dict when it is absent.

    Args:
        path: The file to read.

    Returns:
        The parsed object, or ``{}``.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def build_figures(df: pl.DataFrame, cfg: Config) -> dict[str, Path]:
    """Draw every figure whose underlying data is present.

    Args:
        df: The design table.
        cfg: Pipeline configuration.

    Returns:
        A mapping of figure name to the path written. Missing data means a
        missing key, never an empty chart.
    """
    figures: dict[str, Path] = {}
    out = cfg.results_dir / "figures"
    funnel_data = _read_json(cfg.results_dir / "funnel.json")
    baseline = _read_json(cfg.results_dir / "baseline.json")
    calibration = _read_json(cfg.results_dir / "calibration.json")

    steps = funnel_data.get("steps", [])
    if steps:
        records = [
            (s["step"], s["rows_in"], s["rows_out"], s["seconds"]) for s in steps
        ]
        figures["funnel"] = plots.funnel(records, out / "funnel.png")

    distances = [d for d, _, _ in DISTANCE_CURVE]
    figures["distance_tradeoff"] = plots.distance_tradeoff(
        distances,
        [p for _, p, _ in DISTANCE_CURVE],
        [e for _, _, e in DISTANCE_CURVE],
        out / "distance_tradeoff.png",
    )

    sites = baseline.get("disulfide_candidates", [])
    if sites:
        figures["disulfide"] = plots.disulfide_geometry(
            sites[:6], out / "disulfide_geometry.png"
        )

    affinity = next((s for s in steps if s["step"].startswith("05a")), None)
    if affinity and "holdout_roc_auc" in affinity.get("thresholds", {}):
        # The ROC points are not persisted; recompute only if cheap to do so.
        from .s05a_affinity import validate_holdout

        metrics = validate_holdout(cfg)
        figures["surrogate_roc"] = plots.surrogate_roc(
            metrics.fpr,
            metrics.tpr,
            metrics.roc_auc,
            metrics.test_distance,
            metrics.base_rate,
            out / "surrogate_roc.png",
        )

    spreads = df["model_spread"].drop_nulls().to_numpy()
    if spreads.size:
        stop = next((s for s in steps if s["step"].startswith("12")), None)
        threshold = (
            float(stop["thresholds"].get("spread_threshold", float("nan")))
            if stop
            else float("nan")
        )
        figures["model_spread"] = plots.model_spread(
            spreads, threshold, out / "model_spread.png"
        )

    if calibration.get("cutoff") is not None:
        negatives = df["iptm_primary"].drop_nulls().to_numpy()
        if negatives.size:
            figures["calibration"] = plots.calibration(
                negatives,
                float(calibration["cutoff"]),
                float(calibration.get("achieved_fpr", float("nan"))),
                out / "calibration.png",
            )

    from .. import table as tbl

    final = tbl.alive(df).sort("rank", nulls_last=True).head(cfg.n_submit)
    if final.height:
        designs = [
            {
                "design_id": row["design_id"],
                "affinity": (row["p_high"] or 0) > 0.5,
                "epitope": (row["epitope_recovery"] or 0) >= cfg.min_epitope_recovery,
                "stability": bool(row["disulfide_formed"]),
            }
            for row in final.iter_rows(named=True)
        ]
        figures["tiers"] = plots.tier_triangles(designs, out / "tier_triangles.png")

    return figures


def render_html(df: pl.DataFrame, cfg: Config, figures: dict[str, Path]) -> Path:
    """Write the report page.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        figures: Figures produced by :func:`build_figures`.

    Returns:
        The path written.
    """
    from .. import table as tbl

    funnel_data = _read_json(cfg.results_dir / "funnel.json")
    steps = funnel_data.get("steps", [])
    ran = {s["step"] for s in steps}
    expected = {
        "01_baseline",
        "03a_enumerate",
        "03b_rfd4",
        "04a_novelty",
        "05a_affinity",
        "06a_optimisation",
        "02_guard",
        "07_fold",
        "08_calibrate",
        "09_epitope",
        "10_express",
        "11_rank",
        "12_stop",
        "13_submit",
    }
    missing = sorted(expected - ran)

    final = tbl.alive(df).sort("rank", nulls_last=True).head(cfg.n_submit)

    rows = "\n".join(
        f"<tr><td class='n'>{i}</td><td class='mono'>{r['design_id']}</td>"
        f"<td class='mono'>{r['cdr3_imgt'] or ''}</td>"
        f"<td class='mono'>{r['mutations'] or ''}</td>"
        f"<td class='n'>{'' if r['p_high'] is None else round(r['p_high'], 3)}</td>"
        f"<td>{r['stability_package'] or ''}</td>"
        f"<td>{r['track'] or r['track_predicted'] or ''}</td></tr>"
        for i, r in enumerate(final.iter_rows(named=True), start=1)
    )

    funnel_rows = "\n".join(
        f"<tr><td class='mono'>{s['step']}</td><td class='n'>{s['rows_in']:,}</td>"
        f"<td class='n'>{s['rows_out']:,}</td><td class='n'>{s['seconds']}</td>"
        f"<td>{s.get('notes', '')}</td></tr>"
        for s in steps
    )

    images = "\n".join(
        f"<figure><img src='figures/{p.name}' alt='{name}'></figure>"
        for name, p in figures.items()
    )

    missing_block = (
        f"<p class='warn'>Steps that did not run: {', '.join(missing)}. "
        "Figures and columns that depend on them are absent, not estimated.</p>"
        if missing
        else "<p class='ok'>Every step ran.</p>"
    )

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>HER2 design run</title>
<style>
  :root {{ color-scheme: light; --ink:#0b0b0b; --muted:#52514e; --line:#dcdbd6;
           --surface:#fcfcfb; }}
  body {{ margin:0 auto; max-width:1000px; padding:32px 20px 64px;
          background:var(--surface); color:var(--ink);
          font:15px/1.6 ui-sans-serif, system-ui, sans-serif; }}
  h1 {{ font-size:28px; margin:0 0 4px; }}
  h2 {{ font-size:19px; margin:36px 0 10px; }}
  .mono {{ font-family: ui-monospace, Menlo, monospace; font-size:13px; }}
  .n {{ font-variant-numeric: tabular-nums; text-align:right; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; margin-top:8px; }}
  th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left;
           vertical-align:top; }}
  th {{ font-size:11px; text-transform:uppercase; letter-spacing:.07em;
        color:var(--muted); }}
  figure {{ margin:18px 0; }}
  img {{ max-width:100%; border:1px solid var(--line); }}
  .warn {{ color:#e34948; }} .ok {{ color:#008300; }}
  .lede {{ color:var(--muted); }}
</style>
<h1>HER2 design run</h1>
<p class="lede">Every number below was produced by the run that wrote this page.
Thresholds are recorded beside the counts they produced.</p>
{missing_block}

<h2>Final designs</h2>
<table><thead><tr><th>#</th><th>design</th><th>CDR3 (IMGT)</th><th>mutations</th>
<th>P(high)</th><th>package</th><th>track</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>Funnel</h2>
<table><thead><tr><th>step</th><th>in</th><th>out</th><th>sec</th><th>notes</th>
</tr></thead><tbody>{funnel_rows}</tbody></table>

<h2>Figures</h2>
{images}
</html>
"""
    path = cfg.results_dir / "report.html"
    path.write_text(html)
    return path


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Draw the figures and write the report.

    Args:
        df: The design table.
        cfg: Pipeline configuration.
        funnel: Optional funnel for counts.

    Returns:
        The table, unchanged. This step reports; it does not select.
    """
    figures = build_figures(df, cfg)
    path = render_html(df, cfg, figures)

    if funnel is not None:
        with funnel.step("14_report", rows_in=df.height) as rec:
            rec.rows_out = df.height
            rec.notes = f"wrote {path.name} with {len(figures)} figures"
    return df
