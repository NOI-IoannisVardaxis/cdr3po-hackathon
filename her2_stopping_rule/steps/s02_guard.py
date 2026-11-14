"""Step 2: abguard as a Python call.

abguard is the only authority on whether a design is submittable, and it is the
only filter that ever rejects our constructs: format, length, numbering and
cysteine checks all pass on a trastuzumab framework, leaving prior art. This
module wraps the CLI so every later step can ask for a verdict without a human
running a command.

Verified behaviour, from probing the installed build:

* ``required_cdrh3_mutations`` is **3**, not 4. The task PDF's "under 70% CDRH3
  identity" wording implies 4 on a 13-mer; the tool does not enforce it that way.
* Rejections cite the nearest provided CDR3 **and its source table**, which is
  how the SPR table was confirmed to matter alongside FACS.
* Hand-picked designs fail far more often than random ones, because plausible
  mutations collide with a screen that explored plausible space.
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from ..funnel import Funnel

#: Columns abguard requires, in the order it expects them.
SUBMISSION_COLUMNS = (
    "sequence_id",
    "kind",
    "heavy_sequence",
    "light_sequence",
    "design_description",
)

_ERROR_RE = re.compile(
    r"\[(?P<rule>\w+)\]\s+(?P<sid>[\w.\-]+):\s+(?P<message>.*?)(?:$)", re.MULTILINE
)
#: Errors abguard reports against the file rather than one design, such as a
#: repeated heavy/light pair. These name several ids on one line and have no
#: [rule] prefix, so they need their own pattern or the guard passes a file
#: abguard will reject.
_FILE_ERROR_RE = re.compile(
    r"^\s*-\s+(?P<message>repeated [^:]+):\s*(?P<ids>.+)$", re.MULTILINE
)
_NEAREST_RE = re.compile(
    r"is (?P<dist>\d+) mutation\(s\) from (?P<seq>[A-Z]+) in (?P<source>[\w.\-]+)"
)


@dataclass(frozen=True)
class Verdict:
    """One design's result from ``abguard team-validate``.

    Attributes:
        sequence_id: The design's identifier.
        compliant: Whether abguard accepted it.
        rule: Rule that rejected it, e.g. ``"novelty"``. Empty when compliant.
        message: abguard's own explanation. Empty when compliant.
        nearest_sequence: Nearest provided CDR3, when the rejection names one.
        nearest_source: Table that CDR3 came from.
        nearest_distance: Mutations from that CDR3.
    """

    sequence_id: str
    compliant: bool
    rule: str = ""
    message: str = ""
    nearest_sequence: str = ""
    nearest_source: str = ""
    nearest_distance: int = -1


class AbguardUnavailableError(RuntimeError):
    """Raised when the abguard virtualenv is missing or lacks the novelty gate."""


def check_available(cfg: Config) -> str:
    """Confirm abguard is installed and is the build with the novelty gate.

    Two different wheels on the hackathon VM both call themselves ``0.1.0``. The
    older one validates the hard filters, reports every design compliant and
    never mentions that it skipped prior art. This guard refuses to run against
    it rather than producing a clean, meaningless pass.

    Args:
        cfg: Pipeline configuration, for the abguard path.

    Returns:
        The ``--help`` output, for the record.

    Raises:
        AbguardUnavailableError: If abguard is missing or has no novelty gate.
    """
    exe = cfg.path("abguard")
    if not exe.exists():
        raise AbguardUnavailableError(
            f"{exe} missing. Create it with:\n"
            "  uv venv --python 3.12 .venv-abguard\n"
            "  VIRTUAL_ENV=$PWD/.venv-abguard uv pip install <abguard wheel>"
        )
    out = subprocess.run(
        [str(exe), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    ).stdout
    if "novelty-report" not in out:
        raise AbguardUnavailableError(
            f"{exe} has no novelty gate. This is the older wheel, which reports "
            "every design compliant without checking prior art. Install the build "
            "whose --help lists novelty-report."
        )
    return out


def describe_fallback() -> pl.Expr:
    """Build a design description that is never blank.

    abguard needs one, and ``reason`` is only written at step 11, so anything
    validating earlier has to say something of its own.

    Returns:
        An expression producing a short description per design.
    """
    return (
        pl.when(pl.col("reason").is_not_null() & (pl.col("reason") != ""))
        .then(pl.col("reason"))
        .otherwise(
            pl.format(
                "CDR3 {} ({} mutations), {} framework",
                pl.col("mutations").fill_null("none"),
                pl.col("n_mutations").fill_null(0),
                pl.col("stability_package").fill_null("native"),
            )
        )
    )


def write_submission(df: pl.DataFrame, path: Path) -> Path:
    """Write a design table in abguard's submission format.

    Args:
        df: Rows carrying at least :data:`SUBMISSION_COLUMNS`.
        path: Destination CSV.

    Returns:
        The path written.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cannot write submission, missing columns: {missing}")
    # An empty design_description fails abguard's schema, and it fails it
    # silently: every row is dropped, exit code 0, no error line. Refuse to
    # write one rather than let a whole run disappear into a clean exit.
    blank = (
        df.filter(
            pl.col("design_description").is_null()
            | (pl.col("design_description").str.strip_chars() == "")
        ).height
        if "design_description" in df.columns
        else 0
    )
    if blank:
        raise ValueError(
            f"{blank} design(s) have an empty design_description; abguard drops "
            "those rows silently, so every design would be rejected"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUBMISSION_COLUMNS)
        for row in df.select(SUBMISSION_COLUMNS).iter_rows():
            writer.writerow(["" if v is None else v for v in row])
    return path


def parse_errors(stderr: str) -> dict[str, Verdict]:
    """Turn abguard's error output into per-design verdicts.

    Args:
        stderr: Combined output from ``abguard team-validate``.

    Returns:
        A mapping from ``sequence_id`` to the verdict that rejected it.

    Examples:
        >>> err = ("  - [novelty] d3_probe_1: CDRH3 SRAGGDGEYAPDY is 2 mutation(s) "
        ...        "from SRWGGDGWYAPDY in trastuzumab_her2_spr_all_768; a novel "
        ...        "design needs at least 3")
        >>> v = parse_errors(err)["d3_probe_1"]
        >>> v.rule, v.nearest_distance, v.nearest_source
        ('novelty', 2, 'trastuzumab_her2_spr_all_768')
        >>> v.compliant
        False
        >>> parse_errors("no errors here")
        {}
    """
    verdicts: dict[str, Verdict] = {}
    for match in _FILE_ERROR_RE.finditer(stderr):
        message = match.group("message").strip()
        for sid in (s.strip() for s in match.group("ids").split(",")):
            if sid:
                verdicts[sid] = Verdict(
                    sequence_id=sid, compliant=False, rule="schema", message=message
                )
    for match in _ERROR_RE.finditer(stderr):
        sid = match.group("sid")
        message = match.group("message").strip()
        near = _NEAREST_RE.search(message)
        verdicts[sid] = Verdict(
            sequence_id=sid,
            compliant=False,
            rule=match.group("rule"),
            message=message,
            nearest_sequence=near.group("seq") if near else "",
            nearest_source=near.group("source") if near else "",
            nearest_distance=int(near.group("dist")) if near else -1,
        )
    return verdicts


def validate(df: pl.DataFrame, cfg: Config) -> dict[str, Verdict]:
    """Run ``abguard team-validate`` over a design table.

    Args:
        df: Rows carrying :data:`SUBMISSION_COLUMNS`.
        cfg: Pipeline configuration.

    Returns:
        One verdict per design, compliant designs included.

    Raises:
        AbguardUnavailableError: If abguard is missing or is the wrong build.
    """
    check_available(cfg)
    if df.height == 0:
        return {}

    # Compliance is decided by membership in abguard's own --drop output, not
    # by reading its error text. abguard truncates the error list, so parsing
    # it passes designs that a later, smaller call then rejects: that is
    # exactly how ten patent collisions reached a finished submission.
    # The error text is still read, but only to explain a rejection.
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = write_submission(df, Path(tmp) / "submission.csv")
        kept_path = Path(tmp) / "kept.csv"
        proc = subprocess.run(
            [
                str(cfg.path("abguard")),
                "team-validate",
                str(csv_path),
                "--drop",
                "--output",
                str(kept_path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=900,
        )
        survivors: set[str] = set()
        if kept_path.exists():
            survivors = set(pl.read_csv(kept_path)["sequence_id"].to_list())
        elif proc.returncode == 0:
            # Nothing was dropped, so nothing was written.
            survivors = set(df["sequence_id"].to_list())

    failures = parse_errors(proc.stdout + proc.stderr)
    verdicts: dict[str, Verdict] = {}
    for sid in df["sequence_id"].to_list():
        if sid in survivors:
            verdicts[sid] = Verdict(sequence_id=sid, compliant=True)
        else:
            verdicts[sid] = failures.get(
                sid,
                Verdict(
                    sequence_id=sid,
                    compliant=False,
                    rule="abguard",
                    message="dropped by abguard; reason not in the truncated error list",
                ),
            )
    return verdicts


def run(df: pl.DataFrame, cfg: Config, funnel: Funnel | None = None) -> pl.DataFrame:
    """Validate every live design and record abguard's verdict.

    Args:
        df: The design table. Rows need ``design_id``, ``heavy_sequence``,
            ``light_sequence`` and ``kind``.
        cfg: Pipeline configuration.
        funnel: Unused; present so every step shares one signature.

    Returns:
        The table with ``dropped_by`` set to ``"guard"`` on rejected designs and
        ``liability_flags`` carrying abguard's reason.
    """
    from .. import table as tbl

    live = tbl.alive(df)
    if live.height == 0:
        return df
    payload = live.select(
        pl.col("design_id").alias("sequence_id"),
        pl.col("kind"),
        pl.col("heavy_sequence"),
        pl.col("light_sequence"),
        describe_fallback().alias("design_description"),
    )
    verdicts = validate(payload, cfg)
    rejected = {s for s, v in verdicts.items() if not v.compliant}
    reasons = {
        s: f"{v.rule}: {v.message}" for s, v in verdicts.items() if not v.compliant
    }

    return df.with_columns(
        pl.when(
            pl.col("dropped_by").is_null() & pl.col("design_id").is_in(list(rejected))
        )
        .then(pl.lit("guard"))
        .otherwise(pl.col("dropped_by"))
        .alias("dropped_by"),
        pl.col("design_id")
        .replace_strict(reasons, default=None)
        .alias("liability_flags"),
    )
