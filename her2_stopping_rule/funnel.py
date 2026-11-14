"""Funnel accounting: how many designs each step saw, kept and rejected.

The counts accumulate while the pipeline runs rather than being reconstructed
afterwards, so the numbers on the pitch slide are the numbers that actually ran.
Each record also carries the thresholds in force, because a count without its
threshold cannot be reproduced.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FUNNEL_NAME = "funnel.json"


@dataclass
class StepRecord:
    """One step's contribution to the funnel.

    Attributes:
        step: Step name, e.g. ``"04a_novelty"``.
        rows_in: Rows alive when the step began.
        rows_out: Rows alive when it finished.
        dropped: Rows this step rejected.
        seconds: Wall-clock duration.
        thresholds: Threshold values the step applied.
        notes: Anything worth saying about this run of the step.
    """

    step: str
    rows_in: int = 0
    rows_out: int = 0
    dropped: int = 0
    seconds: float = 0.0
    thresholds: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class Funnel:
    """Append-only record of a pipeline run.

    Examples:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> f = Funnel(d, {"seed": 1})
        >>> with f.step("demo", rows_in=10) as rec:
        ...     rec.rows_out = 7
        >>> f.records[0].dropped
        3
        >>> (d / "funnel.json").exists()
        True
    """

    def __init__(self, results_dir: Path, config: dict[str, Any] | None = None) -> None:
        """Start or resume a funnel record.

        Args:
            results_dir: Directory holding ``funnel.json``.
            config: The run's configuration, stored once alongside the counts.
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path = results_dir / FUNNEL_NAME
        self.config = config or {}
        self.records: list[StepRecord] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.config = self.config or raw.get("config", {})
            self.records = [StepRecord(**r) for r in raw.get("steps", [])]

    @contextmanager
    def step(
        self,
        name: str,
        rows_in: int = 0,
        thresholds: dict[str, Any] | None = None,
    ) -> Iterator[StepRecord]:
        """Time a step and record what it did.

        Replaces any earlier record for the same step name, so re-running a step
        updates the funnel instead of appending a duplicate.

        Args:
            name: Step name.
            rows_in: Rows alive before the step runs.
            thresholds: Thresholds the step is applying.

        Yields:
            The record, for the step to fill in ``rows_out`` and ``notes``.
        """
        rec = StepRecord(step=name, rows_in=rows_in, thresholds=thresholds or {})
        started = time.monotonic()
        try:
            yield rec
        finally:
            rec.seconds = round(time.monotonic() - started, 2)
            rec.dropped = max(rec.rows_in - rec.rows_out, 0)
            self.records = [r for r in self.records if r.step != name] + [rec]
            self.records.sort(key=lambda r: r.step)
            self.save()

    def save(self) -> Path:
        """Write the funnel to disk.

        Returns:
            The path written.
        """
        payload = {
            "config": self.config,
            "steps": [asdict(r) for r in self.records],
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str))
        return self.path

    def summary(self) -> list[tuple[str, int, int, float]]:
        """Compact view for printing or plotting.

        Returns:
            One ``(step, rows_in, rows_out, seconds)`` tuple per step, in order.
        """
        return [(r.step, r.rows_in, r.rows_out, r.seconds) for r in self.records]
