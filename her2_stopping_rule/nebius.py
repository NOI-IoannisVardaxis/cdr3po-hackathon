"""Calling hosted models on the Nebius Scientific AI Platform.

Only ProteinMPNN is used, and only for one job: redesigning the trastuzumab
framework so a design's identity drops below the 70% the de novo track needs,
without touching the backbone that holds the paratope.

A note on what is actually served. The platform's ``proteinmpnn`` is
``v_48_020``, the vanilla checkpoint, and its schema exposes no weight selector:
``input_pdb``, ``input_pdb_chains``, ``num_seq_per_target``, ``omit_AAs``,
``random_seed`` and ``sampling_temp``. The soluble weights the Adaptyv
developability write-up recommends are not available here, so the vanilla score
is used as a sequence-structure compatibility signal and nothing more.

Credentials are read from the path in ``NEBIUS_CREDS`` and never logged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Where the Nebius skill ships its client.
FORGE = Path.home() / ".claude" / "skills" / "nebius-scientific-ai" / "forge.py"

#: forge.py writes its run journal to the *current working directory*, and
#: that journal is how status, resume and fetch find a job. Launching from
#: different folders makes runs invisible to each other, and with first calls
#: taking up to 26 minutes and "never submit the same job twice" as a hard
#: rule, that is how an expensive job gets duplicated. Every call therefore
#: runs from the repository root, whatever the caller's own directory.
RUN_DIR = Path(__file__).resolve().parent.parent


class NebiusUnavailableError(RuntimeError):
    """Raised when the platform cannot be reached or is not configured."""


@dataclass(frozen=True)
class MpnnResult:
    """Sequences ProteinMPNN proposed for a backbone.

    Attributes:
        sequences: One entry per sample, each mapping chain id to its sequence.
        coverage: Positions the model could not design, because the structure
            does not resolve them. Worth checking before trusting a design.
        raw: The platform's reply, kept for the run record.
    """

    sequences: list[dict[str, str]]
    coverage: object
    raw: dict[str, object]


def available() -> bool:
    """Check whether the client and a credentials path are both present.

    Returns:
        True if a call could be attempted.
    """
    creds = os.environ.get("NEBIUS_CREDS")
    if not creds:
        return False
    return FORGE.exists() and Path(creds).exists()


def _parse_mfasta(text: str) -> list[dict[str, str]]:
    """Read ProteinMPNN's multi-chain FASTA into per-sample chain maps.

    The platform returns one record per sampled sequence, chains separated by
    ``/`` in the order they were given.

    Args:
        text: The ``mfasta`` payload.

    Returns:
        One dict per sample, keyed by chain id.

    Examples:
        >>> out = _parse_mfasta(">s1\\nAAAA/CCCC\\n>s2\\nDDDD/EEEE\\n")
        >>> len(out)
        2
        >>> out[0]["A"], out[0]["B"]
        ('AAAA', 'CCCC')
        >>> _parse_mfasta("")
        []
    """
    samples: list[dict[str, str]] = []
    chunks: list[str] = []
    keep = False
    for line in text.splitlines():
        if line.startswith(">"):
            if chunks and keep:
                samples.append(_split_chains("".join(chunks)))
            chunks = []
            # The first record echoes the input backbone's own sequence, with
            # its chains run together and no separator. It is the native, not
            # a design, and including it would hand back trastuzumab.
            keep = not line[1:].lstrip().startswith("input")
        elif line.strip():
            chunks.append(line.strip())
    if chunks and keep:
        samples.append(_split_chains("".join(chunks)))
    return samples


def _split_chains(sequence: str) -> dict[str, str]:
    """Split a slash-separated multi-chain sequence into labelled chains.

    Args:
        sequence: Chains joined by ``/``.

    Returns:
        A mapping from chain letter to sequence, starting at ``A``.

    Examples:
        >>> _split_chains("AAA/CCC") == {"A": "AAA", "B": "CCC"}
        True
        >>> _split_chains("AAA")
        {'A': 'AAA'}
    """
    parts = sequence.split("/")
    return {chr(ord("A") + i): part for i, part in enumerate(parts) if part}


def design_sequences(
    pdb_path: Path,
    chains: list[str],
    n_samples: int,
    sampling_temp: float,
    seed: int,
    timeout: int = 900,
) -> MpnnResult:
    """Ask ProteinMPNN for sequences that fold to a given backbone.

    Args:
        pdb_path: Backbone to design onto.
        chains: Chain ids to redesign.
        n_samples: Sequences to sample.
        sampling_temp: Higher values diversify and drift further from native.
        seed: Random seed.
        timeout: Seconds before the call is abandoned.

    Returns:
        The proposed sequences.

    Raises:
        NebiusUnavailableError: If the platform is unreachable or errors.
    """
    if not available():
        raise NebiusUnavailableError(
            "Nebius not configured. Set NEBIUS_CREDS to the team's "
            "credentials.private.json path."
        )

    args = {
        "input_pdb": {"$upload": str(pdb_path), "media_type": "chemical/x-pdb"},
        "input_pdb_chains": chains,
        "num_seq_per_target": n_samples,
        "sampling_temp": sampling_temp,
        "random_seed": seed,
        # No new cysteines: an unpaired one is an automatic abguard rejection.
        "omit_AAs": ["X", "C"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        args_path = Path(tmp) / "mpnn.json"
        args_path.write_text(json.dumps(args))
        proc = subprocess.run(
            [
                "python3",
                str(FORGE),
                "run",
                "proteinmpnn",
                "--args-file",
                str(args_path),
                # The label doubles as an idempotency key: reusing one with
                # different inputs is refused, so it carries the parameters
                # that distinguish this call.
                "--label",
                f"denovo-T{sampling_temp}-s{seed}-n{n_samples}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=RUN_DIR,
        )

    if proc.returncode != 0:
        raise NebiusUnavailableError(
            f"proteinmpnn call failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-800:]}"
        )

    # `run` streams human-readable progress and leaves the result on the
    # platform; the artifact comes back through a separate `fetch`.
    run_id = _run_id(proc.stdout)
    if run_id is None:
        raise NebiusUnavailableError(
            f"no run id in the platform reply: {proc.stdout[-400:]}"
        )
    return fetch(run_id, timeout=timeout)


_RUN_ID_RE = re.compile(r"\b(op-[0-9a-f]{8,})\b")


def _run_id(stdout: str) -> str | None:
    """Pull the operation id out of the client's progress output.

    Args:
        stdout: What ``forge.py run`` printed.

    Returns:
        The run id, or None if the output does not name one.

    Examples:
        >>> _run_id("Run id: op-6d7f00f5a15051fc  (saved in nebius-runs/x.json)")
        'op-6d7f00f5a15051fc'
        >>> _run_id("nothing here") is None
        True
    """
    match = _RUN_ID_RE.search(stdout)
    return match.group(1) if match else None


def fetch(run_id: str, timeout: int = 900) -> MpnnResult:
    """Download a completed run's artifact and read its sequences.

    Args:
        run_id: The operation id from :func:`design_sequences`.
        timeout: Seconds before the download is abandoned.

    Returns:
        The proposed sequences.

    Raises:
        NebiusUnavailableError: If the artifact cannot be downloaded or read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["python3", str(FORGE), "fetch", run_id, "--dest", tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=RUN_DIR,
        )
        if proc.returncode != 0:
            raise NebiusUnavailableError(
                f"fetch {run_id} failed: {(proc.stderr or proc.stdout)[-600:]}"
            )
        payload: dict[str, object] = {}
        for path in sorted(Path(tmp).glob("*.json")):
            if path.name == "result.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if _find_key(payload, "mfasta") is not None:
                break

    mfasta = _find_key(payload, "mfasta")
    if mfasta is None:
        raise NebiusUnavailableError(f"no sequences in the artifact for {run_id}")
    return MpnnResult(
        sequences=_parse_mfasta(str(mfasta)),
        coverage=_find_key(payload, "backbone_coverage"),
        raw=payload,
    )


def _find_key(payload: object, key: str) -> object | None:
    """Search a nested reply for the first value under a key.

    The platform wraps results differently per model, so the payload shape is
    not fixed.

    Args:
        payload: The decoded reply.
        key: The key to find.

    Returns:
        The first matching value, or None.

    Examples:
        >>> _find_key({"a": {"b": {"mfasta": ">x\\nAA"}}}, "mfasta")
        '>x\\nAA'
        >>> _find_key({"a": [1, {"k": 2}]}, "k")
        2
        >>> _find_key({"a": 1}, "missing") is None
        True
    """
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None
