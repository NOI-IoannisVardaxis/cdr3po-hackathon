"""Running Apheris Foundry models as Docker containers.

Every Foundry model is a Docker image already pulled on the VM. They take the
same inputs and produce the same outputs, so swapping one for another changes
only an image tag. This module is the single place that knows how to invoke
them, so the steps stay free of subprocess plumbing.

The invocation mirrors the organisers' own ``Scripts/run-her2.sh``: mount the
working directory at ``/work``, run as the calling user so outputs are not
root-owned, and give the container a shared-memory segment large enough for the
dataloader.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Apheris image registry.
REGISTRY: Final[str] = "quay.io/apheris/oslo-hackathon"

#: Image tags, verified present on the VM with ``docker images``.
IMAGES: Final[dict[str, str]] = {
    "boltz2": f"{REGISTRY}:apheris-boltz-v1.3.0-boltz2",
    "boltz1": f"{REGISTRY}:apheris-boltz-v1.3.0-boltz1",
    "openfold3": f"{REGISTRY}:apheris-openfold3-v1.3.0-of3ob",
    "openfold3_p2": f"{REGISTRY}:apheris-openfold3-v1.3.0-of3p2",
    "protenix2": f"{REGISTRY}:apheris-protenix-v1.3.0-v2",
    "mol1": f"{REGISTRY}:apheris-mol1-v1.3.0-base",
    "esmfold2": f"{REGISTRY}:apheris-esmfold2-v1.4.0-release",
    "rfd4_antibody": f"{REGISTRY}:apheris-rfd4-proteina-v0.3.0-antibody-2026-10-30",
    "rfd4_her2": f"{REGISTRY}:apheris-rfd4-proteina-v0.3.0-her2-2026-11-07",
    "rfd4_her2_novel": f"{REGISTRY}:apheris-rfd4-proteina-v0.3.0-her2-novel-2026-11-09",
}

#: Folding models that accept a template, which gives a binder a fairer check
#: than folding the target unguided.
TEMPLATE_CAPABLE: Final[frozenset[str]] = frozenset(
    {"boltz2", "openfold3", "openfold3_p2", "protenix2", "mol1"}
)


class FoundryError(RuntimeError):
    """Raised when a Foundry container exits non-zero."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of one container invocation.

    Attributes:
        model: Key from :data:`IMAGES`.
        command: The full argument vector, for the run record.
        returncode: Container exit status.
        stdout: Captured standard output.
        stderr: Captured standard error.
        seconds: Wall-clock duration.
    """

    model: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def available(model: str) -> bool:
    """Check whether a model's image is present locally.

    Args:
        model: Key from :data:`IMAGES`.

    Returns:
        True if ``docker image inspect`` finds it.

    Raises:
        KeyError: If the model name is unknown.
    """
    proc = subprocess.run(
        ["docker", "image", "inspect", IMAGES[model]],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def run(
    model: str,
    args: list[str],
    workdir: Path,
    timeout: int = 7200,
    shm_size: str = "16g",
) -> RunResult:
    """Invoke a Foundry model on the local GPU.

    Args:
        model: Key from :data:`IMAGES`.
        args: Arguments passed to the container's entrypoint. Paths must be
            relative to ``workdir``, which the container sees as ``/work``.
        workdir: Host directory mounted at ``/work``.
        timeout: Seconds before the container is killed.
        shm_size: Shared memory for the dataloader.

    Returns:
        The run result, including timing for the funnel.

    Raises:
        KeyError: If the model name is unknown.
        FoundryError: If the container exits non-zero.
    """
    import time

    uid, gid = os.getuid(), os.getgid()
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        shm_size,
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/tmp",
        "-v",
        "/etc/passwd:/etc/passwd:ro",
        "-v",
        f"{workdir}:/work",
        "-w",
        "/work",
        IMAGES[model],
        *args,
    ]
    started = time.monotonic()
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    result = RunResult(
        model=model,
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        seconds=round(time.monotonic() - started, 1),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-2000:]
        raise FoundryError(
            f"{model} exited {proc.returncode} after {result.seconds}s:\n{tail}"
        )
    return result
