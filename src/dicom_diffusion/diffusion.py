from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DiffusionResult:
    field: np.ndarray
    time_points: list[float]


def _compute_laplacian(field: np.ndarray) -> np.ndarray:
    """Compute the Laplacian with zero-flux (Neumann) boundary conditions."""

    padded = np.pad(field, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        - 4 * center
    )
    return laplacian


def diffuse_image(
    initial_field: np.ndarray,
    diffusion_coefficient: float = 0.1,
    dx: float = 1.0,
    dt: float | None = None,
    total_time: float = 1.0,
    snapshot_times: Iterable[float] | None = None,
) -> DiffusionResult:
    """Simulate isotropic diffusion using an explicit finite-difference scheme.

    Parameters
    ----------
    initial_field:
        2D array representing the starting concentration (e.g., normalized DICOM intensities).
    diffusion_coefficient:
        Diffusion coefficient ``D``.
    dx:
        Spatial resolution. ``dt`` defaults to a stable value based on ``dx`` and ``D``.
    dt:
        Time step. If ``None``, a conservative value is derived from the stability criterion
        ``dt <= dx^2 / (4 * D)``.
    total_time:
        Total duration to simulate.
    snapshot_times:
        Optional iterable of times to record. If omitted, all time points are recorded.

    Returns
    -------
    DiffusionResult
        The evolved field and the sequence of simulation times corresponding to each update.
    """

    if initial_field.ndim != 2:
        raise ValueError("initial_field must be 2D")

    if dt is None:
        dt = 0.9 * (dx**2) / (4 * diffusion_coefficient)

    if dt <= 0:
        raise ValueError("dt must be positive")

    steps = int(math.ceil(total_time / dt))
    field = np.array(initial_field, dtype=np.float64, copy=True)
    times: list[float] = []
    record_snapshots = set(round(t / dt) for t in snapshot_times) if snapshot_times else None

    for step in range(1, steps + 1):
        laplacian = _compute_laplacian(field)
        field = field + diffusion_coefficient * laplacian * dt / (dx**2)
        current_time = step * dt
        if record_snapshots is None or step in record_snapshots:
            times.append(current_time)

    return DiffusionResult(field=field, time_points=times)
