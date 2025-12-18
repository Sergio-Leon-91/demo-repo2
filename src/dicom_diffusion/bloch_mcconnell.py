from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class BlochMcConnellParams:
    r1a: float
    r2a: float
    r1b: float
    r2b: float
    kab: float
    kba: float
    delta_a: float = 0.0
    delta_b: float = 0.0
    omega1: float = 0.0
    m0a: float = 1.0
    m0b: float = 1.0


@dataclass
class BlochMcConnellResult:
    time: np.ndarray
    magnetization: np.ndarray


def _bloch_mcconnell_rhs(time: float, state: np.ndarray, params: BlochMcConnellParams) -> np.ndarray:
    mx_a, my_a, mz_a, mx_b, my_b, mz_b = state

    dmx_a = -params.r2a * mx_a + params.delta_a * my_a
    dmy_a = -params.delta_a * mx_a - params.r2a * my_a + params.omega1 * mz_a
    dmz_a = -params.omega1 * my_a - params.r1a * (mz_a - params.m0a) - params.kab * mz_a + params.kba * mz_b

    dmx_b = -params.r2b * mx_b + params.delta_b * my_b
    dmy_b = -params.delta_b * mx_b - params.r2b * my_b + params.omega1 * mz_b
    dmz_b = -params.omega1 * my_b - params.r1b * (mz_b - params.m0b) + params.kab * mz_a - params.kba * mz_b

    return np.array([dmx_a, dmy_a, dmz_a, dmx_b, dmy_b, dmz_b])


def simulate_bloch_mcconnell(
    initial_magnetization: np.ndarray,
    times: Iterable[float],
    params: BlochMcConnellParams,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> BlochMcConnellResult:
    """Solve the Bloch-McConnell equations for two exchanging pools.

    Parameters
    ----------
    initial_magnetization:
        Array with shape ``(6,)`` or ``(n, 6)`` representing ``[Mx_a, My_a, Mz_a, Mx_b, My_b, Mz_b]``.
        When multiple voxels are provided, each is solved independently.
    times:
        Iterable of monotonic time points for which the solution is desired.
    params:
        Bloch-McConnell coefficients and equilibrium magnetizations.
    rtol, atol:
        Integration tolerances passed to ``solve_ivp``.
    """

    time_array = np.asarray(times, dtype=np.float64)
    if time_array.ndim != 1:
        raise ValueError("times must be a 1D sequence")

    states = np.atleast_2d(initial_magnetization).astype(np.float64)
    if states.shape[1] != 6:
        raise ValueError("initial_magnetization must have shape (6,) or (n, 6)")

    solutions = []
    for state in states:
        solution = solve_ivp(
            fun=lambda t, y: _bloch_mcconnell_rhs(t, y, params),
            t_span=(time_array[0], time_array[-1]),
            y0=state,
            t_eval=time_array,
            rtol=rtol,
            atol=atol,
            vectorized=False,
        )
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        solutions.append(solution.y.T)

    magnetization = np.stack(solutions) if len(solutions) > 1 else solutions[0]
    return BlochMcConnellResult(time=time_array, magnetization=magnetization)
