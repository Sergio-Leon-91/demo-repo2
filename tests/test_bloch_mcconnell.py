import numpy as np

from dicom_diffusion.bloch_mcconnell import BlochMcConnellParams, simulate_bloch_mcconnell


def test_bloch_mcconnell_relaxation_without_exchange():
    params = BlochMcConnellParams(
        r1a=1.0,
        r2a=0.0,
        r1b=1.0,
        r2b=0.0,
        kab=0.0,
        kba=0.0,
        m0a=1.0,
        m0b=1.0,
    )
    times = np.linspace(0.0, 1.0, 5)
    initial_state = np.zeros(6)

    result = simulate_bloch_mcconnell(initial_state, times, params)

    expected_mz = 1 - np.exp(-times)
    simulated_mz_a = result.magnetization[:, 2]
    simulated_mz_b = result.magnetization[:, 5]

    np.testing.assert_allclose(simulated_mz_a, expected_mz, atol=1e-3)
    np.testing.assert_allclose(simulated_mz_b, expected_mz, atol=1e-3)
