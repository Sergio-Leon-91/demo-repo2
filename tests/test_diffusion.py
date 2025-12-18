import numpy as np

from dicom_diffusion.diffusion import diffuse_image


def test_single_step_diffusion_center_peak():
    initial = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = diffuse_image(initial, diffusion_coefficient=1.0, dx=1.0, dt=0.1, total_time=0.1)
    assert np.isclose(result.field[1, 1], 0.6)
