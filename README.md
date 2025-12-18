# Demo 2

Utilities to load a DICOM image, run a finite-difference diffusion solver, and simulate Bloch-McConnell magnetization exchange. The project is packaged with a small CLI that demonstrates both solvers using pixel intensities as initial conditions.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the example

```bash
python -m dicom_diffusion path/to/image.dcm --time 2.0 --diffusion 0.2 --kab 0.5 --kba 0.3
```

The script normalizes the DICOM pixels, evolves them under isotropic diffusion, and then feeds the center voxel into a Bloch-McConnell simulation. Results are printed to stdout.

## Library usage

```python
import numpy as np
from dicom_diffusion import (
    load_dicom,
    normalize_pixels,
    diffuse_image,
    simulate_bloch_mcconnell,
    BlochMcConnellParams,
)

pixels, meta = load_dicom("image.dcm")
normalized = normalize_pixels(pixels)

# Diffusion
result = diffuse_image(normalized, diffusion_coefficient=0.15, total_time=1.0)
print(result.field)

# Bloch-McConnell on a voxel
params = BlochMcConnellParams(
    r1a=1.0,
    r2a=5.0,
    r1b=1.0,
    r2b=5.0,
    kab=0.5,
    kba=0.3,
)
time_points = np.linspace(0, 1, 25)
state0 = np.array([0, 0, result.field[0, 0], 0, 0, 1 - result.field[0, 0]])
solution = simulate_bloch_mcconnell(state0, time_points, params)
print(solution.magnetization)
```
