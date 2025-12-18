import argparse
from pathlib import Path

import numpy as np

from .bloch_mcconnell import BlochMcConnellParams, simulate_bloch_mcconnell
from .diffusion import diffuse_image
from .io import load_dicom, normalize_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diffusion and Bloch-McConnell simulations on a DICOM image.")
    parser.add_argument("dicom", type=Path, help="Path to the DICOM file.")
    parser.add_argument("--time", type=float, default=1.0, help="Total diffusion time (arbitrary units).")
    parser.add_argument("--diffusion", type=float, default=0.1, help="Diffusion coefficient.")
    parser.add_argument("--dx", type=float, default=1.0, help="Spatial step size for diffusion.")
    parser.add_argument("--kab", type=float, default=1.0, help="Exchange rate from pool A to B.")
    parser.add_argument("--kba", type=float, default=1.0, help="Exchange rate from pool B to A.")
    parser.add_argument("--r1", type=float, default=1.0, help="Longitudinal relaxation rate for both pools.")
    parser.add_argument("--r2", type=float, default=5.0, help="Transverse relaxation rate for both pools.")
    parser.add_argument("--omega1", type=float, default=0.0, help="B1 amplitude (rad/s).")
    parser.add_argument("--delta", type=float, default=0.0, help="Off-resonance frequency (rad/s) for both pools.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pixels, _ = load_dicom(args.dicom)
    normalized = normalize_pixels(pixels)

    diffusion_result = diffuse_image(
        normalized,
        diffusion_coefficient=args.diffusion,
        dx=args.dx,
        total_time=args.time,
    )

    center_index = tuple(i // 2 for i in normalized.shape)
    mz_a0 = diffusion_result.field[center_index]
    mz_b0 = max(1e-3, 1.0 - mz_a0)
    initial_state = np.array([0.0, 0.0, mz_a0, 0.0, 0.0, mz_b0])

    params = BlochMcConnellParams(
        r1a=args.r1,
        r2a=args.r2,
        r1b=args.r1,
        r2b=args.r2,
        kab=args.kab,
        kba=args.kba,
        delta_a=args.delta,
        delta_b=args.delta,
        omega1=args.omega1,
        m0a=1.0,
        m0b=1.0,
    )

    time_points = np.linspace(0, args.time, 50)
    bloch_result = simulate_bloch_mcconnell(initial_state, time_points, params)

    print("Diffusion complete.")
    print(f"Final diffusion pixel (center) intensity: {mz_a0:.4f}")
    print("Bloch-McConnell simulation (center voxel):")
    for t, magnetization in zip(bloch_result.time, bloch_result.magnetization):
        mz_a = magnetization[2]
        mz_b = magnetization[5]
        print(f"t={t:6.3f}  Mz_a={mz_a:7.4f}  Mz_b={mz_b:7.4f}")


if __name__ == "__main__":
    main()
