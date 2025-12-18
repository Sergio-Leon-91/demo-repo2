"""DICOM-driven diffusion and Bloch-McConnell simulations."""

from .io import load_dicom, normalize_pixels
from .diffusion import diffuse_image
from .bloch_mcconnell import simulate_bloch_mcconnell, BlochMcConnellParams

__all__ = [
    "load_dicom",
    "normalize_pixels",
    "diffuse_image",
    "simulate_bloch_mcconnell",
    "BlochMcConnellParams",
]
