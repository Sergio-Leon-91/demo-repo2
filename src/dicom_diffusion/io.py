import numpy as np
import pydicom


def load_dicom(path: str) -> tuple[np.ndarray, pydicom.dataset.Dataset]:
    """Load a DICOM file and return the pixel data as float64 plus the dataset.

    Parameters
    ----------
    path:
        Path to the DICOM file.

    Returns
    -------
    tuple[np.ndarray, pydicom.dataset.Dataset]
        The pixel array cast to ``float64`` and the raw dataset.
    """

    dataset = pydicom.dcmread(path)
    pixel_array = dataset.pixel_array.astype(np.float64)
    return pixel_array, dataset


def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    """Normalize pixel values into the range ``[0, 1]``.

    The function is robust to constant images and returns zeros for that case.
    """

    minimum = float(pixels.min())
    maximum = float(pixels.max())
    if maximum == minimum:
        return np.zeros_like(pixels, dtype=np.float64)
    return (pixels - minimum) / (maximum - minimum)
