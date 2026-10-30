"""
WaveFactor: Wavelet-based Bayesian Sparse Factor Model For Spatial Transcriptomics
"""

from pathlib import Path


def _load_version() -> str:
    """Loads WaveFactor package version from repository root or package metadata."""
    # 1. Dev mode: read language-neutral VERSION asset directly from repository root
    repo_version_file = Path(__file__).resolve().parent.parent / "VERSION"
    if repo_version_file.is_file():
        return repo_version_file.read_text(encoding="utf-8").strip()

    # 2. Installed mode: read from package metadata when distributed as a wheel
    try:
        from importlib.metadata import version
        return version("wavefactor")
    except Exception:
        raise ImportError(
            "WaveFactor version could not be determined. "
            "Please ensure the package is installed or the VERSION file is present."
        ) from None


__version__ = _load_version()

from .priors import Priors
from .data import WaveFactorData, prepare_spatial_data
from .results import WaveFactorResult
from .model import WaveFactor

__all__ = [
    "__version__",
    "WaveFactor",
    "WaveFactorResult",
    "WaveFactorData",
    "Priors",
    "prepare_spatial_data",
]
