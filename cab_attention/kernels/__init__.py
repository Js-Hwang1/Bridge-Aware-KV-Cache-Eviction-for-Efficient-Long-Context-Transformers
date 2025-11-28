"""
Triton kernels for efficient CAB attention operations.
"""

from .frc_triton import compute_frc, compute_frc_triton, compute_frc_pytorch

__all__ = [
    "compute_frc",
    "compute_frc_triton",
    "compute_frc_pytorch",
]
