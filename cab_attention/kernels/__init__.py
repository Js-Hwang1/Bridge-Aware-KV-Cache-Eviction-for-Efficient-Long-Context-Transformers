"""
Triton/CUDA kernels for CAB-Attention
"""

from .coarsening import coarsen_qk_max_l2
from .frc_kernel import compute_block_frc

__all__ = ["coarsen_qk_max_l2", "compute_block_frc"]
