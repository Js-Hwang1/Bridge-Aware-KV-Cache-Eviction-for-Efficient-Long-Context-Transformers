"""
CAB-Attention: Curvature-Aware Block-Sparse Attention
A hardware-efficient sparse attention mechanism using Forman-Ricci Curvature
"""

from .coarse_predictor import CoarseCurvaturePredictor
from .cab_attention import CABAttention

__version__ = "0.1.0"
__all__ = ["CoarseCurvaturePredictor", "CABAttention"]
