"""
Coarse Curvature Predictor: The "Brain" of CAB-Attention

This module implements the block-level geometric selector that decides
which blocks to keep and which to prune based on Forman-Ricci Curvature.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .kernels.coarsening import coarsen_qk_max_l2_pytorch, coarsen_qk_max_l2
from .kernels.frc_kernel import compute_block_frc, generate_block_mask_from_frc


class CoarseCurvaturePredictor(nn.Module):
    """
    Predicts which blocks to attend to based on coarse-grained Forman-Ricci Curvature.

    This is the core innovation of CAB-Attention:
    1. Coarsen Q, K from N tokens to M blocks (M = N / block_size)
    2. Compute FRC on the M x M coarse graph
    3. Select blocks with most negative curvature (bridges)
    4. Return a block mask for sparse attention

    Args:
        block_size: Number of tokens per block (default: 64)
        sparsity: Fraction of blocks to prune (default: 0.95 = keep 5%)
        lambda_redundancy: Weight for triangle penalty (default: 0.5)
        use_triton: Use Triton kernels if available (default: True)
        keep_diagonal: Always keep diagonal blocks (local attention)
        causal: Enforce causal masking

    Example:
        >>> predictor = CoarseCurvaturePredictor(block_size=64, sparsity=0.95)
        >>> q = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> k = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> mask = predictor(q, k)
        >>> mask.shape  # (2, 8, 2000, 2000) - block mask
    """

    def __init__(
        self,
        block_size: int = 64,
        sparsity: float = 0.95,
        lambda_redundancy: float = 0.5,
        use_triton: bool = True,
        keep_diagonal: bool = True,
        causal: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.sparsity = sparsity
        self.lambda_redundancy = lambda_redundancy
        self.use_triton = use_triton
        self.keep_diagonal = keep_diagonal
        self.causal = causal
        self.temperature = temperature

        # Check if Triton is available
        self._triton_available = False
        try:
            import triton
            self._triton_available = True
        except ImportError:
            if use_triton:
                print("Warning: Triton not available, falling back to PyTorch implementation")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Generate block mask from Q, K using coarse FRC.

        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            return_diagnostics: If True, return additional debug info

        Returns:
            block_mask: Binary mask [B, H, M, M] where M = ceil(N / block_size)
            diagnostics: (optional) Dict with FRC scores, affinity, etc.
        """
        B, H, N, D = q.shape

        # Stage 1: Coarsening (Task 1.1)
        if self._triton_available and self.use_triton:
            q_coarse, k_coarse = coarsen_qk_max_l2(q, k, self.block_size)
        else:
            q_coarse, k_coarse = coarsen_qk_max_l2_pytorch(q, k, self.block_size)

        # Stage 2: FRC Computation (Task 1.2)
        frc_scores, affinity, triangles = compute_block_frc(
            q_coarse,
            k_coarse,
            temperature=self.temperature,
            use_relu=True,
            lambda_redundancy=self.lambda_redundancy,
        )

        # Stage 3: Mask Generation
        block_mask = generate_block_mask_from_frc(
            frc_scores,
            sparsity=self.sparsity,
            keep_diagonal=self.keep_diagonal,
            causal=self.causal,
        )

        if return_diagnostics:
            diagnostics = {
                'frc_scores': frc_scores,
                'affinity': affinity,
                'triangles': triangles,
                'q_coarse': q_coarse,
                'k_coarse': k_coarse,
                'block_mask': block_mask,
                'effective_sparsity': 1.0 - block_mask.float().mean().item(),
                'num_blocks': q_coarse.shape[2],
            }
            return block_mask, diagnostics

        return block_mask

    def estimate_flops(self, N: int, D: int) -> dict:
        """
        Estimates FLOPs for the predictor.

        This is useful for profiling and understanding overhead.
        """
        M = (N + self.block_size - 1) // self.block_size

        # Coarsening: O(N * D) for L2 norms
        coarsening_flops = N * D

        # Coarse matmul: O(M^2 * D)
        matmul_flops = M * M * D

        # Triangle computation: O(M^3)
        triangle_flops = M * M * M

        # Total
        total_flops = coarsening_flops + matmul_flops + triangle_flops

        return {
            'coarsening': coarsening_flops,
            'matmul': matmul_flops,
            'triangles': triangle_flops,
            'total': total_flops,
            'M': M,
            'compression_ratio': N / M,
        }


class AdaptiveCurvaturePredictor(CoarseCurvaturePredictor):
    """
    Adaptive version that adjusts sparsity based on curvature statistics.

    Instead of fixed sparsity, this dynamically selects a threshold based on
    the distribution of FRC scores. Useful for sequences with varying structure.
    """

    def __init__(
        self,
        block_size: int = 64,
        target_sparsity: float = 0.95,
        percentile_threshold: float = 5.0,  # Keep bottom 5% of FRC scores
        **kwargs
    ):
        super().__init__(block_size=target_sparsity, sparsity=target_sparsity, **kwargs)
        self.percentile_threshold = percentile_threshold

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Generate block mask with adaptive threshold.
        """
        B, H, N, D = q.shape

        # Coarsening
        if self._triton_available and self.use_triton:
            q_coarse, k_coarse = coarsen_qk_max_l2(q, k, self.block_size)
        else:
            q_coarse, k_coarse = coarsen_qk_max_l2_pytorch(q, k, self.block_size)

        # FRC Computation
        frc_scores, affinity, triangles = compute_block_frc(
            q_coarse,
            k_coarse,
            temperature=self.temperature,
            use_relu=True,
            lambda_redundancy=self.lambda_redundancy,
        )

        # Adaptive Threshold: Keep blocks below Nth percentile
        threshold = torch.quantile(frc_scores.flatten(), self.percentile_threshold / 100.0)
        block_mask = frc_scores <= threshold

        # Enforce constraints
        if self.keep_diagonal:
            M = q_coarse.shape[2]
            diag_mask = torch.eye(M, device=q.device, dtype=torch.bool)
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            block_mask = block_mask | diag_mask

        if self.causal:
            M = q_coarse.shape[2]
            causal_mask = torch.tril(torch.ones(M, M, device=q.device, dtype=torch.bool))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            block_mask = block_mask & causal_mask

        if return_diagnostics:
            diagnostics = {
                'frc_scores': frc_scores,
                'affinity': affinity,
                'triangles': triangles,
                'threshold': threshold.item(),
                'effective_sparsity': 1.0 - block_mask.float().mean().item(),
            }
            return block_mask, diagnostics

        return block_mask
