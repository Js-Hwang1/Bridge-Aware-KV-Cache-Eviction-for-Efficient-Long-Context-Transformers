"""
Task 1.1: Coarsening Kernel - Max-L2 Pooling
Reduces Q, K from (B, H, N, D) to (B, H, M, D) where M = N / BLOCK_SIZE

Strategy: For each block of tokens, select the token with the highest L2 norm
as the representative. This preserves "needle" signals better than mean pooling.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _coarsen_max_l2_kernel(
    # Input tensors
    input_ptr,  # [B, H, N, D]
    output_ptr,  # [B, H, M, D]
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # Strides
    stride_b,
    stride_h,
    stride_n,
    stride_d,
    stride_out_b,
    stride_out_h,
    stride_out_m,
    stride_out_d,
):
    """
    For each block of BLOCK_SIZE tokens, find the token with max L2 norm
    and use it as the representative.
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch dimension
    pid_h = tl.program_id(1)  # Head dimension
    pid_m = tl.program_id(2)  # Block (coarse) dimension

    # Calculate the starting token index for this block
    token_start = pid_m * BLOCK_SIZE

    # Load all tokens in this block and compute their L2 norms
    max_norm = -1.0
    max_idx = 0

    # Loop through tokens in this block
    for local_idx in range(BLOCK_SIZE):
        token_idx = token_start + local_idx
        if token_idx < N:
            # Compute L2 norm for this token
            norm_sq = 0.0
            for d in range(D):
                ptr_offset = (pid_b * stride_b + pid_h * stride_h +
                             token_idx * stride_n + d * stride_d)
                val = tl.load(input_ptr + ptr_offset)
                norm_sq += val * val

            norm = tl.sqrt(norm_sq)
            if norm > max_norm:
                max_norm = norm
                max_idx = local_idx

    # Now load the selected token (the one with max L2 norm)
    selected_token_idx = token_start + max_idx

    # Copy the selected token to output
    for d in range(D):
        in_offset = (pid_b * stride_b + pid_h * stride_h +
                    selected_token_idx * stride_n + d * stride_d)
        out_offset = (pid_b * stride_out_b + pid_h * stride_out_h +
                     pid_m * stride_out_m + d * stride_out_d)

        val = tl.load(input_ptr + in_offset)
        tl.store(output_ptr + out_offset, val)


def coarsen_qk_max_l2(q: torch.Tensor, k: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Coarsens Q and K tensors using Max-L2 pooling.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        block_size: Number of tokens per block (default: 64)

    Returns:
        q_coarse: [B, H, M, D] where M = ceil(N / block_size)
        k_coarse: [B, H, M, D]

    Example:
        >>> q = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> k = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=64)
        >>> q_coarse.shape  # (2, 8, 2000, 128)
    """
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D), "Q and K must have same shape"

    # Calculate number of blocks
    M = (N + block_size - 1) // block_size  # Ceiling division

    # Allocate output tensors
    q_coarse = torch.empty(B, H, M, D, dtype=q.dtype, device=q.device)
    k_coarse = torch.empty(B, H, M, D, dtype=k.dtype, device=k.device)

    # Define grid (parallelize over batch, heads, and blocks)
    grid = (B, H, M)

    # Get strides
    strides_q = q.stride()
    strides_k = k.stride()
    strides_q_out = q_coarse.stride()
    strides_k_out = k_coarse.stride()

    # Launch kernel for Q
    _coarsen_max_l2_kernel[grid](
        q, q_coarse,
        B, H, N, D, M, block_size,
        strides_q[0], strides_q[1], strides_q[2], strides_q[3],
        strides_q_out[0], strides_q_out[1], strides_q_out[2], strides_q_out[3]
    )

    # Launch kernel for K
    _coarsen_max_l2_kernel[grid](
        k, k_coarse,
        B, H, N, D, M, block_size,
        strides_k[0], strides_k[1], strides_k[2], strides_k[3],
        strides_k_out[0], strides_k_out[1], strides_k_out[2], strides_k_out[3]
    )

    return q_coarse, k_coarse


def coarsen_qk_max_l2_pytorch(q: torch.Tensor, k: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch fallback implementation of Max-L2 pooling (for testing/debugging).
    This is slower but doesn't require Triton.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        block_size: Number of tokens per block

    Returns:
        q_coarse: [B, H, M, D]
        k_coarse: [B, H, M, D]
    """
    B, H, N, D = q.shape
    M = (N + block_size - 1) // block_size

    # Reshape to expose blocks: [B, H, M, block_size, D]
    # Pad if necessary
    pad_size = M * block_size - N
    if pad_size > 0:
        q_padded = torch.nn.functional.pad(q, (0, 0, 0, pad_size), value=0)
        k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_size), value=0)
    else:
        q_padded = q
        k_padded = k

    q_blocks = q_padded.view(B, H, M, block_size, D)
    k_blocks = k_padded.view(B, H, M, block_size, D)

    # Compute L2 norms for each token in each block
    q_norms = torch.norm(q_blocks, dim=-1)  # [B, H, M, block_size]
    k_norms = torch.norm(k_blocks, dim=-1)  # [B, H, M, block_size]

    # Get indices of max norm tokens
    q_max_indices = torch.argmax(q_norms, dim=-1, keepdim=True)  # [B, H, M, 1]
    k_max_indices = torch.argmax(k_norms, dim=-1, keepdim=True)  # [B, H, M, 1]

    # Expand indices to match D dimension
    q_max_indices = q_max_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, H, M, 1, D]
    k_max_indices = k_max_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B, H, M, 1, D]

    # Gather the selected tokens
    q_coarse = torch.gather(q_blocks, 3, q_max_indices).squeeze(3)  # [B, H, M, D]
    k_coarse = torch.gather(k_blocks, 3, k_max_indices).squeeze(3)  # [B, H, M, D]

    return q_coarse, k_coarse
