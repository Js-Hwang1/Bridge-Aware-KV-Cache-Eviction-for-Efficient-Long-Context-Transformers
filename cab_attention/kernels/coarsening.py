"""
Task 1.1: Coarsening Kernel - Max-L2 Pooling
Reduces Q, K from (B, H, N, D) to (B, H, M, D) where M = N / BLOCK_SIZE

Strategy: For each block of tokens, select the token with the highest L2 norm
as the representative. This preserves "needle" signals better than mean pooling.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}),
        triton.Config({'BLOCK_D': 64}),
        triton.Config({'BLOCK_D': 128}),
        triton.Config({'BLOCK_D': 256}),
    ],
    key=['N', 'D', 'BLOCK_SIZE'],
)
@triton.jit
def _coarsen_max_l2_kernel(
    # Input tensors
    input_ptr,  # [B, H, N, D]
    output_ptr,  # [B, H, M, D]
    # Dimensions
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    Production-quality Max-L2 coarsening kernel with full vectorization.

    Each program handles one (B, H, M) block:
    1. Loads BLOCK_SIZE tokens' embeddings in parallel (BLOCK_SIZE × D matrix)
    2. Computes L2 norms for all tokens in parallel
    3. Uses argmax to select token with max norm
    4. Writes selected token to output

    Key optimizations:
    - Vectorized loads for D dimension (memory coalescing)
    - Parallel norm computation across tokens
    - Single-pass algorithm (no redundant loads)
    - Autotuning to find optimal BLOCK_D for hardware
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch dimension
    pid_h = tl.program_id(1)  # Head dimension
    pid_m = tl.program_id(2)  # Block (coarse) dimension

    # Calculate the starting token index for this block
    token_start = pid_m * BLOCK_SIZE

    # Create token offset range [0, 1, 2, ..., BLOCK_SIZE-1]
    # This allows us to process BLOCK_SIZE tokens in parallel
    token_offsets = tl.arange(0, BLOCK_SIZE)
    token_indices = token_start + token_offsets
    token_mask = token_indices < N  # Handle boundary case (last block may be partial)

    # Base pointer for this (batch, head) position
    # All subsequent loads are relative to this base
    base_ptr = input_ptr + pid_b * stride_b + pid_h * stride_h

    # Initialize accumulator for squared L2 norms [BLOCK_SIZE]
    # Each element will hold ||token_i||^2
    norms_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Process D dimension in chunks of BLOCK_D
    # This loop is sequential but each iteration loads BLOCK_SIZE × BLOCK_D values in parallel
    for d_start in range(0, D, BLOCK_D):
        # D-dimension offsets for this chunk
        d_offsets = tl.arange(0, BLOCK_D)
        d_indices = d_start + d_offsets
        d_mask = d_indices < D  # Handle case where D is not multiple of BLOCK_D

        # Create 2D mask: [BLOCK_SIZE, BLOCK_D]
        # This broadcasts token_mask and d_mask to create a 2D mask
        # True only where both token is valid AND dimension is valid
        mask_2d = token_mask[:, None] & d_mask[None, :]

        # Compute offsets for 2D load: [BLOCK_SIZE, BLOCK_D]
        # This creates a matrix of memory offsets for the entire (BLOCK_SIZE × BLOCK_D) tile
        # Broadcasting rules:
        #   token_indices[:, None] has shape [BLOCK_SIZE, 1]
        #   d_indices[None, :] has shape [1, BLOCK_D]
        #   Result has shape [BLOCK_SIZE, BLOCK_D]
        offsets_2d = (token_indices[:, None] * stride_n +
                      d_indices[None, :] * stride_d)

        # 🚀 KEY OPTIMIZATION: Vectorized 2D load
        # This single tl.load fetches an entire (BLOCK_SIZE × BLOCK_D) tile from memory
        # GPU hardware coalesces these accesses for maximum memory bandwidth
        # Old (slow): for-loop loading D scalars sequentially
        # New (fast): single load of BLOCK_SIZE × BLOCK_D matrix
        vals = tl.load(base_ptr + offsets_2d, mask=mask_2d, other=0.0)

        # Accumulate squared norms: sum over D dimension (axis=1)
        # vals has shape [BLOCK_SIZE, BLOCK_D]
        # vals * vals computes element-wise squares
        # tl.sum(..., axis=1) reduces to [BLOCK_SIZE]
        norms_sq += tl.sum(vals * vals, axis=1)

    # Find token with maximum L2 norm
    norms = tl.sqrt(norms_sq)

    # Mask out invalid tokens (set their norms to -inf)
    norms = tl.where(token_mask, norms, float('-inf'))

    # Get index of max norm token (scalar within this block)
    max_idx = tl.argmax(norms, axis=0)
    selected_token_idx = token_start + max_idx

    # Write selected token to output (vectorized write)
    output_base = (output_ptr + pid_b * stride_out_b +
                   pid_h * stride_out_h + pid_m * stride_out_m)
    input_base = base_ptr + selected_token_idx * stride_n

    # Write D dimension in chunks of BLOCK_D
    for d_start in range(0, D, BLOCK_D):
        d_offsets = tl.arange(0, BLOCK_D)
        d_indices = d_start + d_offsets
        d_mask = d_indices < D

        # Vectorized load from selected token
        vals = tl.load(input_base + d_indices * stride_d, mask=d_mask, other=0.0)

        # Vectorized store to output
        tl.store(output_base + d_indices * stride_out_d, vals, mask=d_mask)


def coarsen_qk_max_l2(q: torch.Tensor, k: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Coarsens Q and K tensors using Max-L2 pooling with optimized Triton kernel.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        block_size: Number of tokens per block (default: 64, must be power of 2)

    Returns:
        q_coarse: [B, H, M, D] where M = ceil(N / block_size)
        k_coarse: [B, H, M, D]

    Performance notes:
        - Fully vectorized kernel with memory coalescing
        - block_size should be power of 2 for optimal performance (32, 64, 128)
        - D dimension is processed in chunks for efficiency

    Example:
        >>> q = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> k = torch.randn(2, 8, 128000, 128, device='cuda')
        >>> q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=64)
        >>> q_coarse.shape  # (2, 8, 2000, 128)
    """
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D), "Q and K must have same shape"
    assert q.is_cuda and k.is_cuda, "Tensors must be on CUDA device"

    # Calculate number of blocks
    M = (N + block_size - 1) // block_size  # Ceiling division

    # Allocate output tensors
    q_coarse = torch.empty(B, H, M, D, dtype=q.dtype, device=q.device)
    k_coarse = torch.empty(B, H, M, D, dtype=k.dtype, device=k.device)

    # Ensure block_size is power of 2 for optimal performance
    assert block_size & (block_size - 1) == 0, "block_size must be power of 2"

    # Define grid (parallelize over batch, heads, and blocks)
    grid = (B, H, M)

    # Get strides
    strides_q = q.stride()
    strides_k = k.stride()
    strides_q_out = q_coarse.stride()
    strides_k_out = k_coarse.stride()

    # Launch kernel for Q (BLOCK_D is autotuned, don't pass it)
    _coarsen_max_l2_kernel[grid](
        q, q_coarse,
        N=N, D=D, BLOCK_SIZE=block_size,
        stride_b=strides_q[0], stride_h=strides_q[1], stride_n=strides_q[2], stride_d=strides_q[3],
        stride_out_b=strides_q_out[0], stride_out_h=strides_q_out[1],
        stride_out_m=strides_q_out[2], stride_out_d=strides_q_out[3]
    )

    # Launch kernel for K (BLOCK_D is autotuned, don't pass it)
    _coarsen_max_l2_kernel[grid](
        k, k_coarse,
        N=N, D=D, BLOCK_SIZE=block_size,
        stride_b=strides_k[0], stride_h=strides_k[1], stride_n=strides_k[2], stride_d=strides_k[3],
        stride_out_b=strides_k_out[0], stride_out_h=strides_k_out[1],
        stride_out_m=strides_k_out[2], stride_out_d=strides_k_out[3]
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
