"""
Example usage of the optimized coarsening kernel in CAB-Attention pipeline.

This demonstrates how to integrate the production kernel into your attention mechanism.
"""

import torch
from coarsening import coarsen_qk_max_l2


def cab_attention_with_coarsening_example():
    """
    Example showing full CAB attention pipeline with optimized coarsening.
    """
    print("="*60)
    print("CAB-Attention with Production Coarsening Kernel")
    print("="*60)

    # Simulated LLM inference setup
    batch_size = 2
    num_heads = 32
    seq_len = 32768  # Long context
    head_dim = 128
    block_size = 64  # Coarsening block size

    print(f"\nInput shape: [B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}]")
    print(f"Block size: {block_size}")

    # Create dummy Q, K, V tensors (in practice, these come from your model)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    print(f"Memory usage (Q+K+V): {(q.numel() + k.numel() + v.numel()) * 2 / 1e9:.2f} GB (FP16)")

    # ========================================
    # Step 1: Coarsen Q and K
    # ========================================
    print("\n" + "-"*60)
    print("Step 1: Coarsening Q and K with Max-L2 pooling")
    print("-"*60)

    # Convert to FP32 for coarsening (FP16 supported but FP32 more stable)
    q_fp32 = q.float()
    k_fp32 = k.float()

    # Apply coarsening kernel
    q_coarse, k_coarse = coarsen_qk_max_l2(q_fp32, k_fp32, block_size=block_size)

    M = seq_len // block_size
    print(f"Output shape: [B={batch_size}, H={num_heads}, M={M}, D={head_dim}]")
    print(f"Compression ratio: {seq_len}/{M} = {seq_len//M}x")
    print(f"Memory reduction: {(q.numel() + k.numel()) / (q_coarse.numel() + k_coarse.numel()):.1f}x")

    # ========================================
    # Step 2: Compute Coarse Attention Scores
    # ========================================
    print("\n" + "-"*60)
    print("Step 2: Computing coarse attention scores")
    print("-"*60)

    # Compute block-level attention: S_coarse = Q_coarse @ K_coarse^T
    # Shape: [B, H, M, M]
    scores_coarse = torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"Coarse attention shape: {scores_coarse.shape}")

    # ========================================
    # Step 3: Compute FRC (Forman-Ricci Curvature)
    # ========================================
    print("\n" + "-"*60)
    print("Step 3: Computing FRC for block selection")
    print("-"*60)

    lambda_redundancy = 0.5

    # Direct attention strength (block-level)
    direct_attention = scores_coarse.abs()  # or softmax(scores_coarse)

    # Redundancy: 2-hop paths (A @ A)
    redundancy = torch.matmul(direct_attention, direct_attention)

    # FRC = direct - λ * redundancy
    frc_scores = direct_attention - lambda_redundancy * redundancy
    print(f"FRC scores shape: {frc_scores.shape}")
    print(f"FRC range: [{frc_scores.min():.4f}, {frc_scores.max():.4f}]")

    # ========================================
    # Step 4: Select Important Blocks (CAB V3)
    # ========================================
    print("\n" + "-"*60)
    print("Step 4: Selecting blocks with HIGH FRC (CAB V3)")
    print("-"*60)

    sparsity = 0.90  # Keep 10% of blocks
    k_keep = max(1, int(M * M * (1 - sparsity)))

    # CAB V3: Select blocks with HIGHEST FRC (not lowest!)
    # This was the breakthrough from the debugging session
    frc_flat = frc_scores.view(batch_size, num_heads, -1)
    threshold = torch.topk(frc_flat, k_keep, dim=-1, largest=True).values[:, :, -1:]
    block_mask = (frc_scores >= threshold.view(batch_size, num_heads, 1, 1))

    blocks_kept = block_mask.sum().item()
    total_blocks = batch_size * num_heads * M * M
    actual_sparsity = 1 - (blocks_kept / total_blocks)

    print(f"Target sparsity: {sparsity*100:.1f}%")
    print(f"Actual sparsity: {actual_sparsity*100:.1f}%")
    print(f"Blocks kept: {blocks_kept}/{total_blocks}")

    # ========================================
    # Step 5: Apply Block-Sparse Attention
    # ========================================
    print("\n" + "-"*60)
    print("Step 5: Computing block-sparse attention")
    print("-"*60)

    # Expand block mask to token-level mask
    token_mask = block_mask.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    print(f"Token-level mask shape: {token_mask.shape}")

    # Compute full attention scores
    scores_full = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) / (head_dim ** 0.5)

    # Apply sparsity mask
    scores_sparse = scores_full.masked_fill(~token_mask, float('-inf'))

    # Softmax and compute attention output
    attn_weights = torch.softmax(scores_sparse, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle -inf

    # Output: [B, H, N, D]
    output = torch.matmul(attn_weights, v.float())

    print(f"Output shape: {output.shape}")
    print(f"Non-zero attention: {(attn_weights > 0).sum().item() / attn_weights.numel() * 100:.1f}%")

    # ========================================
    # Performance Summary
    # ========================================
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    # Compute FLOPs reduction
    dense_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    sparse_flops = 2 * batch_size * num_heads * blocks_kept * block_size * block_size * head_dim

    print(f"Dense attention FLOPs: {dense_flops/1e12:.2f} TFLOPs")
    print(f"Sparse attention FLOPs: {sparse_flops/1e12:.2f} TFLOPs")
    print(f"FLOPs reduction: {dense_flops/sparse_flops:.1f}x")

    # Memory savings
    dense_mem = batch_size * num_heads * seq_len * seq_len * 4  # FP32 attention matrix
    sparse_mem = batch_size * num_heads * blocks_kept * block_size * block_size * 4

    print(f"\nDense attention memory: {dense_mem/1e9:.2f} GB")
    print(f"Sparse attention memory: {sparse_mem/1e9:.2f} GB")
    print(f"Memory reduction: {dense_mem/sparse_mem:.1f}x")

    print("\n✅ CAB-Attention pipeline complete!")

    return output


def minimal_example():
    """
    Minimal example showing just the coarsening kernel usage.
    """
    print("\n" + "="*60)
    print("MINIMAL EXAMPLE: Coarsening Kernel Only")
    print("="*60)

    # Create inputs
    q = torch.randn(1, 8, 2048, 128, device='cuda')
    k = torch.randn(1, 8, 2048, 128, device='cuda')

    print(f"Input:  Q shape = {q.shape}")

    # Apply coarsening
    q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=64)

    print(f"Output: Q_coarse shape = {q_coarse.shape}")
    print(f"Compression: {q.shape[2]//q_coarse.shape[2]}x")
    print("✅ Done!")


if __name__ == "__main__":
    print("CAB-Attention Coarsening Kernel - Usage Examples\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available! These examples require a GPU.")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Run minimal example
    minimal_example()

    # Run full pipeline example
    response = input("\nRun full CAB-Attention pipeline example? (y/n): ")
    if response.lower() == 'y':
        cab_attention_with_coarsening_example()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run test_coarsening.py to benchmark performance")
    print("2. Integrate into your full CAB-Attention implementation")
    print("3. Profile end-to-end inference latency")
