Geometric Inductive Biases in Hardware-Aware Sparse Attention: A Feasibility Study of Forman-Ricci Curvature for Block-Wise Pruning


1. Introduction: The Convergence of Geometry and Hardware Efficiency

The pursuit of infinite context length in Large Language Models (LLMs) faces a fundamental physical barrier: the quadratic computational complexity of the self-attention mechanism. As sequence lengths extend from 32k to 128k, 1M, and beyond, the $O(N^2)$ cost of computing pairwise affinities between tokens transitions from a manageable overhead to a prohibitive bottleneck. While recent architectural innovations like DeepSeek’s Native Sparse Attention (NSA) and Ring Attention have demonstrated that hardware-aligned sparsity—specifically block-structured sparsity—is the only viable path forward for acceleration on modern GPUs, a critical semantic gap remains. The selection mechanisms determining which blocks to retain and which to prune are predominantly heuristic, relying on accumulated attention magnitude (heavy hitters) or static locality (sliding windows). This report argues that these magnitude-based heuristics fundamentally misalign with the information-theoretic needs of reasoning tasks, particularly in "Needle-in-a-Haystack" scenarios where critical information is often low-magnitude but topologically unique.
We posit that the next leap in sparse attention efficiency will not come from purely algorithmic sorting or random projections, but from integrating Geometric Inductive Biases directly into the hardware selection kernel. Specifically, we investigate the feasibility of Forman-Ricci Curvature (FRC) as a lightweight, topological selection criterion for block-sparse attention. Unlike its continuous counterparts or the computationally expensive Ollivier-Ricci Curvature (ORC), FRC offers a unique combination of combinatorial simplicity and geometric expressivity. By measuring the "cliquishness" versus "dispersal" of information flow between token blocks, FRC can theoretically identify structural bottlenecks—the sparse "bridges" that carry high information content—without the crushing latency of optimal transport solvers.
This report provides an exhaustive analysis of this hypothesis, structured to serve as a foundational whitepaper for an ICML submission. We dissect the theoretical underpinnings of curvature in attention graphs, the specific engineering challenges of implementing FRC within the constraints of NVIDIA’s Triton compiler and Tensor Core memory hierarchies, and the experimental rigor required to demonstrate superiority over state-of-the-art baselines like DeepSeek NSA and H2O. The synthesis of these elements suggests a research direction with a High Likelihood of Success, provided the implementation rigorously addresses the tension between irregular graph topology and block-aligned memory access.

1.1 The Hardware Alignment Gap in Sparse Attention

The history of sparse attention is a graveyard of theoretically elegant algorithms that failed to deliver wall-clock speedups. Early approaches like Sparse Transformer, Reformer, and Longformer proposed sophisticated sparsity patterns—logarithmic connections, dilated sliding windows, and random projections—that theoretically reduced Floating Point Operations (FLOPs) from quadratic to linear or log-linear complexity.1 However, these methods often ignored the harsh realities of modern hardware accelerators. Graphics Processing Units (GPUs), particularly the NVIDIA Ampere and Hopper architectures, derive their immense throughput from Tensor Cores designed to perform dense matrix multiplications on aligned tiles (typically $16 \times 16$ or $32 \times 32$ elements).
Unstructured sparsity, where individual weights are zeroed out randomly, leads to memory fragmentation. The GPU must perform scattered memory reads (gather operations) which saturate the memory bandwidth long before the compute units are utilized. As highlighted in recent benchmarks from the "Long Range Arena" 3, many "efficient" transformers exhibited inference latencies higher than standard dense attention due to this hardware misalignment. This reality has forced a convergence toward Block-Structured Sparsity, where sparsity is enforced at the level of contiguous blocks of tokens rather than individual elements.5 By loading and processing entire blocks, kernels can saturate the high-bandwidth memory (HBM) and keep Tensor Cores fed with dense micro-operations.
DeepSeek’s Native Sparse Attention (NSA), introduced in early 2025, represents the crystallization of this hardware-aware philosophy.1 NSA abandons element-wise masks in favor of a hierarchical block strategy. It processes sequences through three parallel paths: a compressed path for global context (coarse-grained), a sliding window for local context, and a "Selected Attention" path for retrieving specific long-range dependencies.6 The success of NSA 2 proves that block-wise operations are the correct abstraction level. However, NSA’s selection mechanism for its "Selected Attention" branch typically relies on magnitude-based top-$k$ selection (Heavy Hitters). This reliance on magnitude is the specific weakness this report targets.

1.2 The Failure of Magnitude Heuristics

Current dynamic sparse attention methods, such as H2O (Heavy Hitter Oracle) 7 and StreamingLLM 8, operate on the premise that "important" tokens are those that have accumulated high attention scores in the past. H2O evicts KV pairs with the lowest accumulated attention mass, while StreamingLLM retains initial tokens (attention sinks) to preserve stability.
While effective for maintaining perplexity in general language modeling, magnitude-based pruning fails catastrophically in reasoning and retrieval tasks involving rare events. In a "Needle-in-a-Haystack" test—where a specific, random passkey is inserted into a long context and queried later—the passkey tokens often have low accumulated attention magnitude because they are semantically disconnected from the surrounding "haystack" text. They only become relevant when the specific query triggers a retrieval. A magnitude-based oracle will likely prune these tokens long before the query arrives, leading to a failure in retrieval.9
This failure mode arises because magnitude measures popularity, not necessity. In graph theoretic terms, high-magnitude nodes are often part of redundant clusters (cliques). Removing a node from a clique rarely disrupts the global flow of information because many alternative paths exist. In contrast, the "needle" often acts as a bridge or a structural bottleneck—a unique pathway to a specific piece of information. Identifying these bridges requires a topological metric, not a magnitude metric.

1.3 The Geometric Alternative: Curvature as Importance

This report explores the application of Ricci curvature to the "Attention Graph"—the dynamic directed graph formed by tokens (nodes) and attention scores (weighted edges). In Riemannian geometry and network science, Ricci curvature quantifies the divergence or convergence of geodesic paths.
Positive Curvature: Indicates a spherical geometry or, in graphs, a dense community (clique). Geodesics converge. Information is redundant.
Negative Curvature: Indicates a hyperbolic geometry or, in graphs, a tree-like structure, bridge, or bottleneck. Geodesics diverge. Information flow is constrained to specific, unique paths.10
Recent theoretical work has characterized the token embedding space of LLMs as a "stratified manifold" with predominantly negative curvature, suggesting that the underlying geometry of language is hyperbolic.12 Furthermore, research into "over-squashing" in Graph Neural Networks (GNNs) has identified edges with negative Ricci curvature as the primary bottlenecks for long-range information propagation.14
By translating this insight to the attention mechanism, we hypothesize that blocks connected by edges with highly negative Forman-Ricci curvature are the structural bottlenecks of the context window. These are the blocks that must be preserved to ensure retrievability, even if their attention magnitude is low. Conversely, blocks with high positive curvature represent redundant information clusters that can be aggressively compressed or pruned without information loss. This defines a new paradigm: Curvature-Aware Block-Sparse Attention (CAB-Attention).

2. Theoretical Foundations: From Manifolds to Triton Blocks

To justify the integration of curvature into a hardware-accelerated kernel, we must first establish the mathematical rigor of the metric and demonstrate its computability. We explicitly choose Forman-Ricci Curvature (FRC) over the more theoretically ubiquitous Ollivier-Ricci Curvature (ORC) due to strictly constrained computational budgets.

2.1 The Computational Intractability of Ollivier-Ricci

Ollivier-Ricci curvature is defined via Optimal Transport. For two nodes $x$ and $y$ in a graph, the curvature $\kappa(x, y)$ is defined by comparing the graph distance $d(x, y)$ to the Wasserstein distance $W_1$ between the probability measures $m_x$ and $m_y$ defined on their neighborhoods 11:


$$\kappa(x, y) = 1 - \frac{W_1(m_x, m_y)}{d(x, y)}$$

Intuitively, if the neighborhoods of $x$ and $y$ are well-connected (forming a triangle or clique), it is "cheap" to transport mass from neighbors of $x$ to neighbors of $y$, resulting in low $W_1$ and positive curvature. If $x$ and $y$ are connected only by a single bridge, mass must traverse the edge $(x, y)$ itself, making transport "expensive" and curvature negative.
While ORC captures deep geometric properties 16, computing $W_1$ (Earth Mover's Distance) requires solving a linear programming problem. The best-case complexity for exact computation is super-cubic $O(k^3 \log k)$ with respect to the node degree $k$, or roughly $O(N^3)$ for the full graph.18 Even Sinkhorn approximations are too slow for the inner loop of a Transformer, which has a latency budget of microseconds per token. Snippet 18 explicitly notes that computing Sinkhorn distances can be slower than exact EMD on dense graphs, rendering ORC strictly infeasible for dynamic attention masking.19

2.2 Forman-Ricci Curvature: The Combinatorial Proxy

Forman-Ricci curvature is a discretization of Ricci curvature based on CW complexes (a generalization of graphs to higher dimensions including faces and volumes). For a 1-dimensional graph complex (nodes and edges), Forman’s formula depends only on the local combinatorial structure—specifically, the weights of the nodes and edges parallel to the edge in question.10
For a weighted graph with edge weights $w_e$ and node weights $w_v$, the augmented Forman-Ricci curvature for an edge $e = (u, v)$ is given by 21:
$$ \text{FRC}(e) = w_e \left( \frac{w_u}{w_e} + \frac{w_v}{w_e} - \sum_{e_u \sim u, e_u \neq e} \frac{w_u}{\sqrt{w_e w_{e_u}}} - \sum_{e_v \sim v, e_v \neq e} \frac{w_v}{\sqrt{w_e w_{e_v}}} \right) $$
Crucially, this formula relies only on the immediate neighbors (1-hop) of nodes $u$ and $v$. It involves basic arithmetic operations: summation, multiplication, and square roots. In the context of the attention mechanism, where $u$ is a Query token (or block) and $v$ is a Key token (or block), the terms in the summation correspond to other high-attention connections shared by $u$ and $v$.
Interpretation in Attention:
Node Weights ($w_u, w_v$): Can be interpreted as the total attention mass entering or leaving a block (conceptually similar to degree).
Edge Weight ($w_e$): The direct attention score between Query block $i$ and Key block $j$.
Summation Terms (Triangles): These terms penalize the curvature score based on the "parallelness" of edges. In a simplified view, if $u$ and $v$ share many common neighbors (triangles), these sums become large, driving the curvature value down (or up, depending on sign convention—Forman’s original formulation yields negative values for triangles, but augmented versions align with Riemannian intuition where cliques = positive).
For the purpose of our hardware implementation, we adopt a simplified "Triangle-Based" proxy for curvature that captures the essential geometric signal:
$$ \text{Curvature}(i, j) \approx \text{DirectConnection}(i, j) - \lambda \times \text{Redundancy}(i, j) $$
Where $\text{Redundancy}(i, j)$ is proportional to the number of shared neighbors (paths of length 2).


$$\text{Redundancy}(i, j) \approx \sum_k A_{ik} \cdot A_{kj} = (A^2)_{ij}$$

This simplification allows us to express curvature estimation as a matrix multiplication operation, which GPUs excel at, rather than a linear programming problem.

2.3 The Stratified Manifold Hypothesis

Recent research into the geometry of LLMs suggests that the embedding space is not a uniform manifold but a stratified manifold—a collection of lower-dimensional manifolds glued together.12 These strata often exhibit uniformly negative Ricci curvature, characteristic of hyperbolic spaces. Hyperbolic geometry is naturally suited for representing hierarchies and trees (e.g., syntax trees, logical entailment).
The "Stratified Manifold Hypothesis" provides the theoretical justification for why FRC is likely to succeed. It suggests that the "true" structure of the data the LLM is processing is tree-like and hierarchical. In such a geometry, "shortcuts" or "bridges" between branches of the tree are the critical semantic links. These bridges are mathematically defined by their negative curvature. Therefore, a curvature-based attention mechanism is effectively aligning the computational graph of the Transformer with the intrinsic geometry of the data it is processing. This is a powerful inductive bias that magnitude-based methods lack.

2.4 Why Block-Level Curvature?

Calculating FRC on individual tokens ($N \times N$) is still $O(N^2)$ or $O(N)$ depending on sparsity, which is expensive. However, to utilize Tensor Cores, we must operate on blocks.
This constraint actually simplifies our problem. We treat a Block of Tokens (e.g., $64 \times 64$) as a Supernode in a coarse-grained graph.
Nodes: Blocks $B_1, B_2, \dots, B_M$ where $M = N/64$.
Edges: Aggregate attention between blocks.
Curvature: Calculated on the $M \times M$ coarse graph.
Since $M \ll N$ (reduced by a factor of 64), the curvature calculation becomes computationally negligible relative to the fine-grained attention. This "Coarsening" strategy is the key to making geometric attention hardware-efficient.

3. System Architecture: Curvature-Aware Block-Sparse Attention (CAB)

We propose Curvature-Aware Block-Sparse Attention (CAB), a drop-in replacement for the sparse attention modules in architectures like DeepSeek NSA. The system consists of three distinct stages: Coarsening, Curvature Estimation (The Predictor), and Sparse Execution.

3.1 Stage 1: The Coarsening Operator

To build the proxy graph, we must reduce the dimensionality of the Query ($Q$) and Key ($K$) tensors. Let the input sequence length be $N$, head dimension $D$, and block size $B_s$.
Input $Q, K \in \mathbb{R}^{B \times H \times N \times D}$.
We reshape these to $\mathbb{R}^{B \times H \times (N/B_s) \times B_s \times D}$.
We must aggregate the $B_s$ tokens in each block into a single representative vector.
Option A: Mean Pooling. $\bar{Q}_i = \frac{1}{B_s} \sum_{t=1}^{B_s} Q_{i,t}$.
Critique: Mean pooling averages out "needles." If a block contains one highly relevant token and 63 irrelevant ones, the mean vector will be dominated by noise.
Option B: Max Pooling. $\bar{Q}_i = \text{Max}_{t=1}^{B_s}(Q_{i,t})$ (element-wise).
Critique: Preserves magnitude but distorts direction.
Option C: Representative Token (Strided). Take the first or middle token.
Critique: Too random.
Proposed Solution: L2-Norm Weighted Pooling.
We compute the $L_2$ norm of each token in the block and use it as a weight for a weighted average, or simply select the token with the max $L_2$ norm as the representative. Given the "Needle" requirement, we propose a Max-Magnitude Representative strategy:


$$\bar{Q}_i = Q_{i, k^*} \quad \text{where} \quad k^* = \text{argmax}_t \|Q_{i,t}\|_2$$

This ensures that the token with the strongest potential signal defines the block's geometric position. This operation can be fused into the previous layer's kernel or executed as a fast reduction in Triton.

3.2 Stage 2: The FRC Predictor Kernel

This is the core novelty. The predictor operates on the coarse block embeddings $\bar{Q}, \bar{K} \in \mathbb{R}^{M \times D}$.
Step 2a: Coarse Adjacency
We compute the coarse affinity matrix:


$$\mathcal{A} = \text{ReLU}(\bar{Q} \bar{K}^T / \sqrt{D})$$

We use ReLU instead of Softmax here to enforce sparsity and avoid the expensive exponential operation on the full coarse matrix. The result $\mathcal{A} \in \mathbb{R}^{M \times M}$ represents the "potential flow" between blocks.
Step 2b: Geometric Filtering (The FRC Score)
We calculate the Forman-Ricci score for each potential connection $(i, j)$ in $\mathcal{A}$.
Using the simplified Triangle Proxy:
Direct Path: $S_{direct} = \mathcal{A}_{ij}$
Redundant Paths (Triangles): $S_{redundant} = (\mathcal{A} \times \mathcal{A}^T)_{ij}$ (for undirected approximation) or $(\mathcal{A} \times \mathcal{A})_{ij}$ (for directed flow).
Note: $(\mathcal{A} \times \mathcal{A})_{ij} = \sum_k \mathcal{A}_{ik} \mathcal{A}_{kj}$. This sum is large if there are many intermediate blocks $k$ connecting $i$ and $j$.
FRC Score:

$$\Gamma_{ij} = S_{direct} - \lambda S_{redundant}$$
If $\Gamma_{ij}$ is high, it means $S_{direct}$ is high and $S_{redundant}$ is low (Strong, Unique connection -> Keep).
If $\Gamma_{ij}$ is low (or negative), it means redundancy is high relative to direct strength (Clique -> Prune).
Wait: Re-evaluating the sign convention. Negative curvature = Bridge. Positive curvature = Clique.
Standard FRC: Clique $\implies$ Positive. Bridge $\implies$ Negative.
We want to Keep Negative Curvature (Bridges).
However, we also want to keep high-magnitude direct connections (local attention).
Therefore, the selection score should be:

$$\text{Score}_{ij} = \alpha \mathcal{A}_{ij} + \beta (1 - \text{Curvature}_{ij})$$

Or simply, using the FRC formula terms directly: We prioritize edges where $\mathcal{A}_{ij}$ is significant AND the number of triangles is LOW.
Step 2c: Dynamic Mask Generation
For each query block $i$, we select the indices $j$ that maximize the Score.


$$\text{Indices}_i = \text{TopK}_j (\Gamma_{ij})$$

This produces a boolean tensor or index list BlockMask to be consumed by the sparse attention kernel.

3.3 Stage 3: Integration with DeepSeek NSA

DeepSeek NSA uses three branches. The "Selected Attention" branch 1 is the target for this upgrade.
Current NSA: Selects blocks based on top-$k$ pre-gating scores or accumulated magnitude.
CAB-NSA: Replaces the selection logic of the "Selected Attention" branch with the FRC Predictor. The Compressed and Sliding Window branches remain untouched, as they handle global coarse context and local context respectively. This hybrid approach ensures that if the FRC predictor misses something (unlikely), the sliding window or compressed global view provides a safety net.

4. Hardware-Aware Implementation: The Triton Kernel

The theoretical elegance of FRC is irrelevant if it creates a pipeline bubble. We analyze the implementation using NVIDIA’s Triton DSL.

4.1 Memory Hierarchy & Data Movement

The bottleneck in any attention kernel is the movement of $Q, K, V$ from HBM (High Bandwidth Memory, e.g., 80GB/s) to SRAM (Streaming Multiprocessor L1 Cache).
Full Attention: Loads $N$ blocks of $K, V$ for each $Q$ block. Total Loads: $N^2$.
Sparse Attention: Loads $k$ blocks of $K, V$ for each $Q$ block. Total Loads: $N \times k$.
The Predictor Overhead:
The predictor requires loading $\bar{Q}, \bar{K}$.
Size of $\bar{Q}$: $(N/B_s) \times D$. For $N=128k, B_s=64, D=128$:
$M = 2048$. Size $\approx 2048 \times 128 \times 2 \text{ bytes (BF16)} \approx 0.5 \text{ MB}$.
This entire coarse representation fits easily into the L2 Cache (typically 40MB-50MB on A100/H100) or even distributed across SRAMs.
Therefore, the "Coarse GEMM" $\bar{Q} \bar{K}^T$ is extremely fast and likely memory-bound only by the initial load.

4.2 Triton Kernel 1: Fused Coarsening

We implement a kernel fused_coarsen_reduce:

Python


@triton.jit
def fused_coarsen_reduce(
    Q_ptr, K_ptr, 
    Q_coarse_ptr, K_coarse_ptr,
    stride_qm, stride_qk,...
    BLOCK_SIZE: tl.constexpr
):
    # Parallelize over blocks
    pid = tl.program_id(0)
    
    # Load BLOCK of Q
    # Apply Max Reduction or L2-Norm selection
    # Store single vector to Q_coarse_ptr


Optimization: This can be fused into the Rotary Embedding (RoPE) kernel typically run before attention.

4.3 Triton Kernel 2: FRC Score & Top-K

Implementing the "Triangle Count" $(\mathcal{A} \times \mathcal{A})$ naively is $O(M^3)$. For $M=2048$, $M^3 \approx 8 \times 10^9$ ops. This is small for a GPU (H100 does $10^{15}$ FLOPS), but latency matters.
Optimization: We do not need the full matrix multiplication. We only need it for the top candidates.
Two-Stage Pipeline:
Candidate Generation: Compute $\mathcal{A} = \bar{Q}\bar{K}^T$. Keep top-$2k$ indices based on magnitude.
Re-Ranking: For these $2k$ edges, compute the triangle term $\sum_l \mathcal{A}_{il} \mathcal{A}_{lj}$. This effectively sparsely samples the matrix multiplication.
This reduces complexity from $O(M^3)$ to $O(M \cdot k \cdot M) = O(M^2 k)$.
Since $k$ is small (e.g., 16 or 32 blocks), this is very fast.

4.4 Triton Kernel 3: Sparse Attention (FlexAttention)

We leverage PyTorch's FlexAttention API 22, which allows passing a BlockMask.
The BlockMask is a tensor of shape (Batch, Heads, M, M) (bitmask).
By generating this mask on the GPU via the FRC kernel and passing it to FlexAttention, we utilize the highly optimized FlashAttention-3 backend without writing raw CUDA assembly.
Snippet 22 confirms that FlexAttention supports block masks efficiently, specifically optimized for $128 \times 128$ blocks, though $64 \times 64$ is also supported.

5. Experimental Protocol: Validating the Hypothesis

To secure an ICML acceptance, the experiments must prove that CAB-Attention solves the "Needle" problem better than H2O without regressing on speed.

Experiment 1: Synthetic Topology Recovery (Sanity Check)

Hypothesis: FRC can recover ground-truth bridges in a synthetic graph where magnitude is inversely correlated with importance.
Data: Generate random graphs using a Stochastic Block Model (SBM) with two dense communities connected by a single bridge edge.
Signal: Assign high weights to intra-community edges and low weights to the bridge edge.
Task: Retrieve a node across the bridge.
Baselines: Top-K Magnitude (H2O) vs. Top-K FRC.
Metric: Bridge Retrieval Rate (%).
Expected Result: H2O should fail (0%) as it chases high weights. FRC should succeed (~100%) as it detects the bottleneck.

Experiment 2: Needle-in-a-Haystack (The Killer App)

Hypothesis: CAB-Attention outperforms H2O on retrieval accuracy at high sparsity levels (95%+).
Dataset: "Needle In A Haystack" (NIAH) benchmark.24
Setup:
Context Lengths: 32k, 64k, 128k.
Needle Depth: 0% to 100% (distributed uniformly).
Needle Type: Random UUIDs, passkeys, and "reasoning needles" (facts required for a chain of thought).
Baselines:
Full Attention (Upper Bound).
DeepSeek NSA (Standard).
H2O (Magnitude).
CAB-Attention (Ours).
Metric: Retrieval Accuracy vs. Sparsity Ratio.
Visualization: A Heatmap of Retrieval Accuracy (X-axis: Context Length, Y-axis: Needle Depth).
Success Condition: CAB-Attention maintains "Green" (100% acc) regions in the heatmap significantly larger than H2O.

Experiment 3: Perplexity vs. Speed Trade-off

Hypothesis: CAB-Attention achieves better perplexity than NSA/H2O at the same FLOP budget.
Dataset: PG-19 (Books) and LongBench (Summarization).
Model: Fine-tune a LLaMA-3-8B or Mistral-7B checkpoint with the CAB-Attention adapters.
Metric: Perplexity (lower is better) vs. Inference Throughput (tokens/sec).
Analysis: Plot the Pareto frontier. CAB-Attention should push the frontier outward (better PPL for same Speed).

Experiment 4: Wall-Clock Profiling

Hypothesis: The overhead of the FRC predictor is $< 10\%$ of the total attention time.
Tool: NVIDIA Nsight Compute.
Measurement: Break down the forward pass into:
Coarsening Kernel Time.
FRC Predictor Time.
Sparse Attention Kernel Time.
Comparison: Compare total time against FlashAttention-2.
Target: Break-even point (where Sparse becomes faster than Dense) should be $< 8k$ tokens.

Experiment 5: Physics-Informed Rewiring Comparison

Context: Snippets 25 and 26 discuss "PIORF" (Physics-Informed Ollivier-Ricci Flow) for graph rewiring in fluid dynamics.
Adaptation: While we don't simulate physics, we can compare our "Graph Rewiring" (Attention Masking) against a "PIORF-inspired" baseline that uses iterative rewiring (running the predictor multiple times to refine the graph).
Goal: Determine if iterative refinement (Flow) improves attention quality or if single-step estimation (Curvature) is sufficient.

6. Strategic Roadmap for ICML Submission


6.1 Positioning

The paper should be titled: "Geometric Inductive Biases in Hardware-Aware Sparse Attention."
It positions itself at the intersection of three trendy fields:
Systems for ML: (Triton, Sparse Kernels).
Geometric Deep Learning: (Ricci Curvature, Manifolds).
Long-Context LLMs: (Needle in a Haystack).

6.2 Novelty Arguments

First Hardware-Efficient Curvature: We bridge the gap between theoretical ORC (slow) and practical attention (fast) using Coarse FRC.
Topological "Needle" Theory: We provide a theoretical explanation for why H2O fails on needles (needles are bridges, bridges have negative curvature) and validate it.
Open Source Triton Kernels: We provide the first open implementation of a geometric attention selector in Triton.

6.3 Risk Assessment & Pre-empting Reviewers

Critique: "The coarse graph ($64 \times 64$ blocks) is too blurry to capture fine interactions."
Defense: Use the Max-Pooling ablation in Exp 5 to show that max-pooling preserves the signal of the "needle" token even in a coarse block.
Critique: "Why not just use a learned router (MoE)?"
Defense: Learned routers add parameters and training instability. FRC is a parameter-free inductive bias that generalizes zero-shot to infinite contexts.

7. Conclusion

Forman-Ricci curvature represents a powerful, underutilized tool for the optimization of Transformer architectures. By shifting the selection paradigm from Magnitude (Heavy Hitters) to Topology (Bridges), we can address the critical weakness of current sparse attention models: their inability to reliably retrieve low-probability, high-information content. The "Coarse-Grained FRC" approach proposed here makes this theoretically sophisticated metric computationally practically feasible on modern GPUs via block-wise operations in Triton.
This report outlines a comprehensive path to validating this hypothesis. If the experimental results confirm that FRC recovers "needles" that H2O misses, while maintaining the hardware-aligned throughput of DeepSeek NSA, the resulting publication would be a seminal contribution to the field of efficient long-context modeling.

References & Citations

Detailed citations are integrated into the text using the source IDs provided (e.g.1).
Table 1: Comparative Analysis of Sparse Attention Selection Metrics
Metric
Basis
Complexity (Token)
Complexity (Block)
Inductive Bias
Needle Retrieval
Hardware Fit
H2O
Attention Magnitude ($L_1$)
$O(N \log k)$
$O(M \log k)$
Frequency / Recency
Poor (Drops low-mag needles)
High
StreamingLLM
Positional Heuristic
$O(1)$
$O(1)$
Locality / Sinks
Fail (Cannot see middle)
High
Ollivier-Ricci
Optimal Transport ($W_1$)
$O(N^3)$
$O(M^3)$
Mass Diffusion
Good
Infeasible
CAB (FRC)
Combinatorial Topology
N/A
$O(M^2)$ (Approx)
Bottlenecks / Bridges
Excellent (Preserves unique paths)
High (Triton)

M = Number of Blocks ($N/B_s$)
Works cited
Hardware-Aligned and Natively Trainable Sparse Attention - arXiv, accessed November 26, 2025, https://arxiv.org/html/2502.11089v1
Hardware-Aligned and Natively Trainable Sparse Attention - ACL Anthology, accessed November 26, 2025, https://aclanthology.org/2025.acl-long.1126.pdf
Long Range Arena for Benchmarking Efficient Transformers - GitHub, accessed November 26, 2025, https://github.com/google-research/long-range-arena
LONG RANGE ARENA:ABENCHMARK FOR EFFICIENT TRANSFORMERS - OpenReview, accessed November 26, 2025, https://openreview.net/pdf/c7ddcda9fb422b91032d80ebd1564c35dd6f9fa8.pdf
Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores, accessed November 26, 2025, https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/
Hardware-Aligned and Natively Trainable Sparse Attention - arXiv, accessed November 26, 2025, https://arxiv.org/pdf/2502.11089
[2306.14048] H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models - arXiv, accessed November 26, 2025, https://arxiv.org/abs/2306.14048
Efficient streaming language models with at - arXiv, accessed November 26, 2025, https://arxiv.org/pdf/2309.17453
Natively Sparse Attention (NSA) for Efficient Long-Context LLMs - Ajith's AI Pulse, accessed November 26, 2025, https://ajithp.com/2025/02/21/natively-sparse-attention-nsa-the-future-of-efficient-long-context-modeling-in-large-language-models/
Network Geometry of Borsa Istanbul: Analyzing Sectoral Dynamics with Forman–Ricci Curvature - PMC - PubMed Central, accessed November 26, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11941537/
Graph Pooling via Ricci Flow - OpenReview, accessed November 26, 2025, https://openreview.net/pdf?id=xpBHp9WFvk
The structure of the token space for large language models - arXiv, accessed November 26, 2025, https://arxiv.org/html/2410.08993v1
[2410.08993] The structure of the token space for large language models - arXiv, accessed November 26, 2025, https://arxiv.org/abs/2410.08993
Curvature and over-squashing in Graph Neural Networks, accessed November 26, 2025, https://www.sci.unich.it/geodeep2022/slides/GRAFF_presentation%20(18).pdf
Understanding over-squashing and bottlenecks on graphs via curvature - OpenReview, accessed November 26, 2025, https://openreview.net/forum?id=7UmjRGzp-A
CurvAGN: Curvature-based Adaptive Graph Neural Networks for Predicting Protein-Ligand Binding Affinity - PMC - NIH, accessed November 26, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10557336/
(PDF) Graph Pooling via Ricci Flow - ResearchGate, accessed November 26, 2025, https://www.researchgate.net/publication/382065254_Graph_Pooling_via_Ricci_Flow
Graph Pooling via Ricci Flow - arXiv, accessed November 26, 2025, https://arxiv.org/html/2407.04236v1
Lower Ricci Curvature for Hypergraphs - arXiv, accessed November 26, 2025, https://arxiv.org/html/2506.03943v1
On the Ricci curvature of attention maps and transformers training and robustness (Workshop Report) - NSF Public Access Repository, accessed November 26, 2025, https://par.nsf.gov/biblio/10627697
accessed November 26, 2025, https://arxiv.org/html/2307.13275v2#:~:text=However%2C%20the%20calculation%20of%20Forman,directed%20and%20undirected%20weighted%20graphs.
torch.nn.attention.flex_attention — PyTorch 2.9 documentation, accessed November 26, 2025, https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention, accessed November 26, 2025, https://pytorch.org/blog/flexattention/
[2510.24606] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs - arXiv, accessed November 26, 2025, https://arxiv.org/abs/2510.24606
[2504.04052] PIORF: Physics-Informed Ollivier-Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks - arXiv, accessed November 26, 2025, https://arxiv.org/abs/2504.04052
Adaptive Graph Rewiring in Mesh GNNs - Emergent Mind, accessed November 26, 2025, https://www.emergentmind.com/topics/adaptive-graph-rewiring-in-mesh-based-graph-neural-networks-adameshnet
