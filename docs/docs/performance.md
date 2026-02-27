# Performance

ChebyshevSharp achieves high performance through native BLAS integration and careful memory management in the evaluation hot path. On low-dimensional problems (1D--3D), C# is 17--42x faster than the Python reference implementation; on high-dimensional problems where BLAS dominates, it remains 1.4--1.5x faster.

## BLAS Integration

The core evaluation loop in `VectorizedEval` contracts an N-dimensional tensor one axis at a time, reducing it to a scalar. Each contraction is a matrix-vector multiply (GEMV) or matrix-matrix multiply (GEMM), routed through native OpenBLAS via the [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas) NuGet package.

This is the same algorithmic approach used by PyChebyshev's `vectorized_eval()`, which reshapes the tensor contraction into `numpy.dot` calls backed by the system BLAS. The difference is that C# eliminates Python's per-call interpreter overhead (object allocation, GIL acquisition, reference counting), which is significant when the BLAS work itself is small.

No system BLAS installation is required. The BlasSharp.OpenBlas package bundles pre-built OpenBLAS binaries for all supported platforms:

| Runtime | Architecture |
|---------|-------------|
| Windows | x64, x86 |
| Linux   | x64, arm64 |
| macOS   | x64, arm64 (Apple Silicon) |

Adding the NuGet package is sufficient -- there is no runtime library probing or fallback logic.

## Key Optimizations

### BLAS GEMV for Barycentric Contraction

`MatmulLastAxis` calls OpenBLAS `cblas_dgemv` with `CblasRowMajor` layout to contract the tensor's last axis with a barycentric weight vector. The flat `double[]` tensor data is passed directly to BLAS via pinned pointers with no copy or transpose.

### BLAS GEMM for Differentiation

When computing derivatives, `MatmulLastAxisMatrixFlat` calls OpenBLAS `cblas_dgemm` to multiply the tensor by a differentiation matrix. The differentiation matrix is pre-flattened to a row-major `double[]` at build time, enabling zero-copy BLAS calls with no per-eval allocation.

### Pre-transposed Differentiation Matrices

Differentiation matrices are transposed and flattened into contiguous `double[]` arrays (`DiffMatricesTFlat`) in a single pass during `Build()` or `Load()`. This eliminates O(n^2) transpose and flatten operations that would otherwise run on every evaluation involving derivatives.

### Shape Allocation Elimination

The evaluation loop tracks tensor dimensions as two integers (`leadSize` and `lastDim`) instead of cloning an `int[]` shape array per dimension. This removes one array allocation per dimension per eval call.

### Inline Barycentric Weights

Normalized barycentric weights are computed directly in the evaluation loop rather than allocated into a separate array, reducing allocation pressure on the hot path.

### FFT-based DCT-II

Chebyshev coefficient extraction for `ErrorEstimate()` uses an O(n log n) DCT-II via FFT (MathNet.Numerics Fourier transform) for polynomial orders above 32. For small orders (n <= 32), a direct O(n^2) summation avoids the FFT setup overhead.

## Performance Comparison: C# vs Python

Both implementations use OpenBLAS for the underlying linear algebra. Benchmarks were run on the same hardware (12th Gen Intel Core i7-12700K) with .NET 10.0 (X64 RyuJIT AVX2) and Python + NumPy.

| Benchmark | Python | C# | Speedup |
|-----------|-------:|---:|--------:|
| 1D value eval | 3,482 ns | 83 ns | 42x |
| 1D first derivative | 4,351 ns | 178 ns | 24x |
| 3D value eval | 10,363 ns | 544 ns | 19x |
| 3D delta (derivative) | 11,334 ns | 662 ns | 17x |
| 5D value eval | 55,455 ns | 39,082 ns | 1.4x |
| 5D delta (derivative) | 56,747 ns | 38,835 ns | 1.5x |
| 3D batch (100 points) | 1,251,969 ns | 54,117 ns | 23x |
| 3D multi (price + Greeks) | 17,908 ns | 1,921 ns | 9.3x |

### Why the Gap Varies by Dimension

**1D--3D (17--42x faster):** Python's per-call overhead dominates. Each call to `vectorized_eval` in Python enters the interpreter, allocates NumPy arrays for weights, passes through the GIL, and performs reference counting. When the actual BLAS work is sub-microsecond, this overhead is the bottleneck. C# JIT-compiles to native code with no interpreter overhead.

**5D (1.4--1.5x faster):** The 161,051-element tensor contraction saturates the BLAS GEMV call. Both languages call the same OpenBLAS `dgemv` routine, so the speedup narrows to what C# saves in loop overhead and memory management.

**Batch 3D (23x faster):** Python's `vectorized_eval_batch` loops over points in the interpreter; C# loops in JIT-compiled native code. Each individual eval is fast, so the loop overhead matters.

## Memory Allocation

The optimized implementation reduces per-eval memory allocation significantly for low-dimensional problems:

| Benchmark | Before | After | Reduction |
|-----------|-------:|------:|----------:|
| 1D value eval | 640 B | 216 B | -66% |
| 1D first derivative | 4,096 B | 432 B | -89% |
| 3D value eval | 2,872 B | 2,008 B | -30% |
| 3D delta | 4,856 B | 2,152 B | -56% |

For 5D problems, allocation is dominated by the large intermediate tensors required for contraction and remains roughly the same (~129 KB per eval).

## ChebyshevSpline Performance

`ChebyshevSpline` evaluation adds a thin routing layer on top of `ChebyshevApproximation`. The overhead consists of one `Array.BinarySearch` per dimension (O(log k) where k is the number of knots) followed by a standard `VectorizedEval` on the selected piece.

Since each piece has fewer nodes than a global interpolant of equivalent accuracy, per-eval BLAS work is smaller. The trade-off is more function evaluations at build time (one full tensor grid per piece) and slightly higher per-eval overhead from piece routing.

**When to expect similar performance:** For functions with singularities, the spline achieves target accuracy with far fewer nodes per piece than a global interpolant would need. The reduced per-piece tensor size often more than compensates for the routing overhead.

**When to expect overhead:** For smooth functions where a global interpolant already works well, the spline adds routing cost with no accuracy benefit. Use `ChebyshevApproximation` in this case.

Spline benchmarks are available in the `SplineBenchmarks` class in the benchmark project. Run them with:

```bash
dotnet run -c Release --project benchmarks/ChebyshevSharp.Benchmarks -- --filter '*Spline*'
```

## ChebyshevSlider Performance

`ChebyshevSlider` trades approximation fidelity for dramatically reduced build cost in high dimensions. Instead of a single tensor grid with $\prod_i n_i$ evaluations, it builds one small `ChebyshevApproximation` per partition group, with total build cost $\sum_g \prod_{i \in g} n_i$.

For example, a 6D function with 10 nodes per dimension and partition `[[0,1],[2,3],[4,5]]`:
- Full tensor grid: $10^6 = 1{,}000{,}000$ evaluations
- Slider (three 2D groups): $3 \times 10^2 = 300$ evaluations

Evaluation involves summing the contributions of each slide (one `VectorizedEval` per group) plus the pivot value. For k groups, each eval is approximately k times the cost of evaluating a single low-dimensional `ChebyshevApproximation`. The overhead from the additive decomposition is negligible compared to the build cost savings.

**When to expect good accuracy:** Functions that are additively separable or nearly so — where cross-group interactions are weak relative to within-group effects. The error estimate (`ErrorEstimate()`) reports the sum of per-slide errors, providing a conservative bound.

**When to expect reduced accuracy:** Functions with strong interactions between dimensions in different partition groups. The sliding technique cannot capture cross-group coupling beyond the pivot point. Consider using `ChebyshevApproximation` (for moderate dimensions) or [`ChebyshevTT`](tensor-train.md) (for high dimensions with coupling).

## ChebyshevTT Performance

`ChebyshevTT` uses TT-Cross to build from $O(d \cdot n \cdot r^2)$ function evaluations instead of the full $n^d$ tensor grid. Evaluation contracts Chebyshev polynomial basis vectors against the TT coefficient cores with cost $O(d \cdot n \cdot r^2)$ per point.

**Build cost comparison** (5D, 11 nodes per dim):
- Full tensor (`ChebyshevApproximation`): 161,051 evaluations
- TT-Cross (rank 15): ~7,400 evaluations (22x fewer)

**Eval cost:** For typical financial applications (d = 5--10, n = 7--15, r = 5--15), evaluation takes 1--10 microseconds per point. The cores are small enough that the computation is CPU-cache-bound rather than BLAS-bound. GPU acceleration is not beneficial at these sizes -- kernel launch overhead would dominate the actual computation.

**Batch eval:** `EvalBatch` vectorizes the contraction across all points, providing 15--20x speedup over calling `Eval` in a loop. This is valuable for Monte Carlo or grid-based downstream calculations.

**When to use TT vs. Slider:** If the function is nearly additively separable, `ChebyshevSlider` provides analytical derivatives and simpler error analysis. If cross-variable coupling is significant (e.g., $S \cdot \sigma$ interactions in Black-Scholes), `ChebyshevTT` captures it through higher TT rank at the cost of finite-difference derivatives.

## References

1. Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
2. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
3. Oseledets, I. V. (2011). "Tensor-Train Decomposition." *SIAM Journal on Scientific Computing* 33(5):2295--2317.
