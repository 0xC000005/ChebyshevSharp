# Optimized Benchmarks (Post-Optimization)

**Date**: 2026-02-24
**System**: 12th Gen Intel Core i7-12700K, 20 logical / 12 physical cores
**Runtime**: .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2
**BLAS**: OpenBLAS via BlasSharp.OpenBlas NuGet package

## Optimizations Applied

1. **Native BLAS GEMV/GEMM**: BlasSharp.OpenBlas NuGet package provides pre-built OpenBLAS bindings for all platforms. No runtime library probing or fallback logic needed.
2. **Pre-transposed DiffMatrices**: `DiffMatricesTFlat` — transposed and flattened in one pass at Build/Load time, eliminates O(n^2) transpose allocation per eval and avoids per-call flattening for BLAS GEMM.
3. **Shape allocation elimination**: Track `leadSize`/`lastDim` integers instead of cloning shape arrays per dimension per eval.
4. **Inline barycentric weights**: Compute normalized weights directly in VectorizedEval without separate allocation.
5. **FFT-based DCT-II**: O(n log n) via MathNet Fourier for n > 32 (replaces O(n^2) naive DCT).

## Eval Benchmarks

| Method           | Mean          | Allocated |
|----------------- |--------------:|----------:|
| Eval_1D_Value    |      83.29 ns |     216 B |
| Eval_1D_Deriv1   |     178.03 ns |     432 B |
| Eval_3D_Value    |     543.60 ns |    2008 B |
| Eval_3D_Delta    |     662.15 ns |    2152 B |
| Eval_5D_Value    |  39,081.73 ns |  129220 B |
| Eval_5D_Delta    |  38,835.29 ns |  129321 B |
| Batch_3D_100pts  |  54,116.46 ns |  201624 B |
| Multi_3D_Greeks  |   1,921.45 ns |    6576 B |
| ErrorEstimate_3D | 482,798.80 ns |  219024 B |

## Kernel Micro-Benchmarks

| Method                 | Mean            | Allocated |
|----------------------- |----------------:|----------:|
| GEMV_Small_10x10       |        53.83 ns |     104 B |
| GEMV_Medium_180x10     |       345.00 ns |    1464 B |
| GEMV_Large_14641x11    |    36,767.20 ns |  117175 B |
| GEMM_Small_10x10x10    |       146.32 ns |     824 B |
| GEMM_Medium_180x10x10  |     1,520.99 ns |   14424 B |
| GEMM_Large_14641x11x11 |   100,945.51 ns | 1288594 B |
| DCT_n10                |       686.78 ns |     208 B |
| DCT_n20                |     2,877.26 ns |     368 B |
| DCT_n50                |     7,652.05 ns |    8135 B |

## Cross-Language Comparison

| Method           | Python (ns) | C# (ns) | C# Speedup |
|----------------- |------------:|--------:|-----------:|
| Eval_1D_Value    |       3,482 |      83 |    **42x** |
| Eval_1D_Deriv1   |       4,351 |     178 |    **24x** |
| Eval_3D_Value    |      10,363 |     544 |    **19x** |
| Eval_3D_Delta    |      11,334 |     662 |    **17x** |
| Eval_5D_Value    |      55,455 |  39,082 |   **1.4x** |
| Eval_5D_Delta    |      56,747 |  38,835 |   **1.5x** |
| Batch_3D_100pts  |   1,251,969 |  54,117 |    **23x** |
| Multi_3D_Greeks  |      17,908 |   1,921 |   **9.3x** |
