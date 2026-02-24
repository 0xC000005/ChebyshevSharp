# Benchmark Comparison: Baseline vs Optimized

**Date**: 2026-02-24
**System**: 12th Gen Intel Core i7-12700K, .NET 10.0.3, X64 RyuJIT AVX2
**BLAS**: OpenBLAS via BlasSharp.OpenBlas NuGet package

## Optimizations Applied

1. **Native BLAS GEMV/GEMM** via BlasSharp.OpenBlas NuGet package: Provides pre-built OpenBLAS bindings for all platforms (Linux, macOS, Windows). No runtime probing or fallback logic needed. Uses `CblasRowMajor` -- zero-copy, no transpose tricks.
2. **Pre-transposed DiffMatrices** (`DiffMatricesTFlat`): Transposed and flattened in one pass at Build/Load. Eliminates per-eval transpose and BLAS flattening overhead.
3. **Shape allocation elimination**: Track `leadSize`/`lastDim` integers instead of `(int[])NNodes.Clone()` per dimension.
4. **Inline barycentric weights**: Compute normalized weights directly in eval loop.
5. **FFT-based DCT-II**: O(n log n) via MathNet Fourier for n > 32.

## Eval Benchmarks

| Method           | Baseline (ns) | Optimized (ns) | Speedup  | Alloc Before | Alloc After | Reduction |
|----------------- |---------------:|---------------:|---------:|-------------:|------------:|----------:|
| Eval_1D_Value    |          134.3 |          83.3  | **1.6x** |        640 B |       216 B |   **-66%** |
| Eval_1D_Deriv1   |          966.1 |         178.0  | **5.4x** |      4,096 B |       432 B |   **-89%** |
| Eval_3D_Value    |        1,184.7 |         543.6  | **2.2x** |      2,872 B |     2,008 B |     -30%   |
| Eval_3D_Delta    |        1,693.8 |         662.2  | **2.6x** |      4,856 B |     2,152 B |   **-56%** |
| Eval_5D_Value    |       90,032.1 |      39,081.7  | **2.3x** |    130,503 B |   129,220 B |      -1%   |
| Eval_5D_Delta    |       93,266.7 |      38,835.3  | **2.4x** |    131,624 B |   129,321 B |      -2%   |
| Batch_3D_100pts  |      116,484.6 |      54,116.5  | **2.2x** |    288,024 B |   201,624 B |     -30%   |
| Multi_3D_Greeks  |        4,455.6 |       1,921.5  | **2.3x** |     10,640 B |     6,576 B |     -38%   |
| ErrorEstimate_3D |      469,523.2 |     482,798.8  |   1.0x   |    215,144 B |   219,024 B |      +2%   |

## Cross-Language Comparison (C# vs PyChebyshev)

PyChebyshev uses NumPy + OpenBLAS. Same BLAS library, same hardware.

| Method           | Python (ns)  | C# (ns)    | C# Speedup   |
|----------------- |-------------:|-----------:|:-------------|
| Eval_1D_Value    |        3,482 |         83 | **42x faster** |
| Eval_1D_Deriv1   |        4,351 |        178 | **24x faster** |
| Eval_3D_Value    |       10,363 |        544 | **19x faster** |
| Eval_3D_Delta    |       11,334 |        662 | **17x faster** |
| Eval_5D_Value    |       55,455 |     39,082 | **1.4x faster** |
| Eval_5D_Delta    |       56,747 |     38,835 | **1.5x faster** |
| Batch_3D_100pts  |    1,251,969 |     54,117 | **23x faster** |
| Multi_3D_Greeks  |       17,908 |      1,921 | **9.3x faster** |

## Analysis

### Why C# Beats Python

- **1D-3D: 17-42x faster** -- Python's per-call overhead (~3.5us to enter vectorized_eval, allocate weight arrays, etc.) dwarfs the actual computation. C# JIT compiles to native code with zero interpreter overhead.
- **5D: 1.4-1.5x faster** -- Both use the same OpenBLAS GEMV for the 161K-element contraction. The advantage comes from C#'s lower overhead in the eval loop (no Python object allocation, no GIL, no reference counting).
- **Batch 3D: 23x faster** -- Python's `vectorized_eval_batch` loops in Python; C# loops in native code.

### BLAS Integration

ChebyshevSharp uses the [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas) NuGet package for native BLAS operations. This package bundles pre-built OpenBLAS binaries for all supported platforms (Linux, macOS, Windows) -- no system BLAS installation required, no runtime library probing, no fallback logic. Just add the NuGet package and BLAS is available.
