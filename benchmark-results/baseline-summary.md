# Baseline Benchmarks (Pre-Optimization)

**Date**: 2026-02-24
**System**: 12th Gen Intel Core i7-12700K, 20 logical / 12 physical cores
**Runtime**: .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2

## Eval Benchmarks

| Method           | Mean          | Error      | StdDev     | Allocated |
|----------------- |--------------:|-----------:|-----------:|----------:|
| Eval_1D_Value    |     134.3 ns  |   0.94 ns  |   0.84 ns  |     640 B |
| Eval_1D_Deriv1   |     966.1 ns  |   8.59 ns  |   8.03 ns  |    4096 B |
| Eval_3D_Value    |   1,184.7 ns  |   6.39 ns  |   5.98 ns  |    2872 B |
| Eval_3D_Delta    |   1,693.8 ns  |   6.64 ns  |   6.21 ns  |    4856 B |
| Eval_5D_Value    |  90,032.1 ns  | 497.78 ns  | 465.63 ns  |  130503 B |
| Eval_5D_Delta    |  93,266.7 ns  | 493.15 ns  | 461.29 ns  |  131624 B |
| Batch_3D_100pts  | 116,484.6 ns  | 409.62 ns  | 363.12 ns  |  288024 B |
| Multi_3D_Greeks  |   4,455.6 ns  |  39.52 ns  |  36.97 ns  |   10640 B |
| ErrorEstimate_3D | 469,523.2 ns  | 593.83 ns  | 526.41 ns  |  215144 B |

## Kernel Micro-Benchmarks

| Method                 | Mean            | Error        | StdDev       | Allocated |
|----------------------- |----------------:|-------------:|-------------:|----------:|
| GEMV_Small_10x10       |        57.14 ns |     0.721 ns |     0.674 ns |     104 B |
| GEMV_Medium_180x10     |     1,038.75 ns |     8.173 ns |     7.645 ns |    1464 B |
| GEMV_Large_14641x11    |    78,862.29 ns |   590.995 ns |   552.817 ns |  117175 B |
| GEMM_Small_10x10x10    |       692.87 ns |     2.704 ns |     2.529 ns |     824 B |
| GEMM_Medium_180x10x10  |    12,375.45 ns |    64.012 ns |    59.876 ns |   14424 B |
| GEMM_Large_14641x11x11 | 1,258,823.63 ns | 9,592.731 ns | 8,973.047 ns | 1288594 B |
| DCT_n10                |       665.45 ns |     2.188 ns |     1.939 ns |     208 B |
| DCT_n20                |     2,795.16 ns |     4.721 ns |     4.185 ns |     368 B |
| DCT_n50                |    18,291.35 ns |    78.358 ns |    73.296 ns |     848 B |

## Key Observations

- **5D eval dominates**: 90us per call due to 161,051-element tensor contractions
- **GEMM is the bottleneck**: Large GEMM (14641x11x11) takes 1.26ms — this is the diff matrix multiply in VectorizedEval for 5D
- **Heavy allocations**: 5D eval allocates 130KB per call; GEMM_Large allocates 1.29MB
- **3D is fast enough**: 1.2us per eval, 1.7us with derivatives
- **ErrorEstimate expensive**: 470us due to repeated DCT-II over all 1D slices
- **DCT scales as O(n^2)**: n=10: 665ns, n=20: 2.8us (4.2x), n=50: 18.3us (27.5x)
