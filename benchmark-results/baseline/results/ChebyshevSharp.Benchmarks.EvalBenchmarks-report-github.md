```

BenchmarkDotNet v0.14.0, Ubuntu 24.04.4 LTS (Noble Numbat)
12th Gen Intel Core i7-12700K, 1 CPU, 20 logical and 12 physical cores
.NET SDK 10.0.103
  [Host]     : .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2
  DefaultJob : .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2


```
| Method           | Mean         | Error     | StdDev    | Gen0    | Gen1    | Gen2    | Allocated |
|----------------- |-------------:|----------:|----------:|--------:|--------:|--------:|----------:|
| Eval_1D_Value    |     134.3 ns |   0.94 ns |   0.84 ns |  0.0489 |       - |       - |     640 B |
| Eval_1D_Deriv1   |     966.1 ns |   8.59 ns |   8.03 ns |  0.3128 |       - |       - |    4096 B |
| Eval_3D_Value    |   1,184.7 ns |   6.39 ns |   5.98 ns |  0.2193 |       - |       - |    2872 B |
| Eval_3D_Delta    |   1,693.8 ns |   6.64 ns |   6.21 ns |  0.3700 |       - |       - |    4856 B |
| Eval_5D_Value    |  90,032.1 ns | 497.78 ns | 465.63 ns | 36.6211 | 35.6445 | 35.6445 |  130503 B |
| Eval_5D_Delta    |  93,266.7 ns | 493.15 ns | 461.29 ns | 36.6211 | 35.5225 | 35.5225 |  131624 B |
| Batch_3D_100pts  | 116,484.6 ns | 409.62 ns | 363.12 ns | 21.9727 |  0.1221 |       - |  288024 B |
| Multi_3D_Greeks  |   4,455.6 ns |  39.52 ns |  36.97 ns |  0.8087 |       - |       - |   10640 B |
| ErrorEstimate_3D | 469,523.2 ns | 593.83 ns | 526.41 ns | 16.1133 |  0.9766 |       - |  215144 B |
