```

BenchmarkDotNet v0.14.0, Ubuntu 24.04.4 LTS (Noble Numbat)
12th Gen Intel Core i7-12700K, 1 CPU, 20 logical and 12 physical cores
.NET SDK 10.0.103
  [Host]     : .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2
  DefaultJob : .NET 10.0.3 (10.0.326.7603), X64 RyuJIT AVX2


```
| Method                 | Mean            | Error        | StdDev       | Gen0     | Gen1     | Gen2     | Allocated |
|----------------------- |----------------:|-------------:|-------------:|---------:|---------:|---------:|----------:|
| GEMV_Small_10x10       |        57.14 ns |     0.721 ns |     0.674 ns |   0.0079 |        - |        - |     104 B |
| GEMV_Medium_180x10     |     1,038.75 ns |     8.173 ns |     7.645 ns |   0.1106 |        - |        - |    1464 B |
| GEMV_Large_14641x11    |    78,862.29 ns |   590.995 ns |   552.817 ns |  35.6445 |  35.6445 |  35.6445 |  117175 B |
| GEMM_Small_10x10x10    |       692.87 ns |     2.704 ns |     2.529 ns |   0.0629 |        - |        - |     824 B |
| GEMM_Medium_180x10x10  |    12,375.45 ns |    64.012 ns |    59.876 ns |   1.0986 |        - |        - |   14424 B |
| GEMM_Large_14641x11x11 | 1,258,823.63 ns | 9,592.731 ns | 8,973.047 ns | 248.0469 | 248.0469 | 248.0469 | 1288594 B |
| DCT_n10                |       665.45 ns |     2.188 ns |     1.939 ns |   0.0153 |        - |        - |     208 B |
| DCT_n20                |     2,795.16 ns |     4.721 ns |     4.185 ns |   0.0267 |        - |        - |     368 B |
| DCT_n50                |    18,291.35 ns |    78.358 ns |    73.296 ns |   0.0610 |        - |        - |     848 B |
