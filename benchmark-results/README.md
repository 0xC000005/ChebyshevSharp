# Benchmark Result Provenance

These files are historical benchmark artifacts, not CI-generated reports.

- `baseline-summary.md` and `baseline/results/` record the pre-optimization run from 2026-02-24.
- `optimized-summary.md` and `COMPARISON.md` record the post-optimization summary from the same review cycle.
- Baseline raw BenchmarkDotNet exports are committed under `baseline/results/`; matching optimized raw exports were not committed.

BenchmarkDotNet writes fresh exports to `BenchmarkDotNet.Artifacts/results/`, which is ignored by git. To regenerate current results:

```bash
dotnet run --configuration Release --project benchmarks/ChebyshevSharp.Benchmarks/ChebyshevSharp.Benchmarks.csproj
```

For a focused run:

```bash
dotnet run --configuration Release --project benchmarks/ChebyshevSharp.Benchmarks/ChebyshevSharp.Benchmarks.csproj -- --filter '*Eval*'
```

When updating published performance claims, record the date, hardware, OS, .NET SDK/runtime, BenchmarkDotNet version, BLAS package version, command line, and the raw exported result files used to derive any summary tables.
