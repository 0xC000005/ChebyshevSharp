# Phase 13 Accuracy And Speed Pilot Plan

> **For agentic workers:** Phase 13 is a benchmark-and-optimization phase.
> Preserve the public wrapper `curve bumps[60], coupon, maturity -> dirty PV`.
> Do not claim a general arbitrary-bond replacement; this remains a supported
> fixed-rate bullet family case study.

**Goal:** Decide whether the Phase 12 schedule-resolved cashflow Chebyshev
recipe can be both risk-accurate and materially faster than the QLNet-backed
reference path, and identify whether the next speed gains belong in the
example model or the ChebyshevSharp library.

## Why This Phase Exists

Phase 12 found the first accurate recipe: resolve the maturity schedule, keep
coupon/notional algebraic, and use local 1D/2D Chebyshev kernels for smooth
discount factors. That model reached roundoff-level PV and risk accuracy, but
the ad hoc stopwatch measurement showed only about `2.5x` scalar evaluation
speedup. Phase 13 replaces that rough timing with benchmark evidence and pilots
the most defensible hot-path improvements before proposing library API work.

## Related-Work Refresh

- MoCaX research frames Chebyshev tensors as a way to accelerate repeated risk
  and dynamic-sensitivity calculations, not only one scalar price.
- Orthogonal Chebyshev Sliding and Tensor Train examples motivate comparing
  compressed models only after held-out PV and risk validation.
- BenchmarkDotNet's `MemoryDiagnoser` is the project-standard way to measure
  runtime and managed allocations without relying on hand-written stopwatch
  loops.
- Tensor Train Cross literature supports sampled high-dimensional models only
  when held-out validation confirms the sampled object is compressible enough.

## Benchmark Targets

Measure all candidates in Release mode with BenchmarkDotNet:

1. QLNet reference adapter, value-only.
2. Current schedule-resolved Chebyshev kernel pricer, value-only.
3. Optimized schedule-resolved Chebyshev kernel pricer, value-only.
4. QLNet reference all-pillar DV01 by finite differences.
5. Schedule-resolved Chebyshev all-pillar DV01 and rate-coupon mixed terms by
   analytic discount-kernel derivatives.
6. Batch scenario pricing for a small deterministic scenario set.

## Product-Side Pilot Changes

Implement these before touching the core library:

- cache schedule templates together with their discount kernels;
- avoid per-cashflow dictionary lookups on the hot path;
- add a validated public entry point and an internal unchecked hot path;
- compute discount-kernel first derivatives analytically where the interpolation
  target is `exp(-t z(x))`;
- keep maturity finite differences explicit because maturity changes the
  cashflow schedule.

## ChebyshevSharp-Side Decision Gate

Only open or implement core-library optimization work if benchmarks show the
remaining bottleneck is the general Chebyshev evaluator. Candidate future
library items include allocation-free `ReadOnlySpan<double>` evaluation,
workspace-based derivative APIs, and batched evaluation helpers.

## Exit Gate

Phase 13 is complete when:

1. tests compare the optimized PV/risk path against the existing reference
   diagnostics;
2. BenchmarkDotNet reports value-only, risk-vector, and batch-scenario timings
   with managed-allocation columns;
3. the phase report states which optimizations worked and which did not;
4. the tracking issue is updated with benchmark evidence;
5. `dotnet test`, the fixed-rate bond example mode, `docfx`, `dotnet format`,
   and `git diff --check` pass locally;
6. one coherent Phase 13 PR is opened only after the local exit gate passes.

## Sources To Cite In The Report

- MoCaX research resources: <https://mocaxintelligence.com/research-resources/>
- BenchmarkDotNet diagnosers: <https://benchmarkdotnet.org/articles/configs/diagnosers.html>
- Tensor Train Cross error analysis: <https://arxiv.org/abs/2207.04327>
- Low-rank tensor approximation for Chebyshev interpolation: <https://arxiv.org/abs/1902.04367>
