# Phase 13 Report: Accuracy And Hot-Path Speed Pilot

## Objective

Phase 13 tests whether the Phase 12 schedule-resolved cashflow Chebyshev
recipe can be both risk-accurate and faster than the QLNet-backed reference
path. The public wrapper remains:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

This phase does not turn the example into a general fixed-income library. It
benchmarks the supported regular fixed-rate bullet family and decides where the
next optimization work should live.

## Related Work

MoCaX's research materials frame Chebyshev tensors as tools for repeated risk
and dynamic-sensitivity calculations, where replacing many pricing-function
calls can produce large computational gains. BenchmarkDotNet's documentation
recommends diagnosers such as `MemoryDiagnoser` for allocation and GC evidence.
Tensor Train Cross error analysis reinforces the earlier workflow rule: a
sampled high-dimensional model is only useful after held-out validation, so
speed is not evaluated without the Phase 12 accuracy gates.

## Implementation Changes

The schedule-resolved pricer now caches each resolved cashflow template together
with its local discount kernel. The previous hot path resolved the template and
then performed a dictionary lookup for the discount kernel on every cashflow.
The new hot path stores the kernel directly on the cashflow component.

The pricer also exposes a risk snapshot path:

```csharp
double dirtyPrice = model.EvalRisk(
    fullPoint,
    curveBumpGradient,
    rateCouponMixed,
    out double couponDerivative);
```

This computes dirty price, all-pillar curve-bump gradients, coupon derivative,
and all-pillar rate-coupon mixed terms in one pass. The derivative formula comes
from the local discount kernel. For a payment between curve pillars `j` and
`j+1`,

$$
D_k(x)=\exp(-t_k z_x(t_k)),
$$

so

$$
\frac{\partial D_k}{\partial x_j}
  = -t_k(1-w_k)10^{-4}D_k(x),
\qquad
\frac{\partial D_k}{\partial x_{j+1}}
  = -t_k w_k10^{-4}D_k(x).
$$

Therefore

$$
\frac{\partial Q}{\partial x_j}
  = \frac{100}{N_0}
    \sum_k \left(R_k+cA_k\right)
    \frac{\partial D_k}{\partial x_j},
$$

and

$$
\frac{\partial^2 Q}{\partial x_j\,\partial c}
  = \frac{100}{N_0}
    \sum_k A_k
    \frac{\partial D_k}{\partial x_j}.
$$

Maturity sensitivity remains a finite-difference quantity because changing
maturity can change the resolved cashflow schedule.

## Benchmark Results

Command:

```bash
dotnet run --project benchmarks/ChebyshevSharp.Benchmarks/ChebyshevSharp.Benchmarks.csproj \
  -c Release -- --filter '*FixedRateBondSurrogateBenchmarks*' \
  --job Short --warmupCount 1 --iterationCount 3
```

Environment: BenchmarkDotNet `0.15.8`, .NET `10.0.7`, Ubuntu `24.04.4`, Intel
Core i7-12700K. This was a short directional run, not a final publication-grade
benchmark.

| Method | Mean | Allocated | Interpretation |
| --- | ---: | ---: | --- |
| QLNet value only | `11,500.0 ns` | `28,321 B` | Baseline reference path. |
| Cached exact cashflow value only | `108.4 ns` | `0 B` | Specialized exact fixed-schedule control. |
| Chebyshev kernel value only | `1,520.5 ns` | `0 B` | Safe public Chebyshev evaluation path. |
| Chebyshev kernel value only, unchecked | `1,489.9 ns` | `0 B` | Hot path after caller-side validation. |
| QLNet all-pillar DV01 by finite difference | `1,382,441.2 ns` | `3,464,729 B` | 120 reference-pricer calls. |
| Chebyshev all-pillar risk analytically | `1,625.6 ns` | `32 B` | PV, all-pillar gradient, coupon derivative, and rate-coupon mixed terms. |
| QLNet batch 32 value only | `529,871.1 ns` | `1,150,518 B` | 32 reference-pricer calls. |
| Chebyshev batch 32 value only | `75,150.8 ns` | `1,024 B` | 32 cached-kernel evaluations. |

Derived ratios:

| Comparison | Speedup |
| --- | ---: |
| Chebyshev scalar value vs QLNet scalar value | `7.6x` |
| Chebyshev unchecked scalar value vs QLNet scalar value | `7.7x` |
| Chebyshev analytic all-pillar risk vs QLNet finite-difference all-pillar DV01 | `850.4x` |
| Chebyshev batch 32 scalar values vs QLNet batch 32 scalar values | `7.1x` |
| Cached exact fixed-schedule value vs QLNet scalar value | `106.1x` |

The CLI accuracy harness also improved after removing per-cashflow dictionary
lookups:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj \
  -c Release -- --accuracy-recipe-search
```

It now reports `9.1x` measured scalar evaluation speedup for the
schedule-resolved cashflow Chebyshev kernels, while preserving the Phase 12
roundoff-level PV and risk errors.

## Interpretation

The accurate formula-aware Chebyshev clone now provides a useful scalar speedup
against the QLNet reference path, but the scalar value-only result is not yet
the `10x` to `100x` target. The largest demonstrated win is the risk-vector
path: computing all-pillar curve risk and rate-coupon mixed terms analytically
through the resolved discount kernels avoids dozens of reference-pricer finite
differences and reaches about `850x` speedup in the short benchmark.

The cached exact cashflow control is critical. It prices the same fixed schedule
in about `108 ns`, much faster than the Chebyshev interpolation kernel. That
means the immediate bottleneck is not QLNet alone; for this direct-zero fixed
bond, once the schedule is known, exact cashflow summation is extremely cheap.
Chebyshev is valuable here primarily as a public demonstration of decomposition
and as a route to fast smooth kernels and risk snapshots. For production scalar
fixed-bond pricing, an exact cached pricer may be the stronger baseline.

The next optimization work should therefore be staged:

1. keep the schedule-resolved Chebyshev recipe as the accurate case-study clone;
2. present the risk-vector speedup as the main acceleration result;
3. benchmark an exact cached fixed-rate bond pricer as a serious baseline in
   any future production comparison;
4. only add core ChebyshevSharp hot-path APIs after profiling a broader class of
   expensive smooth kernels where interpolation, not schedule resolution, is the
   bottleneck.

## Verification

- Focused risk tests:
  `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondScheduleResolvedRiskTests" --verbosity minimal`
  passed `2` tests.
- Existing Phase 12 accuracy tests:
  `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondAccuracyRecipeSearchTests" --verbosity minimal`
  passed `12` tests.
- Accuracy recipe CLI completed and reported `9.1x` schedule-resolved scalar
  speedup with the existing `99`-point validation bank.
- BenchmarkDotNet short run completed all `8` fixed-rate bond benchmark methods.

## Sources

- MoCaX research resources: <https://mocaxintelligence.com/research-resources/>
- BenchmarkDotNet diagnosers: <https://benchmarkdotnet.org/articles/configs/diagnosers.html>
- Tensor Train Cross error analysis: <https://arxiv.org/abs/2207.04327>
