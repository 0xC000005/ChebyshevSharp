# Phase 7 Report: Structured Alternatives

## Objective

Phase 7 asks whether common practitioner modelling choices improve the Phase 6
naive full-wrapper failure before introducing a new library feature or a bond-
specific analytical decomposition.

The public contract remains unchanged for every candidate:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

Each candidate is allowed to route, reorder, compress, or bucket internally, but
the validation points still arrive as 60 direct zero-rate bump coordinates, a
coupon, and a maturity. This keeps the experiment faithful to the goal of
cloning a reference fixed-rate bond pricer rather than solving a smaller
selected-pillar problem.

## Research Basis

The tested families mirror standard high-dimensional approximation practice:

- **Tensor Train:** use TT-Cross instead of a dense `n^d` grid when the sampled
  function has manageable numerical ranks.
- **Slider:** approximate a high-dimensional function as a sum of lower-
  dimensional slides around a pivot. Interacting variables must be grouped, or
  cross-group mixed terms are zero by construction.
- **Curve-factor compression:** project the full curve vector into a small
  level/slope/curvature factor space before fitting the Chebyshev object. This
  is a transparent PCA-style proxy, not a historical PCA calibration.
- **Piecewise maturity routing:** split a nonsmooth coordinate into smaller
  smooth pieces. Phase 6 found maturity-sensitivity spikes around semiannual
  schedule regions, so this phase tests 1Y and 0.5Y maturity buckets.

MoCaX public materials describe Chebyshev tensors, tensor extension algorithms,
and high-dimensional risk-factor acceleration as a model-space problem. The
Ruiz-Zeron reference text and PyChebyshev guides use the same vocabulary for
Chebyshev tensors, Tensor Train compression, Slider partitions, and special
points. Chebfun's guide and edge-detection examples provide the piecewise-smooth
precedent: global Chebyshev approximation works best on smooth pieces, while
nonsmooth coordinates should be split or detected.

## Candidate Configuration

| Candidate | Internal model | Wrapper dims | Internal dims | Buckets | Build evals |
| --- | --- | ---: | ---: | ---: | ---: |
| Stronger global TT | 62D TT-Cross, `n=4`, `maxRank=8`, `maxSweeps=5` | 62 | 62 | 1 | 21,569 |
| Auto-ordered global TT | 62D TT-Cross with one random auto-order trial | 62 | 62 | 1 | 4,272 |
| Grouped Slider | Slider with 5Y, 10Y, 20Y, 30Y, coupon, maturity in one slide | 62 | 62 | 1 | 897 |
| Curve-factor tensor | Dense 5D tensor over level/slope/curvature, coupon, maturity | 62 | 5 | 1 | 675 |
| Bucketed curve-factor tensor | Same 5D tensor routed through 1Y maturity buckets | 62 | 5 | 28 | 12,096 |
| Semiannual bucketed curve-factor tensor | Same 5D tensor routed through 0.5Y maturity buckets | 62 | 5 | 56 | 24,192 |

The curve-factor basis is deterministic:

```text
level      = 1
slope      = linearly increasing tenor coordinate
curvature  = quadratic tenor coordinate
```

It is deliberately used as a transparent stand-in for a first curve-factor
model. A true historical PCA version should pin a public scenario panel before
claiming explained variance.

## Validation Sets

Two held-out sets are reported:

- **Clone validation:** the 12 Phase 6 full-wrapper points, including flat
  bumps, alternating bumps, slope-shaped bumps, sinusoidal bumps, coupon
  extremes, and maturity-boundary points.
- **Factor-aligned validation:** 7 points generated from moderate
  level/slope/curvature coordinates and then expanded back to the same 62D
  wrapper. These test whether factor compression works when the input actually
  lies in the modelled factor space.

Reported finite-difference quantities remain the Phase 6 bank: PV, selected
zero-pillar DV01, coupon derivative, maturity sensitivity, rate-coupon mixed,
rate-maturity mixed, rate-rate mixed, and coupon-maturity mixed.

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
```

Measured on the dense Federal Reserve semiannual zero-curve fixture:

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity rel. error | Max factor-aligned PV rel. error |
| --- | ---: | ---: | ---: | ---: |
| Stronger global TT | 12.36% | 256.28% | 80.99% | 11.85% |
| Auto-ordered global TT | 17.72% | 461.43% | 49.10% | 8.30% |
| Grouped Slider | 8.48% | 327.65% | 75.20% | 5.85% |
| Curve-factor tensor | 4.70% | 90.42% | 59.04% | 0.59% |
| Bucketed curve-factor tensor | 4.73% | 128.96% | 59.76% | 0.60% |
| Semiannual bucketed curve-factor tensor | 4.73% | 87.72% | 56.94% | 0.58% |

## Interpretation

The stronger global TT improves the naive Phase 6 PV error, but it does not make
the clone reliable: maturity sensitivity and coupon-maturity mixed error remain
large. Increasing global effort alone is therefore not a satisfying answer.

The auto-order trial did not improve the result in this run. Its retained order
started with the canonical dimension order, so Phase 7 does not have evidence
that ordering alone fixes the bond-pricer structure.

The grouped Slider is useful as a contrast case. Grouping coupon and maturity
with the major long-tenor pillars improves PV versus the singleton Slider from
Phase 6, but the remaining maturity and mixed-term errors are still too large.
This matches the Slider model assumption: only interactions inside a slide are
represented.

The curve-factor tensor is the best common-practice candidate tested here. It
reduces max clone PV error to `4.70%` and reaches `0.59%` max PV error on
factor-aligned points. That is strong evidence that model-space compression is
useful when the production scenario set is factor-like. It is also evidence that
arbitrary 60-pillar direct-bump cloning still carries projection error.

The 1Y and 0.5Y bucketed factor tensors do not materially improve PV. The 0.5Y
bucket does reduce maturity-sensitivity error relative to the 1Y bucket and the
global factor tensor, but the derivative and mixed-term errors remain too high
for a faithful risk clone. Maturity routing helps but is not sufficient in this
simple factor-space form.

## Decision

Phase 7 exhausts the first common-practice alternatives enough to guide the next
phase:

1. Do not keep trying larger global full-wrapper TT builds as the main route.
2. Keep factor-space compression as a valid tutorial branch, but label it as a
   factor-scenario surrogate, not an arbitrary 60-pillar clone.
3. Keep Slider as an explanatory contrast and for defended local partitions, not
   as the default full-pricer clone for cross Greeks.
4. Move the next design phase toward either high-dimensional piecewise support
   for selected nonsmooth coordinates, true special-point/edge-detected routing,
   or the analytical coupon decomposition already justified by the fixed-rate
   bond identity.

## Verification

Fresh local checks:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
dotnet format --verify-no-changes --verbosity minimal
dotnet test --filter "FullyQualifiedName~FixedRateBondStructuredAlternativesTests"
dotnet test --filter "FullyQualifiedName~FixedRateBond"
dotnet build --configuration Release --no-restore
docfx docs/docfx.json
git diff --check
dotnet test --configuration Release --no-build --verbosity minimal
```

Results:

- structured-alternatives example completed successfully and printed the table
  above;
- formatting check exited with no changes;
- focused Phase 7 tests passed `3` tests with `0` failures;
- fixed-rate bond test slice passed `59` tests with `0` failures;
- Release build succeeded with `0` warnings and `0` errors;
- DocFX built successfully with `0` warnings and `0` errors;
- `git diff --check` reported no whitespace errors;
- Release test suite passed `1708` tests with `0` failures.

Build times vary slightly by machine load; the evaluation counts are
deterministic.

## Sources

- MoCaX Intelligence, "Research & Resources": <https://mocaxintelligence.com/research-resources/>
- Ruiz, I. and Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View*. Wiley Finance.
- Chebfun project, "First steps in Chebfun": <https://www.chebfun.org/docs/guide/guide01.html>
- Chebfun project, "Edge detection in Chebfun": <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- Phase 6 report: [Naive Dense-Baseline Surrogate Discovery](phase-6-naive-surrogate-discovery.md)
