---
title: Fixed-Rate Bond Surrogate Case Study
---

# Fixed-Rate Bond Surrogate Case Study

This case study shows how to use ChebyshevSharp to approximate an expensive
fixed-rate bond pricing function, and why a naive high-dimensional clone is not
enough for risk sensitivities.

The example is deliberately public and restricted. It uses QLNet as the
reference pricer, a pinned Federal Reserve nominal-yield-curve fixture, and a
regular fixed-rate bullet bond. It is not a general fixed-income library and it
is not a replacement for arbitrary bond products.

## Problem Setup

The runnable harness lives in `examples/FixedRateBondSurrogate`. The reference
function maps a curve scenario and product parameters to dirty PV:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

The 60 curve coordinates are semiannual direct zero-rate bumps from 0.5Y to
30Y. The valuation-date anchor is part of the QLNet curve object but is not a
Chebyshev input coordinate. Notional is kept outside the surrogate because the
restricted bond family scales linearly with notional.

The default baseline uses:

| Item | Choice |
| --- | --- |
| Pricer | QLNet `FixedRateBond` with `DiscountingBondEngine` |
| Curve fixture | Federal Reserve fitted nominal yield curve, 2026-05-15 |
| Curve points | valuation-date anchor plus 60 semiannual zero-rate pillars |
| Coupon schedule | semiannual |
| Coupon day count | 30/360 USA |
| Calendar | U.S. Government Bond |
| Business-day rule | Modified Following |
| Curve day count | Actual/365 Fixed |
| Curve interpolation | linear zero-rate interpolation |
| Curve compounding | continuous annual |
| Settlement assumption | valuation date equals effective date |

The default 30Y, 4.5% coupon bond has dirty price `89.26423408` against this
upward-sloping curve. That is economically sensible because the fitted 30Y zero
yield is around 5.33%.

Run the baseline:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
```

## Why Not Build the Full Tensor?

A dense Chebyshev tensor with even three nodes per scalar coordinate would need
`3^62` source-function evaluations:

```text
381,520,424,476,945,831,628,649,898,809
```

That is the reason the case study starts with high-dimensional approximations:
`ChebyshevTT` and `ChebyshevSlider`. Both can be called through the same
62-coordinate wrapper, but they make different modelling assumptions.

## Naive Global TT and Slider

The first experiment asks a deliberately simple question: what happens if a new
user tries to clone the full bond price directly?

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery
```

The result is the motivating failure bank:

| Model | Build evals | Max PV rel. error | Max maturity-sensitivity rel. error | Max coupon-maturity mixed rel. error |
| --- | ---: | ---: | ---: | ---: |
| TensorTrain | 5,274 | 17.72% | 461.43% | 49.10% |
| Slider | 186 | 92.64% | 154.35% | 100.00% |

The TensorTrain is a valid global TT-Cross probe, but the low-node global model
does not reproduce the risk quantities reliably. The Slider is also a valid
62D model, but the singleton partition is an anchored additive approximation.
Cross-group mixed derivatives are zero by construction, so it cannot represent
coupon-maturity or rate-maturity interactions unless the interacting variables
are placed in the same slide.

The structural sanity checks are important. A 10Y bond has zero direct
sensitivity to an unsupported 30Y zero-rate pillar in both the baseline and the
surrogates. That means the failure is not a simple post-maturity exposure bug;
it is a modelling error in PV, maturity sensitivity, coupon sensitivity, and
mixed terms at harder points.

## Maturity Is the Hard Axis

For fixed curve and fixed maturity schedule, coupon is smooth and linear.
Maturity is different because changing maturity can regenerate the cashflow
schedule. A small date move can change coupon count, final accrual behavior, or
business-day-adjusted payment dates.

The harness scans one-day windows around semiannual maturity regions. In the
largest Phase 6 window, dirty PV looks visually mild, while the finite
difference maturity sensitivity moves abruptly:

![Maturity sensitivity near a semiannual schedule boundary](../research/fixed-rate-bond-surrogate/images/phase-6-maturity-sensitivity.svg)

Representative spike evidence:

| Maturity date | Cashflows | Second difference | Left slope/year | Right slope/year |
| --- | ---: | ---: | ---: | ---: |
| 2039-11-11 | 28 | `7.339039E-003` | `-2.650493E+000` | `2.825619E-002` |
| 2040-11-10 | 30 | `7.191154E-003` | `-2.605554E+000` | `1.921722E-002` |
| 2038-05-15 | 25 | `6.116018E-003` | `-1.953303E+000` | `2.790432E-001` |

This is the practical meaning of piecewise smoothness in the bond example: the
price can be continuous enough to look benign, while derivatives and mixed terms
are poor targets for one global polynomial surrogate.

## Common Improvements

The next experiment keeps the full public wrapper but tests common modelling
choices:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
```

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity mixed rel. error | Max factor-aligned PV rel. error |
| --- | ---: | ---: | ---: | ---: |
| Stronger global TT | 12.36% | 256.28% | 80.99% | 11.85% |
| Grouped Slider | 8.48% | 327.65% | 75.20% | 5.85% |
| Curve-factor tensor | 4.70% | 90.42% | 59.04% | 0.59% |
| Semiannual bucketed curve-factor tensor | 4.73% | 87.72% | 56.94% | 0.58% |

The important distinction is **clone accuracy** versus **factor-scenario
accuracy**. A level/slope/curvature factor tensor is strong when validation
points are generated from that same factor space. It is not a faithful clone of
arbitrary 60-pillar bump vectors. This is a useful tutorial result because many
real risk workflows are factor-like, but the approximation contract must say so.

## Coupon Decomposition

For this restricted regular fixed-rate bullet bond, coupon should not be a
Chebyshev coordinate. The baseline confirms the identity:

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

Run the check:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
```

The exact decomposition oracle has max absolute error `8.526513E-014` across
the validation bank. This is numerical roundoff, so the identity is valid for
the restricted product family.

This improves the model design:

- coupon delta becomes the annuity surface;
- rate-coupon mixed terms become first derivatives of the annuity surface;
- the internal Chebyshev models drop one scalar coordinate.

It does not solve maturity nonsmoothness or arbitrary curve-bump projection
error. A global decomposed TT still has `456.94%` max maturity-sensitivity
relative error in the Phase 8 benchmark.

## Schedule-Aware Routing

Piecewise routing is the natural next idea. The maturity-special-points
experiment compares uniform buckets, schedule-derived split points, automatic
second-difference candidates, and a hybrid set:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity mixed rel. error |
| --- | ---: | ---: | ---: |
| Global decomposed curve-factor tensor | 5.12% | 142.62% | 74.94% |
| Semiannual uniform bucketed tensor | 4.70% | 96.44% | 55.52% |
| Schedule-aware special-point tensor | 4.73% | 89.21% | 48.75% |
| Automatic-detector special-point tensor | 4.73% | 274.41% | 59.50% |
| Hybrid special-point tensor | 4.73% | 399.72% | 48.75% |

Schedule-aware routing is better than the uniform bucket control on the two
remaining derivative-style metrics, but it is not a finished risk clone.
Automatic detection is also not ready: more split points are not automatically
better.

The explicit router benchmark turns the schedule-aware split points into a
half-open piece router:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

It preserves the full wrapper and makes dispatch semantics auditable, but it
reproduces the Phase 9 metrics rather than improving them:

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity mixed rel. error |
| --- | ---: | ---: | ---: |
| Schedule-aware router decomposed factor tensor | 4.73% | 89.21% | 48.75% |

The Phase 10 conclusion is therefore conservative: keep the router example-local
for now. The evidence supports a future schedule-aware high-dimensional
piecewise design discussion, not a generic automatic kink-detection API.

## Practical Workflow

Use the case study as a workflow template:

1. Pick a trusted reference pricer and keep it behind a small adapter.
2. Pin market data fixtures so CI and docs are deterministic.
3. Define the public surrogate input contract before reducing dimensions
   internally.
4. Build the naive global surrogate first to expose failure modes.
5. Validate PV and risk quantities, not just interpolation error estimates.
6. Add structural sanity checks that a domain user would expect.
7. Use domain structure only after the naive evidence is recorded.
8. State whether a result is a faithful clone, a factor-scenario model, or a
   research benchmark.

For this bond example, the current best lesson is not "TT fixes the pricer." The
lesson is that high-dimensional Chebyshev modelling works only when the input
contract, product structure, sensitivity definitions, and smoothness assumptions
are explicit.

## Sources and Reports

Core public references are listed in [Citations](citations.md), especially the
sections on Tensor Train algorithms, sensitivity analysis, piecewise smoothness,
fixed-income baseline libraries, and public market data.

Detailed phase reports:

- [Phase 5: Realistic Baseline](../research/fixed-rate-bond-surrogate/reports/phase-5-realistic-baseline.md)
- [Phase 6: Naive Dense-Baseline Surrogate Discovery](../research/fixed-rate-bond-surrogate/reports/phase-6-naive-surrogate-discovery.md)
- [Phase 7: Structured Alternatives](../research/fixed-rate-bond-surrogate/reports/phase-7-structured-alternatives.md)
- [Phase 8: Analytic Coupon Decomposition](../research/fixed-rate-bond-surrogate/reports/phase-8-analytic-coupon-decomposition.md)
- [Phase 9: Maturity Special Points](../research/fixed-rate-bond-surrogate/reports/phase-9-maturity-special-points.md)
- [Phase 10: Schedule-Aware Piecewise Router](../research/fixed-rate-bond-surrogate/reports/phase-10-schedule-aware-router.md)
