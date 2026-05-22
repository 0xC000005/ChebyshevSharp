---
title: Fixed-Rate Bond Surrogate Case Study
---

# Fixed-Rate Bond Surrogate Case Study

This page is a worked technical case study: start with a trusted fixed-rate bond
pricing function, try to clone it with high-dimensional Chebyshev models, explain
where the naive models fail, and end with the first proof-of-concept method that
is accurate enough for the supported bond family.

The example is deliberately restricted and public. It uses QLNet as the
reference pricer, a pinned Federal Reserve nominal-yield-curve fixture, and a
regular fixed-rate bullet bond. It is not a general fixed-income library and it
does not claim to price arbitrary bond products.

## Product and Data Scope

The runnable harness lives in `examples/FixedRateBondSurrogate`.

| Item | Choice |
| --- | --- |
| Reference pricer | QLNet `FixedRateBond` with `DiscountingBondEngine` |
| Curve fixture | Federal Reserve fitted nominal yield curve, 2026-05-15 |
| Curve grid | valuation-date anchor plus 60 semiannual zero-rate pillars |
| Curve convention | Actual/365 Fixed timing, continuous compounding, linear zero-rate interpolation |
| Bond family | regular fixed-rate bullet bond |
| Coupon schedule | semiannual |
| Coupon day count | 30/360 USA |
| Calendar | U.S. Government Bond |
| Business-day rule | Modified Following |
| Settlement assumption | valuation date equals effective date |
| Excluded features | amortization, callability, ex-coupon logic, stubs, arbitrary settlement dates |

Run the reference-pricer baseline:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
```

The default 30Y, 4.5% coupon bond has dirty price `89.26423408` against the
upward-sloping fixture. This is economically plausible because the fitted 30Y
zero yield is around 5.33%, above the coupon.

## Public Wrapper

The surrogate wrapper exposed by the case study is:

```text
curve bumps[60], coupon, maturity -> dirty price per 100 notional
```

The 60 curve coordinates are basis-point bumps to the semiannual zero-rate
pillars from 0.5Y to 30Y. Coupon and maturity stay in the public wrapper because
the example is a parametric new-bond surface, not only a fixed-trade scenario
surface. Notional is handled algebraically: dirty price per 100 does not need a
Chebyshev coordinate, and cash value is obtained by multiplying by
`notional / 100`.

## Error Metrics

The tables below report held-out errors against the QLNet-backed reference
function. A small PV error alone is not enough for a risk use case, so the
benchmark also checks derivatives and mixed terms.

| Metric | Meaning |
| --- | --- |
| PV relative error | `abs(PV_model - PV_ref) / max(abs(PV_ref), floor)` over validation points. |
| DV01 | Finite-difference sensitivity of dirty price to a one-pillar zero-rate bump. The harness also checks the full 60-pillar DV01 vector. |
| Maturity sensitivity | Finite-difference slope of dirty price with respect to the maturity coordinate, using a 7-day step unless stated otherwise. |
| Coupon derivative | Finite-difference slope of dirty price with respect to coupon. |
| Rate-coupon mixed | Finite-difference estimate of `d^2 PV / (d rate_i d coupon)`. It checks whether coupon exposure changes correctly when rates move. |
| Rate-maturity mixed | Finite-difference estimate of `d^2 PV / (d rate_i d maturity)`. It checks whether rate exposure changes correctly when maturity moves. |
| Coupon-maturity mixed | Finite-difference estimate of `d^2 PV / (d coupon d maturity)`. It checks whether annuity-like coupon exposure changes correctly when maturity moves. |
| Rate-rate mixed | Finite-difference estimate of `d^2 PV / (d rate_i d rate_j)`. It should be small or localized where the discount factor depends on the same cashflow timing. |

Relative errors can look large when the reference derivative is near zero. For
that reason the final candidate is accepted mainly on absolute PV, all-pillar
DV01, and mixed-risk absolute errors.

## Why a Full Dense Tensor Is Impossible

A dense Chebyshev tensor with only three nodes per scalar coordinate would need
`3^62` source-function evaluations:

```text
381,520,424,476,945,831,628,649,898,809
```

This is why the first experiments use `ChebyshevTT` and `ChebyshevSlider`.
Both can sit behind the same 62-coordinate wrapper, but they make different
compression assumptions.

## Naive Global Clone Fails

The naive experiment asks what happens if a user directly clones the full bond
price with one global high-dimensional model:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery
```

| Model | Build evals | Max PV relative error | Max maturity-sensitivity relative error | Max coupon-maturity mixed relative error |
| --- | ---: | ---: | ---: | ---: |
| TensorTrain | 5,274 | 17.72% | 461.43% | 49.10% |
| Slider | 186 | 92.64% | 154.35% | 100.00% |

The TensorTrain probe is useful evidence, but this low-node global model does
not reproduce risk quantities reliably. The Slider result is also instructive:
with singleton partitions it is an anchored additive approximation, so
cross-group mixed derivatives are zero by construction. That explains the
100% coupon-maturity mixed relative error.

The structural sanity checks pass: a 10Y bond has zero direct sensitivity to an
unsupported 30Y zero-rate pillar in the reference and in the surrogate probes.
So the failure is not a simple post-maturity curve-exposure bug. The problem is
that one global smooth object is trying to learn schedule-sensitive behavior and
cross terms that it is not resolving well.

## Maturity Is Schedule Sensitive

For a fixed schedule, coupon is smooth and linear. Maturity is different because
changing maturity can regenerate the cashflow schedule. A small date move can
change coupon count, final accrual behavior, or business-day-adjusted payment
dates.

The harness scans one-day windows around semiannual maturity regions. Dirty PV
can look visually mild while finite-difference maturity sensitivity jumps:

![Maturity sensitivity near a semiannual schedule boundary](../images/fixed-rate-bond-maturity-sensitivity.svg)

Representative spike evidence:

| Maturity date | Future cashflows | Second difference | Left slope/year | Right slope/year |
| --- | ---: | ---: | ---: | ---: |
| 2039-11-11 | 28 | `7.339039E-003` | `-2.650493E+000` | `2.825619E-002` |
| 2040-11-10 | 30 | `7.191154E-003` | `-2.605554E+000` | `1.921722E-002` |
| 2038-05-15 | 25 | `6.116018E-003` | `-1.953303E+000` | `2.790432E-001` |

This is the practical meaning of piecewise smoothness in this case study. The
price can be continuous enough to look benign, while derivatives and mixed terms
remain poor targets for one global polynomial surrogate.

## Common Improvements Are Not Enough

The next experiment keeps the full public wrapper while trying common modelling
choices: a stronger global TT, a grouped Slider, low-dimensional curve factors,
and maturity buckets.

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
```

| Model | Max PV relative error on arbitrary 60-pillar bumps | Max maturity-sensitivity relative error | Max coupon-maturity mixed relative error | Max PV relative error on factor-aligned scenarios |
| --- | ---: | ---: | ---: | ---: |
| Stronger global TT | 12.36% | 256.28% | 80.99% | 11.85% |
| Grouped Slider | 8.48% | 327.65% | 75.20% | 5.85% |
| Curve-factor tensor | 4.70% | 90.42% | 59.04% | 0.59% |
| Semiannual bucketed curve-factor tensor | 4.73% | 87.72% | 56.94% | 0.58% |

The factor tensor is strong when validation scenarios are generated from the
same level/slope/curvature factor space. It is not a faithful clone of arbitrary
60-pillar bump vectors. This distinction matters: factor compression can be a
good risk workflow if the input contract is factor scenarios, but it is not a
drop-in replacement for all curve-bump inputs.

## Coupon Is Algebraic

For the supported fixed-rate bullet family, coupon cashflows are linear in the
coupon rate. With schedule fixed,

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

Run the check:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
```

The identity holds to numerical roundoff: max absolute error is
`8.526513E-014` across the validation bank. This explains why coupon should not
be treated as a difficult nonlinear Chebyshev axis for this restricted product.
It also gives a useful derivative identity:

```text
d^2 PV / (d rate_i d coupon) = d AnnuityPV / d rate_i
```

Coupon decomposition improves the modelling story, but it does not by itself
solve the maturity problem. A global decomposed TT still has `456.94%` max
maturity-sensitivity relative error in the benchmark.

## Schedule Routing Helps but Does Not Finish the Clone

The schedule-aware routing experiment uses maturity split points derived from
cashflow-schedule behavior:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

| Model | Max PV relative error | Max maturity-sensitivity relative error | Max coupon-maturity mixed relative error |
| --- | ---: | ---: | ---: |
| Global decomposed curve-factor tensor | 5.12% | 142.62% | 74.94% |
| Semiannual uniform bucketed tensor | 4.70% | 96.44% | 55.52% |
| Schedule-aware special-point tensor | 4.73% | 89.21% | 48.75% |
| Automatic-detector special-point tensor | 4.73% | 274.41% | 59.50% |
| Hybrid special-point tensor | 4.73% | 399.72% | 48.75% |

Schedule-aware routing improves the derivative-style metrics versus a uniform
bucket control, but it is still not a finished clone. Automatic split detection
also needs more evidence: more split points are not automatically better.

## Proof-of-Concept Method That Works

The successful proof of concept changes the modelling premise. It keeps the
same public wrapper, but it stops asking one global TT to rediscover the bond
pricing formula.

The implementation is
`examples/FixedRateBondSurrogate/ScheduleResolvedCashflowChebyshevBondPricer.cs`.
It does three things:

1. Resolve the maturity coordinate to the supported bond schedule and cache the
   future cashflow template.
2. Keep coupon and notional algebraic in the cashflow amount.
3. Use local Chebyshev kernels only for the smooth discount-factor pieces.

For a cashflow paid at time `t_k`, the direct zero curve uses linear
interpolation between adjacent zero-rate pillars. If the payment lies between
pillars `j` and `j+1`, then

```text
z_k(x) = (1 - w_k) * (z_j + 1e-4 * x_j)
       + w_k       * (z_{j+1} + 1e-4 * x_{j+1})

D_k(x) = exp(-t_k * z_k(x))
```

where `x_j` and `x_{j+1}` are basis-point bump coordinates. Therefore each
cashflow discount factor depends on at most two public curve coordinates. The
case study builds a 1D or 2D Chebyshev kernel for `D_k(x)` instead of building
one 62D object for the whole bond.

The dirty price per 100 notional is then

```text
Q(x, c, T) = 100 / N0 * sum_k (R_k(T) + c * A_k(T)) * D_k(x)
```

where `R_k(T)` is any principal redemption in the resolved schedule and
`A_k(T)` is the coupon multiplier produced by the unit-coupon schedule. This
formula explains why the method captures the important mixed terms:

```text
d^2 Q / (d x_j d c)
  = 100 / N0 * sum_k A_k(T) * d D_k(x) / d x_j
```

Maturity is no longer treated as a single smooth polynomial axis. It is a
schedule-routing input that chooses the cashflow template; the Chebyshev
approximation is only used after the smooth discount-kernel problem has been
isolated.

Run the final evidence mode:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --accuracy-recipe-search
```

| Result | Value |
| --- | ---: |
| Public wrapper dimension | 62 |
| Max internal kernel dimension | 2 |
| Validation points | 99 |
| Build evaluations | 39,699 |
| Measured evaluation speedup | 2.5x |
| Max PV absolute error | `1.348184E-010` |
| Max all-pillar DV01 absolute error | `4.263256E-010` |
| Max 10Y rate-coupon mixed absolute error | `2.842171E-006` |
| Max 10Y rate-maturity mixed absolute error | `1.112253E-008` |
| Max 10Y-10.5Y rate-rate mixed absolute error | `3.197442E-006` |
| Non-100 notional dirty-price max absolute error | `4.243361E-011` |

This is the first method in the case study that behaves like a practical clone
for the supported family: it accepts the full request-level wrapper, preserves
schedule-sensitive maturity behavior by resolving cashflows, captures coupon
and rate cross terms through the cashflow formula, rejects out-of-domain curve
bumps instead of silently clamping, and avoids modelling inactive post-maturity
curve pillars.

The tradeoff is important: this is a formula-aware surrogate for a supported
regular fixed-rate bullet family. It is not a blind black-box TT replacement for
arbitrary bond products.

## Practical Workflow

Use this case study as a workflow template:

1. Pick a trusted reference pricer and keep it behind a small adapter.
2. Pin market data fixtures so tests and docs are deterministic.
3. Define the public surrogate input contract before reducing dimensions
   internally.
4. Build the naive global surrogate first to expose failure modes.
5. Validate PV and risk quantities, not just interpolation error estimates.
6. Add structural checks that a risk user would expect, such as zero
   post-maturity pillar DV01.
7. Use domain structure only after the naive evidence is recorded.
8. State whether a result is a faithful clone, a factor-scenario model, or a
   research benchmark.

## Sources

Core references are listed in [Citations](citations.md), especially the sections
on Chebyshev interpolation, Tensor Train algorithms, sensitivity analysis,
piecewise smoothness, fixed-income baseline libraries, and public market data.
