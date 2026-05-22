# Phase 12 Report: Accuracy Recipe Search

## Objective

Phase 12 investigates why the fixed-rate bond surrogate still has material
accuracy error after schedule-aware routing. The public wrapper remains:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

The phase is not a public API phase. It is a controlled search for the next
defensible modelling recipe.

## Starting Evidence

The Phase 10 router made maturity dispatch auditable but did not improve the
residual numerical clone:

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity rel. error |
| --- | ---: | ---: | ---: |
| Schedule-aware router decomposed factor tensor | `4.73%` | `89.21%` | `48.75%` |

The likely remaining causes are curve-factor projection error, local piece
resolution, finite-difference derivative measurement, or the broader requirement
to clone arbitrary 60-pillar bumps.

## Related Work

Chebfun's edge-detection example supports splitting piecewise-smooth functions
into smooth pieces. OpenGamma Strata documents present value and bucketed PV01
as core fixed-income outputs, which matches this phase's validation targets.
Federal Reserve H.15 describes constant-maturity Treasury yields as values read
from an interpolated public curve. Yield-curve PCA references support
level/slope/curvature as common compression factors, but Phase 12 treats that as
a hypothesis to test, not as a guarantee. Tensor Train Cross references support
compressed high-dimensional sampling only when held-out validation confirms the
sampled function is compressible enough.

## Planned Diagnostics

1. Projection oracle: measure how much error comes from projecting arbitrary
   60-pillar shocks into a lower-dimensional factor curve.
2. Derivative oracle: measure whether DV01 and maturity-slope errors are stable
   under finite-difference step changes.
3. Schedule oracle: confirm that piece dispatch is not the hidden source of the
   Phase 10 residual error.
4. Stronger candidates: compare richer factor tensors, active-pillar local TT
   pieces, fixed-trade curve-only controls, and a formula-aware full-wrapper
   cashflow-kernel Chebyshev model.

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --accuracy-recipe-search
```

Initial oracle results:

| Diagnostic | Result |
| --- | ---: |
| Clone validation points | `12` |
| Factor-aligned validation points | `7` |
| Projection oracle, clone max PV abs. error | `4.467694E+000` |
| Projection oracle, clone max PV rel. error | `4.73%` |
| Projection oracle, factor-aligned max PV abs. error | `0.000000E+000` |
| Projection oracle, factor-aligned max PV rel. error | `0.00%` |
| Five-factor deterministic basis, clone max PV abs. error | `5.291768E+000` |
| Five-factor deterministic basis, clone max PV rel. error | `4.81%` |
| 10Y pillar DV01, rate step `1e-4` | `-6.363631E-002` |
| 10Y pillar DV01, rate step `5e-5` | `-6.363631E-002` |
| 10Y pillar DV01, rate step `1e-5` | `-6.363632E-002` |
| 1-day maturity central slope | `-7.757807E-001` |
| 3-day maturity central slope | `-6.556768E-001` |
| 7-day maturity central slope | `-6.005639E-001` |
| Post-maturity unsupported 30Y pillar DV01 | `0.000000E+000` |
| Active-support oracle max PV abs. error | `0.000000E+000` |
| Active-support oracle max PV rel. error | `0.00%` |
| Active-support min active curve-bump dimensions | `5` |
| Active-support max active curve-bump dimensions | `60` |
| 10Y active-pillar TT internal dimensions | `23` |
| 10Y active-pillar TT build evaluations | `2014` |
| 10Y active-pillar TT max PV abs. error | `1.650403E+000` |
| 10Y active-pillar TT max PV rel. error | `1.53%` |
| 10Y active-pillar TT max 10Y DV01 rel. error | `9.83%` |
| 10Y active-pillar TT max coupon derivative rel. error | `3.66%` |
| 10Y active-pillar TT max maturity sensitivity rel. error | `132.88%` |
| 10Y active-pillar TT max coupon-maturity mixed rel. error | `120.33%` |
| 10Y narrow active-pillar TT internal dimensions | `23` |
| 10Y narrow active-pillar TT build evaluations | `3566` |
| 10Y narrow active-pillar TT max PV rel. error | `0.48%` |
| 10Y narrow active-pillar TT max 10Y DV01 rel. error | `1.03%` |
| 10Y narrow active-pillar TT max coupon derivative rel. error | `1.03%` |
| 10Y narrow active-pillar TT max maturity sensitivity rel. error | `161.90%` |
| 10Y narrow active-pillar TT max maturity left-slope rel. error | `1513.27%` |
| 10Y narrow active-pillar TT max maturity right-slope rel. error | `892.27%` |
| 10Y narrow active-pillar TT max coupon-maturity mixed rel. error | `10.98%` |
| 10Y fixed-trade curve-only TT internal dimensions | `21` |
| 10Y fixed-trade curve-only TT build evaluations | `837` |
| 10Y fixed-trade curve-only TT max PV abs. error | `1.796590E-004` |
| 10Y fixed-trade curve-only TT max PV rel. error | `0.00%` |
| 10Y fixed-trade curve-only TT max 10Y DV01 rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels internal dimensions | `2` |
| Schedule-resolved cashflow Chebyshev kernels build evaluations | `39699` |
| Schedule-resolved cashflow Chebyshev kernels validation points | `99` |
| Schedule-resolved cashflow Chebyshev kernels measured eval speedup | `2.5x` |
| Schedule-resolved cashflow Chebyshev kernels max PV abs. error | `1.348184E-010` |
| Schedule-resolved cashflow Chebyshev kernels max PV rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max all-pillar DV01 abs. error | `4.263256E-010` |
| Schedule-resolved cashflow Chebyshev kernels max all-pillar DV01 rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max 10Y DV01 rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max coupon derivative rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max maturity sensitivity rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max 10Y rate-coupon mixed abs. error | `2.842171E-006` |
| Schedule-resolved cashflow Chebyshev kernels max 10Y rate-coupon mixed rel. error | `0.01%` |
| Schedule-resolved cashflow Chebyshev kernels max 10Y rate-maturity mixed abs. error | `1.112253E-008` |
| Schedule-resolved cashflow Chebyshev kernels max coupon-maturity mixed rel. error | `0.00%` |
| Schedule-resolved cashflow Chebyshev kernels max 10Y-10.5Y rate-rate mixed abs. error | `3.197442E-006` |
| Non-100 notional dirty-price max abs. error | `4.243361E-011` |

## Interpretation

The first oracle result is already informative. The deterministic
level/slope/curvature projection creates a `4.73%` max PV error on the same
arbitrary clone validation bank used by the previous phases, while the
factor-aligned points reconstruct exactly. That means the Phase 10 factor-router
error floor can be explained before adding more TT complexity: the current
factor representation is not a faithful clone of arbitrary 60-pillar shocks.

The first richer deterministic basis check does not fix this. A five-factor
polynomial basis increases the current max clone PV error slightly to `4.81%`.
That does not prove every richer basis is unhelpful, but it does show that
adding a few smooth factors blindly is not the missing recipe.

The rate-step derivative diagnostic is stable for the tested 10Y pillar, so the
initial evidence does not point to rate finite-difference step size as the main
problem. Maturity central differences are materially step-dependent, which is
consistent with the schedule-boundary evidence from earlier phases. The
post-maturity 30Y pillar DV01 remains numerically zero for a 10Y bond, so the
current baseline and wrapper preserve the expected curve-support sanity check.

The active-support oracle is exact on the current validation bank when curve
bumps after the maturity neighbourhood are removed. Active curve dimensions
range from `5` to `60`, depending on maturity. This supports a schedule-aware
active-pillar recipe: it can preserve the full public wrapper while each local
piece avoids modelling provably inactive post-maturity pillars. The long end
still reaches all `60` curve-bump dimensions, so active support is necessary but
not by itself a low-dimensional recipe for every maturity.

The first active-pillar TT candidate is a 10Y local piece with `23` internal
coordinates: active curve pillars through the maturity neighbourhood, coupon,
and maturity. It reduces local max PV relative error to `1.53%` with `2014`
build evaluations. That is a meaningful improvement over the factor projection
floor. However, the derivative surface is still not acceptable: the same
candidate reports `9.83%` max 10Y DV01 relative error, `132.88%` max maturity
sensitivity relative error, and `120.33%` max coupon-maturity mixed relative
error. Active support fixes the inactive-pillar problem and improves PV, but it
does not by itself solve the slope/cross-term problem.

Narrowing the 10Y active-pillar piece to `[9.95Y, 10.05Y]` and increasing the
coupon/maturity resolution improves several outputs: max PV relative error falls
to `0.48%`, max 10Y DV01 relative error falls to `1.03%`, and max
coupon-maturity mixed relative error falls to `10.98%`. The remaining blocker is
maturity sensitivity itself, which stays very large at `161.90%`. This suggests
the active-pillar recipe is promising for PV and rate/coupon risk, but maturity
derivatives need special handling instead of being treated as ordinary smooth
central differences.

One-sided maturity slopes do not remove the problem in this first candidate.
For the narrow 10Y active-pillar TT, max left-slope and right-slope relative
errors are `1513.27%` and `892.27%`. The absolute one-sided slope errors are
also material in the CLI output, so this is not only a denominator artifact. The
current evidence says the maturity coordinate is still the hard axis.

The fixed-trade curve-only control is the clearest positive result in this
phase. With coupon and maturity fixed and only active curve pillars varied, the
10Y TT uses `21` internal dimensions, `837` build evaluations, and reaches
`1.796590E-004` max PV absolute error with effectively zero displayed PV and
10Y DV01 relative error. This supports a practical recipe for portfolio/risk
use cases where trades are fixed and repeated scenarios vary the curve. It does
not solve the separate problem of a parametric new-bond surface over maturity.

The schedule-resolved cashflow Chebyshev-kernel candidate changes the modelling
premise. It keeps the public `curve bumps[60], coupon, maturity` wrapper, but it
does not ask a single TT to rediscover the bond formula. Instead, it resolves
the maturity date to the corresponding coupon/redemption cashflows, keeps coupon
linearity in the cashflow amount, and prices each cashflow through a local
Chebyshev discount kernel. Because linear zero-rate interpolation makes a
single cashflow discount factor depend on only one or two adjacent curve
pillars, each local kernel is at most 2D. On a broadened 99-point full-wrapper
validation bank spanning coupons, maturities, parallel shifts, slopes, sinusoidal
shocks, and local 10Y bumps, this candidate reports `1.348184E-010` max PV
absolute error, `4.263256E-010` max all-pillar DV01 absolute error, and `2.5x` measured
evaluation speedup over the QLNet baseline path after schedules/kernels are
cached. A separate non-100 notional check at `250` notional reports
`4.243361E-011` max dirty-price absolute error, which confirms that notional is
handled algebraically rather than becoming a hidden Chebyshev coordinate.
Additional mixed-risk diagnostics remain small in absolute terms:
`2.842171E-006` for the 10Y rate-coupon mixed term, `1.112253E-008` for the 10Y
rate-maturity mixed term, and `3.197442E-006` for the 10Y-10.5Y rate-rate mixed
term. The rate-rate relative error is intentionally not used as the acceptance
criterion because the baseline mixed term is near zero on many validation
points.

This is the first candidate in Phase 12 that satisfies the intended replacement
shape: it accepts the full wrapper, preserves schedule-sensitive maturity
finite differences by routing through resolved cashflows, captures coupon/rate
cross terms through the coupon-weighted cashflow amount, and avoids modelling
inactive post-maturity curve pillars. The implementation uses a small
allocation-free barycentric Chebyshev evaluator for each discount kernel rather
than the general-purpose dense tensor evaluator on every cashflow. The tradeoff
is that it is formula-aware. It is a correct recipe for the supported
fixed-coupon bond family, not a blind black-box surrogate for arbitrary
products.

The candidate is now exposed as
`ScheduleResolvedCashflowChebyshevBondPricer` in the fixed-rate bond example.
That class can price a full `FixedRateBondRequest` by mapping the request back
to the public 62-coordinate wrapper, and the request-level test matches the
QLNet reference dirty price to 8 decimal places under a non-flat 60-pillar bump
shape, changed coupon, changed maturity, and non-100 notional. It also rejects
incompatible dates, curve pillar layouts, and curve bumps outside the supported
`[-150, 150]` bp domain, so the example does not rely on silent clamping.

## Decision

Current working decision: factor compression can remain a factor-scenario
recipe, but it should not be presented as the faithful arbitrary-pillar clone.
The leading full-wrapper replacement recipe is now schedule-resolved cashflow
decomposition plus local Chebyshev discount kernels. This keeps maturity as a
schedule-routing input instead of an ordinary smooth Chebyshev axis, keeps
coupon/notional algebraic, and uses low-dimensional Chebyshev tensors only for
the smooth discount-factor kernels. The next Phase 12 task is to broaden this
candidate's validation bank and decide whether any additional supported-family
eligibility checks are needed before presenting this as the recommended
bond-pricer acceleration pattern.

## Sources

- Chebfun, "Edge detection in Chebfun": <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- Federal Reserve Board, "H.15 Selected Interest Rates": <https://www.federalreserve.gov/releases/h15/>
- OpenGamma Strata, `FixedCouponBond`: <https://strata.opengamma.io/apidocs/com/opengamma/strata/product/bond/FixedCouponBond.html>
- OpenGamma Strata, `FixedCouponBondTradeCalculations`: <https://strata.opengamma.io/apidocs/com/opengamma/strata/measure/bond/FixedCouponBondTradeCalculations.html>
- OpenGamma, "Strata and multi-curve calibration and bucketed PV01": <https://opengamma.com/strata-and-multi-curve-calibration-and-bucketed-pv01/>
- QuantLib Guide, "Cash-flow analysis": <https://www.quantlibguide.com/Cash-flow%20analysis.html>
- Qin et al., "Error Analysis of Tensor-Train Cross Approximation": <https://arxiv.org/abs/2207.04327>
- Gantner et al., "On the level-slope-curvature effect in yield curves": <https://ri.conicet.gov.ar/handle/11336/30774>
