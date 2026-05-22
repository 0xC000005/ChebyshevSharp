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
   pieces, analytic-coupon active-pillar pieces, and fixed-trade curve-only
   controls.

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

## Decision

Current working decision: separate factor-scenario accuracy from arbitrary
60-pillar clone accuracy before trying larger TT builds. The next candidate
should move to schedule-aware active-pillar models that keep the full public
wrapper but do not compress every shock into three factors. Factor compression
can remain a factor-scenario recipe, but it should not be presented as the
faithful arbitrary-pillar clone.

## Sources

- Chebfun, "Edge detection in Chebfun": <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- Federal Reserve Board, "H.15 Selected Interest Rates": <https://www.federalreserve.gov/releases/h15/>
- OpenGamma Strata, `FixedCouponBondTradeCalculations`: <https://strata.opengamma.io/apidocs/com/opengamma/strata/measure/bond/FixedCouponBondTradeCalculations.html>
- OpenGamma, "Strata and multi-curve calibration and bucketed PV01": <https://opengamma.com/strata-and-multi-curve-calibration-and-bucketed-pv01/>
- QuantLib Guide, "Cash-flow analysis": <https://www.quantlibguide.com/Cash-flow%20analysis.html>
- Qin et al., "Error Analysis of Tensor-Train Cross Approximation": <https://arxiv.org/abs/2207.04327>
- Gantner et al., "On the level-slope-curvature effect in yield curves": <https://ri.conicet.gov.ar/handle/11336/30774>
