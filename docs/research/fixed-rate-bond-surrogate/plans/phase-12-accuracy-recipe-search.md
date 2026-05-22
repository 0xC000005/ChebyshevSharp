# Phase 12 Accuracy Recipe Search Plan

> **For agentic workers:** Phase 12 is an evidence-gathering phase. Preserve the
> public wrapper `curve bumps[60], coupon, maturity -> dirty PV`. Do not add a
> reusable ChebyshevSharp API, bump versions, or publish packages in this phase.

**Goal:** Identify the dominant reason the fixed-rate bond surrogate is still
not accurate enough, then choose the next defensible modelling recipe.

## Why This Phase Exists

Phases 6-10 proved that a naive global clone is not enough, but they did not
produce a finished recipe. The best current routed factor model still reports
material sensitivity error. Phase 12 therefore starts with failure isolation
before adding more modelling complexity.

Current evidence points to four possible error sources:

1. **Curve projection error:** level/slope/curvature factors may be too low
   order for arbitrary 60-pillar shocks.
2. **Local model resolution:** each maturity piece may need a stronger local
   Chebyshev object than the current small dense factor tensor.
3. **Derivative measurement error:** finite-difference step choice may distort
   DV01, maturity slope, and mixed-term comparisons.
4. **Problem framing:** a blind faithful arbitrary 60-pillar parametric bond
   clone may be less defensible than a formula-aware replacement that preserves
   the full wrapper while decomposing the fixed-coupon bond cashflows.

## Related-Work Refresh

- Chebfun uses breakpoints/splitting to represent piecewise-smooth functions;
  this supports schedule-aware maturity pieces, but not a generic API until the
  local model is accurate.
- OpenGamma Strata treats present value and bucketed PV01 as first-class fixed
  income outputs; this supports validating both PV and per-pillar risk.
- The Federal Reserve H.15 notes that constant-maturity Treasury rates are read
  from an interpolated curve at fixed maturities; this supports documenting the
  fixture as public curve data, not a traded-bond bootstrap.
- Yield-curve PCA literature supports level/slope/curvature as a common
  compression idea, but Phase 12 must measure its projection error instead of
  assuming it is faithful.
- Tensor Train Cross literature supports sampling high-dimensional tensors from
  selected entries, but only when the sampled object is low-rank enough and
  held-out validation confirms accuracy.

## Required Wrapper Contract

Every candidate must be callable through the same 62-coordinate boundary:

```text
double Eval(double[] fullPoint)
// fullPoint[0..59] = zero-rate bumps
// fullPoint[60]    = coupon
// fullPoint[61]    = maturity years
```

Internal models may project, route, decompose coupon, or select active pillars,
but validation points must enter through the full wrapper.

## Phase Tasks

### Task 1: Rebuild The Validation Bank

- Keep the Phase 6/7 clone validation points.
- Add absolute-error floors beside relative errors for Greeks and mixed terms.
- Report PV, selected pillar DV01, all-pillar DV01 summary, coupon sensitivity,
  maturity sensitivity, rate-coupon, rate-maturity, coupon-maturity, and
  rate-rate mixed terms.
- Keep structural checks: post-maturity DV01 must be numerically zero.

### Task 2: Add Error-Source Oracles

- **Projection oracle:** compare the baseline price under the original 60D bump
  vector against the baseline price under a reconstructed factor curve.
- **Derivative oracle:** run central, one-sided, and multiple step-size
  finite-difference comparisons for DV01 and maturity slope.
- **Schedule oracle:** confirm that the router chooses the expected maturity
  piece and that errors are not caused by split dispatch.

### Task 3: Test Stronger Recipe Candidates

Measure candidates in this order:

1. richer deterministic curve factors, such as level/slope/curvature plus
   long-end and belly hump factors;
2. schedule-aware active-pillar TT per maturity piece while preserving the 62D
   wrapper;
3. analytic-coupon plus active-pillar TT;
4. fixed-trade curve-only surrogate as a production-control case;
5. schedule-resolved cashflow Chebyshev kernels: resolve the maturity schedule,
   keep coupon/notional algebraic, and use local 1D/2D Chebyshev tensors for
   smooth discount-factor kernels.

Stop adding candidates once the dominant error source is clear.

### Task 4: Decide The Recipe

The phase report must make one of these decisions:

- a candidate materially improves PV and risk accuracy and should become the
  next implementation path;
- a blind arbitrary 60-pillar parametric clone is not viable at current build
  cost, but a formula-aware replacement is viable for the supported bond family;
- the bottleneck is a ChebyshevSharp capability gap that needs a separate
  library-design phase.

## Exit Gate

Phase 12 is complete when:

1. the dominant error source is quantified;
2. at least one stronger candidate is compared against the Phase 10 router;
3. all candidates preserve the full 62-coordinate wrapper;
4. tests cover the new diagnostics and structural checks;
5. the Phase 12 report documents results, limitations, and the next recipe;
6. `dotnet format`, focused fixed-rate bond tests, DocFX, and `git diff --check`
   pass locally;
7. one coherent Phase 12 PR is opened only after the local exit gate passes.

## Sources To Cite In The Report

- Chebfun edge detection: <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- Federal Reserve H.15 constant-maturity description: <https://www.federalreserve.gov/releases/h15/>
- OpenGamma Strata fixed-coupon bond calculations: <https://strata.opengamma.io/apidocs/com/opengamma/strata/measure/bond/FixedCouponBondTradeCalculations.html>
- OpenGamma bucketed PV01 discussion: <https://opengamma.com/strata-and-multi-curve-calibration-and-bucketed-pv01/>
- QuantLib Guide cash-flow analysis: <https://www.quantlibguide.com/Cash-flow%20analysis.html>
- Tensor Train Cross error analysis: <https://arxiv.org/abs/2207.04327>
- Level-slope-curvature yield-curve reference: <https://ri.conicet.gov.ar/handle/11336/30774>
