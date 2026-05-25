---
title: Examples
---

# Examples

The repository includes runnable console projects. They are intended to be
small enough to read in one sitting and concrete enough to verify that the local
toolchain, package reference, and public API are working.

Run the examples from the repository root:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/SliderPartitionValidation/SliderPartitionValidation.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --surrogate-reproduction
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --accuracy-recipe-search
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-maturity-scan-csv
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --naive-surrogate-discovery
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --structured-alternatives
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --risk-acceptance
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --risk-acceptance-heavy
```

## QuickStart

`examples/QuickStart` demonstrates the standard dense workflow:

1. Define a smooth 2D function on a bounded domain.
2. Build a `ChebyshevApproximation` on a tensor-product Chebyshev grid.
3. Evaluate the value and first derivatives at an interior point.
4. Compare the interpolated values with the original function.
5. Print the `ErrorEstimate()` diagnostic.

Use this example before adapting the larger snippets in
[Getting Started](getting-started.md).

## SliderPartitionValidation

`examples/SliderPartitionValidation` demonstrates the high-dimensional Slider
workflow:

1. Define an 8D model with known pairwise interactions.
2. Build one `ChebyshevSlider` with the correct pair grouping.
3. Build a second slider with an intentionally weak singleton grouping.
4. Compare build costs against the dense grid size.
5. Validate both sliders on held-out points and compare a derivative.

Use this example when deciding whether a proposed partition is a modelling
assumption you can defend. A small `ErrorEstimate()` does not prove the partition
captures cross-group interactions; held-out points far from the pivot are the
more important check.

## TensorTrainHighDim

`examples/TensorTrainHighDim` demonstrates the high-dimensional TT-Cross
workflow:

1. Define a smooth 7D coupled model.
2. Build a `ChebyshevTT` with `method: "cross"`.
3. Compare the dense grid size with the number of TT-Cross evaluations.
4. Inspect TT ranks and compression ratio.
5. Check one held-out point against the original function.

Use this example when the dense grid size `prod(n_i)` is the limiting factor and
the model is expected to have moderate numerical TT ranks. If the TT ranks grow
to the rank cap or the held-out error is too high, increase the rank cap, adjust
the node counts, or use a dense/spline/slider representation instead.

## FixedRateBondSurrogate

`examples/FixedRateBondSurrogate` is the finance research harness used to build
a public fixed-rate bond tutorial. It currently prices a restricted regular
fixed-rate bullet bond against a pinned Federal Reserve nominal-yield-curve
fixture:

1. Load `Data/fed-nominal-yield-curve-semiannual-2026-05-15.json`.
2. Price it with QLNet through a small `IFixedRateBondReferencePricer` boundary.
3. Report the curve fixture, curve date, curve-pillar count, conventions, dirty
   price, clean price, accrued amount, NPV, and cashflow count.
4. Keep the direct zero-rate curve separate from later Chebyshev surrogates.

For the full modelling narrative, including the naive global TT/Slider failure
evidence and the structured follow-up experiments, see the
[Fixed-Rate Bond Surrogate Case Study](fixed-rate-bond-surrogate.md).

### QLNet-backed reference pricer

The baseline is QLNet, not an in-repository bond-pricing implementation. The
example code only adapts a compact research input into QLNet objects:

```text
FixedRateBondRequest
  -> QLNet Schedule
  -> QLNet FixedRateBond
  -> QLNet InterpolatedZeroCurve<Linear>
  -> QLNet DiscountingBondEngine
  -> FixedRateBondResult
```

This boundary is intentional. The case study is about learning a fast
Chebyshev approximation of a trusted pricing function, not about replacing
fixed-income conventions, schedule generation, accrued-interest logic, or
discounting rules. Future TensorTrain and Slider experiments should treat
`IFixedRateBondReferencePricer.Price()` as the ground-truth black box, then
measure where a surrogate reproduces PV, DV01, coupon sensitivity, maturity
sensitivity, and mixed terms.

The current baseline assumptions are deliberately narrow:

- fixed-rate bullet bond;
- valuation date equals effective date;
- semiannual coupon schedule;
- 30/360 USA coupon day count;
- U.S. Government Bond calendar;
- Modified Following business-day adjustment;
- backward schedule generation;
- direct zero-rate curve with Actual/365 Fixed timing;
- continuous annual compounding and linear zero-rate interpolation;
- redemption of `100.0`;
- no amortization, callability, ex-coupon handling, stubs, or arbitrary
  settlement-date modelling.

These assumptions make the example reproducible and auditable. They are also
the eligibility domain for later Chebyshev surrogate validation; outside this
domain, the correct reference remains the underlying fixed-income pricer.

The default run uses a 30-year semiannual 4.5% coupon bullet bond with
61 zero-rate pillars: the valuation-date anchor plus 60 semiannual points from
0.5Y to 30Y. The semiannual fixture is derived from the Federal Reserve fitted
nominal yield-curve parameters for the pinned curve date, using Actual/365
year fractions to match the QLNet dated zero curve. The compact annual fixture
remains available for the earlier surrogate reproduction experiment.

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
```

Run the diagnostics mode to inspect baseline coupon linearity, zero-pillar
DV01, and maturity-date spike candidates before fitting a Chebyshev surrogate:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics
```

Run the surrogate reproduction mode to fit the first compact full-PV
TensorTrain and Slider models against the public reference pricer, then compare
PV, zero-pillar DV01, coupon, maturity, and mixed finite-difference errors:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --surrogate-reproduction
```

Run the naive discovery mode to use the dense fixture, estimate why the full
dense tensor is infeasible, and compare full-input 62D TensorTrain and Slider
probes before any decomposition or maturity splitting:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery
```

The naive discovery report also prints structural sanity checks. For example,
a bond maturing at the 10Y curve pillar should have zero direct sensitivity to
the 30Y zero-rate pillar under the current direct-zero interpolation setup. If
that check fails, the surrogate is inventing post-maturity curve exposure before
any finer modelling question matters.

Run the structured alternatives mode after the naive failure is reproduced. It
keeps the same 62-coordinate wrapper while comparing common modelling choices:
trying a stronger global TensorTrain, testing TT auto-ordering, grouping
interacting variables in a Slider, projecting the 60 curve bumps into
level/slope/curvature factors, and routing factor models through 1Y and 0.5Y
maturity buckets.

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives
```

Read this mode as an evidence generator, not a finished pricing architecture.
The factor tensor performs well on factor-aligned scenarios, while arbitrary
60-pillar bump vectors still expose projection error. The bucketed variants
test whether maturity splitting helps, but they do not by themselves make the
mixed sensitivities reliable.

Run the analytic coupon decomposition mode to test the fixed-rate bond identity
used by later candidate models:

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
```

This mode keeps the full 62-coordinate public wrapper but builds internal
principal and annuity models without coupon as a tensor coordinate. The result
is useful for deciding whether coupon should be analytical in a tutorial model;
it does not remove the separate maturity-smoothness problem.

Run the maturity-special-points mode after the analytic coupon result. It keeps
the full public wrapper, scans one-day windows around semiannual maturity
regions, and compares global, uniform-bucket, schedule-aware, detector, and
hybrid maturity routing:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

Read this as a research benchmark. The current result supports
schedule-aware maturity routing over a blind global fit or a simple uniform
bucket rule, but it does not yet make automatic kink detection a public library
feature.

Run the schedule-aware router mode after the maturity-special-points benchmark.
It converts the schedule-derived maturity split points into an explicit
half-open piecewise router, keeps the 62-coordinate public wrapper, and reports
one-sided maturity sensitivities around split points:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

Read this as a router-design benchmark. The current explicit router reproduces
the schedule-aware special-point benchmark and clarifies dispatch semantics, but
it does not yet remove the residual sensitivity errors. The result supports
further piece-model research before promoting a reusable ChebyshevSharp router
API.

Run the accuracy recipe search mode to compare the remaining candidate recipes
and the schedule-resolved cashflow-kernel proof of concept:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --accuracy-recipe-search
```

This mode reports the first full-wrapper proof of concept that reaches
near-roundoff PV error for the supported regular fixed-rate bullet family. It is
formula-aware: it resolves cashflows by maturity, keeps coupon and notional
algebraic, and uses local one- or two-dimensional Chebyshev kernels only for
smooth discount factors.

To regenerate the maturity-sensitivity plot used by the case study, run:

```bash
python tools/PlotFixedRateBondEvidence/plot_maturity_sensitivity.py
```

The script invokes the example's `--naive-maturity-scan-csv` mode, writes the
deterministic CSV artifact, and renders the SVG used by the public case study.

This example is intentionally restricted. It is a baseline for later surrogate
validation, not a general fixed-income library. It does not download market data
at runtime; the optional refresh script under
`tools/RefreshFixedRateBondMarketData/` regenerates the pinned JSON fixtures
from the Federal Reserve public CSV. The public case study page tracks formulas,
data provenance, citations, source limitations, and validation results.

## CallableBondSurrogate

`examples/CallableBondSurrogate` extends the finance examples to a callable
fixed-rate bond where the QLNet baseline is expensive enough for acceleration to
matter. It uses the same pinned semiannual Federal Reserve curve fixture, but
prices a callable fixed-rate bond with QLNet's `CallableFixedRateBond`,
`HullWhite`, and `TreeCallableFixedRateBondEngine` classes.

The public wrapper intentionally stays full-dimensional:

```text
curveBumps[60], coupon, maturity, firstCall, callPrice, sigma
    -> callable dirty price per 100 notional
```

The default mode prints the QLNet baseline price, comparable straight-bond
price, embedded call value, and call-date count:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj
```

Run the naive discovery mode before trying structured models. It compares a
full-input 65D TensorTrain probe with a singleton Slider probe and records why
the direct global clone is not accurate enough for PV, DV01, and mixed-risk
checks:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --naive-surrogate-discovery
```

Run the structured alternatives mode to compare factor-risk and formula-aware
surrogates behind the same 65D public wrapper. This mode reports build cost,
evaluation cost, break-even count, PV errors, factor-direction sensitivities,
local key-rate DV01 checks, and volatility sensitivity checks:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --structured-alternatives
```

Run the risk-acceptance mode to test the current candidate family against a
broader risk-manager style validation bank. It reports PV, factor sensitivities,
full 60-pillar DV01 vector error, product Greeks, selected mixed terms, and the
full-DV01 timing for the reference-semantics tree clone, smoothed lattice
tangent, hybrid DV01 correction, HDMR, residual, and Tensor Train probes:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --risk-acceptance
```

Use the heavy mode only when you need to reproduce the stronger full-pillar TT
probe. It is intentionally slower because that candidate uses a larger 65D
TT-Cross build:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --risk-acceptance-heavy
```

For the full narrative, including the financial formula, trial rationale,
failure evidence, acceleration economics, and current limitations, see the
[Callable Bond Case Study](callable-bond-surrogate.md).

## When to run them

Run the examples when changing public workflows, constructor defaults,
serialization behavior, domain validation, or documentation snippets that show
first-use code. CI runs the examples in the `Format, Pack, and Docs` job, so a
PR that breaks the examples should fail before merge.
