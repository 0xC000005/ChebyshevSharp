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

This example is intentionally restricted. It is a baseline for later surrogate
validation, not a general fixed-income library. It does not download market data
at runtime; the optional refresh script under
`tools/RefreshFixedRateBondMarketData/` regenerates the pinned JSON fixtures
from the Federal Reserve public CSV. The research notes in
`docs/research/fixed-rate-bond-surrogate/` track formulas, data provenance,
citations, source limitations, and validation results as the workflow progresses.

## When to run them

Run the examples when changing public workflows, constructor defaults,
serialization behavior, domain validation, or documentation snippets that show
first-use code. CI runs the examples in the `Format, Pack, and Docs` job, so a
PR that breaks the examples should fail before merge.
