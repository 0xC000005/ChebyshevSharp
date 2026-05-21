# Phase 1 Report: Baseline Pricer Adapter

## Objective

Select a deterministic fixed-rate bond reference-pricer path and expose it through a small adapter for later Chebyshev surrogate experiments.

## Baseline Selection

Phase 1 uses QLNet `1.13.1` as the first C# baseline. QLNet is a C# quantitative finance library derived from QuantLib. It provides the exact components needed for the restricted example: `FixedRateBond`, `Schedule`, `InterpolatedZeroCurve<Linear>`, and `DiscountingBondEngine`.

Python QuantLib remains a useful independent cross-check and research/data-refresh tool, but it is not required by the C# test or example path. The NuGet `QuantLib` wrapper is available, but it wraps the C++ library and its package notes mention thread-safety limits, so it is not the first CI dependency for this phase.

## Restricted Product

The current adapter covers only the first tutorial family:

- fixed-rate bullet bond;
- semiannual schedule;
- 30/360 USA coupon day count;
- U.S. government-bond calendar;
- Modified Following payment adjustment;
- settlement days defaulting to zero;
- direct continuously compounded zero-rate curve;
- dirty price, clean price, accrued amount, NPV, settlement value, and cashflow diagnostics.

This is not a general fixed-income library and does not model arbitrary stubs, callability, amortization, ex-coupon logic, floating coupons, or market-quote bootstrapping.

## Formula Under Test

For a regular fixed-rate bullet bond, dirty PV is linear in coupon:

$$
PV(N,c,T,r) = N \left(P(r,T) + c A(r,T)\right)
$$

where `P` is the discounted principal component and `A` is the fixed-leg annuity component. Phase 1 tests this identity using the QLNet baseline before any Chebyshev approximation is introduced.

## Implementation

- Example project: `examples/FixedRateBondSurrogate/`.
- Adapter interface: `IFixedRateBondReferencePricer`.
- QLNet adapter: `QlNetFixedRateBondReferencePricer`.
- Regression tests: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondReferencePricerTests.cs`.

The adapter uses a direct zero-rate curve, so later sensitivities are zero-pillar sensitivities. They are not bootstrapped market-quote sensitivities.

## Validation

Focused tests:

```bash
dotnet test --filter "FullyQualifiedName~FixedRateBondReferencePricerTests"
```

Result:

```text
Passed: 5, Failed: 0, Skipped: 0
```

Full verification:

```bash
rg -n "VTA|proprietary|internal product|private object|company confidential|internal-only|private assessment" examples/FixedRateBondSurrogate tests/ChebyshevSharp.Tests/Finance docs/research/fixed-rate-bond-surrogate docs/docs/citations.md docs/docs/examples.md docs/docs/testing-and-validation.md .github/workflows/test.yml ChebyshevSharp.slnx
git diff --check
dotnet restore
dotnet format --verify-no-changes --verbosity minimal
dotnet build --configuration Release --no-restore
dotnet test --configuration Release --no-build --verbosity minimal -- RunConfiguration.DisableParallelization=true
dotnet pack src/ChebyshevSharp --configuration Release --no-build --output artifacts/packages
dotnet build --no-restore
dotnet test
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/SliderPartitionValidation/SliderPartitionValidation.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
dotnet run --configuration Release --no-build --project examples/QuickStart/QuickStart.csproj
dotnet run --configuration Release --no-build --project examples/SliderPartitionValidation/SliderPartitionValidation.csproj
dotnet run --configuration Release --no-build --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
docfx docs/docfx.json
```

Results:

```text
private-name scan: no proprietary system/interface names; matches are guardrail language only
git diff --check: no whitespace errors
dotnet format: 0 formatting changes required
dotnet build --configuration Release --no-restore: 0 warnings, 0 errors
dotnet test --configuration Release --no-build: Passed 1654, Failed 0, Skipped 0
dotnet build --no-restore: 0 warnings, 0 errors
dotnet test: Passed 1654, Failed 0, Skipped 0
dotnet pack: created ChebyshevSharp.0.13.3.nupkg
DocFX: 0 warnings, 0 errors
All runnable examples completed in Debug and CI-style Release/no-build mode.
```

The tests verify:

- finite price outputs for a deterministic regular fixed-rate bond;
- coupon linearity;
- principal/annuity recombination;
- zero-coupon principal component behavior;
- matured-bond zero value and zero rate sensitivity under the direct-zero curve setup.

## Citation and Source Verification

References were added to `docs/docs/citations.md` for QLNet, QuantLib-Python, NuGet QuantLib, and public market-data sources. During setup, the environment also verified:

- Python QuantLib imports through `uv`;
- QLNet restores from NuGet;
- C# `HttpClient` can reach the Federal Reserve nominal yield curve CSV, Treasury XML feed, and New York Fed SOFR averages/index page.

## Next Phase

Phase 2 should build the pinned public curve fixture pipeline. The fixture must record source URL, download date, curve date, field names, units, compounding convention, interpolation convention, and whether the inputs are fitted zero yields, par yields, SOFR rates, or synthetic transforms.
