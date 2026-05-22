# Phase 8 Report: Analytic Coupon Decomposition

## Objective

Phase 8 tests whether coupon should remain a Chebyshev dimension for the
restricted regular fixed-rate bullet bond harness.

The public wrapper remains:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

Internally, this phase tests the fixed-rate bond identity:

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

where `PrincipalPV` is the dirty PV at zero coupon and `AnnuityPV` is the dirty
PV slope with respect to the decimal annual coupon rate. This is a modelling
decision for the restricted product family only; it is not a claim about
amortizing, callable, floating-rate, ex-coupon, or irregular-stub products.

## Research Basis

OpenGamma Strata describes a fixed coupon bond as periodic fixed payments plus
a final nominal payment. The QLNet/QuantLib baseline discounts future fixed
bond cashflows using a discounting bond engine. Under the restricted regular
bullet assumptions used by this harness, the future coupon cashflows are
proportional to coupon, so coupon is linear once the curve and schedule are
fixed.

Phase 8 deliberately keeps automatic kink detection and special-point routing
out of scope. Maturity remains the nonsmooth coordinate because changing
maturity can regenerate cashflow dates and accrual periods. That belongs to
Phase 9.

## Candidate Configuration

| Candidate | Internal model | Wrapper dims | Internal dims | Buckets | Build evals |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact coupon decomposition oracle | Reference-pricer principal and annuity calls | 62 | 61 | 1 | 0 |
| Global decomposed TT | Two 61D TT-Cross models, `n=4`, `maxRank=8` | 62 | 61 | 1 | 21,023 |
| Curve-factor decomposed tensor | Two dense 4D tensors over factors + maturity | 62 | 4 | 1 | 270 |
| Bucketed decomposed curve-factor tensor | Same 4D tensors routed through 1Y maturity buckets | 62 | 4 | 28 | 6,048 |
| Semiannual bucketed decomposed curve-factor tensor | Same 4D tensors routed through 0.5Y maturity buckets | 62 | 4 | 56 | 12,096 |

The annuity is estimated from the reference pricer as:

```text
AnnuityPV(curve, T) =
  (PV(curve, coupon = 0.12, T) - PV(curve, coupon = 0, T)) / 0.12
```

The validation bank contains the Phase 6 clone points and the Phase 7
factor-aligned points.

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
```

Coupon-linearity identity check:

| Validation points | Max absolute error | Max relative error |
| ---: | ---: | ---: |
| 19 | `8.526513E-014` | `0.000000%` |

Measured model errors:

| Model | Max clone PV rel. error | Max clone coupon rel. error | Max clone maturity rel. error | Max clone coupon-maturity rel. error | Max factor-aligned PV rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact coupon decomposition oracle | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Global decomposed TT | 14.34% | 5.28% | 456.94% | 51.28% | 11.97% |
| Curve-factor decomposed tensor | 4.70% | 1.80% | 90.42% | 59.04% | 0.59% |
| Bucketed decomposed curve-factor tensor | 4.73% | 1.79% | 128.96% | 59.76% | 0.60% |
| Semiannual bucketed decomposed curve-factor tensor | 4.73% | 1.80% | 87.72% | 56.94% | 0.58% |

## Interpretation

The exact decomposition oracle confirms that coupon is analytically linear for
this restricted baseline. The formula is not the source of the earlier
coupon-related failures.

Removing coupon from the global TT is not enough. Even with a comparable
`n=4`, `maxRank=8` budget, the decomposed global TT still has `14.34%` max PV
relative error and `456.94%` max maturity-sensitivity relative error. This
supports the Phase 7 conclusion that the global high-dimensional clone remains
hard even after removing an easy dimension.

The decomposed factor tensor matches the Phase 7 factor PV behavior while
reducing build evaluations from `675` to `270`. This is the main practical win:
the formula turns a 5D full-PV factor tensor into two 4D tensors and makes the
coupon derivative directly interpretable.

The 1Y and 0.5Y bucketed decomposed factor tensors repeat the Phase 7 pattern.
Simple schedule-cadence buckets reduce neither arbitrary-bump projection error
nor the remaining coupon-maturity mixed error enough to be a final risk clone.
The 0.5Y bucket remains slightly better on maturity sensitivity than the 1Y
bucket, but it does not eliminate the maturity nonsmoothness problem.

## Decision

Phase 8 establishes the next modelling boundary:

1. Coupon should not be a tensor dimension for this restricted regular fixed
   coupon bond family.
2. Analytic coupon decomposition is valid and should be used in the tutorial
   when the product eligibility gate guarantees the same assumptions.
3. Decomposition improves interpretability and reduces build cost, but it does
   not solve arbitrary 60-pillar curve projection error or maturity
   nonsmoothness.
4. Phase 9 should focus on high-dimensional piecewise treatment of maturity:
   special points, schedule-aware routing, or automatic edge/kink detection.

## Verification

Fresh checks run during Phase 8 implementation:

```bash
dotnet format --verify-no-changes --verbosity minimal
dotnet test --configuration Release --filter "FullyQualifiedName~FixedRateBondAnalyticCouponDecompositionTests" --collect:"XPlat Code Coverage" --results-directory TestResults/phase8-coverage-final
dotnet test --filter "FullyQualifiedName~FixedRateBond"
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --analytic-coupon-decomposition
dotnet build --configuration Release --no-restore
docfx docs/docfx.json
dotnet test --configuration Release --verbosity minimal
git diff --check
```

Results:

- formatting check exited with no changes;
- focused Phase 8 tests passed `3` tests with `0` failures;
- focused coverage found no missing or partial lines in `AnalyticCouponDecompositionBenchmark.cs`;
- fixed-rate bond test slice passed `62` tests with `0` failures;
- analytic-coupon example completed successfully and printed the results above;
- Release build succeeded with `0` warnings and `0` errors;
- DocFX built successfully with `0` warnings and `0` errors;
- Release test suite passed `1711` tests with `0` failures;
- `git diff --check` reported no whitespace errors;
- PR CI passed `Format, Pack, and Docs`, `.NET 8 library build`, `.NET 10 tests`, `All Tests Passed`, and `codecov/patch`;
- public-surface scan found no private implementation names.

## Sources

- OpenGamma Strata `FixedCouponBond` API documentation: <https://strata.opengamma.io/apidocs/com/opengamma/strata/product/bond/FixedCouponBond.html>
- OpenGamma Strata `DiscountingFixedCouponBondProductPricer` API documentation: <https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/bond/DiscountingFixedCouponBondProductPricer.html>
- QuantLib Guide, "Vanilla bonds": <https://www.quantlibguide.com/Vanilla%20bonds.html>
- Phase 7 report: [Structured Alternatives](phase-7-structured-alternatives.md)
