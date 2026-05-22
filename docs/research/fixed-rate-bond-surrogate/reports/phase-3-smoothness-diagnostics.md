# Phase 3 Report: Smoothness Diagnostics

## Objective

Measure baseline fixed-rate bond smoothness before fitting any Chebyshev surrogate. This phase asks whether present value, coupon sensitivity, zero-pillar DV01, and maturity-date behavior are smooth enough for global interpolation or should be treated as piecewise smooth.

## Baseline

- Reference pricer: QLNet `FixedRateBond` through `QlNetFixedRateBondReferencePricer`.
- Curve fixture: `fed-nominal-yield-curve-2026-05-15`.
- Curve fields: Federal Reserve `SVENYXX` continuously compounded zero-coupon yields.
- Product shape: restricted regular fixed-rate bullet bond, semiannual coupon, 30/360 USA coupon day count, Actual/365 zero curve, settlement equal to valuation date.

The diagnostics intentionally operate on the exact baseline only. TT and Slider surrogates start in Phase 4.

## Finite-Difference Rules

Rate derivative:

```text
dPV/dr_i approx (PV(r_i + h) - PV(r_i - h)) / (2h), h = 1 bp
zero_pillar_dv01 = dPV/dr_i * 1 bp
```

Coupon derivative:

```text
dPV/dc approx (PV(c + h_c) - PV(c - h_c)) / (2h_c), h_c = 1 bp
```

Maturity-date slope:

```text
dPV/dT approx (PV(T + 1 day) - PV(T - 1 day)) / (2 / 365)
```

The maturity diagnostic uses dates, not decimal-year maturity inputs. The second-difference spike score is:

```text
abs(PV(T+1 day) - 2 PV(T) + PV(T-1 day))
```

This is a breakpoint signal, not proof of a mathematical discontinuity.

## Coupon Slice

The coupon dimension is numerically linear for this regular bullet family.

| Coupon | Dirty PV | dPV/dc | Second difference |
| ---: | ---: | ---: | ---: |
| 0.00% | 62.53989812 | 799.89848657 | 1.492140E-013 |
| 2.00% | 78.53786785 | 799.89848657 | -2.131628E-013 |
| 4.50% | 98.53533001 | 799.89848657 | 1.705303E-013 |
| 8.00% | 126.53177704 | 799.89848657 | -2.273737E-013 |
| 12.00% | 158.52771651 | 799.89848657 | -1.421085E-013 |

Conclusion: coupon should remain an analytical parameter in later phases unless a more complex product family breaks this identity.

## Zero-Pillar DV01

| Pillar | zero-pillar DV01 | Local second difference |
| ---: | ---: | ---: |
| 1Y | -4.313883E-004 | 3.495055E-008 |
| 5Y | -3.615172E-003 | 1.260325E-006 |
| 10Y | -6.730724E-002 | 6.590982E-005 |
| 20Y | 0.000000E+000 | 0.000000E+000 |
| 30Y | 0.000000E+000 | 0.000000E+000 |

Conclusion: the dominant zero-pillar sensitivity is the 10Y pillar, as expected for a 10Y bond. The 20Y and 30Y pillars have zero sensitivity in this setup because the linear zero curve has no interpolation support from those pillars on remaining cashflow dates.

The diagnostics also evaluates the five selected pillars over `[-150, -75, 0, 75, 150]` bp, for 25 deterministic rate-bump slice points. These are baseline samples for Phase 4 surrogate error checks, not a pruning rule.

## Maturity-Date Slice

Top daily second-difference spike candidates:

| Maturity date | Offset from scanned boundary | Cashflows | Spike |
| --- | ---: | ---: | ---: |
| 2028-05-13 | -2d | 5 | -2.184913E-003 |
| 2028-05-20 | 5d | 6 | -2.163606E-003 |
| 2031-05-10 | -5d | 11 | 1.912040E-003 |
| 2031-05-16 | 1d | 12 | -1.870143E-003 |
| 2031-05-17 | 2d | 12 | 1.737497E-003 |

Conclusion: maturity should be treated as piecewise smooth in later surrogate phases. The diagnostic does not prove that every semiannual date must become a split point, but it shows that maturity-date scans should drive the Phase 6 split comparison.

## Validation

Focused tests:

```bash
dotnet test --filter "FullyQualifiedName~FixedRateBondSmoothnessDiagnosticsTests"
```

Result:

```text
Passed: 7, Failed: 0, Skipped: 0
```

Diagnostics command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics
```

The diagnostics mode does not download market data at runtime. It consumes the pinned Phase 2 fixture.

Local closeout checks:

```bash
dotnet format --verify-no-changes --verbosity minimal
dotnet build --configuration Release --no-restore
dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage" -- RunConfiguration.DisableParallelization=true
dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics
docfx docs/docfx.json
```

Results:

- public-surface scan found no private implementation names;
- formatter verification exited with no changes;
- Release build succeeded with 0 warnings and 0 errors;
- Release coverage test passed 1683 tests with 0 failures and 0 skipped tests;
- local coverage inspection found no uncovered or partial lines in `SmoothnessDiagnostics.cs`; the only uncovered `Program.cs` line in the full report is the unchanged console entrypoint;
- both default and diagnostics example modes ran successfully;
- DocFX build succeeded with 0 warnings and 0 errors.

## References

- Chebfun documentation motivates splitting piecewise-smooth functions into smooth intervals and using edge detection as a breakpoint signal.
- SciPy finite-difference documentation motivates recording derivative step sizes and treating boundary stencils explicitly.
- QuantLib-Python and QLNet references document the fixed-rate bond baseline family and bond pricing outputs.
- Federal Reserve nominal yield curve documentation identifies the public zero-yield data source.

Citation links verified on 2026-05-21:

- Chebfun Guide: <https://www.chebfun.org/docs/guide/guide01.html>
- Chebfun edge detection example: <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- SciPy finite-difference differentiation: <https://docs.scipy.org/doc/scipy/reference/differentiate.html>
- QuantLib-Python instruments: <https://quantlib-python-docs.readthedocs.io/en/latest/instruments.html>
- Federal Reserve nominal yield curve: <https://www.federalreserve.gov/data/nominal-yield-curve.htm>
