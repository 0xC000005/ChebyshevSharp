# Phase 5 Report: Realistic Baseline

## Objective

Harden the fixed-rate bond baseline before additional Chebyshev surrogate work. The earlier sparse curve fixture was useful for compact reproduction, but it was too sparse for a realistic semiannual bond example. This phase adds a dense direct zero-rate fixture, explicit conventions, and baseline sanity tests.

## Data Fixture

New fixture:

```text
examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-semiannual-2026-05-15.json
```

The fixture uses the same Federal Reserve nominal yield-curve source and curve date as the earlier annual fixture. Instead of using only selected annual `SVENYXX` columns, it samples the fitted curve every six months from 0.5Y to 30Y using the published `BETA0`, `BETA1`, `BETA2`, `BETA3`, `TAU1`, and `TAU2` fields. The time variable is the Actual/365 year fraction from the curve date to the semiannual pillar date, matching the QLNet zero-curve day counter:

```text
L(t, tau) = (1 - exp(-t / tau)) / (t / tau)
y(t) = beta0
     + beta1 * L(t, tau1)
     + beta2 * (L(t, tau1) - exp(-t / tau1))
     + beta3 * (L(t, tau2) - exp(-t / tau2))
```

The generated annual points remain close to the published annual `SVENYXX` columns. They are not forced to match exactly after the first year because the fixture now uses dated Actual/365 year fractions rather than nominal integer tenors:

| Maturity | Dense sample | Published field |
| --- | ---: | ---: |
| 1Y | 3.8924992577 | 3.8925 |
| 2Y | 3.9895344732 | 3.9893 |
| 5Y | 4.2730714057 | 4.2728 |
| 10Y | 4.6903819201 | 4.6898 |
| 30Y | 5.3320833655 | 5.3322 |

The fixture contains 60 semiannual points. `FixedRateBondMarketData.ToZeroRatePillars()` adds a valuation-date anchor, so the QLNet curve receives 61 pillars.

## Pricing Conventions

The QLNet adapter now exposes its convention summary:

| Convention | Value |
| --- | --- |
| Calendar | `UnitedStates.GovernmentBond` |
| Schedule | Semiannual, backward generation |
| Coupon day count | `30/360 USA` |
| Curve day count | `Actual/365 Fixed` |
| Business-day rule | `ModifiedFollowing` |
| End of month | `false` |
| Curve interpolation | Linear zero-rate interpolation |
| Curve compounding | Continuous annual |
| Redemption | `100.0` |

These conventions keep the example narrow and explainable. It is still not a general fixed-income pricer.

## Baseline Sanity Checks

For the default 30Y fixed-rate bullet example:

```text
Valuation date : 2026-05-15
Effective date : 2026-05-15
Maturity date  : 2056-05-15
Coupon         : 4.50 %
Curve pillars  : 61
Cashflows      : 61
Dirty price    : 89.26423408
Clean price    : 89.26423408
Accrued amount : 0.00000000
```

The price is economically sensible for a 4.5% coupon bond priced against an upward-sloping fitted zero curve with 30Y zero yield around 5.33%. Tests also verify that coupon ordering is monotone and NPV scales with notional.

Two additional hardening checks were added:

- A zero-coupon midpoint test prices a one-cashflow bond between the 1Y and 2Y curve pillars and matches a manual linear zero-rate interpolation formula:

```text
r(t) = r0 + (r1 - r0) * (t - t0) / (t1 - t0)
PV = 100 * exp(-r(t) * t)
```

- Public Treasury auction sanity checks cover six note/bond auctions. Each uses the auction issue date as valuation/effective date, coupon, maturity, high yield, price, and no accrued interest from the TreasuryDirect result. Feeding the QLNet adapter a flat continuous-zero curve converted from the auction high yield produces prices within `0.10` to `0.20` price points. These checks are not presented as exact replications of the Treasury auction-yield formula; they are convention sanity checks that the external pricer, cashflow schedule, and discounting path are in the right neighborhood against actual public prices.

| CUSIP | Term | Coupon | Maturity | High yield | Price | Source |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `91282CQL8` | 2Y | 3.750% | 2028-04-30 | 3.812% | 99.881691 | [R_20260427_1](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260427_1.pdf) |
| `91282CQR5` | 3Y | 3.875% | 2029-05-15 | 3.965% | 99.747787 | [R_20260511_3](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260511_3.pdf) |
| `91282CQK0` | 5Y | 3.875% | 2031-04-30 | 3.955% | 99.640273 | [R_20260427_4](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260427_4.pdf) |
| `91282CQN4` | 7Y | 4.125% | 2033-04-30 | 4.175% | 99.699199 | [R_20260428_3](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260428_3.pdf) |
| `91282CQQ7` | 10Y | 4.375% | 2036-05-15 | 4.468% | 99.256552 | [R_20260512_3](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260512_3.pdf) |
| `912810UU0` | 30Y | 5.000% | 2056-05-15 | 5.046% | 99.292811 | [R_20260513_2](https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260513_2.pdf) |

## Implementation

- Dense fixture support: `examples/FixedRateBondSurrogate/MarketData.cs`.
- Convention summary: `examples/FixedRateBondSurrogate/ReferencePricer.cs`.
- Default example output: `examples/FixedRateBondSurrogate/Program.cs`.
- Refresh script: `tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py`.
- Regression tests: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondRealisticBaselineTests.cs` and existing fixed-rate bond tests.

## Verification

Completed during implementation:

```bash
uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --help
uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --curve-date 2026-05-15 --download-date 2026-05-20 --density semiannual-svensson --input-csv /tmp/feds200628.csv --output /tmp/fed-dense-check.json
uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --curve-date 2026-05-15 --download-date 2026-05-20 --density selected-annual --input-csv /tmp/feds200628.csv --output /tmp/fed-annual-check.json
dotnet test --filter "FullyQualifiedName~FixedRateBond"
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
```

Results so far:

- Dense fixture regenerated byte-for-byte from the local copy of the Federal Reserve CSV.
- Existing annual fixture regenerated byte-for-byte from the same script.
- Focused fixed-rate bond test suite passed 50 tests with 0 failures.
- Default example ran against the dense fixture and printed the 30Y baseline above.

Closeout verification:

```bash
dotnet format --verify-no-changes --verbosity minimal
dotnet build --configuration Release --no-restore
dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage" -- RunConfiguration.DisableParallelization=true
dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
docfx docs/docfx.json
git diff --check
```

Results:

- Public-surface scan found no private implementation names.
- Formatting verification exited with no changes.
- Release build succeeded with 0 warnings and 0 errors.
- Release coverage test run passed 1699 tests with 0 failures and 0 skipped tests.
- Release example output matched the dense 30Y baseline shown above.
- DocFX built 57 models successfully with 0 warnings and 0 errors.
- `git diff --check` reported no whitespace errors.

## Sources

- Federal Reserve nominal yield curve data page: <https://www.federalreserve.gov/data/nominal-yield-curve.htm>
- Federal Reserve nominal yield curve table/CSV: <https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv>
- Federal Reserve FEDS 2006-28 paper: <https://www.federalreserve.gov/pubs/feds/2006/200628/200628pap.pdf>
- Treasury auction result for 30Y CUSIP 912810UU0: <https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260513_2.pdf>
- QLNet `FixedRateBond`: <https://github.com/amaggiulli/QLNet/blob/develop/src/QLNet/Instruments/Bonds/FixedRateBond.cs>
