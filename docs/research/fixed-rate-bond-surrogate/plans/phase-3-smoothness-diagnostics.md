# Phase 3 Smoothness Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where the public fixed-rate bond baseline is smooth, piecewise smooth, or nonsmooth before building TT/Slider surrogates.

**Architecture:** Add deterministic C# diagnostics around the existing QLNet-backed reference pricer and pinned Federal Reserve zero-yield fixture. Phase 3 should not build Chebyshev surrogates yet; it should produce baseline PV, finite-difference, and schedule-boundary evidence that later phases can target.

**Tech Stack:** C#/.NET 10, QLNet `1.13.1`, xUnit, DocFX, optional public-reference checks via browser/search.

---

## Research Inputs

- Chebfun represents piecewise-smooth functions by splitting a domain into smooth subintervals; this motivates detecting maturity breakpoints before using global Chebyshev approximations.
- Chebfun edge detection introduces breakpoints for nonsmooth or rapidly changing one-dimensional slices; use this as inspiration only, not as a production adaptive splitter yet.
- SciPy's finite-difference documentation uses central differences in the interior and one-sided steps near boundaries; Phase 3 should record step sizes and stencil direction for every derivative-like metric.
- QuantLib/QLNet bond outputs distinguish dirty price, clean price, accrued amount, NPV, settlement value, and cashflow diagnostics; Phase 3 should report these separately so accrued/schedule effects are visible.
- Phase 2 fixture source remains the Federal Reserve nominal yield curve CSV using `SVENYXX` continuously compounded zero-coupon yields.

Reference links to cite in the report if used:

- <https://www.chebfun.org/docs/guide/guide01.html>
- <https://www.chebfun.org/examples/approx/EdgeDetection.html>
- <https://docs.scipy.org/doc/scipy/reference/differentiate.html>
- <https://quantlib-python-docs.readthedocs.io/en/latest/instruments.html>
- <https://www.federalreserve.gov/data/nominal-yield-curve.htm>

## Files

- Create `examples/FixedRateBondSurrogate/SmoothnessDiagnostics.cs`.
- Modify `examples/FixedRateBondSurrogate/Program.cs` to support a `--diagnostics` mode while preserving the default example output.
- Create `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSmoothnessDiagnosticsTests.cs`.
- Create `docs/research/fixed-rate-bond-surrogate/reports/phase-3-smoothness-diagnostics.md`.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md`.
- Modify `docs/docs/examples.md` only if the diagnostics mode becomes stable enough to mention publicly.

## Diagnostic Definitions

Use dirty price as the primary PV value and record clean price/accrued amount as diagnostics.

Rate derivative:

```text
dPV/dr_i approx (PV(r_i + h) - PV(r_i - h)) / (2h)
```

Use `h = 1e-4` decimal rate, i.e. one basis point. Report `zero_pillar_dv01 = dPV/dr_i * 1e-4`.

Coupon derivative:

```text
dPV/dc approx (PV(c + h_c) - PV(c - h_c)) / (2h_c)
```

Use `h_c = 1e-4`. Coupon is expected to be linear for the regular bullet family, so second differences should be near zero.

Rate-coupon mixed derivative:

```text
d2PV/(dr_i dc) approx
  (PV(r_i+h, c+h_c) - PV(r_i+h, c-h_c)
 - PV(r_i-h, c+h_c) + PV(r_i-h, c-h_c)) / (4 h h_c)
```

Maturity date slope:

```text
dPV/dT approx (PV(T + 1 day) - PV(T - 1 day)) / (2 / 365)
```

Use date-based maturity points, not decimal-year maturity inputs. When `T - 1 day` is not valid for a boundary case, record a one-sided stencil and label it.

Second-difference spike score:

```text
spike(T_i) = abs(PV(T_{i+1}) - 2 PV(T_i) + PV(T_{i-1}))
```

Use this as a breakpoint signal, not as proof of a true discontinuity.

## Task 1: Finite-Difference and Bump Helpers

- [ ] Create `SmoothnessDiagnostics.cs` records:
  - `RateSensitivityPoint`
  - `CouponSlicePoint`
  - `MaturitySlicePoint`
  - `SmoothnessDiagnosticReport`
- [ ] Add helper methods:
  - `BumpZeroRate(FixedRateBondRequest request, int pillarIndex, double bump)`
  - `WithCoupon(FixedRateBondRequest request, double coupon)`
  - `WithMaturity(FixedRateBondRequest request, DateTime maturityDate)`
- [ ] Add derivative helpers for rate, coupon, rate-coupon mixed derivative, and date-based maturity slope.
- [ ] Reject invalid central stencils with clear exceptions rather than silently clamping.

## Task 2: Baseline Slice Generators

- [ ] Add `SmoothnessDiagnostics.RunDefault(IFixedRateBondReferencePricer pricer)` that loads the Phase 2 fixture and builds the regular 10Y request.
- [ ] Generate coupon slices at `0%`, `2%`, `4.5%`, `8%`, and `12%`.
- [ ] Generate zero-rate bump slices for selected pillars `1Y`, `5Y`, `10Y`, `20Y`, and `30Y` over `[-150, -75, 0, 75, 150]` basis points.
- [ ] Generate maturity-date slices around semiannual schedule boundaries from 2Y through 5Y with daily points in a `[-7,+7]` day window.
- [ ] Record cashflow count, coupon cashflow count, dirty price, clean price, accrued amount, first future cashflow date, and final cashflow date for every maturity point.

## Task 3: Regression Tests

- [ ] Add `Coupon_slice_has_near_zero_second_difference`.
- [ ] Add `Rate_bump_slice_is_finite_and_locally_smooth_for_supported_pillars`.
- [ ] Add `Pillars_without_cashflow_interpolation_support_have_zero_dv01`.
- [ ] Add `Maturity_slice_records_schedule_count_changes_near_boundaries`.
- [ ] Add `Diagnostics_mode_writes_summary_without_live_downloads`.
- [ ] Run `dotnet test --filter "FullyQualifiedName~FixedRateBondSmoothnessDiagnosticsTests"`.

## Task 4: Example Diagnostics Mode

- [ ] Update `Program.cs` so `dotnet run --project examples/FixedRateBondSurrogate -- --diagnostics` prints a compact deterministic summary.
- [ ] Keep default `dotnet run --project examples/FixedRateBondSurrogate` output unchanged except for intentional formatting improvements.
- [ ] Include top maturity spike candidates and the largest absolute zero-pillar DV01 values in the diagnostics output.
- [ ] Do not write generated CSV/JSON artifacts by default.

## Task 5: Phase Report and Status

- [ ] Create `phase-3-smoothness-diagnostics.md` with:
  - exact fixture and baseline pricer used;
  - finite-difference step sizes and stencil rules;
  - coupon linearity table;
  - rate-bump smoothness table;
  - maturity boundary table with cashflow-count changes;
  - conclusion on whether PV, slope, DV01, coupon sensitivity, and maturity sensitivity are smooth or piecewise smooth.
- [ ] Update `status.md` with Phase 3 status, files changed, commands run, and next task.
- [ ] Add citations only for public references actually used in the report.
- [ ] Confirm no proprietary names or private details are introduced.

## Task 6: Verification and Phase Closeout

- [ ] Run `rg -n "VTA|proprietary|internal product|private object|company confidential|internal-only|private assessment" examples/FixedRateBondSurrogate tests/ChebyshevSharp.Tests/Finance docs/research/fixed-rate-bond-surrogate docs/docs/examples.md`.
- [ ] Run focused Phase 3 tests.
- [ ] Run `dotnet format --verify-no-changes --verbosity minimal`.
- [ ] Run `dotnet build --configuration Release --no-restore`.
- [ ] Run `dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage" -- RunConfiguration.DisableParallelization=true`.
- [ ] Run `dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj`.
- [ ] Run `dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics`.
- [ ] Run `docfx docs/docfx.json`.
- [ ] Commit Phase 3 work.
- [ ] Push `bond-surrogate-research`.
- [ ] Open one coherent Phase 3 PR only after the local exit gate is satisfied.
- [ ] Keep Phase 3 review fixes inside that same PR; do not start Phase 4 implementation while it is open.
- [ ] Wait for CI/review feedback, then merge the Phase 3 PR or explicitly close it without merge.
- [ ] Record the PR outcome in `status.md` and tracking issue `#191` before starting Phase 4.

