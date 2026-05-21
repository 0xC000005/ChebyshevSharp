# Phase 2 Public Data Fixture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic public yield-curve fixture pipeline for the fixed-rate bond surrogate harness.

**Architecture:** Keep live downloads out of tests and examples. Add an optional Python refresh script under `tools/`, commit a small normalized Federal Reserve nominal-yield-curve JSON fixture under `examples/FixedRateBondSurrogate/Data/`, and load that fixture through the example adapter. The C# harness should consume only pinned fixture data during CI.

**Tech Stack:** Python stdlib CSV/JSON/urllib for refresh, C# `System.Text.Json`, xUnit, QLNet `1.13.1`, DocFX.

---

## Research Inputs

- Federal Reserve nominal yield curve data publishes fitted nominal yield-curve parameters and smoothed yields for hypothetical Treasury securities from 1961 to present. The data page labels this as a staff research product, not an official statistical release.
- The Federal Reserve CSV header identifies `SVENYXX` as continuously compounded zero-coupon yields, `SVENPYXX` as coupon-equivalent par yields, `SVENFXX` as continuously compounded instantaneous forward rates, and `SVEN1FXX` as coupon-equivalent one-year forward rates.
- Gürkaynak, Sack, and Wright define continuously compounded zero-coupon yields by `y_t(n) = -ln(d_t(n)) / n`, with `d_t(n) = exp(-y_t(n)n)`.
- Treasury XML data is official and useful for par-yield examples, but Phase 2 should not bootstrap par yields yet.
- New York Fed SOFR data is useful for later overnight-rate context, but it is not a full term zero curve for the first direct-zero harness.

## Files

- Create `tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py`.
- Create `examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-2026-05-15.json`.
- Modify `examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj` to copy JSON fixture files to output.
- Create `examples/FixedRateBondSurrogate/MarketData.cs` with fixture records and loader methods.
- Modify `examples/FixedRateBondSurrogate/Program.cs` to price from the pinned fixture.
- Modify `tests/ChebyshevSharp.Tests/Finance/FixedRateBondReferencePricerTests.cs` to verify fixture metadata and pricing.
- Create `docs/research/fixed-rate-bond-surrogate/reports/phase-2-public-data-fixtures.md`.
- Modify `docs/docs/examples.md`, `docs/docs/citations.md`, and `docs/research/fixed-rate-bond-surrogate/status.md`.

## Task 1: Pinned Fixture Contract

- [x] Create `MarketData.cs` records:
  - `YieldCurveFixture`
  - `YieldCurveSourceMetadata`
  - `YieldCurvePoint`
- [x] Add `FixedRateBondMarketData.LoadDefaultCurveFixture()` that loads the copied JSON fixture from `AppContext.BaseDirectory`.
- [x] Add `FixedRateBondMarketData.ToZeroRatePillars(YieldCurveFixture fixture, DateTime valuationDate)` that converts percent zero yields to decimal `ZeroRatePillar` values and includes a valuation-date anchor rate.
- [x] Add validation that fixture kind is `zero_coupon_yield`, compounding is `continuous`, units are `percent`, maturities are strictly increasing, and all selected yields are finite.

## Task 2: Data Refresh Tool

- [x] Create `refresh_fed_nominal_yield_curve.py` using only Python stdlib.
- [x] The script must download `https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv`.
- [x] It must skip the CSV note/series metadata rows, parse the real `Date,...` header row, select either a user-supplied `--curve-date` or the latest row with complete `SVENY01` to `SVENY30`, and write normalized JSON.
- [x] The JSON must record:
  - source institution;
  - source URL;
  - download date;
  - curve date;
  - original fields;
  - units;
  - compounding;
  - interpolation convention used by the C# harness;
  - selected maturities and yields.
- [x] Run the tool once to create the pinned 2026-05-15 fixture.

## Task 3: C# Fixture Tests

- [x] Add `Default_curve_fixture_has_expected_metadata`.
- [x] Add `Default_curve_fixture_converts_percent_yields_to_decimal_pillars`.
- [x] Add `Reference_pricer_uses_pinned_public_curve_fixture`.
- [x] Add at least one invalid-fixture test for non-increasing maturities or non-finite values.
- [x] Run focused finance tests.

## Task 4: Example and Documentation

- [x] Update `Program.cs` output to include curve date, source label, and fixture name.
- [x] Update `docs/docs/examples.md` to explain that the finance example uses a pinned Federal Reserve nominal zero-yield fixture and does not download data at runtime.
- [x] Update `docs/docs/citations.md` only if Phase 2 needs additional citation detail beyond the existing public data sources.
- [x] Create the Phase 2 report with source verification, fixture schema, generated file path, commands, and limitations.
- [x] Update `status.md` with Phase 2 status and next task.

## Task 5: Verification and Phase Closeout

- [x] Run `rg -n "VTA|proprietary|internal product|private object|company confidential|internal-only|private assessment" examples/FixedRateBondSurrogate tests/ChebyshevSharp.Tests/Finance tools/RefreshFixedRateBondMarketData docs/research/fixed-rate-bond-surrogate docs/docs/citations.md docs/docs/examples.md`.
- [x] Run `uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --help`.
- [x] Run the refresh script for `--curve-date 2026-05-15` and verify the committed fixture is reproducible.
- [x] Run `dotnet format --verify-no-changes --verbosity minimal`.
- [x] Run `dotnet build --configuration Release --no-restore`.
- [x] Run `dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage" -- RunConfiguration.DisableParallelization=true`.
- [x] Run `dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj`.
- [x] Run `docfx docs/docfx.json`.
- [x] Commit Phase 2 work.
- [x] Push `bond-surrogate-research`.
- [x] Open one coherent Phase 2 PR only after the exit gate is satisfied locally.
- [ ] Keep Phase 2 review fixes inside that same PR; do not open Phase 3 PRs or implementation PRs while it is open.
- [ ] Wait for required CI/review feedback, then merge the Phase 2 PR or explicitly close it without merge.
- [ ] Record the PR outcome, remaining follow-ups, and tracking issue update before starting Phase 3 implementation.
