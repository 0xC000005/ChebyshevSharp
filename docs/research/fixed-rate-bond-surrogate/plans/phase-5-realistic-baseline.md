# Phase 5 Realistic Baseline Implementation Plan

> **For agentic workers:** Implement this phase before any further surrogate work. The exit gate is a realistic, audited fixed-rate bond baseline with dense curve support, explicit conventions, tests, and documentation.

**Goal:** Replace the sparse tutorial baseline with a more realistic direct zero-rate baseline that uses semiannual curve support, explicit fixed-rate bond conventions, and deterministic validation against QLNet.

**Scope boundary:** Do not implement analytic coupon decomposition, maturity splitting, or adaptive Chebyshev fixes in this phase. The purpose is to make the exact baseline credible before fitting any more surrogates.

**Tech stack:** C#/.NET 10 example project, QLNet reference pricer, xUnit tests, pinned Federal Reserve nominal yield-curve data, optional Python refresh tool, DocFX docs.

## Source Decisions

- Use the Federal Reserve nominal yield-curve CSV as the public source.
- Keep the existing sparse annual fixture for the compact Phase 4 surrogate reproduction.
- Add a dense fixture sampled every six months from the published fitted curve parameters for the same pinned curve date.
- Keep live downloads out of C# tests and examples. The Python refresh tool may download or read a local CSV and write pinned JSON.
- Treat the direct zero-rate curve as the first sensitivity state. Bootstrapped market-quote DV01 remains out of scope.

## Tasks

- [x] Add failing tests for a dense semiannual fixture, explicit convention metadata, regular 30Y cashflows, price sanity, coupon ordering, and notional scaling.
- [x] Add `maturity_months` support while preserving the existing annual fixture schema.
- [x] Add `LoadDenseSemiannualCurveFixture()` and `RegularThirtyYearFromDenseFixture()`.
- [x] Expose the exact QLNet convention summary used by the adapter.
- [x] Add the dense fixture `fed-nominal-yield-curve-semiannual-2026-05-15.json`.
- [x] Align dense fixture sampling with the QLNet Actual/365 curve-time convention.
- [x] Add manual zero-rate interpolation and public Treasury auction price sanity checks.
- [x] Extend the refresh tool with `--density semiannual-svensson` and `--input-csv`.
- [x] Update the public example to use the dense 30Y baseline by default.
- [x] Update research status, report, and docs examples.
- [x] Run closeout verification.

## Validation Commands

```bash
uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --help
uv run tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py --curve-date 2026-05-15 --download-date 2026-05-20 --density semiannual-svensson --input-csv /tmp/feds200628.csv --output /tmp/fed-dense-check.json
dotnet test --filter "FullyQualifiedName~FixedRateBond"
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj
```

Closeout should additionally run formatting, Release build/tests, private-name scan, and DocFX.

## Exit Gate

The phase is complete when the dense baseline is reproducible, the example prints a 30Y semiannual bond with 61 curve pillars and 61 cashflows, the QLNet conventions are explicit, and all fixed-rate bond tests pass. Stop after this gate and wait before resuming surrogate development.
