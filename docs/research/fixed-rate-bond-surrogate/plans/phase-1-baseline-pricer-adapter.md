# Phase 1 Baseline Pricer Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select and integrate a deterministic fixed-rate bond reference-pricer path that Chebyshev examples can call through a small adapter.

**Architecture:** Keep all finance-specific baseline code outside the core `src/ChebyshevSharp` package. Put the runnable harness in `examples/FixedRateBondSurrogate/` and deterministic regression tests in `tests/ChebyshevSharp.Tests/Finance/`. Use QLNet as the first C# baseline candidate because it is pure C#, available from NuGet, derived from QuantLib, and supports `FixedRateBond`, `ZeroCurve`, and `DiscountingBondEngine`.

**Tech Stack:** C# `net10.0`, xUnit, QLNet `1.13.1`, DocFX, public references recorded in `docs/docs/citations.md`.

---

## Research Inputs

- QLNet project page says QLNet is a C# financial library.
- QLNet quick-start documents NuGet installation with `dotnet add package QLNet`.
- QLNet source exposes `FixedRateBond` constructors with schedule, coupons, day counter, payment convention, redemption, and issue date.
- QLNet source exposes `DiscountingBondEngine`, which discounts future bond cashflows with a `YieldTermStructure`.
- QuantLib-Python documentation shows the analogous `DiscountingBondEngine(discountCurve)` pattern.
- NuGet QuantLib exists but wraps the C++ library and notes thread-safety limits; this makes it useful as an optional cross-check but not the first C# CI dependency.

## Files

- Create `examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj`.
- Create `examples/FixedRateBondSurrogate/ReferencePricer.cs`.
- Create `examples/FixedRateBondSurrogate/Program.cs`.
- Modify `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` to reference QLNet and the example project.
- Create `tests/ChebyshevSharp.Tests/Finance/FixedRateBondReferencePricerTests.cs`.
- Modify `docs/docs/citations.md` with finance-baseline references.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md` with phase progress and commands.
- Create `docs/research/fixed-rate-bond-surrogate/reports/phase-1-baseline-pricer.md`.

## Task 1: Example Project Skeleton

- [x] Create `examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj` as a `net10.0` executable that references `src/ChebyshevSharp` and QLNet `1.13.1`.
- [x] Create `examples/FixedRateBondSurrogate/Program.cs` that calls the reference pricer once and prints dirty price, clean price, accrued amount, and cashflow count.
- [x] Run `dotnet build examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj` and verify it compiles.

## Task 2: Reference-Pricer Boundary

- [x] Create immutable records in `ReferencePricer.cs`: `ZeroRatePillar`, `FixedRateBondRequest`, `CashflowInfo`, and `FixedRateBondResult`.
- [x] Create `IFixedRateBondReferencePricer` with `FixedRateBondResult Price(FixedRateBondRequest request)`.
- [x] Implement `QlNetFixedRateBondReferencePricer` using:
  - `Settings.setEvaluationDate`
  - `Schedule`
  - `FixedRateBond`
  - `InterpolatedZeroCurve<Linear>`
  - `DiscountingBondEngine`
- [x] The adapter must expose a deterministic direct-zero curve state, not bootstrapped market-quote risk.
- [x] The adapter must return dirty price, clean price, accrued amount, NPV, settlement value, and cashflow diagnostics.

## Task 3: Baseline Regression Tests

- [x] Add QLNet `1.13.1` and a project reference to `FixedRateBondSurrogate` in the test project.
- [x] Write `Price_returns_finite_outputs_for_regular_fixed_rate_bond`.
- [x] Write `Coupon_dependence_is_linear_for_dirty_price`.
- [x] Write `Principal_and_annuity_recombine_to_coupon_price`.
- [x] Write `Zero_coupon_case_matches_principal_component`.
- [x] Write `Matured_bond_has_zero_value_and_rate_sensitivity`.
- [x] Run the focused finance tests.
- [x] Run the full `dotnet test`.

## Task 4: Documentation and Research Report

- [x] Add citation entries for QLNet, QuantLib, QuantLib-Python bond engine docs, and QLNet source files used for API verification.
- [x] Create `phase-1-baseline-pricer.md` with selected baseline, rejected alternatives, formulas, scope restrictions, and commands run.
- [x] Update `status.md` with Phase 1 status, results, and next task.
- [x] Run `docfx docs/docfx.json`.

## Task 5: Commit Phase 1 Increment

- [x] Run `rg -n "VTA|proprietary|internal product|private object" examples/FixedRateBondSurrogate tests/ChebyshevSharp.Tests/Finance docs/research/fixed-rate-bond-surrogate`.
- [x] Run `dotnet build --no-restore`.
- [x] Run `dotnet test`.
- [x] Run `docfx docs/docfx.json`.
- [x] Commit with `feat: add fixed-rate bond reference pricer harness`.
- [x] Push `bond-surrogate-research`.
- [x] Comment on issue `#191` with the commit, tests, and Phase 1 report path.
