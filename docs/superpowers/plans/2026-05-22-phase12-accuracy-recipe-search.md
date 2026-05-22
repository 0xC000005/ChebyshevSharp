# Phase 12 Accuracy Recipe Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build diagnostics that identify why the fixed-rate bond surrogate
still has material PV and sensitivity error, then compare stronger candidate
recipes without changing the public 62-coordinate wrapper.

**Architecture:** Keep the Phase 12 work inside the fixed-rate bond example
and tests. Add one benchmark module for accuracy-recipe diagnostics, wire it to
the existing example CLI, and document findings in the research report.

**Tech Stack:** C#/.NET 10 tests, QLNet-backed reference pricer, ChebyshevSharp
TensorTrain/Slider objects where needed, DocFX documentation.

---

## File Map

- Create `examples/FixedRateBondSurrogate/AccuracyRecipeSearch.cs`.
- Modify `examples/FixedRateBondSurrogate/Program.cs`.
- Create `tests/ChebyshevSharp.Tests/Finance/FixedRateBondAccuracyRecipeSearchTests.cs`.
- Modify `docs/research/fixed-rate-bond-surrogate/reports/phase-12-accuracy-recipe-search.md`.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md`.

## Task 1: Add Accuracy Report Types

- [x] Create `AccuracyRecipeSearch.cs` with immutable report records:
  `AccuracyRecipeSearchReport`, `AccuracyProjectionOracleSummary`,
  `AccuracyDerivativeOracleSummary`, and `AccuracyScheduleDispatchSummary`.
- [x] Include public fields for model name, build evaluations, PV max absolute
  error, PV max relative error, maturity-slope max absolute error, maturity
  slope max relative error, coupon-maturity max absolute error, and
  coupon-maturity max relative error.
- [x] Add a `RunDefault(IFixedRateBondReferencePricer pricer)` entry point that
  loads `FixedRateBondMarketData.LoadDenseSemiannualCurveFixture()` and uses
  `RegularThirtyYearFromDenseFixture()`.

## Task 2: Add Projection Oracle

- [x] Implement deterministic validation points by reusing the same full 60D
  bump shapes used by Phase 6 and Phase 7.
- [x] Implement the level/slope/curvature projection and reconstruction already
  used by the structured-alternatives benchmark.
- [x] For each validation point, price the original 60D bumped curve and the
  reconstructed curve with the QLNet reference pricer.
- [x] Summarize max PV absolute and relative projection error.
- [x] Add a test asserting the projection oracle reports nonzero error on at
  least one arbitrary clone point and near-zero error on a factor-aligned point.

## Task 3: Add Derivative Oracle

- [x] Implement DV01 and maturity-slope finite differences with at least three
  step sizes: `1e-4`, `5e-5`, and `1e-5` for rate bumps, and one-day,
  three-day, and seven-day maturity steps.
- [x] Report how much the baseline derivative changes as the step changes.
- [ ] Add one-sided maturity slopes around schedule split candidates from Phase
  9 so central differences do not silently cross pricing regimes.
- [x] Add a test asserting post-maturity pillar DV01 stays within numerical
  tolerance for the baseline derivative oracle.

## Task 4: Add Stronger Candidate Recipes

- [x] Implement a richer deterministic factor projection candidate with five
  smooth polynomial factors before building another tensor candidate.
- [x] Implement a schedule-aware active-support oracle that preserves
  `Eval(double[] fullPoint)` evidence and selects pillars around cashflow
  support for the current maturity neighbourhood.
- [x] Implement a first 10Y active-pillar TT candidate that preserves the full
  public wrapper and measures local PV and Greek errors.
- [x] Implement a narrowed/higher-resolution 10Y active-pillar TT candidate to
  test whether rank/window/resolution are driving the Greek errors.
- [x] Add one-sided maturity slope metrics to check whether central differences
  are the only reason maturity sensitivity looks bad.
- [x] Skip the analytic-coupon active-pillar TT as a separate local-TT variant
  after the stronger formula-aware cashflow-kernel candidate supersedes it.
- [x] Add fixed-trade curve-only control results to distinguish a production
  scenario surrogate from the parametric new-bond clone.
- [x] Implement a schedule-resolved cashflow Chebyshev-kernel candidate that
  preserves the 62-coordinate wrapper, resolves cashflow schedules from
  maturity, keeps coupon/notional algebraic, and uses local 1D/2D Chebyshev
  discount kernels.
- [x] Broaden the cashflow-kernel validation bank across coupons, maturities,
  parallel shifts, slopes, sinusoidal shocks, and local 10Y curve bumps.

## Task 5: Wire CLI And Tests

- [x] Add `--accuracy-recipe-search` to `FixedRateBondExample.Run`.
- [x] Print a compact table comparing the projection oracle, derivative oracle,
  richer factor candidate, active-pillar candidates, fixed-trade control, and
  cashflow-kernel candidate.
- [x] Add focused tests that run the CLI mode and assert the report contains
  the expected model names and nonempty diagnostics.
- [x] Add focused tests requiring the cashflow-kernel candidate to preserve the
  full wrapper, validate at least 80 points, match PV/risk metrics, and report
  measured acceleration.
- [x] Run:

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondAccuracyRecipeSearchTests" --verbosity minimal
```

## Task 6: Document Findings

- [x] Update `reports/phase-12-accuracy-recipe-search.md` with the initial oracle table,
  oracle interpretations, and next-recipe decision.
- [x] Update `status.md` with the Phase 12 branch, verification commands, and
  current decision.
- [x] Add only public-facing evidence to the report; keep workflow-only details
  in the plan/status files.

## Task 7: Local Exit Gate

- [ ] Run:

```bash
dotnet format --verify-no-changes --verbosity minimal
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBond" --verbosity minimal
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --accuracy-recipe-search
docfx docs/docfx.json
git diff --check
```

- [x] Record initial oracle results in `status.md` and the Phase 12 report.
- [x] Push `phase12-accuracy-recipe-search`.
- [x] Update issue #191 with the branch, plan, and evidence.
- [x] Open one coherent Phase 12 PR only after local verification passes.
