# Phase 9 Maturity Special Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the Phase 9 fixed-rate bond maturity-special-point benchmark, then decide whether ChebyshevSharp needs a reusable high-dimensional piecewise routing feature.

**Architecture:** Keep the public research contract as `curve bumps[60], coupon, maturity -> dirty PV`. Internally, reuse Phase 8's analytic coupon decomposition and compare maturity treatments: Phase 8 controls, schedule-aware special points, automatic detector candidates, and a hybrid only if justified by evidence.

**Tech Stack:** C#/.NET 10 tests, QLNet reference pricer, ChebyshevSharp dense/TT models, DocFX research docs, GitHub issue #191 for tracking.

---

## File Map

- Create `examples/FixedRateBondSurrogate/MaturitySpecialPointsBenchmark.cs`: Phase 9 report records, maturity breakpoint inventory, candidate generation, and model comparisons.
- Modify `examples/FixedRateBondSurrogate/Program.cs`: add `--maturity-special-points` CLI mode and concise console output.
- Create `tests/ChebyshevSharp.Tests/Finance/FixedRateBondMaturitySpecialPointTests.cs`: structural tests, CLI tests, finite-value checks, and candidate sanity checks.
- Create `docs/research/fixed-rate-bond-surrogate/reports/phase-9-maturity-special-points.md`: evidence report and decision.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md`: Phase 9 progress, verification, PR/issue links.
- Modify `docs/docs/examples.md`: add the Phase 9 example command only after the CLI/report is working.

## Task 1: Report Skeleton and CLI Mode

**Files:**
- Create: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondMaturitySpecialPointTests.cs`
- Create: `examples/FixedRateBondSurrogate/MaturitySpecialPointsBenchmark.cs`
- Modify: `examples/FixedRateBondSurrogate/Program.cs`

- [ ] **Step 1: Write failing tests for the Phase 9 report shape**

```csharp
[Fact]
public void Phase9_report_preserves_full_public_wrapper()
{
    MaturitySpecialPointsReport report = MaturitySpecialPointsBenchmark.RunDefault(Pricer);

    Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
    Assert.Equal(62, report.PublicInputDimensionCount);
    Assert.NotEmpty(report.BreakpointInventory);
    Assert.Contains(report.Candidates, candidate => candidate.Name == "Schedule-aware special points");
    Assert.Contains(report.Candidates, candidate => candidate.Name == "Automatic detector candidates");
}
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondMaturitySpecialPointTests" --verbosity minimal
```

Expected: compile failure because `MaturitySpecialPointsBenchmark` does not exist.

- [ ] **Step 3: Add minimal report records and a deterministic stub report**

Implement `MaturitySpecialPointsReport`, `MaturityBreakpointInventoryPoint`, and `MaturitySpecialPointCandidateSummary` with real wrapper metadata and a placeholder inventory generated from the reference request.

- [ ] **Step 4: Add `--maturity-special-points` CLI mode**

Console output must include:

```text
Fixed-rate bond maturity special points
full wrapper
Breakpoint inventory
Schedule-aware special points
Automatic detector candidates
```

- [ ] **Step 5: Run focused tests until GREEN**

Run the focused command from Step 2. Expected: all Phase 9 tests pass.

## Task 2: Maturity Breakpoint Inventory

**Files:**
- Modify: `examples/FixedRateBondSurrogate/MaturitySpecialPointsBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondMaturitySpecialPointTests.cs`

- [ ] **Step 1: Add failing tests for schedule-regime evidence**

Test that the inventory contains semiannual boundary windows, finite left/right slopes, finite second differences, positive cashflow counts, and at least one point where cashflow count or final accrual metadata changes.

- [ ] **Step 2: Implement inventory scan**

Use the dense semiannual fixture, valuation date, and maturities from 2Y to 30Y. Around each semiannual schedule candidate, scan offsets `-7` through `+7` days, price each maturity through `IFixedRateBondReferencePricer`, and record:

```text
boundary date, offset days, maturity date, maturity years, cashflow count,
coupon cashflow count, final cashflow date, final coupon accrual period,
dirty price, left slope/year, right slope/year, central slope/year,
second difference, schedule-regime changed flag
```

- [ ] **Step 3: Add defensive interpretation**

The report must say these are one-dimensional maturity diagnostics over the fixed restricted product family, not proof that all bond products have the same breakpoints.

- [ ] **Step 4: Run focused tests**

Expected: finite metrics and a non-empty schedule-regime evidence set.

## Task 3: Candidate Generation

**Files:**
- Modify: `examples/FixedRateBondSurrogate/MaturitySpecialPointsBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondMaturitySpecialPointTests.cs`

- [ ] **Step 1: Test schedule-aware candidates**

Assert schedule candidates are sorted, inside `[2.0, 30.0]`, deduplicated, and derived from inventory points with schedule-regime changes.

- [ ] **Step 2: Test automatic detector candidates**

Assert detector candidates are sorted, inside `[2.0, 30.0]`, deduplicated, and correspond to the largest absolute second differences after minimum-distance filtering.

- [ ] **Step 3: Implement candidate generation**

Add schedule-aware, automatic detector, and hybrid candidate summaries. The hybrid candidate is allowed only when both schedule-aware and detector lists are non-empty.

## Task 4: Candidate Model Comparison

**Files:**
- Modify: `examples/FixedRateBondSurrogate/MaturitySpecialPointsBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondMaturitySpecialPointTests.cs`

- [ ] **Step 1: Add tests for model summaries**

Assert every model summary has public dimension count `62`, finite metrics, non-negative build evaluations, positive bucket/piece count, and a clear interpretation.

- [ ] **Step 2: Implement controls**

Include Phase 8 controls by rerunning or reusing:

```text
Curve-factor decomposed tensor
Semiannual bucketed decomposed curve-factor tensor
```

- [ ] **Step 3: Implement special-point piecewise factor models**

Build internal decomposed factor tensors over level/slope/curvature/maturity per maturity piece. Keep the outer evaluator a full 62-coordinate wrapper.

- [ ] **Step 4: Validate metrics**

Reuse the Phase 6-8 metric bank: PV, selected zero-pillar DV01, coupon derivative, maturity sensitivity, rate-coupon mixed, rate-maturity mixed, rate-rate mixed, and coupon-maturity mixed.

## Task 5: Documentation and Decision

**Files:**
- Create: `docs/research/fixed-rate-bond-surrogate/reports/phase-9-maturity-special-points.md`
- Modify: `docs/research/fixed-rate-bond-surrogate/status.md`
- Modify: `docs/docs/examples.md`

- [ ] **Step 1: Write the Phase 9 report**

Report the inventory, candidate lists, model metrics, worst points, and decision. State explicitly whether evidence supports a future `PiecewiseChebyshevTT` or similar API.

- [ ] **Step 2: Update user-facing examples**

Add:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

- [ ] **Step 3: Update status and issue tracking**

Update the Phase 9 status entry and add a tracking comment to issue #191 summarizing results and the PR link once opened.

## Task 6: Verification and PR Gate

**Files:**
- All Phase 9 files

- [ ] **Step 1: Run focused tests**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondMaturitySpecialPointTests" --verbosity minimal
```

- [ ] **Step 2: Run fixed-rate bond test slice**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBond" --verbosity minimal
```

- [ ] **Step 3: Run the example**

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

- [ ] **Step 4: Run docs**

```bash
docfx docs/docfx.json
```

- [ ] **Step 5: Run full verification**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --verbosity minimal
git diff --check
```

- [ ] **Step 6: Open one coherent Phase 9 PR**

Only after local exit gates pass, open one PR for Phase 9. Do not create separate PRs for skeleton, inventory, candidates, and docs.

## Exit Criteria

Phase 9 is complete only when the report answers:

1. whether maturity failures align with schedule-regime changes;
2. whether schedule-aware special points improve maturity and mixed-term errors versus uniform buckets;
3. whether automatic detection finds useful breakpoints without excessive pieces;
4. whether a hybrid router is materially better than schedule-only routing;
5. whether a future ChebyshevSharp library feature is justified, and what minimal API that feature should expose.
