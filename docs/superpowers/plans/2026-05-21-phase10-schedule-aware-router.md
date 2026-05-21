# Phase 10 Schedule-Aware Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a schedule-aware high-dimensional piecewise router prototype for the fixed-rate bond surrogate harness, validate one-sided sensitivity behavior, and decide whether a ChebyshevSharp library feature is justified.

**Architecture:** Keep the public wrapper as `curve bumps[60], coupon, maturity -> dirty PV`. Internally, route by maturity into schedule-derived pieces, use Phase 8 analytic coupon recombination, and compare against Phase 9 controls using the same metric bank plus one-sided maturity diagnostics.

**Tech Stack:** C#/.NET 10 tests, QLNet reference pricer, ChebyshevSharp dense factor tensors, DocFX research docs, GitHub issue #191 for tracking.

---

## File Map

- Create `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`: report records, router prototype, one-sided metrics, and control comparison.
- Modify `examples/FixedRateBondSurrogate/Program.cs`: add `--schedule-aware-router`.
- Create `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`: router shape, breakpoint provenance, one-sided metric, and CLI tests.
- Create `docs/research/fixed-rate-bond-surrogate/reports/phase-10-schedule-aware-router.md`: results and API decision.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md`: phase status, verification, PR and issue links.
- Modify `docs/docs/examples.md`: add the new command only after the CLI passes.

## Task 1: CLI and Report Skeleton

**Files:**
- Create: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`
- Create: `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`
- Modify: `examples/FixedRateBondSurrogate/Program.cs`

- [x] **Step 1: Write the failing wrapper/report test**

```csharp
[Fact]
public void Phase10_report_preserves_full_public_wrapper()
{
    ScheduleAwareRouterReport report = ScheduleAwareRouterBenchmark.RunDefault(Pricer);

    Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
    Assert.Equal(62, report.PublicInputDimensionCount);
    Assert.NotEmpty(report.Pieces);
    Assert.Contains(report.Models, model => model.ModelName == "Schedule-aware router decomposed factor tensor");
}
```

- [x] **Step 2: Run the focused test and confirm RED**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondScheduleAwareRouterTests" --verbosity minimal
```

Expected: compile failure because `ScheduleAwareRouterBenchmark` does not exist.

- [x] **Step 3: Add minimal report records and CLI mode**

Define `ScheduleAwareRouterReport`, `ScheduleAwareRouterPieceSummary`, and `ScheduleAwareRouterDecision`. Add `--schedule-aware-router` output containing:

```text
Fixed-rate bond schedule-aware router
full wrapper
Pieces
One-sided maturity diagnostics
Decision
```

- [x] **Step 4: Run focused tests until GREEN**

Expected: focused Phase 10 tests pass.

## Task 2: Router Core

**Files:**
- Modify: `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`

- [x] **Step 1: Add router piece tests**

Assert piece intervals are sorted, non-overlapping, cover `[2.0, 30.0]`, and route exact boundary points by half-open intervals except for the final closed interval.

- [x] **Step 2: Implement router**

Implement an internal `ScheduleAwarePiecewiseRouter` with:

```csharp
public int PieceCount { get; }
public int BuildEvaluations { get; }
public double BuildSeconds { get; }
public double Eval(double[] fullPoint);
public ScheduleAwareRouterPieceSummary Route(double maturityYears);
```

- [x] **Step 3: Run focused tests**

Expected: piece coverage, boundary routing, and wrapper preservation tests pass.

## Task 3: Schedule Breakpoint Source

**Files:**
- Modify: `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`

- [x] **Step 1: Add provenance tests**

Assert breakpoints are derived from Phase 9 schedule-aware candidates, not automatic detector candidates:

```csharp
Assert.All(report.ScheduleBreakpoints, bp =>
    Assert.Contains(report.Phase9ScheduleCandidateYears, y => Math.Abs(y - bp) < 1e-8));
```

- [x] **Step 2: Implement source extraction**

Reuse `MaturitySpecialPointsBenchmark.RunDefault(pricer)` to obtain schedule candidates, then filter to `(2.0, 30.0)`.

- [x] **Step 3: Run focused tests**

Expected: provenance and domain tests pass.

## Task 4: One-Sided Sensitivity Diagnostics

**Files:**
- Modify: `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`

- [x] **Step 1: Add one-sided metric tests**

Assert the report includes left and right maturity sensitivity near at least five schedule breakpoints, and all values are finite.

- [x] **Step 2: Implement one-sided metrics**

For a breakpoint `T`, evaluate:

```text
left slope  = (f(T - eps) - f(T - 2eps)) / eps
right slope = (f(T + 2eps) - f(T + eps)) / eps
```

Use `eps = 7.0 / 365.25`, and clamp sample points inside `[2.0, 30.0]`.

- [x] **Step 3: Run focused tests**

Expected: one-sided diagnostics are finite and named by breakpoint.

## Task 5: Model Comparison and Decision

**Files:**
- Modify: `examples/FixedRateBondSurrogate/ScheduleAwareRouterBenchmark.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondScheduleAwareRouterTests.cs`

- [x] **Step 1: Add comparison tests**

Assert the model bank contains:

```text
Phase 9 global decomposed factor control
Phase 9 uniform 0.5Y control
Phase 9 schedule-aware special-point control
Schedule-aware router decomposed factor tensor
```

- [x] **Step 2: Implement comparison**

Reuse the Phase 9 validation bank and metric summaries. The new router must report PV, selected zero-pillar DV01, coupon derivative, maturity sensitivity, rate-coupon mixed, rate-maturity mixed, rate-rate mixed, and coupon-maturity mixed.

- [x] **Step 3: Implement decision**

The decision must state whether the router should remain example-local, become a follow-up public API issue, or needs another modelling phase first.

## Task 6: Documentation and Tracking

**Files:**
- Create: `docs/research/fixed-rate-bond-surrogate/reports/phase-10-schedule-aware-router.md`
- Modify: `docs/research/fixed-rate-bond-surrogate/status.md`
- Modify: `docs/docs/examples.md`

- [x] **Step 1: Write the Phase 10 report**

Include wrapper contract, architecture, schedule-breakpoint provenance, one-sided sensitivity results, model comparison, and decision.

- [x] **Step 2: Update examples docs**

Add:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

- [x] **Step 3: Update status and issue #191**

Record the phase branch, report path, verification commands, and PR link once opened.

## Task 7: Verification and PR Gate

**Files:**
- All Phase 10 files

- [x] **Step 1: Run focused tests**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondScheduleAwareRouterTests" --verbosity minimal
```

- [x] **Step 2: Run fixed-rate bond slice**

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBond" --verbosity minimal
```

- [x] **Step 3: Run example**

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

- [x] **Step 4: Run closeout checks**

```bash
dotnet format --verify-no-changes --verbosity minimal
docfx docs/docfx.json
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --verbosity minimal
git diff --check
```

- [x] **Step 5: Open one coherent Phase 10 PR**

Open the PR only after local verification passes. Keep all Phase 10 review fixes in that PR.

## Exit Criteria

Phase 10 is complete only when the report answers:

1. whether the explicit router improves or clarifies the Phase 9 schedule-aware result;
2. whether one-sided split-point diagnostics are more meaningful than central cross-boundary derivatives;
3. whether the remaining errors are due to routing, curve factor projection, derivative estimation, or the restricted product model;
4. whether the next step is a public ChebyshevSharp API issue, another example-local modelling phase, or tutorial documentation.
