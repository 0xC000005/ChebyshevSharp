# Phase 4 Surrogate Problem Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public, deterministic reproduction harness showing whether full-PV Chebyshev TT and Slider surrogates can have acceptable PV error while producing weaker DV01, coupon, maturity, or mixed-term estimates.

**Architecture:** Keep this phase deliberately restricted: train direct full-PV surrogates against the existing QLNet reference pricer on the pinned Federal Reserve nominal zero-yield fixture. Use finite differences around the reference pricer and around each surrogate for all reported sensitivities so the comparison is tool-agnostic and easy to audit. Do not implement analytic coupon decomposition or maturity splitting yet; those are later phases.

**Tech Stack:** C#/.NET 10 example project, QLNet reference pricer, ChebyshevSharp `ChebyshevTT`, ChebyshevSharp `ChebyshevSlider`, xUnit tests, pinned public Fed fixture, DocFX research docs.

---

## Scope Decisions

- Use only public-safe names: fixed-rate bond, reference pricer, exact pricer, TT surrogate, Slider surrogate.
- Use the committed Fed fixture `examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-2026-05-15.json`; live Yahoo Finance, FRED, or Federal Reserve downloads are out of scope for this deterministic phase.
- Approximate a compact domain around one regular ten-year bond:
  - selected zero-rate pillar bumps in basis points,
  - coupon,
  - maturity in years converted to maturity dates for the reference pricer.
- Keep the first reproduction small enough for CI by using selected curve pillars rather than every curve pillar.
- Record build cost, sample count, max/mean PV error, and sensitivity errors.
- Treat a negative result as useful. Tests verify harness behavior and invariants, not that one model must fail.

## Task 1: Failing Tests for Surrogate Experiment Records

**Files:**
- Create: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSurrogateReproductionTests.cs`
- Create later: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`

- [x] Add tests that call `FixedRateBondSurrogateExperiment.RunDefault(new QlNetFixedRateBondReferencePricer())`.
- [x] Assert the report contains two model summaries named `TensorTrain` and `Slider`.
- [x] Assert every metric is finite, every validation point is inside the configured domain, and build evaluation counts are positive.
- [x] Assert the selected 20Y and 30Y pillars are not part of Phase 4's compact curve input set for the ten-year example, because Phase 3 showed they have zero direct support in the current interpolation setup.
- [x] Run:
  - `dotnet test --filter "FullyQualifiedName~FixedRateBondSurrogateReproductionTests"`
  - Expected: fail because `FixedRateBondSurrogateExperiment` does not exist.

## Task 2: Implement Experiment Domain and Baseline Mapping

**Files:**
- Create: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`

- [x] Define immutable records:
  - `SurrogateInputDimension`
  - `SurrogateValidationPoint`
  - `SurrogateMetricSummary`
  - `SurrogateModelSummary`
  - `SurrogateExperimentReport`
- [x] Define `FixedRateBondSurrogateExperiment.RunDefault(IFixedRateBondReferencePricer pricer)`.
- [x] Use dimensions:
  - 1Y zero-rate bump in `[-150, 150]` bp,
  - 5Y zero-rate bump in `[-150, 150]` bp,
  - 10Y zero-rate bump in `[-150, 150]` bp,
  - coupon in `[0.00, 0.12]`,
  - maturity years in `[8.0, 12.0]`.
- [x] Convert point coordinates to `FixedRateBondRequest` by applying bump dimensions to the Fed zero curve, setting coupon directly, and mapping maturity years to `valuationDate.AddDays(Math.Round(365.25 * maturityYears))`.
- [x] Add validation points from a deterministic set: center, rate corners, high coupon, low coupon, near maturity endpoints, and deterministic scenario points.
- [x] Add baseline finite-difference helpers for PV, zero-pillar DV01, coupon derivative, maturity slope, rate-coupon mixed derivative, and rate-maturity mixed derivative.
- [x] Run the focused test and verify the failures move from missing type to missing model construction.

## Task 3: Build TT and Slider Full-PV Surrogates

**Files:**
- Modify: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`

- [x] Build a `ChebyshevTT` over the five-dimensional full-PV function with `method: "cross"`, deterministic seed, modest node counts, and a rank cap suitable for CI.
- [x] Build a `ChebyshevSlider` over the same function with partition `[[0, 1, 2], [3], [4]]` and pivot at the domain midpoint.
- [x] Evaluate each model at the deterministic validation points.
- [x] Compute model-side finite differences using the same step sizes as the baseline, with central differences where the shifted point stays inside the domain and one-sided differences near boundaries.
- [x] Fill summary metrics for:
  - absolute and relative PV error,
  - zero-pillar DV01 error,
  - coupon derivative error,
  - maturity slope error,
  - rate-coupon mixed derivative error,
  - rate-maturity mixed derivative error.
- [x] Run the focused test and fix only this experiment code until it passes.

## Task 4: Add CLI Output and Research Report

**Files:**
- Modify: `examples/FixedRateBondSurrogate/Program.cs`
- Create: `docs/research/fixed-rate-bond-surrogate/reports/phase-4-surrogate-reproduction.md`
- Modify: `docs/research/fixed-rate-bond-surrogate/status.md`
- Modify if useful: `docs/docs/examples.md`

- [x] Add a `--surrogate-reproduction` CLI mode that prints the fixture ID, dimensions, model build costs, and compact metric table.
- [x] Write the report with formulas for the compared quantities:
  - PV error,
  - zero-pillar DV01,
  - coupon derivative,
  - maturity slope,
  - mixed finite differences.
- [x] Explain why this phase intentionally uses a full-PV tensor before testing the later analytic coupon decomposition.
- [x] Cite only verified public sources and local ChebyshevSharp docs where needed.
- [x] Update `status.md` to mark Phase 4 in progress, include the plan/report links, and preserve the one-active-PR gate.

## Task 5: Closeout Verification and One Phase PR

**Files:**
- Modify this plan file as steps complete.

- [ ] Run private-name scan:
  - `rg -n "VTA|proprietary|internal product|private object|company confidential|internal-only|private assessment" examples/FixedRateBondSurrogate tests/ChebyshevSharp.Tests/Finance docs/research/fixed-rate-bond-surrogate docs/docs/examples.md`
- [ ] Run focused tests:
  - `dotnet test --filter "FullyQualifiedName~FixedRateBondSurrogateReproductionTests"`
- [ ] Run full verification:
  - `dotnet format --verify-no-changes --verbosity minimal`
  - `dotnet build --configuration Release --no-restore`
  - `dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage" -- RunConfiguration.DisableParallelization=true`
  - `dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --surrogate-reproduction`
  - `docfx docs/docfx.json`
- [ ] Open exactly one Phase 4 PR after local verification.
- [ ] Do not begin Phase 5 implementation until the Phase 4 PR is merged or explicitly closed.
