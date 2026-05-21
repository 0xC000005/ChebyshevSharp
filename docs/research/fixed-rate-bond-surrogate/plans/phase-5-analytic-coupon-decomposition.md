# Phase 5 Analytic Coupon Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether removing coupon from the Chebyshev tensor improves fixed-rate bond surrogate sensitivity behavior.

**Architecture:** Keep the Phase 4 direct full-PV experiment intact, then add a separate decomposition experiment. For each curve/maturity point, derive principal PV and fixed-leg annuity from the QLNet reference pricer, build lower-dimensional surrogates for those two components, and reconstruct price as `PV = Principal + coupon * Annuity`. Compare the reconstructed model against the same finite-difference metrics used in Phase 4.

**Tech Stack:** C#/.NET 10 example project, QLNet reference pricer, ChebyshevSharp `ChebyshevTT`, ChebyshevSharp `ChebyshevSlider`, xUnit tests, pinned Federal Reserve fixture, DocFX research docs.

---

## Scope Decisions

- Do not replace the Phase 4 full-PV harness; use it as the baseline failure case.
- Keep the same public Fed fixture and same selected zero-rate pillars: 1Y, 5Y, and 10Y.
- Remove coupon from the surrogate input domain. Component surrogates use four dimensions: 1Y bump, 5Y bump, 10Y bump, and maturity.
- Derive components from the reference pricer instead of implementing independent cashflow math:
  - `Principal(curve, T) = Price(coupon = 0)`.
  - `Annuity(curve, T) = (Price(coupon = c_ref) - Price(coupon = 0)) / c_ref`, with `c_ref = 0.01`.
- Reconstruct price at validation time:
  - `PV(curve, c, T) = Principal(curve, T) + c * Annuity(curve, T)`.
- Treat this as a research result. If errors improve, the next phase still needs maturity splitting; if errors do not improve, record that and inspect whether maturity nonsmoothness dominates.

## Task 1: Failing Tests for Component Identity

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSurrogateReproductionTests.cs`
- Modify later: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`

- [ ] Add a test that calls a public component helper on deterministic Phase 4 validation points.
- [ ] Assert `principal + coupon * annuity` matches the QLNet dirty price within `1e-8`.
- [ ] Assert `annuity` is positive and finite for every validation point.
- [ ] Run:
  - `dotnet test --filter "FullyQualifiedName~FixedRateBondSurrogateReproductionTests"`
  - Expected: fail because the component helper does not exist yet.

## Task 2: Add Component Pricing Helper

**Files:**
- Modify: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`

- [ ] Add `BondCouponDecomposition` with `PrincipalDirtyPrice`, `AnnuityDirtyPrice`, and `ReferenceCoupon`.
- [ ] Add `FixedRateBondSurrogateExperiment.DecomposeCoupon(IFixedRateBondReferencePricer pricer, FixedRateBondRequest request)`.
- [ ] Compute `principal` from `coupon = 0`.
- [ ] Compute `annuity` from `coupon = 0.01` minus `principal`, divided by `0.01`.
- [ ] Keep this helper public enough for tests and documentation examples, but keep lower-level request mapping internal to the example.
- [ ] Run the focused test and verify the identity test passes.

## Task 3: Build Decomposed Surrogate Experiment

**Files:**
- Modify: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`
- Modify: `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSurrogateReproductionTests.cs`

- [ ] Add `FixedRateBondSurrogateExperiment.RunCouponDecomposition(IFixedRateBondReferencePricer pricer)`.
- [ ] Build two `ChebyshevTT` models over the four-dimensional component domain:
  - one for `Principal(curve, T)`;
  - one for `Annuity(curve, T)`.
- [ ] Build two matching `ChebyshevSlider` models over the same component domain with partition `[[0, 1, 2], [3]]`.
- [ ] Reuse the Phase 4 validation coordinates by dropping coupon for component evaluation and reintroducing coupon only in the reconstructed PV function.
- [ ] Compare reconstructed model metrics with the same finite-difference quantities reported in Phase 4.
- [ ] Add tests asserting the report contains `DecomposedTensorTrain` and `DecomposedSlider`, all metrics are finite, and each component model has positive build evaluation counts.

## Task 4: Add CLI and Report

**Files:**
- Modify: `examples/FixedRateBondSurrogate/Program.cs`
- Create: `docs/research/fixed-rate-bond-surrogate/reports/phase-5-analytic-coupon-decomposition.md`
- Modify: `docs/research/fixed-rate-bond-surrogate/status.md`
- Modify if useful: `docs/docs/examples.md`

- [ ] Add `--coupon-decomposition` CLI mode.
- [ ] Print direct Phase 4 headline metrics beside decomposed headline metrics.
- [ ] Write the report with:
  - the decomposition formula;
  - the component-building method from two reference-pricer calls;
  - finite-difference metric table;
  - interpretation of whether coupon removal helped PV, DV01, coupon derivative, and mixed terms.
- [ ] Cite verified public sources for fixed-rate coupon cashflow structure and the existing verified Federal Reserve/QuantLib sources where reused.
- [ ] Update `status.md` with Phase 5 files, commands, findings, and current PR gate.

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
  - `dotnet run --configuration Release --no-build --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --coupon-decomposition`
  - `docfx docs/docfx.json`
- [ ] Open exactly one Phase 5 PR after local verification.
- [ ] Do not begin Phase 6 implementation until the Phase 5 PR is merged or explicitly closed.
