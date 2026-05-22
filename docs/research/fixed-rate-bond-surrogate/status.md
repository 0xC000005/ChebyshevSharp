# Fixed-Rate Bond Surrogate Research Status

## Current Objective

Create a public, non-proprietary reproduction harness inside ChebyshevSharp that tests Chebyshev tensor surrogates around a trusted fixed-rate bond baseline pricer.

Design spec: [Fixed-Rate Bond Surrogate Reproduction Harness Design](design.md)

This is the primary scope for the next small version update.

Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191)

Working branch: `phase12-accuracy-recipe-search`

Last completed phase PR: [#202](https://github.com/0xC000005/ChebyshevSharp/pull/202), merged on 2026-05-22 as `f4dafca`.

## Confidentiality Guardrail

Do not mention proprietary systems, proprietary interface names, internal product names, or private object models in code, docs, reports, issues, PRs, branch names, or commits.

Use generic public terms:

- fixed-rate bond
- reference pricer
- legacy pricer
- surrogate
- exact pricer
- public finance example

## Phase Checklist

| Phase | Status | Exit Gate |
| --- | --- | --- |
| 0. Setup and guardrails | Complete | Plan, status file, tracking issue, working branch, and report folders exist with public-safe language |
| 1. Baseline pricer selection and adapter | Complete | QuantLib/QLNet/Python baseline selected and callable behind generic adapter |
| 2. Data fixture pipeline | Complete | Public curve fixture generated, pinned, and documented; no live downloads in CI |
| 3. Smoothness diagnostics | Complete | Report identifies PV/slope/sensitivity smoothness and maturity breakpoints |
| 4. Reproduce surrogate problem | Complete | TT/Slider report confirms or rejects PV-good/Greeks-bad behavior |
| 5. Realistic dense-curve baseline | Complete | Dense semiannual curve fixture, explicit conventions, 30Y baseline tests, and docs complete |
| 6. Naive dense-baseline surrogate discovery | Complete | Dense-baseline naive TT/Slider failure evidence and maturity-smoothness evidence documented |
| 7. Structured alternatives benchmark | Complete | Controlled alternatives compared against the Phase 6 evidence bank |
| 8. Analytic coupon decomposition | Complete | Principal/annuity surrogate comparison completed |
| 9. Maturity splitting and adaptive knots | Complete | Schedule-aware and detector special-point comparison completed |
| 10. Schedule-aware high-dimensional router | Complete | PR #201 merged; router remains example-local and does not justify a generic kink-detection API yet |
| 11. Tutorial and documentation | Complete | PR #202 merged; public tutorial and case-study documentation are on main |
| 12. Accuracy recipe search | In progress | Dominant residual error source identified and next modelling recipe selected |
| 13. Library improvement issues | Not started | Evidence-backed issues opened only where needed |

## Environment Readiness

Last checked: 2026-05-20.

- Last completed branch/PR: `bond-surrogate-research`, PR [#196](https://github.com/0xC000005/ChebyshevSharp/pull/196), merged into `main` on 2026-05-21.
- Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191).
- Python tooling: `uv 0.9.2` at `/home/max/.local/bin/uv`; Python `3.13.9`.
- Python data stack smoke: `uv run --with pandas --with pandas-datareader --with requests ...` imports `pandas`, `pandas_datareader`, and `requests`.
- Python baseline candidate smoke: `uv run --with QuantLib ...` imports QuantLib `1.42.1`.
- .NET tooling: SDK `10.0.107`; DocFX `2.78.4`.
- .NET baseline candidate smoke: temporary NuGet restore installed QLNet `1.13.1` successfully.
- Baseline tests: `dotnet test` passed 1649 tests with 0 failures from the workflow worktree.
- Baseline build: `dotnet build --no-restore` succeeded with 0 warnings and 0 errors.
- Documentation build: `docfx docs/docfx.json` succeeded with 0 warnings and 0 errors.
- Public source reachability: C# `HttpClient` received HTTP 200 from Federal Reserve nominal yield curve CSV, Treasury daily yield curve XML feed, and New York Fed SOFR averages/index page.
- Sandbox note: `dotnet` and `docfx` need unrestricted execution for MSBuild/Roslyn IPC; sandboxed runs can fail with socket or named-pipe permission errors.

## Next Task

Start Phase 12 on `phase12-accuracy-recipe-search`: isolate why the fixed-rate bond surrogate remains inaccurate after schedule-aware routing, then compare stronger candidate recipes while preserving the full 62-coordinate wrapper.

## Phase 12 Notes

- Plan: [Phase 12 Accuracy Recipe Search Plan](plans/phase-12-accuracy-recipe-search.md).
- Implementation plan path: `docs/superpowers/plans/2026-05-22-phase12-accuracy-recipe-search.md`.
- Report draft: [Phase 12 Report: Accuracy Recipe Search](reports/phase-12-accuracy-recipe-search.md).
- Working branch: `phase12-accuracy-recipe-search`.
- Phase PR: [#203](https://github.com/0xC000005/ChebyshevSharp/pull/203).
- Scope boundary: do not add reusable ChebyshevSharp APIs, bump versions, or release packages in this phase.
- Required wrapper contract: every candidate remains callable as the full 62-coordinate interface, `curve bumps[60] + coupon + maturity`.
- Research refresh: Chebfun supports splitting piecewise-smooth functions; OpenGamma Strata validates PV and bucketed PV01 as core outputs; Federal Reserve H.15 documents public constant-maturity curve construction; yield-curve PCA references support level/slope/curvature as a common compression idea; Tensor Train Cross references require held-out validation rather than rank assumptions.
- Required first diagnostics: projection oracle, derivative-step oracle, schedule-dispatch oracle, richer factor candidate, schedule-aware active-pillar candidate, fixed-trade curve-only control, and a full-wrapper schedule-resolved cashflow Chebyshev-kernel candidate.
- Initial oracle implementation: `examples/FixedRateBondSurrogate/AccuracyRecipeSearch.cs`, CLI flag `--accuracy-recipe-search`, and `FixedRateBondAccuracyRecipeSearchTests`.
- Current evidence: projection from arbitrary 60-pillar clone points into deterministic level/slope/curvature factors creates `4.467694E+000` max PV absolute error and `4.73%` max PV relative error, while factor-aligned points reconstruct at `0.00%` error. A five-factor deterministic polynomial basis reports `5.291768E+000` max PV absolute error and `4.81%` max PV relative error, so simply adding smooth factors is not the recipe. The active-support oracle is exact on the current validation bank after removing post-maturity-neighbourhood curve bumps; active curve dimensions range from `5` to `60`. A first 10Y active-pillar TT uses `23` internal dimensions, `2014` build evaluations, and reports `1.53%` max PV relative error, but still has `132.88%` max maturity-sensitivity relative error. A narrowed/higher-resolution 10Y active-pillar TT uses `3566` build evaluations and improves max PV relative error to `0.48%`, max 10Y DV01 relative error to `1.03%`, max coupon derivative relative error to `1.03%`, and max coupon-maturity mixed relative error to `10.98%`; maturity-sensitivity relative error remains high at `161.90%`, and one-sided maturity slope errors remain large. A fixed-trade curve-only TT uses `21` internal dimensions and `837` build evaluations, with `1.796590E-004` max PV absolute error and effectively zero displayed PV/DV01 relative error. The schedule-resolved cashflow Chebyshev-kernel candidate keeps the full 62-coordinate wrapper, uses at-most-2D local discount kernels, validates on `99` full-wrapper points, reports `1.348184E-010` max PV absolute error, `4.263256E-010` max all-pillar DV01 absolute error, small mixed-risk absolute errors (`2.842171E-006` for 10Y rate-coupon, `1.112253E-008` for 10Y rate-maturity, and `3.197442E-006` for 10Y-10.5Y rate-rate), and `2.2x` measured evaluation speedup. A non-100 notional check at `250` notional reports `4.243361E-011` max dirty-price absolute error, confirming notional is algebraic rather than a hidden Chebyshev dimension. The tested 10Y pillar DV01 is stable across rate steps, one-day/three-day/seven-day maturity slopes differ materially, and post-maturity unsupported 30Y pillar DV01 is numerically zero.
- Verification so far: focused Phase 12 tests passed 9 tests; the fixed-rate bond Release slice passed 87 tests; the CI-style net10 coverage run passed 1736 tests; `--accuracy-recipe-search` CLI run completed with the 99-point cashflow-kernel validation bank and the added all-pillar/mixed-risk metrics; `dotnet format --verify-no-changes --verbosity minimal`, `docfx docs/docfx.json`, and `git diff --check` passed.
- Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191).

## Phase 6 Notes

- Plan: [Phase 6 Naive Surrogate Discovery Implementation Plan](plans/phase-6-naive-surrogate-discovery.md).
- Report draft: [Phase 6 Report: Naive Dense-Baseline Surrogate Discovery](reports/phase-6-naive-surrogate-discovery.md).
- Merge outcome: PR [#196](https://github.com/0xC000005/ChebyshevSharp/pull/196) merged into `main` on 2026-05-21 with merge commit `74a8d8abb2e36018a2d362cd94cca223f6de2ef0`.
- Scope boundary: this is a discovery phase. It may build naive TT/Slider models, but it must not implement the next modelling fix.
- Conceptual inputs are curve, coupon, maturity, and notional. Chebyshev dimensions count scalar coordinates, so the dense fixture creates 60 curve-bump dimensions; excluding notional, the naive full-PV surrogate is 62-dimensional.
- Correction: selected-pillar surrogate inputs are not faithful evidence for the clone objective. All Phase 6 and later surrogate tests must expose the full 62-coordinate input at the wrapper boundary, even if an internal model partitions, routes, or ignores some coordinates.
- Evidence targets: full dense tensor infeasibility, naive TT/Slider PV error, zero-pillar DV01 error, coupon and maturity finite-difference error, rate-coupon/rate-maturity/coupon-maturity/rate-rate mixed terms, structural post-maturity support checks, and maturity-date second-difference spikes.
- Implementation files: `examples/FixedRateBondSurrogate/NaiveSurrogateDiscovery.cs`, `examples/FixedRateBondSurrogate/Program.cs`, and `tests/ChebyshevSharp.Tests/Finance/FixedRateBondNaiveSurrogateDiscoveryTests.cs`.
- First measured result: a dense full tensor would need `3^62 = 381,520,424,476,945,831,628,649,898,809` nodes even with only three nodes per scalar coordinate.
- Corrected full-input naive probe: all 60 semiannual zero-rate bumps plus coupon and maturity, with no decomposition or bucket splitting.
- Preliminary findings from `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery`: TensorTrain max PV relative error `17.72%`, maturity-sensitivity relative error `461.43%`, and coupon-maturity mixed relative error `49.10%`; Slider max PV relative error `92.64%`, maturity-sensitivity relative error `154.35%`, and coupon-maturity mixed relative error `100.00%`.
- Structural support checks now pass: a 10Y bond has zero baseline, TT, and numerical-zero Slider sensitivity to the unsupported 30Y zero-rate pillar; the unsupported 20Y-30Y rate-rate mixed sensitivity is also zero. This narrows the naive failure to PV, maturity sensitivity, coupon/coupon-maturity behaviour, and difficult active-pillar DV01 rather than a blanket curve-support bug.
- The maturity scan found one-day slope flips around schedule-boundary candidates, with the current largest local second difference near the 2039-11-15 semiannual boundary at `2039-11-11`: left slope/year `-2.650493E+000`, right slope/year `2.825619E-002`, and second difference `7.339039E-003`.
- Maturity schedule-sensitivity evidence is reproducible: `tools/PlotFixedRateBondEvidence/plot_phase6_maturity.py` regenerates `docs/research/fixed-rate-bond-surrogate/data/phase-6-maturity-scan.csv` and `docs/research/fixed-rate-bond-surrogate/images/phase-6-maturity-sensitivity.svg`.
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBondNaiveSurrogateDiscoveryTests"` passed 6 tests with 0 failures.
- Fixed-rate bond test slice run: `dotnet test --filter "FullyQualifiedName~FixedRateBond"` passed 55 tests with 0 failures.
- Local closeout checks so far: `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1703 tests with 0 failures; `docfx docs/docfx.json` passed with 0 warnings/errors; `git diff --check` passed; private-name scan matched only pre-existing guardrail/search-term text.
- Superseded tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4509528109) recorded the earlier selected-pillar probe and should not be used as clone evidence.
- Correction tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4509648601).
- Closeout tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4510805560).

## Phase 7 Notes

- Plan: [Phase 7 Structured Alternatives Benchmark Plan](plans/phase-7-structured-alternatives.md).
- Report draft: [Phase 7 Report: Structured Alternatives](reports/phase-7-structured-alternatives.md).
- Phase PR: [#198](https://github.com/0xC000005/ChebyshevSharp/pull/198).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4511793914).
- Merge outcome: PR [#198](https://github.com/0xC000005/ChebyshevSharp/pull/198) merged into `main` on 2026-05-21 with merge commit `22331ea87f274b320b01bc6cacacbdce2eff8fbd`.
- Scope boundary: compare fixes against the Phase 6 evidence bank without claiming a final bond-pricer replacement design.
- Required wrapper contract: every tested candidate must be callable as the full 62-coordinate interface, `curve bumps[60] + coupon + maturity`, even if the implementation internally routes, buckets, partitions, or ignores unsupported coordinates.
- Candidate families measured: stronger global TT settings, TT auto-ordering, grouped Slider partitions, level/slope/curvature curve-factor compression, 1Y bucketed curve-factor routing, and 0.5Y semiannual bucketed curve-factor routing.
- Fresh benchmark command: `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --structured-alternatives`.
- Current result: stronger global TT improves Phase 6 PV error but still has max clone maturity-sensitivity relative error `256.28%` and coupon-maturity mixed relative error `80.99%`.
- Grouped Slider reduces PV error versus the Phase 6 singleton Slider, but max clone maturity-sensitivity relative error remains `327.65%` and coupon-maturity mixed relative error remains `75.20%`.
- Curve-factor tensor is the strongest common-practice candidate: max PV relative error is `4.70%` on arbitrary clone points and `0.59%` on factor-aligned points. This supports factor-space compression for factor-like scenario sets, but not a general arbitrary 60-pillar clone.
- 0.5Y semiannual bucketed curve-factor routing improves maturity-sensitivity error versus the 1Y bucket and global factor tensor, but remaining derivative and mixed-term errors are too large for a faithful risk clone.
- Phase 7 decision: stop treating larger global TT/Slider tuning as the main route. The next design phase should evaluate true high-dimensional piecewise/special-point support and analytical coupon decomposition as explicit designs.
- Local verification: focused Phase 7 tests passed 3 tests; fixed-rate bond slice passed 59 tests; Release build passed with 0 warnings/errors; DocFX passed with 0 warnings/errors; Release tests passed 1708 tests; `git diff --check` passed; private-name scan matched only existing guardrail/checklist text.
- CI outcome: `.NET 10 tests`, `.NET 8 library build`, `All Tests Passed`, `Format, Pack, and Docs`, and `codecov/patch` passed before merge.

## Phase 8 Notes

- Plan: [Phase 8 Analytic Coupon Decomposition Plan](plans/phase-8-analytic-coupon-decomposition.md).
- Report draft: [Phase 8 Report: Analytic Coupon Decomposition](reports/phase-8-analytic-coupon-decomposition.md).
- Phase PR: [#199](https://github.com/0xC000005/ChebyshevSharp/pull/199).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4512018897).
- Merge outcome: PR [#199](https://github.com/0xC000005/ChebyshevSharp/pull/199) merged into `main` on 2026-05-21 with merge commit `ae41f2de30e5ce34adb0b51aaa03a3de0fac852d`.
- Scope boundary: validate and benchmark analytic coupon recombination; keep automatic kink detection and library-level special-point routing for Phase 9.
- Required wrapper contract: every tested candidate remains callable as the full 62-coordinate interface, `curve bumps[60] + coupon + maturity`, even when coupon is removed internally.
- Core formula: `PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)`.
- Coupon-linearity identity result: max absolute error `8.526513E-014` and max relative error `0.000000%` over 19 clone/factor validation points.
- Candidate families measured: exact decomposition oracle, global decomposed TT, curve-factor decomposed tensor, 1Y bucketed decomposed factor tensor, and 0.5Y semiannual bucketed decomposed factor tensor.
- Current result: global decomposed TT still has max clone PV relative error `14.34%`, maturity-sensitivity relative error `456.94%`, and coupon-maturity mixed relative error `51.28%`; removing coupon alone does not solve the global clone problem.
- Curve-factor decomposed tensor matches Phase 7 factor-space PV behavior while reducing build evaluations from `675` to `270`; max factor-aligned PV relative error remains `0.59%`.
- 0.5Y semiannual bucketed decomposed factor routing keeps max factor-aligned PV relative error at `0.58%` but leaves coupon-maturity mixed relative error `56.94%`, so maturity nonsmoothness remains a Phase 9 problem.
- Local verification: focused Phase 8 tests passed 3 tests; focused Phase 8 coverage showed no missing or partial lines in `AnalyticCouponDecompositionBenchmark.cs`; fixed-rate bond slice passed 62 tests; analytic-coupon example ran and printed the report table; Release build passed with 0 warnings/errors; DocFX passed with 0 warnings/errors; Release tests passed 1711 tests; `git diff --check` passed; private-name scan matched only existing guardrail/checklist text.
- CI outcome before merge: `Format, Pack, and Docs`, `.NET 8 library build`, `.NET 10 tests`, `All Tests Passed`, and `codecov/patch` passed on the amended Phase 8 commit.

## Phase 9 Notes

- Plan: [Phase 9 Maturity Special Points Plan](plans/phase-9-maturity-special-points.md).
- Implementation plan path: `docs/superpowers/plans/2026-05-21-phase9-maturity-special-points.md`.
- Report draft: [Phase 9 Report: Maturity Special Points](reports/phase-9-maturity-special-points.md).
- Phase PR: [#200](https://github.com/0xC000005/ChebyshevSharp/pull/200).
- Working branch: `phase9-maturity-special-points`.
- Scope boundary: test maturity special points, schedule-aware routing, and automatic detector candidates before adding reusable library-level APIs.
- Required wrapper contract: every tested candidate remains callable as the full 62-coordinate interface, `curve bumps[60] + coupon + maturity`.
- Default internal formula: use the Phase 8 principal/annuity decomposition unless validation proves the restricted product family no longer satisfies coupon linearity.
- Evidence basis: Phase 6 maturity second-difference spikes, Phase 7 bucket/factor limitations, and Phase 8 proof that coupon is analytical but maturity errors remain.
- Current inventory: 167 one-day maturity-window points around semiannual schedule regions; largest spike is `2038-05-15`, with second difference `6.116018E-003` and one-day slope jump `2.233876E+000`.
- Candidate families measured: global decomposed factor control, uniform 0.5Y bucket control, schedule-aware special points, automatic detector special points, and a hybrid union.
- Current result: schedule-aware routing improves maturity relative error from `96.44%` to `89.21%` and coupon-maturity mixed relative error from `55.52%` to `48.75%` versus the uniform 0.5Y control, while detector-only and hybrid routing do not improve maturity sensitivity.
- Phase 9 decision so far: evidence supports a future schedule-aware high-dimensional piecewise-router design, but not a generic automatic kink-detection API yet.
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4512527963).
- Focused tests run: `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondMaturitySpecialPointTests" --verbosity minimal` passed 8 tests with 0 failures.
- Fixed-rate bond test slice run: `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBond" --verbosity minimal` passed 70 tests with 0 failures.
- Benchmark command run: `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points`.
- Local verification so far: `dotnet format --verify-no-changes --verbosity minimal` passed; `docfx docs/docfx.json` passed with 0 warnings/errors; Release tests passed 1719 tests; `git diff --check` passed.
- CI outcome on PR #200: `Format, Pack, and Docs`, `.NET 8 library build`, `.NET 10 tests`, and `All Tests Passed` passed; Dependabot was skipped.
- Codecov closeout: focused local coverage found no missing or partial lines in `MaturitySpecialPointsBenchmark.cs`; final PR check `codecov/patch` passed.
- Merge outcome: PR [#200](https://github.com/0xC000005/ChebyshevSharp/pull/200) merged into `main` on 2026-05-21 with merge commit `e2ac6acfd172bee280e9486dab8bec5a92c4324c`.
- Closeout tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4512938294).

## Phase 10 Notes

- Plan: [Phase 10 Schedule-Aware Piecewise Router Plan](plans/phase-10-schedule-aware-router.md).
- Implementation plan path: `docs/superpowers/plans/2026-05-21-phase10-schedule-aware-router.md`.
- Phase PR: [#201](https://github.com/0xC000005/ChebyshevSharp/pull/201).
- Working branch: `phase10-schedule-aware-router`.
- Scope boundary: prototype a schedule-aware high-dimensional router inside the fixed-rate bond harness first; do not add a generic automatic kink-detection API in this phase.
- Required wrapper contract: every tested candidate remains callable as the full 62-coordinate interface, `curve bumps[60] + coupon + maturity`.
- Default internal formula: keep Phase 8 principal/annuity decomposition and Phase 9 schedule-derived breakpoints.
- Required new evidence: one-sided maturity sensitivities around split points, piece-routing diagnostics, comparison against Phase 9 global/uniform/schedule-aware controls, and an explicit public-API decision.
- Report draft: [Phase 10 Report: Schedule-Aware Piecewise Router](reports/phase-10-schedule-aware-router.md).
- Current implementation: explicit `ScheduleAwarePiecewiseRouter` with half-open maturity pieces, full-wrapper `Eval(double[] fullPoint)`, build diagnostics, and schedule-candidate provenance.
- Current result: the explicit router reproduces the Phase 9 schedule-aware clone metrics (`4.73%` max PV rel error, `89.21%` max maturity rel error, `48.75%` max coupon-maturity mixed rel error) while making the dispatch semantics auditable.
- Current one-sided evidence: first split diagnostics show left/right maturity-slope absolute errors around `1.0E-001` to `2.0E-001`, so the router clarifies validation but does not solve residual sensitivity error.
- Current public-API decision: keep the router example-local; evidence supports a future schedule-aware router design discussion, not a generic automatic kink-detection API.
- Focused tests run: `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondScheduleAwareRouterTests" --verbosity minimal` passed 8 tests with 0 failures.
- Focused coverage run found no missing or partial lines in `ScheduleAwareRouterBenchmark.cs`.
- Fixed-rate bond test slice run: `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBond" --verbosity minimal` passed 77 tests with 0 failures.
- Closeout verification so far: `dotnet format --verify-no-changes --verbosity minimal` passed; `docfx docs/docfx.json` passed with 0 warnings/errors; full Release tests passed 1,727 tests with 0 failures; `git diff --check` passed.
- Benchmark command run: `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router`.
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4513211570).
- Merge outcome: PR [#201](https://github.com/0xC000005/ChebyshevSharp/pull/201) merged into `main` on 2026-05-22 with merge commit `1e9d66e29eee7028d5c7e7918f9d8d3b39ac260b`.
- Closeout tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4519494565).

## Phase 11 Notes

- Plan: [Phase 11 Fixed-Rate Bond Tutorial Plan](plans/phase-11-fixed-rate-bond-tutorial.md).
- Implementation plan path: `docs/superpowers/plans/2026-05-22-phase11-fixed-rate-bond-tutorial.md`.
- Working branch: `phase11-fixed-rate-bond-tutorial`.
- Scope boundary: create public-facing tutorial and documentation only; do not change surrogate algorithms, expose new APIs, bump versions, or release packages in this phase.
- Required narrative: start from the QLNet reference pricer and pinned Federal Reserve fixture, then show why a naive global surrogate is insufficient before introducing progressively more structured models.
- Required guardrail: avoid proprietary names and avoid language implying a universal replacement for arbitrary fixed-income products.
- Required evidence reuse: Phase 5 baseline assumptions, Phase 6 naive global TT/Slider failures, Phase 7 structured alternatives, Phase 8 analytic coupon identity, Phase 9 schedule-special-point evidence, and Phase 10 router decision.
- Planned output: a documentation case study, README/example links if needed, and a concise tracking-issue update before opening one coherent Phase 11 PR.
- Current implementation: added the public tutorial page at `docs/docs/fixed-rate-bond-surrogate.md`, linked it from the user-guide TOC and examples page, and kept the detailed evidence in the phase reports.
- Verification so far: CLI flags quoted by the tutorial match `examples/FixedRateBondSurrogate/Program.cs`; public source links checked with `curl` returned HTTP 200 for the Federal Reserve nominal curve page/CSV, QLNet `FixedRateBond`, Chebfun guide/edge-detection pages, OpenGamma fixed-coupon-bond/pricer pages, and QuantLib Guide vanilla-bonds page; `docfx docs/docfx.json` passed with 0 warnings/errors; `git diff --check` passed; fixed-rate bond Release test slice passed 78 tests with 0 failures.
- Phase PR: [#202](https://github.com/0xC000005/ChebyshevSharp/pull/202).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4519611297).
- CI outcome: `.NET 10 tests`, `.NET 8 library build`, `All Tests Passed`, and `Format, Pack, and Docs` passed; Dependabot skipped.
- Public-surface audit: `docs/research/fixed-rate-bond-surrogate/plans/**`, `status.md`, `design.md`, and early setup/prototype reports from Phases 1-4 were being published by DocFX. `docs/docfx.json` now excludes them while leaving the public case-study page, linked Phase 5-10 evidence reports, figures, and data available.

## Phase 5 Notes

- Plan: [Phase 5 Realistic Baseline Implementation Plan](plans/phase-5-realistic-baseline.md).
- Report draft: [Phase 5 Report: Realistic Baseline](reports/phase-5-realistic-baseline.md).
- Scope boundary: do not implement analytic coupon decomposition, maturity splitting, adaptive splitting, or new Chebyshev surrogate fixes in this phase.
- New dense fixture: `examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-semiannual-2026-05-15.json`.
- Data method: sample the Federal Reserve fitted nominal zero-yield curve every six months from the published `BETA0`, `BETA1`, `BETA2`, `BETA3`, `TAU1`, and `TAU2` parameters for 2026-05-15, using the same Actual/365 year fraction that QLNet uses for the dated zero curve.
- Baseline conventions are exposed from `QlNetFixedRateBondReferencePricer.SupportedConventions`: United States Government Bond calendar, semiannual schedule, 30/360 USA coupon day count, Actual/365 Fixed curve day count, Modified Following business-day adjustment, backward schedule generation, linear zero-rate interpolation, continuous annual compounding, and 100 redemption.
- Default example now prices a regular 30Y 4.5% coupon bullet from the dense fixture. It uses 61 curve pillars and produces 61 cashflows, dirty price `89.26423408`, clean price `89.26423408`, and zero accrued amount at valuation/effective date.
- Added baseline hardening tests for manual linear zero-rate interpolation and six public Treasury auction sanity checks across 2Y, 3Y, 5Y, 7Y, 10Y, and 30Y notes/bonds. The QLNet continuous-zero approximation prices all six auction examples within `0.10` to `0.20` price points, which is sufficient for a convention sanity check rather than an exact Treasury-yield formula clone.
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBond"` passed 50 tests with 0 failures.
- Fixture regeneration checks using `--input-csv /tmp/feds200628.csv` matched both the dense semiannual fixture and the existing annual fixture byte-for-byte.
- Closeout verification: private-name scan found only guardrail/search-term text; stale Phase 5/deleted-worktree scan found no matches; `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1699 tests with 0 failures; the Release default example ran against the dense baseline; `docfx docs/docfx.json` passed with 0 warnings/errors; `git diff --check` passed.

## Phase 4 Notes

- Plan: [Phase 4 Surrogate Problem Reproduction Implementation Plan](plans/phase-4-reproduce-surrogate-problem.md).
- Report draft: [Phase 4 Report: Full-PV Surrogate Reproduction](reports/phase-4-surrogate-reproduction.md).
- Implementation files: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`, `examples/FixedRateBondSurrogate/Program.cs`, and `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSurrogateReproductionTests.cs`.
- Implementation commits: `39c22c3` and `8c88454`.
- Phase PR: [#195](https://github.com/0xC000005/ChebyshevSharp/pull/195).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4504622669).
- Data-source decision: use the pinned Federal Reserve nominal zero-yield fixture for deterministic tests and documentation. Yahoo Finance, FRED, and live Federal Reserve downloads remain optional future refresh paths; no API key or live download is required for this phase.
- Scope boundary: build direct full-PV surrogates first. Do not implement analytic coupon decomposition, maturity splitting, or adaptive splitting until later phases so the reproduction isolates the problem before proposing fixes.
- Preliminary findings: TensorTrain max PV relative error is 0.35% on the compact validation set, while maturity-sensitivity relative error reaches 398.88%, rate-coupon mixed relative error reaches 23.62%, and rate-maturity mixed relative error reaches 150.84%. Slider is weaker on this partition, including 100% relative error for the reported mixed terms.
- Local verification: private-name scan produced only guardrail/search-term matches; focused Phase 4 tests passed 3 tests; focused coverage found no missing lines in `SurrogateReproduction.cs`; `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1686 tests with 0 failures; the Release `--surrogate-reproduction` example ran; `docfx docs/docfx.json` passed with 0 warnings/errors.
- CI/review outcome: PR [#195](https://github.com/0xC000005/ChebyshevSharp/pull/195) merged on 2026-05-21 after required checks passed, including `Format, Pack, and Docs`, `.NET 8 library build`, `.NET 10 tests`, and `All Tests Passed`.
- Merge commit: `c44278a`.

## Phase PR Cadence Gate

Use exactly one active phase PR for this workflow. After a phase PR opens, all review fixes for that phase stay in that PR. Do not start the next phase implementation, open another phase PR, or accumulate unrelated follow-up PRs until the current phase PR is merged or explicitly closed without merge. Record the outcome in this status file and in tracking issue [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191) before moving on.

## Phase 1 Notes

- Plan: [Phase 1 Baseline Pricer Adapter Implementation Plan](plans/phase-1-baseline-pricer-adapter.md).
- Report draft: [Phase 1 Report: Baseline Pricer Adapter](reports/phase-1-baseline-pricer.md).
- Selected first C# baseline path: QLNet `1.13.1` in the example and tests only.
- Optional cross-check path: Python QuantLib through `uv`.
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBondReferencePricerTests"` passed 16 tests with 0 failures.
- Full verification run: `dotnet build --no-restore` passed with 0 warnings/errors; `dotnet test` passed 1665 tests with 0 failures; Release coverage run passed 1665 tests with 0 failures; all examples ran; `docfx docs/docfx.json` passed with 0 warnings/errors.
- Local coverage evidence for the Codecov follow-up: `FixedRateBondSurrogate` line-rate `99.29%`, branch-rate `100%`.
- Phase PR: [#192](https://github.com/0xC000005/ChebyshevSharp/pull/192).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4504103137).
- Merge outcome: PR [#192](https://github.com/0xC000005/ChebyshevSharp/pull/192) merged on 2026-05-21 after required checks passed, including `All Tests Passed` and `codecov/patch`.

## Phase 2 Notes

- Plan: [Phase 2 Public Data Fixture Pipeline Implementation Plan](plans/phase-2-public-data-fixtures.md).
- Report draft: [Phase 2 Report: Public Data Fixtures](reports/phase-2-public-data-fixtures.md).
- Primary data source selected for the first fixture: Federal Reserve nominal yield curve CSV, using `SVENY01` to `SVENY30` continuously compounded zero-coupon yields.
- Pinned fixture: `examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-2026-05-15.json`.
- Optional refresh tool: `tools/RefreshFixedRateBondMarketData/refresh_fed_nominal_yield_curve.py`.
- C# fixture loader: `examples/FixedRateBondSurrogate/MarketData.cs`.
- Implementation commit: `35f9c33`.
- Phase PR: [#193](https://github.com/0xC000005/ChebyshevSharp/pull/193).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4504300102).
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBondReferencePricerTests"` passed 27 tests with 0 failures.
- Local closeout verification: private-name scan produced only guardrail/search-term matches; fixture regeneration matched the committed JSON byte-for-byte; `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1676 tests with 0 failures; local coverage inspection found no uncovered or partial lines in `MarketData.cs`; the fixed-rate bond example ran against the pinned fixture; `docfx docs/docfx.json` passed with 0 warnings/errors.
- CI/review outcome: PR [#193](https://github.com/0xC000005/ChebyshevSharp/pull/193) merged on 2026-05-21 after required checks passed, including `All Tests Passed` and `codecov/patch`. Codecov reported that all modified and coverable lines were covered by tests.
- Merge commit: `052eb82`.
- Official source checks completed:
  - Federal Reserve nominal yield curve page confirms the data are fitted nominal yield-curve parameters and smoothed yields from 1961 to present, and that the model is a staff research product rather than an official statistical release.
  - Federal Reserve CSV schema confirms `SVENYXX` are continuously compounded zero-coupon yields, `SVENPYXX` are coupon-equivalent par yields, `SVENFXX` are continuously compounded instantaneous forwards, and `SVEN1FXX` are coupon-equivalent one-year forwards.
  - Treasury XML feed remains a later par-yield source, not the first direct-zero fixture.
  - New York Fed SOFR remains a later overnight-rate source, not the first term zero-curve fixture.

## Phase 3 Notes

- Plan: [Phase 3 Smoothness Diagnostics Implementation Plan](plans/phase-3-smoothness-diagnostics.md).
- Report draft: [Phase 3 Report: Smoothness Diagnostics](reports/phase-3-smoothness-diagnostics.md).
- Implementation files: `examples/FixedRateBondSurrogate/SmoothnessDiagnostics.cs`, `examples/FixedRateBondSurrogate/Program.cs`, and `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSmoothnessDiagnosticsTests.cs`.
- Implementation commit: `00845f4`.
- Phase PR: [#194](https://github.com/0xC000005/ChebyshevSharp/pull/194).
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4504452110).
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBondSmoothnessDiagnosticsTests"` passed 7 tests with 0 failures.
- Local closeout verification: private-name scan produced only guardrail/search-term matches; `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1683 tests with 0 failures; local coverage inspection found no uncovered or partial lines in `SmoothnessDiagnostics.cs`; the only uncovered `Program.cs` line in the full report is the unchanged console entrypoint; both fixed-rate bond example modes ran successfully; `docfx docs/docfx.json` passed with 0 warnings/errors.
- CI/review outcome: PR [#194](https://github.com/0xC000005/ChebyshevSharp/pull/194) merged on 2026-05-21 after required checks passed, including `All Tests Passed` and `codecov/patch`. Codecov reported that all modified and coverable lines were covered by tests.
- Merge commit: `396684d`.
- Diagnostics command run: `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --diagnostics`.
- Preliminary findings: coupon second differences are near numerical zero, 10Y is the largest zero-pillar DV01 for the 10Y bond, 20Y/30Y DV01 are zero under the current interpolation support, the diagnostics records 25 rate-bump slice points, and maturity-date slices show daily second-difference spikes near schedule-boundary regions.

## Notes for Future Sessions

- Keep the harness inside ChebyshevSharp under `examples/`, `tests/`, and `docs/research/`.
- Use phase-level PRs. Do not open a PR for every small task. A phase PR should satisfy the phase exit gate and include tests, reports, docs, and status updates.
- Keep at most one phase PR open for this workflow. Finish review and merge or explicitly close that PR before starting the next phase implementation.
- Treat PR closeout as part of the phase: record CI/review results, review fixes, merge/close outcome, and any remaining follow-up in the phase report or tracking issue.
- Update documentation while each phase is running. Phase work must record formulas, concepts, usage examples, citations, citation verification, and data provenance as soon as they become part of the evidence.
- Use the tracking issue and phase reports for routine observations. Create separate issues only for real bugs, blockers, or follow-up features that need independent tracking. If a discovered bug is required for the current phase, fix it in the phase branch and reference the issue in the phase PR.
- The first implementation should be deliberately restricted and transparent, not a general fixed-income library.
- Do not implement a full bond pricer unless external baseline integration blocks deterministic CI or a tiny transparent sanity oracle is needed.
- Preferred baseline strategy: use QuantLib/QLNet/QuantLib Python as the trusted pricing function and train/validate Chebyshev surrogates against it.
- Compute DV01 and cross-terms by finite differences around the baseline pricer, with recorded step sizes and boundary handling.
- Include structural sanity checks: matured bonds with no remaining cashflows must have zero PV and zero rate sensitivity; curve bumps with no interpolation support on remaining cashflow discount factors must have zero DV01.
- The central hypothesis is `PV = N * (Principal(curve, T) + coupon * Annuity(curve, T))`; coupon should be tested as an analytical parameter before being treated as a tensor dimension.
- Maturity should be treated as the likely piecewise-smooth variable because changing maturity can change cashflow dates, accrual fractions, and schedule regimes.
- Slider is expected to be useful as a contrast case because it cannot represent cross-group mixed terms when interacting variables are split across slides.
- Use a direct zero-rate curve for the first harness. Report sensitivities as zero-pillar DV01, not market-quote DV01. Bootstrapped market-quote DV01 is a later experiment.
- Public data candidates: Federal Reserve nominal yield curve data for zero-coupon-style fixtures, U.S. Treasury par yield curve data when clearly labeled or bootstrapped, and New York Fed SOFR/SOFR Averages only for overnight/compounded benchmark examples.
- Use Python for optional public-data refresh scripts and fixture generation; C# tests/examples should consume pinned fixtures.
- Documentation plots and headline business-demonstration results should use real public market-data fixtures where possible. Synthetic coupons, maturities, and shocks are allowed only when clearly labeled as scenario design choices around a real curve.
