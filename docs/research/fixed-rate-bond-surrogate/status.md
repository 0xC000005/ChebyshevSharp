# Fixed-Rate Bond Surrogate Research Status

## Current Objective

Create a public, non-proprietary reproduction harness inside ChebyshevSharp that tests Chebyshev tensor surrogates around a trusted fixed-rate bond baseline pricer.

Design spec: [Fixed-Rate Bond Surrogate Reproduction Harness Design](design.md)

This is the primary scope for the next small version update.

Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191)

Working branch: `bond-surrogate-research`

Last completed phase PR: [#195](https://github.com/0xC000005/ChebyshevSharp/pull/195), merged on 2026-05-21 as `c44278a`.

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
| 5. Realistic dense-curve baseline | Complete locally | Dense semiannual curve fixture, explicit conventions, 30Y baseline tests, and docs complete |
| 6. Naive dense-baseline surrogate discovery | In progress | Dense-baseline naive TT/Slider and maturity-smoothness evidence documented |
| 7. Analytic coupon decomposition | Deferred | Principal/annuity surrogate comparison completed |
| 8. Maturity splitting | Not started | No split vs 1Y vs 0.5Y vs schedule-aware split comparison completed |
| 9. Adaptive splitting research | Not started | Decision on whether adaptive splitting is needed |
| 10. Tutorial and documentation | Not started | Public tutorial merged into documentation site |
| 11. Library improvement issues | Not started | Evidence-backed issues opened only where needed |

## Environment Readiness

Last checked: 2026-05-20.

- Branch/worktree: `bond-surrogate-research` at `/home/max/Documents/ChebyshevSharp/.worktrees/bond-surrogate-research`.
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

Run Phase 6 naive dense-baseline surrogate discovery. Start from the QLNet-backed dense baseline and collect evidence for or against naive full-PV Chebyshev modelling. Stop after documenting the naive failure modes; do not implement analytic coupon decomposition, maturity splitting, adaptive splitting, or portfolio-specific modelling in this phase.

## Phase 6 Notes

- Plan: [Phase 6 Naive Surrogate Discovery Implementation Plan](plans/phase-6-naive-surrogate-discovery.md).
- Report draft: [Phase 6 Report: Naive Dense-Baseline Surrogate Discovery](reports/phase-6-naive-surrogate-discovery.md).
- Scope boundary: this is a discovery phase. It may build limited naive TT/Slider models, but it must not implement the next modelling fix.
- Conceptual inputs are curve, coupon, maturity, and notional. Chebyshev dimensions count scalar coordinates, so the dense fixture creates 60 curve-bump dimensions; excluding notional, the naive full-PV surrogate is 62-dimensional.
- Evidence targets: full dense tensor infeasibility, naive TT/Slider PV error, zero-pillar DV01 error, coupon and maturity finite-difference error, rate-coupon/rate-maturity/coupon-maturity mixed terms, and maturity-date second-difference spikes.
- Implementation files: `examples/FixedRateBondSurrogate/NaiveSurrogateDiscovery.cs`, `examples/FixedRateBondSurrogate/Program.cs`, and `tests/ChebyshevSharp.Tests/Finance/FixedRateBondNaiveSurrogateDiscoveryTests.cs`.
- First measured result: a dense full tensor would need `3^62 = 381,520,424,476,945,831,628,649,898,809` nodes even with only three nodes per scalar coordinate.
- Limited naive probe: selected 1Y, 5Y, 10Y, 20Y, and 30Y zero-rate bumps plus coupon and maturity, with no decomposition or bucket splitting.
- Preliminary findings from `dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery`: TensorTrain max PV relative error `2.98%`, maturity-slope relative error `115.74%`, and coupon-maturity mixed relative error `16.24%`; Slider max PV relative error `93.87%`, maturity-slope relative error `288.39%`, and coupon-maturity mixed relative error `100.00%`.
- The maturity scan found one-day slope flips around schedule-boundary candidates, with the largest second difference at `2038-05-15`: left slope/year `-1.953303E+000`, right slope/year `2.790432E-001`, and second difference `6.116018E-003`.
- Focused tests run: `dotnet test --filter "FullyQualifiedName~FixedRateBondNaiveSurrogateDiscoveryTests"` passed 4 tests with 0 failures.
- Fixed-rate bond test slice run: `dotnet test --filter "FullyQualifiedName~FixedRateBond"` passed 54 tests with 0 failures.
- Local closeout checks so far: `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1703 tests with 0 failures; `docfx docs/docfx.json` passed with 0 warnings/errors; `git diff --check` passed; private-name scan matched only pre-existing guardrail/search-term text.
- Tracking issue update: [#191 comment](https://github.com/0xC000005/ChebyshevSharp/issues/191#issuecomment-4509528109).

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
- Preliminary findings: TensorTrain max PV relative error is 0.35% on the compact validation set, while maturity-slope relative error reaches 398.88%, rate-coupon mixed relative error reaches 23.62%, and rate-maturity mixed relative error reaches 150.84%. Slider is weaker on this partition, including 100% relative error for the reported mixed terms.
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
