# Fixed-Rate Bond Surrogate Research Status

## Current Objective

Create a public, non-proprietary reproduction harness inside ChebyshevSharp that tests Chebyshev tensor surrogates around a trusted fixed-rate bond baseline pricer.

Design spec: [Fixed-Rate Bond Surrogate Reproduction Harness Design](design.md)

This is the primary scope for the next small version update.

Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191)

Working branch: `bond-surrogate-research`

Last completed phase PR: [#194](https://github.com/0xC000005/ChebyshevSharp/pull/194), merged on 2026-05-21 as `396684d`.

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
| 4. Reproduce surrogate problem | In progress | TT/Slider report confirms or rejects PV-good/Greeks-bad behavior |
| 5. Analytic coupon decomposition | Not started | Principal/annuity surrogate comparison completed |
| 6. Maturity splitting | Not started | No split vs 1Y vs 0.5Y vs schedule-aware split comparison completed |
| 7. Adaptive splitting research | Not started | Decision on whether adaptive splitting is needed |
| 8. Tutorial and documentation | Not started | Public tutorial merged into documentation site |
| 9. Library improvement issues | Not started | Evidence-backed issues opened only where needed |

## Environment Readiness

Last checked: 2026-05-20.

- Branch/worktree: `bond-surrogate-research` at `/tmp/ChebyshevSharp-worktrees/fixed-rate-bond-surrogate`.
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

Execute Phase 4 surrogate reproduction from the written plan. Phase 4 should build the first full PV TT and Slider surrogates against the public baseline and test whether acceptable PV can coexist with unacceptable DV01, coupon, maturity, or mixed-term errors.

## Phase 4 Notes

- Plan: [Phase 4 Surrogate Problem Reproduction Implementation Plan](plans/phase-4-reproduce-surrogate-problem.md).
- Report draft: [Phase 4 Report: Full-PV Surrogate Reproduction](reports/phase-4-surrogate-reproduction.md).
- Implementation files: `examples/FixedRateBondSurrogate/SurrogateReproduction.cs`, `examples/FixedRateBondSurrogate/Program.cs`, and `tests/ChebyshevSharp.Tests/Finance/FixedRateBondSurrogateReproductionTests.cs`.
- Implementation commits: `39c22c3` and `8c88454`.
- Data-source decision: use the pinned Federal Reserve nominal zero-yield fixture for deterministic tests and documentation. Yahoo Finance, FRED, and live Federal Reserve downloads remain optional future refresh paths; no API key or live download is required for this phase.
- Scope boundary: build direct full-PV surrogates first. Do not implement analytic coupon decomposition, maturity splitting, or adaptive splitting until later phases so the reproduction isolates the problem before proposing fixes.
- Preliminary findings: TensorTrain max PV relative error is 0.35% on the compact validation set, while maturity-slope relative error reaches 398.88%, rate-coupon mixed relative error reaches 23.62%, and rate-maturity mixed relative error reaches 150.84%. Slider is weaker on this partition, including 100% relative error for the reported mixed terms.
- Local verification: private-name scan produced only guardrail/search-term matches; focused Phase 4 tests passed 3 tests; focused coverage found no missing lines in `SurrogateReproduction.cs`; `dotnet format --verify-no-changes --verbosity minimal` passed; `dotnet build --configuration Release --no-restore` passed with 0 warnings/errors; Release coverage tests passed 1686 tests with 0 failures; the Release `--surrogate-reproduction` example ran; `docfx docs/docfx.json` passed with 0 warnings/errors.

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
