# Fixed-Rate Bond Surrogate Research Status

## Current Objective

Create a public, non-proprietary reproduction harness inside ChebyshevSharp that tests Chebyshev tensor surrogates around a trusted fixed-rate bond baseline pricer.

Design spec: [Fixed-Rate Bond Surrogate Reproduction Harness Design](design.md)

This is the primary scope for the next small version update.

Tracking issue: [#191](https://github.com/0xC000005/ChebyshevSharp/issues/191)

Working branch: `bond-surrogate-research`

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
| 1. Baseline pricer selection and adapter | Not started | QuantLib/QLNet/Python baseline selected and callable behind generic adapter |
| 2. Data fixture pipeline | Not started | Public curve fixture generated, pinned, and documented; no live downloads in CI |
| 3. Smoothness diagnostics | Not started | Report identifies PV/slope/sensitivity smoothness and maturity breakpoints |
| 4. Reproduce surrogate problem | Not started | TT/Slider report confirms or rejects PV-good/Greeks-bad behavior |
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

Wait for the explicit order to start continuous execution. The first execution step is to write the Phase 1 implementation plan before selecting or integrating the external baseline pricer.

## Notes for Future Sessions

- Keep the harness inside ChebyshevSharp under `examples/`, `tests/`, and `docs/research/`.
- Use phase-level PRs. Do not open a PR for every small task. A phase PR should satisfy the phase exit gate and include tests, reports, docs, and status updates.
- Use issues for real bugs, blockers, and follow-up features. If a discovered bug is required for the current phase, fix it in the phase branch and reference the issue in the phase PR.
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
