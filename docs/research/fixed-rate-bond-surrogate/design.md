---
title: Fixed-Rate Bond Surrogate Reproduction Harness Design
date: 2026-05-20
status: draft
---

# Fixed-Rate Bond Surrogate Reproduction Harness Design

## Objective

Build a clean, public, non-proprietary reproduction harness inside ChebyshevSharp that demonstrates how Chebyshev tensor methods can accelerate an existing fixed-rate bond pricer, and where the surrogate fails for sensitivities.

The final tutorial should read as an actual financial use case and business demonstration, not a toy numerical example. Use real public market-data fixtures where practical, and clearly label any synthetic dimensions or scenario ranges.

The harness should answer four questions with local evidence:

1. Can `ChebyshevTT` and `ChebyshevSlider` reproduce present value from a trusted baseline fixed-rate bond pricer?
2. Do DV01, coupon sensitivity, maturity sensitivity, and mixed cross-terms fail even when PV looks acceptable?
3. Does analytic coupon decomposition improve the approximation?
4. Does piecewise maturity splitting reduce errors caused by schedule/accrual regime changes?

This work must not mention proprietary systems or interfaces. Use generic terms such as "legacy pricer", "reference pricer", "fixed-rate bond", and "surrogate".

## Scope

The first version is a research/tutorial harness, not a production bond library. It should wrap a trusted baseline pricer for a restricted public example:

- Fixed-rate bullet bond.
- Unit notional with scalar notional applied after pricing.
- Deterministic zero curve.
- Regular schedule.
- Semiannual coupons.
- 30/360 day count.
- Weekend-only Modified Following calendar unless a fuller public calendar is added later.
- Settlement equal to valuation date.
- Dirty PV as the primary output; clean PV and accrued interest may be reported for diagnostics.

Out of scope for the first pass: amortization, callable features, ex-coupon logic, arbitrary stubs, arbitrary calendars, settlement-date modeling, inflation/float legs, and private object models.

## Proposed Location

- `examples/FixedRateBondSurrogate/` for the runnable harness.
- `tests/ChebyshevSharp.Tests/Finance/` for deterministic regression tests.
- `docs/docs/fixed-rate-bond-surrogate.md` for the tutorial.
- `docs/research/fixed-rate-bond-surrogate/` for experiment reports, validation tables, and checkpoint notes.

The example should be source-controlled and reproducible. Generated large artifacts should either be omitted or written under an ignored results folder.

## Architecture

### Baseline Pricer Adapter

Prefer an external, established fixed-income library as the baseline pricer. The tutorial is about accelerating a generic expensive pricing function, not about reimplementing bond mathematics.

Candidate baselines:

1. QuantLib Python for research scripts and fast iteration.
2. QuantLib C# NuGet package for a .NET-side validation path.
3. QLNet as a pure C# fallback if QuantLib C# integration is too heavy.
4. A tiny transparent in-repo reference pricer only as a sanity oracle if external-library setup blocks CI.

The production code path should depend only on a small adapter interface:

```text
IBondPricingBaseline.Price(input) -> dirty PV, clean PV, accrued, optional cashflow diagnostics
```

Surrogate training and validation should call that adapter as a black-box function. This keeps the example generic: users can replace the baseline with QuantLib, QLNet, an in-house pricer, or any other pricing engine without changing the Chebyshev workflow.

The key formula for a regular bullet bond is:

```text
PV = N * (P(curve, T) + coupon * A(curve, T))
```

where `P` is discounted principal and `A` is fixed-leg annuity. This identity should be used as a diagnostic and possible decomposition strategy even when the baseline pricer is external. The core hypothesis is that coupon should be handled analytically rather than used as a tensor dimension.

### Sensitivity Harness

Once the baseline pricer is reliable, compute all required sensitivities from the baseline by finite differences:

```text
dPV / dr_i
dPV / dc
dPV / dT
d2PV / dr_i dc
d2PV / dr_i dT
```

Use central differences away from boundaries and one-sided or nudged stencils near boundaries. The report must record step sizes because derivative results can be dominated by stencil choice when maturity crosses schedule/accrual breakpoints.

For the decomposed model, validate the analytic relation:

```text
d2PV / dr_i dc = N * dA / dr_i
```

This should be preferred over estimating the same term as a noisy second mixed finite difference when the bond family is regular enough for the identity to hold.

### Structural Sanity Checks

Validation must include financial invariants, not only surrogate-vs-baseline error statistics:

- A matured bond with no remaining cashflows should have zero dirty PV, zero clean PV, and zero rate sensitivities.
- A curve node or curve segment with no interpolation support before the final cashflow should have zero DV01 for that bond.
- Coupon sensitivity should equal notional times annuity for a regular fixed-rate bullet bond.
- Zero-coupon cases should remove the coupon-leg contribution.
- Principal-only and annuity-only components should recombine to the full baseline PV.

Be precise about the curve-sensitivity definition. A pillar beyond maturity can still matter if the curve interpolation method uses that pillar to interpolate discount factors before the final cashflow. The invariant is not "all later tenors are always zero" for every interpolation scheme; it is "cashflows after maturity do not exist, and curve bumps with no support on remaining cashflow discount factors must have zero sensitivity."

### Surrogate Variants

The harness should compare:

1. Full PV TT: `PV(curve bumps, coupon, maturity)`.
2. Full PV Slider: same input space, with curve/coupon/maturity partitioning.
3. Decomposed TT: `Principal(curve bumps, maturity)` and `Annuity(curve bumps, maturity)`, then `PV = N * (Principal + coupon * Annuity)`.
4. Optional dense `ChebyshevApproximation` on low-dimensional reduced cases as a reference.

Slider is expected to fail cross-group mixed terms when interacting variables are split across groups. TT is expected to capture more coupling but may still produce weak derivatives because current TT derivatives use finite differences on the interpolant.

## Research Workflow

Each cycle must update a durable checkpoint in `docs/research/fixed-rate-bond-surrogate/status.md`:

- Current objective.
- Hypothesis under test.
- Files changed.
- Commands run.
- Key metrics.
- Failure modes found.
- Next task.

Each experiment should produce a compact report in `docs/research/fixed-rate-bond-surrogate/reports/` with:

- Configuration: curve pillars, bump domain, coupon range, maturity range, nodes, ranks, split strategy, random seed.
- Accuracy metrics: PV, DV01, coupon delta, maturity finite difference, rate-coupon mixed term, rate-maturity mixed term.
- Worst cases: input point, exact value, surrogate value, absolute error, relative error.
- Build metrics: evaluations, time, ranks, compression ratio.
- Conclusion: accepted, rejected, or inconclusive.

This gives future conversations a restart point and prevents repeated audits of the same ground.

## Auto-Research Execution Loop

This project is the main scope for the next small version update. The loop for each large phase is:

1. Read the current design spec, status file, and prior phase reports.
2. Run a focused design brainstorm for the phase before implementation.
3. Search public documentation and references relevant to that phase.
4. Implement the smallest useful harness/test/reporting increment for that phase.
5. Validate with deterministic tests and phase-specific research reports.
6. Update `docs/research/fixed-rate-bond-surrogate/status.md`.
7. Stop only when the phase exit gate is satisfied or a blocker is recorded.

Do not treat each small task as a standalone PR. Small fixes, chores, and documentation edits that support the current phase should accumulate on the phase branch and be reviewed together.

## Issue, Branch, and PR Cadence

Use issue-backed, phase-level development:

- Create one tracking issue for the fixed-rate bond surrogate small-version objective.
- Create additional focused issues only for real bugs, design blockers, or follow-up features discovered during research.
- Work on a dedicated branch for this effort, then keep phase work on that branch or on phase sub-branches as needed.
- Open one PR per completed large phase, not one PR per trivial task.
- A phase PR should include implementation, tests, reports, documentation updates, and status-file updates needed to satisfy that phase's exit gate.
- Do not merge a phase PR until validation passes and the report explains what was learned.
- Release notes should reference the phase PR and any issues closed by that PR.

If a bug is discovered while working on a phase:

1. Open an issue with reproduction notes and impact.
2. If it is directly required for the current phase, fix it in the phase branch and reference the issue in the PR.
3. If it is unrelated or risky, leave it as a tracked follow-up unless it blocks validation.

## Private Source Material Handling

The public repository should contain only sanitized requirements, public references, public data sources, and reproducible example code.

If private assessment notes or conversation transcripts are needed for continuity, store them outside version control. Preferred options:

1. A private folder outside the repository.
2. A local ignored folder such as `.private/` if the user explicitly approves adding that folder to `.gitignore`.

Private material should be used only to extract public-safe requirements. Do not copy proprietary names, private interface signatures, internal object names, production logs, or confidential market data into examples, docs, issues, PRs, commits, or release notes.

The public durable record is `docs/research/fixed-rate-bond-surrogate/status.md`; it should summarize only generic assumptions and decisions.

## Market Data and Baseline Strategy

Use public data only for the tutorial and reproducibility harness. Prefer real market-data fixtures for public plots and documentation so the example is credible as a business demonstration:

- Federal Reserve nominal yield curve data for public zero-coupon-style curve inputs. These data provide fitted nominal yield curve parameters and smoothed yields on hypothetical Treasury securities from 1961 to present.
- U.S. Treasury Daily Treasury Par Yield Curve Rates as an official public par-yield source. These are par/CMT yields, not zero rates, so either bootstrap discount factors explicitly or label them as par-yield inputs.
- New York Fed SOFR and SOFR Averages/Index data for official overnight benchmark examples. These are overnight/compounded historical rates, not a full forward SOFR OIS zero curve.
- Synthetic coupons and maturities for controlled experiments. The coupon and maturity ranges are scenario dimensions, not necessarily actual issued-bond records.

Real-data fixtures should record:

- source institution and source URL;
- curve date and download date;
- raw field names and units;
- whether rates are par yields, fitted zero yields, SOFR overnight rates, SOFR averages/index values, or synthetic transforms;
- compounding and interpolation assumptions used by the example;
- any transformation from raw market data to the direct zero-rate curve used by the surrogate.

If synthetic shocks, coupons, or maturities are used, label them as scenario design choices around the real market curve rather than as observed market trades.

For correctness baselines:

1. Prefer QuantLib or QLNet as the primary baseline pricer for the example.
2. Avoid making QuantLib or QLNet a required runtime dependency for the ChebyshevSharp library package. Keep them inside the example, an optional validation tool, or a separate test project.
3. Use a tiny transparent in-repo exact pricer only if needed to keep CI deterministic or to explain the principal/annuity identity.
4. Pin a small public curve fixture in the repo so CI is deterministic. Live market downloads should be optional refresh tools, not unit-test dependencies.

Never compare a surrogate to another surrogate as the only baseline. Each experiment must retain an exact-pricer comparison.

Data refresh should be Python-first unless there is a clear .NET advantage. Python has better lightweight tooling for public market-data pulls, CSV normalization, pandas inspection, and report generation. The repository can include optional scripts that fetch public data and write pinned JSON/CSV fixtures. The C# example should consume those fixtures rather than call live web APIs during tests.

The first harness should use a direct zero-rate curve as its state variable. This gives a transparent sensitivity definition:

```text
zero-pillar DV01 = price sensitivity to bumping one zero-rate pillar, then interpolating the zero curve
```

This is intentionally different from market-quote DV01:

```text
market-quote DV01 = price sensitivity to bumping an input deposit/future/swap/OIS quote, rebuilding the bootstrapped curve, then repricing
```

Market-quote DV01 is closer to production curve-risk systems, but it introduces bootstrapping, instrument selection, interpolation choices, and "trickle-down" behavior where one quote bump changes many downstream discount factors. Keep that out of the first tutorial. It can be a later experiment after the direct-zero harness is understood.

## Task Breakdown

### Phase 0: Setup and Guardrails

- Confirm no proprietary names appear in code, docs, reports, branches, commits, or issue titles.
- Create the durable research folder and status file.
- Add a checklist that tracks phase status across conversation compactions.
- Decide where private source material lives, if any, without committing it.

Exit gate: repository contains a public-safe research plan and empty report structure.

### Phase 1: Baseline Pricer Selection and Adapter

- Evaluate QuantLib C#, QLNet, and QuantLib Python for the fixed-rate bond use case.
- Select the baseline path for the tutorial.
- Implement a small adapter boundary so the Chebyshev harness calls a generic pricing function, not library-specific types.
- Add smoke tests for dirty PV, clean PV, accrued interest, cashflow count, and basic convention handling.
- Add optional sanity tests for coupon linearity and principal/annuity decomposition if the selected baseline exposes enough cashflow detail.

Exit gate: a trusted baseline pricer can be called deterministically from the harness, with public fixtures and no Chebyshev approximation yet.

### Phase 2: Data Fixture Pipeline

- Build an optional Python data-refresh script for public curve data.
- Normalize public curve data into pinned fixtures consumed by C#.
- Record source URL, download date, curve date, units, compounding assumption, interpolation assumption, and whether the input is zero-rate, par-yield, SOFR, or synthetic.
- Prefer real public market fixtures for documentation plots and business-demonstration outputs; reserve synthetic curves for controlled edge-case tests.
- Keep live downloads out of CI.

Exit gate: the harness has deterministic public fixtures, enough metadata to explain what the curve represents, and at least one real-market fixture suitable for documentation plots.

### Phase 3: Smoothness Diagnostics

- Scan one-dimensional slices for curve bumps, coupon, and maturity.
- Measure PV continuity, finite-difference slope, second-difference spikes, accrued-interest discontinuities, and schedule-boundary effects.
- Compare maturity slices using calendar dates rather than only decimal years.

Exit gate: report identifies whether PV, slope, and sensitivities are smooth or piecewise smooth, and where breakpoints occur.

### Phase 4: Reproduce the Surrogate Problem

- Build the full PV TT and full PV Slider.
- Validate on held-out random, boundary, and schedule-boundary samples.
- Report PV, DV01, coupon delta, maturity sensitivity, and mixed cross-term errors.
- Confirm whether acceptable PV can coexist with unacceptable Greeks.
- Run structural sanity checks on both the baseline and surrogates, including matured-bond zero PV/DV01 and post-final-cashflow curve-support checks.

Exit gate: a report either reproduces the observed failure or explains why the local model does not.

### Phase 5: Analytic Coupon Decomposition

- Build principal and annuity surrogates over `(curve bumps, maturity)`.
- Compare against the full PV surrogate.
- Replace rate-coupon mixed second derivative with first derivative of the annuity surrogate:

```text
d2PV / dr_i dc = N * dA / dr_i
```

Exit gate: report shows whether removing coupon as a tensor dimension improves PV and sensitivity stability.

### Phase 6: Maturity Splitting

- Compare no split, 1Y buckets, 0.5Y buckets, and schedule-boundary-aware buckets.
- Use half-open intervals except the final bucket.
- Evaluate both PV and derivative errors near bucket edges.

Exit gate: choose the simplest maturity split that meets validation thresholds, or document why adaptive splitting is required.

### Phase 7: Adaptive Splitting Research

- Prototype only after fixed splits are tested.
- Use second-difference spikes, coefficient-tail diagnostics, and held-out validation error as candidate split signals.
- Keep this as research code unless it clearly generalizes to a library feature.

Exit gate: decision on whether ChebyshevSharp needs a future `PiecewiseChebyshevTT` or selected-dimension auto-splitting API.

### Phase 8: Tutorial and Documentation

- Add a finance tutorial explaining the restricted product, exact formula, Chebyshev model choices, and validation results.
- Explain that the baseline price is trusted because it comes from an established open-source financial library, while the Chebyshev object is only a surrogate for that pricing function.
- Use real public market fixtures for headline prices, sensitivities, and plots where possible; label synthetic scenario dimensions clearly.
- Explain why coupon is analytical, why maturity is piecewise, why Slider misses cross-group interactions, and why TT sensitivities need validation.
- Include Sobol/variance attribution as a diagnostic, not as the sole pruning rule.
- Link to `ChebyshevTT`, `ChebyshevSlider`, `ChebyshevSpline`, adaptive refinement, and testing guidance.

Exit gate: documentation is public-safe, reproducible, and useful as a tutorial.

### Phase 9: Library Improvement Issues

Open follow-up issues only after evidence supports them:

- `PiecewiseChebyshevTT` or `BucketedTT`.
- Spectral/analytic TT derivatives.
- Built-in validation reports.
- Domain policy diagnostics.
- Built-in Sobol pruning workflow.
- Multi-output TT for principal/annuity style decompositions.

Exit gate: issues cite local reproduction reports and avoid speculative implementation.

## Acceptance Criteria

The work is successful if it produces:

- A public-safe fixed-rate bond example with no proprietary references.
- A trusted external baseline pricer path with deterministic adapter tests.
- Optional transparent reference-pricer checks only where useful for explanation or CI stability.
- A public-data fixture pipeline with pinned fixtures and no live downloads in CI.
- Local evidence explaining PV versus Greek error behavior.
- A comparison of Slider, full PV TT, and decomposed TT.
- A maturity-splitting recommendation backed by reports.
- A tutorial suitable for a small documentation-focused version update.
- A durable status file that future sessions can resume from without re-discovery.

## Open Decisions

- Whether the first harness should use weekend-only calendars or include a small public holiday calendar.
- Whether maturity input should be exposed to the example as a `DateTime`, decimal years, or both with clear warnings.
- Whether generated reports should be committed or only summarized in documentation.
- What numerical thresholds should define "acceptable" PV, DV01, and mixed-term errors for the tutorial.
- Whether the first external baseline should be QuantLib C#, QLNet, or QuantLib Python with generated fixtures.
