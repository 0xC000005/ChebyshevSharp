# Phase 10 Schedule-Aware Piecewise Router Plan

> **For agentic workers:** Phase 10 starts from the merged Phase 9 evidence.
> Do not implement a generic automatic kink detector in this phase. Build and
> validate a schedule-aware high-dimensional piecewise router first.

**Goal:** Prototype a schedule-aware high-dimensional piecewise router for the
fixed-rate bond surrogate harness and decide whether the abstraction should
become a ChebyshevSharp library feature.

**Public wrapper contract:**

```text
curve bumps[60], coupon, maturity -> dirty PV
```

The wrapper remains 62-dimensional even if the internal router projects curve
bumps, removes coupon analytically, or dispatches maturity to pieces.

## Evidence Being Addressed

- Phase 9 showed schedule-aware routing improves the uniform 0.5Y bucket
  control: maturity relative error moved from `96.44%` to `89.21%`, and
  coupon-maturity mixed relative error moved from `55.52%` to `48.75%`.
- Phase 9 also showed detector-only and hybrid routing are not enough:
  detector-only maturity relative error was `274.41%`, and hybrid maturity
  relative error was `399.72%`.
- The next step is therefore not automatic kink detection. It is an explicit
  schedule-aware router with validated one-sided behavior near maturity split
  points.

## Scope Decisions

- Keep this phase inside the fixed-rate bond harness first.
- Build a reusable internal router shape, but do not expose public
  ChebyshevSharp API until the benchmark justifies it.
- Keep analytic coupon decomposition from Phase 8.
- Keep curve factor compression as the first internal model, because Phase 7-9
  showed it is the strongest practical candidate for factor-aligned scenarios.
- Add one-sided derivative validation near split points; central finite
  differences across piece boundaries should not be treated as the headline
  sensitivity.
- Preserve the restricted-product language. This is not a general fixed-income
  replacement.

## Candidate Architecture

1. **Router core**
   - A small internal router maps the full wrapper point to a maturity piece.
   - Each piece owns its model domain, build diagnostics, and evaluation
     delegate.
   - The router records which piece handled each validation point.

2. **Schedule piece source**
   - The first source is the Phase 9 schedule-derived breakpoint inventory.
   - Breakpoints are sorted, deduplicated, and filtered to the open interval
     `(2Y, 30Y)`.
   - Detector candidates remain diagnostic metadata, not router inputs.

3. **Piece models**
   - First implementation: decomposed curve-factor tensors per piece.
   - Optional comparison: TT per piece only if the dense factor piece cannot
     answer the validation question.

4. **Sensitivity validation**
   - Report one-sided maturity sensitivity at or near split points.
   - Keep the existing PV, DV01, coupon, maturity, rate-coupon, rate-maturity,
     rate-rate, and coupon-maturity metric bank for continuity.

## Tasks

- [ ] Add a `--schedule-aware-router` CLI/report mode with a failing test.
- [ ] Extract the Phase 9 piecewise factor logic into an internal router type
      with explicit piece diagnostics.
- [ ] Add schedule breakpoint source tests: sorted, unique, in-domain, and
      traceable to inventory evidence.
- [ ] Add one-sided maturity sensitivity metrics around split points.
- [ ] Compare router results against Phase 9 global, uniform 0.5Y, and
      schedule-aware controls.
- [ ] Document whether the router reduces maturity and coupon-maturity errors
      enough to justify a public API.
- [ ] If justified, open a follow-up issue for a minimal library API design;
      if not justified, document why the example should remain local.
- [ ] Open one coherent Phase 10 PR only after local verification passes.

## Exit Gate

Phase 10 is complete when the report can answer:

1. whether an explicit schedule-aware router is better than the Phase 9
   example-local special-point implementation;
2. whether one-sided sensitivity diagnostics avoid misleading cross-boundary
   finite-difference errors;
3. whether remaining errors are caused by router shape, curve factor projection,
   derivative estimation, or the product-family approximation itself;
4. whether a public ChebyshevSharp feature is justified, and what its minimal
   API should be.
