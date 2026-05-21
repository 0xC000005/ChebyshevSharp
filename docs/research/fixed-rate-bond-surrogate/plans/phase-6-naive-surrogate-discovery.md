# Phase 6 Naive Surrogate Discovery Implementation Plan

> **For agentic workers:** Use this phase to discover and document naive
> full-PV surrogate failure modes. Do not implement analytic coupon
> decomposition, maturity buckets, adaptive splitting, or portfolio-specific
> modelling in this phase.

**Goal:** Build evidence for what breaks when a fixed-rate bond pricer is
approximated naively as one full-PV Chebyshev function.

**Architecture:** Keep QLNet as the reference pricer. First estimate the
infeasibility of a full dense Chebyshev tensor over the dense curve fixture.
Then run naive full-PV TensorTrain and Slider comparisons whose public callable
input is the full 62-coordinate vector: 60 curve-bump coordinates plus coupon
and maturity. Report PV, DV01, coupon,
maturity sensitivity, and mixed finite-difference errors, plus structural
post-maturity support checks and maturity-date smoothness diagnostics.

**Tech stack:** C#/.NET 10 example project, QLNet reference pricer,
ChebyshevSharp `ChebyshevTT`, ChebyshevSharp `ChebyshevSlider`, xUnit, pinned
Federal Reserve dense zero-curve fixture, DocFX research docs.

## Scope Decisions

- Treat the conceptual input blocks as curve, coupon, maturity, and notional.
- Count Chebyshev dimensions by scalar coordinates: each curve pillar is one
  dimension. The dense fixture has 60 semiannual market points; excluding
  notional, the naive surrogate has 62 dimensions.
- Hold notional fixed at `100.0` because Phase 5 already verifies linear
  notional scaling. This keeps the failure search focused on curve, coupon,
  and maturity.
- Do not train a dense full tensor. Record its node count to show why it is not
  a viable starting point.
- Do not use selected-pillar surrogate inputs as evidence for the clone
  objective. Any TT, Slider, bucketed, decomposed, or routed model must expose
  the full 62-coordinate input at the wrapper boundary.
- Internal models may partition, route, or ignore coordinates where the design
  justifies it, but tests and reports must state that distinction explicitly.
- Treat user hypotheses as unproven until measured: DV01 weakness, mixed-term
  weakness, and maturity non-smoothness must be supported by local evidence.
- Use "maturity sensitivity" for `dPV/dT`, where `T` is the contractual
  maturity parameter and the valuation date is fixed. Do not call this theta or
  roll-down.
- Keep direct zero-pillar DV01 separate from bootstrapped market-quote DV01.
  This phase bumps direct zero-rate nodes in the pinned curve fixture.
- The naive TT should remain a full-input, low-node, canonical-order TT-Cross
  probe. Do not use `WithAutoOrder()`, Sobol pruning, decomposition, or maturity
  buckets until the naive failure evidence is recorded.
- The naive Slider should remain a full-input singleton-partition contrast case.
  Its low build evaluation count is expected because it builds 62 one-dimensional
  slides with 3 nodes each.

## Tasks

- [x] Add tests for a Phase 6 report that uses the dense fixture, reports full
  tensor infeasibility, builds TensorTrain and Slider summaries, and includes
  maturity-smoothness evidence.
- [x] Implement a `NaiveSurrogateDiscovery` harness with deterministic domains,
  validation points, finite-difference metrics, and a maturity boundary scan.
- [x] Add a CLI mode that prints the Phase 6 findings without changing the
  default pricing example.
- [x] Run the focused Phase 6 tests and CLI mode. Record the actual metrics in
  the Phase 6 report.
- [x] Update `status.md`, public examples, and the meta issue with the evidence.
- [x] Add risk-terminology citations and post-maturity support checks after the
  full-input correction.
- [x] Run closeout verification before calling the phase complete.

## Exit Gate

The phase is complete when the repo contains a deterministic report answering:

1. why the full dense tensor is infeasible;
2. whether naive TT and Slider reproduce PV;
3. whether DV01 errors are acceptable or problematic;
4. whether coupon/maturity/rate mixed terms are weak;
5. whether unsupported post-maturity direct-zero sensitivities are zero;
6. whether the baseline maturity scan shows sensitivity or second-difference spikes;
7. what evidence should drive the next modelling phase.

Stop at this gate. Do not implement the next modelling approach in this phase.
