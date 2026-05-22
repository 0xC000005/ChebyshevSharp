# Phase 11 Fixed-Rate Bond Tutorial Plan

> **For agentic workers:** Phase 11 is a documentation and tutorial phase.
> Do not change the surrogate algorithms, add public APIs, bump versions, or
> publish packages unless the phase scope is explicitly changed.

**Goal:** Turn the fixed-rate bond surrogate research into a public,
non-proprietary tutorial that explains how to use ChebyshevSharp as a
surrogate-modeling tool around a trusted bond pricer, and why naive global
cloning is not enough for this example.

**Public wrapper contract:**

```text
curve bumps[60], coupon, maturity -> dirty PV
```

The tutorial must preserve this full wrapper framing even when explaining
internal models that project curve factors, use analytical coupon recombination,
or route by maturity.

## Storyline

1. **Reference problem**
   - Define the restricted regular fixed-rate bullet bond example.
   - State the public data fixture and reference-pricer assumptions.
   - Explain that the objective is an educational surrogate case study, not a
     universal fixed-income replacement.

2. **Naive global clone**
   - Explain why full dense tensors are infeasible in 62 scalar coordinates.
   - Present global TensorTrain and Slider evidence from Phase 6.
   - Emphasize PV, DV01, maturity sensitivity, and mixed-term diagnostics.

3. **Common practitioner improvements**
   - Show factor compression, bucket/routing ideas, and Slider/TT tradeoffs.
   - Explain when factor-space accuracy is meaningful and when it is not a
     faithful arbitrary-pillar clone.

4. **Bond-specific structure**
   - Explain the coupon-linearity identity:
     `PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)`.
   - Explain why maturity is harder because changing maturity changes the
     cashflow schedule.

5. **Piecewise routing evidence**
   - Summarize schedule-aware special points and the Phase 10 router.
   - State that the router made dispatch auditable but did not solve the
     residual sensitivity error.

6. **Practical conclusion**
   - List what the example demonstrates today.
   - List what it does not yet justify.
   - Point to future work: stronger schedule-aware high-dimensional pieces,
     better derivative diagnostics, and possible library features only after
     more evidence.

## Required Evidence Sources

- Phase 5 report: realistic baseline, dense Federal Reserve fixture, and QLNet
  assumptions.
- Phase 6 report: naive dense-baseline TT/Slider failure evidence and maturity
  schedule-sensitivity plot.
- Phase 7 report: structured alternatives and factor-compression limitations.
- Phase 8 report: analytic coupon decomposition and proof-by-test.
- Phase 9 report: maturity special-point and detector evidence.
- Phase 10 report: explicit schedule-aware router and one-sided diagnostics.

## Documentation Targets

- Add a public case-study/tutorial page under the existing documentation
  structure.
- Add links from the existing examples or research index pages if the tutorial
  would otherwise be hard to discover.
- Keep the status file and tracking issue current.
- Avoid duplicating large result tables where a concise narrative plus links to
  phase reports is clearer.

## Exit Gate

Phase 11 is complete when:

1. the documentation site contains a coherent fixed-rate bond surrogate tutorial;
2. the tutorial makes all baseline assumptions and limitations explicit;
3. the tutorial cites the public data source and reference pricer materials
   already used by the harness;
4. examples, commands, and result claims match the committed reports;
5. DocFX builds cleanly;
6. exactly one Phase 11 PR is opened after local verification passes.
