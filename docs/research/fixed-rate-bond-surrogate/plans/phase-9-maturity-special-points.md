# Phase 9 Maturity Special Points Plan

> **For agentic workers:** Phase 9 starts from the merged Phase 8 evidence. Do
> not implement a final production bond surrogate in this phase. First measure
> whether maturity-aware piecewise routing fixes the remaining sensitivity and
> mixed-term failures.

**Goal:** Test whether maturity special points, schedule-aware routing, or
automatic kink detection improve the fixed-rate bond surrogate while preserving
the public full-wrapper interface.

**Public wrapper contract:**

```text
curve bumps[60], coupon, maturity -> dirty PV
```

**Internal default from Phase 8:**

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

Coupon stays analytical unless a validation check proves the restricted product
family no longer satisfies the fixed-coupon identity. Phase 9 focuses on
maturity because changing contractual maturity can regenerate cashflow dates,
business-day adjustments, accrual periods, and cashflow count.

## Evidence Being Addressed

- Phase 6 found full-input naive failures: TensorTrain max PV relative error
  `17.72%`, maturity-sensitivity relative error `461.43%`, and
  coupon-maturity mixed relative error `49.10%`; singleton Slider reached
  `100.00%` coupon-maturity mixed relative error.
- Phase 6 maturity scans found one-day slope flips around semiannual schedule
  regions. The largest recorded local second difference was near the
  2039-11-15 schedule region, at `2039-11-11`, with left slope/year
  `-2.650493E+000` and right slope/year `2.825619E-002`.
- Phase 7 showed that common modelling improvements help but do not finish the
  clone. Stronger global TT still had maturity-sensitivity relative error
  `256.28%`; semiannual bucketed factor routing improved maturity sensitivity
  versus 1Y buckets but still left `87.72%` maturity error and `56.94%`
  coupon-maturity mixed error.
- Phase 8 proved coupon is not the root cause: the coupon-linearity identity
  held to max absolute error `8.526513E-014`, while the global decomposed TT
  still had maturity-sensitivity relative error `456.94%`.

## Scope Decisions

- Keep the full 62-coordinate wrapper at the public boundary.
- Use Phase 8 principal/annuity recombination as the default internal model.
- Treat uniform 1Y and 0.5Y buckets as controls, not as the proposed final
  method.
- Test schedule-aware and detector-driven maturity breakpoints separately
  before combining them.
- Keep automatic library-level APIs such as `PiecewiseChebyshevTT` out of scope
  unless the benchmark clearly shows the example needs a reusable primitive.
- Do not claim general fixed-income support. The harness remains restricted to
  the documented regular fixed-rate bullet product family.

## Candidate Families

1. **Phase 8 baseline control**
   - Reuse the decomposed factor tensor and semiannual bucketed decomposed
     factor tensor.
   - This anchors Phase 9 to the previous evidence.

2. **Breakpoint inventory**
   - Scan maturities by date around semiannual schedule regions.
   - Record maturity date, adjusted final payment date, cashflow count,
     final accrual fraction, dirty PV, maturity sensitivity, and local second
     difference.
   - This separates actual schedule-regime changes from decimal-year bucket
     guesses.

3. **Declared schedule special points**
   - Build piecewise models whose maturity pieces are split at known
     schedule-regime candidates.
   - Compare against uniform 1Y and 0.5Y routing.

4. **Automatic detector candidates**
   - Use a maturity-axis scan based on finite-difference spikes and/or
     held-out validation error to propose candidate special points.
   - Validate proposed points before accepting them. Detection is evidence, not
     proof.

5. **Hybrid router**
   - Combine schedule-derived special points with detector refinements only if
     each ingredient improves the benchmark independently.

## Metrics

Report the same Phase 6-8 metric bank:

- PV;
- selected zero-pillar DV01;
- coupon derivative;
- maturity sensitivity;
- rate-coupon mixed;
- rate-maturity mixed;
- rate-rate mixed;
- coupon-maturity mixed.

For every candidate, also report build evaluations, build seconds, bucket or
piece count, worst validation point, and whether the worst point lies near a
schedule boundary. Derivatives exactly at special points should be treated with
one-sided diagnostics; do not use undefined central-knot derivatives as a
headline metric.

## Tasks

- [ ] Update the Phase 9 branch, status file, and tracking issue before
  implementation.
- [ ] Add a failing test and CLI/report skeleton for the maturity-special-point
  benchmark.
- [ ] Implement the maturity breakpoint inventory and document the evidence.
- [ ] Add schedule-aware declared special-point routing.
- [ ] Add automatic detector candidates and validate their proposed breakpoints.
- [ ] Add the hybrid router only if the previous two candidates justify it.
- [ ] Run the full benchmark and write the Phase 9 report.
- [ ] Update examples docs, citations, and status notes.
- [ ] Open one coherent Phase 9 PR after local verification.

## Exit Gate

Phase 9 is complete when the report can answer:

1. whether the maturity failures align with schedule-regime changes;
2. whether schedule-aware special points improve maturity and mixed-term errors
   versus uniform buckets;
3. whether automatic detection finds useful candidate breakpoints without
   adding excessive pieces;
4. whether a hybrid router is materially better than schedule-only routing;
5. whether evidence justifies a future library-level feature such as
   high-dimensional piecewise TT routing.

