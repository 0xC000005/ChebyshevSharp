# Phase 7 Structured Alternatives Benchmark Plan

> **For agentic workers:** Phase 7 starts from the Phase 6 evidence bank. It
> measures candidate modelling changes; it does not implement the final
> production architecture.

**Goal:** Compare controlled alternatives that may fix the naive global
fixed-rate bond surrogate failures measured in Phase 6.

**Baseline:** QLNet-backed dense semiannual fixed-rate bond pricer from the
Phase 5 fixture and Phase 6 validation points.

**Required wrapper contract:** Every candidate must expose the same public
62-coordinate input used in Phase 6:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

Internal implementations may route, bucket, partition, reorder, or ignore
coordinates where mathematically justified, but the tests and reports must keep
the full-wrapper input boundary visible.

## Scope Decisions

- Use the Phase 6 metrics as the comparison bank: PV, direct zero-pillar DV01,
  coupon derivative, maturity sensitivity, rate-coupon mixed, rate-maturity
  mixed, coupon-maturity mixed, rate-rate mixed, post-maturity support checks,
  and maturity-sensitivity spike evidence.
- Do not use selected-pillar inputs as clone evidence.
- Do not claim any candidate is production-ready until it passes held-out PV,
  sensitivity, and structural sanity checks.
- Keep direct zero-pillar DV01 separate from bootstrapped market-quote PV01.
- Record whether a candidate reduces error, increases build cost, or merely
  hides the failure by dropping dimensions.

## Candidate Families

1. **Stronger global TT baseline**
   - Increase rank cap and/or node counts on the same 62D input.
   - Record build evaluations, ranks, PV errors, and sensitivity errors.
   - Use this as the controlled "try harder globally" baseline.

2. **TT dimension ordering and Sobol diagnostics**
   - Test `WithAutoOrder()` or explicit orderings as a modelling variant.
   - Use Sobol outputs diagnostically, but do not drop wrapper inputs unless the
     wrapper still accepts the full 62D vector and documents the internal policy.

3. **Grouped Slider partitions**
   - Compare singleton Slider against partitions that group interacting
     variables, such as coupon+maturity or local curve+maturity groups.
   - Confirm cross-group mixed terms remain zero when variables are separated.

4. **Maturity-aware routing**
   - Test a wrapper that accepts the full 62D input and routes internally by
     maturity bucket.
   - Compare no split, 1Y buckets, 0.5Y buckets, and schedule-boundary-aware
     windows where feasible.

5. **Deferred analytic coupon decomposition**
   - Keep this as a later candidate unless Phase 7 evidence shows coupon
     variation is the dominant blocker.
   - If attempted, it must still be evaluated through the full wrapper contract.

## Tasks

- [ ] Create a fresh Phase 7 branch from `origin/main`.
- [ ] Add tests that load the Phase 6 evidence thresholds and require each
  candidate report to include the full 62D wrapper contract.
- [ ] Implement one candidate family at a time with deterministic validation
  points and no private/proprietary references.
- [ ] Update the Phase 7 report after each candidate with measured errors,
  build cost, and interpretation.
- [ ] Keep the tracking issue updated after each coherent candidate result.
- [ ] Stop after the benchmark comparison; do not merge a final architecture
  without a separate design decision.

## Exit Gate

Phase 7 is complete when the report can answer:

1. whether stronger global TT settings materially improve the naive failure;
2. whether ordering/Sobol diagnostics explain rank or dimension effects;
3. whether grouped Slider partitions recover relevant mixed terms;
4. whether maturity-aware routing improves maturity sensitivity and PV without
   violating the 62D wrapper contract;
5. which candidate should become the next implementation/design phase.
