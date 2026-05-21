# Phase 8 Analytic Coupon Decomposition Plan

> **For agentic workers:** Phase 8 starts from the merged Phase 7 evidence. It
> tests the fixed-rate bond identity before adding library-level special-point
> or automatic kink detection features.

**Goal:** Test whether removing coupon as a Chebyshev dimension improves the
fixed-rate bond surrogate while preserving the public full-wrapper interface.

**Public wrapper contract:**

```text
curve bumps[60], coupon, maturity -> dirty PV
```

**Core identity:**

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

For the restricted regular fixed-rate bullet bond used by this harness, coupon
payments are fixed cashflows proportional to the coupon rate. Coupon should
therefore be handled analytically unless the identity fails under the reference
pricer/conventions.

## Scope Decisions

- Keep automatic kink detection, special-point TT routing, and library API work
  out of Phase 8. Those belong to Phase 9.
- Keep the full 62-coordinate wrapper at the public boundary. Internal models
  may drop coupon only because the wrapper recombines it analytically.
- Compare against Phase 6 and Phase 7 evidence using the same clone and
  factor-aligned validation point families.
- Report PV, zero-pillar DV01, coupon derivative, maturity sensitivity,
  rate-coupon mixed, rate-maturity mixed, rate-rate mixed, and coupon-maturity
  mixed terms.
- Treat the decomposition as accepted only if it improves the coupon derivative
  and coupon-related mixed terms without hiding PV or maturity errors.

## Candidate Families

1. **Exact decomposition oracle**
   - Compute principal from the reference pricer at coupon `0`.
   - Compute annuity as the finite coupon slope from two reference-pricer calls.
   - Recombine as `Principal + coupon * Annuity`.
   - This should match the reference pricer to numerical tolerance and validates
     the formula before fitting a Chebyshev object.

2. **Global decomposed TT**
   - Build one TT for principal and one TT for annuity over
     `curve bumps[60] + maturity`.
   - Recombine through the same 62D wrapper.
   - This tests whether removing coupon alone improves the full global TT.

3. **Curve-factor decomposed tensor**
   - Project curve bumps into the Phase 7 level/slope/curvature factor basis.
   - Build dense 4D tensors for principal and annuity over
     `level, slope, curvature, maturity`.
   - Recombine through the full 62D wrapper.

4. **Bucketed decomposed curve-factor tensor**
   - Repeat the factor model in 1Y and 0.5Y maturity buckets.
   - This checks whether decomposition plus simple maturity routing is enough
     before Phase 9 special-point or edge-detection work.

## Tasks

- [x] Add failing Phase 8 tests requiring the report, decomposition identity,
  model names, and CLI mode.
- [x] Implement the exact decomposition oracle and verify identity error.
- [x] Implement global decomposed TT.
- [x] Implement decomposed curve-factor tensor and bucketed variants.
- [x] Run the benchmark and record results in a Phase 8 report.
- [x] Update `status.md`, examples docs, and citations if new sources are used.
- [x] Update tracking issue #191 with the Phase 8 evidence.
- [x] Open one coherent Phase 8 PR after local verification.

## Local Outcome

The QLNet-backed restricted baseline is coupon-linear to numerical precision:
the maximum absolute identity error across the Phase 6/7 validation bank is
`8.526513E-014`.

The decomposed factor tensor keeps the Phase 7 factor-space PV result while
reducing build evaluations from `675` to `270`. Removing coupon from the global
TT does not solve the global clone problem; max clone PV relative error remains
`14.34%` and maturity-sensitivity relative error remains `456.94%`.

The next phase should keep automatic kink detection and special-point routing
in Phase 9, focused on maturity rather than coupon.

## Exit Gate

Phase 8 is complete when the report can answer:

1. whether the QLNet-backed baseline is coupon-linear under the restricted
   product family;
2. whether analytic coupon recombination improves coupon derivative and
   coupon-maturity mixed terms;
3. whether removing coupon alone fixes the global TT failure;
4. whether factor compression plus decomposition is a better tutorial path than
   Phase 7's full-PV factor tensor;
5. what remains for Phase 9 special-point or automatic kink detection work.
