# Phase 9 Report: Maturity Special Points

## Objective

Phase 9 tests whether the remaining maturity-sensitivity failures can be
reduced by splitting the maturity axis at evidence-backed special points while
preserving the public wrapper:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

The internal model keeps Phase 8's analytic coupon identity:

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

This phase does not add a reusable ChebyshevSharp API. It builds an example-local
router first, then decides whether a library feature is justified.

## Research Basis

Chebfun motivates the modelling idea: piecewise-smooth functions can be
represented as smooth pieces with breakpoints, and automatic splitting can be
useful when a global Chebyshev representation fails. Chebfun also warns that
splitting can add unnecessary pieces, so it must be validated rather than used
blindly.

PyChebyshev already exposes this idea for dense splines through
`special_points`: declaring a kink dispatches `ChebyshevApproximation` to a
piecewise `ChebyshevSpline`. Phase 9 asks whether the same concept is needed for
the high-dimensional fixed-rate bond harness, where maturity changes can alter
cashflow count and final accrual behavior.

## Breakpoint Inventory

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

The inventory scanned one-day windows around semiannual maturity regions from
2Y to 30Y using the QLNet reference pricer.

| Inventory metric | Value |
| --- | ---: |
| Inventory points | `167` |
| Largest spike date | `2038-05-15` |
| Max absolute second difference | `6.116018E-003` |
| Max one-day slope jump | `2.233876E+000` |

Top raw spike candidates:

| Maturity date | Offset | Cashflows | Second difference | Left slope/year | Right slope/year |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2038-05-15` | `0d` | `25` | `6.116018E-003` | `-1.954641E+000` | `2.792344E-001` |
| `2036-11-15` | `0d` | `22` | `5.901720E-003` | `-1.879544E+000` | `2.760599E-001` |
| `2042-11-15` | `0d` | `34` | `5.656000E-003` | `-1.785514E+000` | `2.803401E-001` |
| `2039-05-14` | `-1d` | `27` | `5.471144E-003` | `-1.976359E+000` | `2.197626E-002` |
| `2037-11-14` | `-1d` | `24` | `5.400643E-003` | `-1.953188E+000` | `1.939687E-002` |

These are not arbitrary numerical wiggles. The scan records structural reasons
such as cashflow-count and coupon-count changes, while also recording final
coupon accrual behavior for audit.

## Candidate Configuration

All candidates keep the full public 62-coordinate wrapper. Internally, they use
two dense 4D tensors over level/slope/curvature curve factors and maturity:
one tensor for principal PV and one tensor for annuity PV.

| Candidate | Pieces | Build evals | Purpose |
| --- | ---: | ---: | --- |
| Global decomposed curve-factor tensor | `1` | `162` | No maturity splitting control |
| Semiannual uniform bucketed decomposed factor tensor | `56` | `9,072` | Fixed 0.5Y bucket control |
| Schedule-aware special-point decomposed factor tensor | `57` | `9,234` | Split at schedule-derived maturity candidates |
| Automatic-detector special-point decomposed factor tensor | `33` | `5,346` | Split at spaced second-difference candidates |
| Hybrid special-point decomposed factor tensor | `81` | `13,122` | Union of schedule and detector candidates |

Candidate counts:

| Candidate list | Count | Source |
| --- | ---: | --- |
| Schedule-aware special points | `56` | Semiannual schedule boundary inventory |
| Automatic detector candidates | `32` | Largest spaced maturity-axis second differences |
| Hybrid special points | `80` | Union of schedule-aware and detector candidates |

## Results

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity rel. error | Max factor-aligned PV rel. error |
| --- | ---: | ---: | ---: | ---: |
| Global decomposed curve-factor tensor | `5.12%` | `142.62%` | `74.94%` | `2.23%` |
| Semiannual uniform bucketed decomposed factor tensor | `4.70%` | `96.44%` | `55.52%` | `0.57%` |
| Schedule-aware special-point decomposed factor tensor | `4.73%` | `89.21%` | `48.75%` | `0.57%` |
| Automatic-detector special-point decomposed factor tensor | `4.73%` | `274.41%` | `59.50%` | `0.58%` |
| Hybrid special-point decomposed factor tensor | `4.73%` | `399.72%` | `48.75%` | `0.57%` |

## Interpretation

Schedule-aware routing improves the two key residual errors relative to the
uniform 0.5Y control: maturity relative error falls from `96.44%` to `89.21%`,
and coupon-maturity mixed relative error falls from `55.52%` to `48.75%`.
This is a real improvement, but it is not a finished risk clone.

Detector-only splitting is not ready. Even after spacing the top second-
difference candidates, it leaves maturity relative error at `274.41%`. The
hybrid keeps the schedule-aware coupon-maturity result but worsens maturity
error, so more breakpoints are not automatically better.

The result supports a narrow next step: design a schedule-aware high-dimensional
piecewise router and validate one-sided sensitivity behavior near split points.
It does not yet justify a generic automatic kink-detection API.

## Decision

1. Keep the full public wrapper and analytic coupon decomposition.
2. Prefer schedule-aware maturity special-point routing over uniform buckets for
   the next modelling iteration.
3. Treat automatic detection as a diagnostic source only until it beats the
   schedule-aware control on held-out sensitivities.
4. Open a future library design only for a minimal high-dimensional piecewise
   router, not a broad automatic splitting feature.

## Verification

Fresh checks run during Phase 9 implementation:

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondMaturitySpecialPointTests" --verbosity minimal
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --maturity-special-points
```

Results:

- focused Phase 9 tests passed `8` tests with `0` failures;
- the Phase 9 example completed successfully and printed the results above.

## Sources

- Chebfun Guide, "Piecewise smooth chebfuns": <https://www.chebfun.org/docs/guide/guide01.html>
- Pachón, R., Platte, R. B., & Trefethen, L. N. (2010). "Piecewise-smooth chebfuns." *IMA Journal of Numerical Analysis*, 30(4), 898-916. <https://doi.org/10.1093/imanum/drp008>
- Oxford Mathematical Institute seminar page, "Optimal domain splitting in Chebyshev collocation": <https://www.maths.ox.ac.uk/node/10792>
- PyChebyshev special-points guide: <https://0xc000005.github.io/PyChebyshev/user-guide/special-points/>
- Phase 8 report: [Analytic Coupon Decomposition](phase-8-analytic-coupon-decomposition.md)
