# Phase 10 Report: Schedule-Aware Piecewise Router

## Objective

Phase 10 turns the Phase 9 maturity-special-point evidence into an explicit
piecewise router while preserving the public fixed-rate bond wrapper:

```text
curve bumps[60], coupon, maturity -> dirty PV
```

Internally, the router still uses the Phase 8 coupon identity:

```text
PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)
```

Each maturity piece owns two dense 4D Chebyshev factor tensors over
level/slope/curvature curve factors and maturity. The router is example-local in
this phase; it is not a public ChebyshevSharp API.

## Router Design

The router uses the Phase 9 schedule-aware maturity candidates as split points.
Detector-only candidates remain diagnostic evidence and are not router inputs.

| Router property | Value |
| --- | ---: |
| Public input dimension | `62` |
| Schedule breakpoints | `56` |
| Routed pieces | `57` |
| Piece interval rule | half-open `[lo, hi)`, final piece closed |
| Router model build evals | `9,234` |

The first pieces are:

| Piece | Interval | Source |
| ---: | --- | --- |
| `0` | `[2.000000, 2.001369)` | Phase 9 schedule-aware candidate |
| `1` | `[2.001369, 2.505133)` | Phase 9 schedule-aware candidate |
| `2` | `[2.505133, 3.000684)` | Phase 9 schedule-aware candidate |
| `3` | `[3.000684, 3.504449)` | Phase 9 schedule-aware candidate |
| `4` | `[3.504449, 4.000000)` | Phase 9 schedule-aware candidate |

## One-Sided Maturity Diagnostics

Central maturity differences can straddle a schedule split and mix two local
pricing regimes. Phase 10 therefore reports one-sided slopes around split
points using `eps = 7 / 365.25` years:

```text
left slope  = (f(T - eps) - f(T - 2eps)) / eps
right slope = (f(T + 2eps) - f(T + eps)) / eps
```

Representative diagnostics from the default run:

| Split | T | Baseline left | Router left | Left abs error | Right abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `split-1` | `2.505133` | `1.852654E-001` | `8.402875E-002` | `1.012366E-001` | `1.972767E-001` |
| `split-2` | `3.000684` | `9.707601E-002` | `2.276867E-001` | `1.306107E-001` | `1.186468E-001` |
| `split-3` | `3.504449` | `1.388640E-002` | `-8.708157E-002` | `1.009680E-001` | `1.991534E-001` |
| `split-4` | `4.000000` | `-6.420540E-002` | `6.359928E-002` | `1.278047E-001` | `1.155446E-001` |
| `split-5` | `4.503765` | `-1.368917E-001` | `-2.379406E-001` | `1.010489E-001` | `1.954785E-001` |

The one-sided diagnostics are more interpretable than cross-boundary central
differences, but they still show material residual slope error near schedule
splits.

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

| Model | Max clone PV rel. error | Max clone maturity rel. error | Max clone coupon-maturity rel. error |
| --- | ---: | ---: | ---: |
| Phase 9 global decomposed factor control | `5.12%` | `142.62%` | `74.94%` |
| Phase 9 uniform 0.5Y control | `4.70%` | `96.44%` | `55.52%` |
| Phase 9 schedule-aware special-point control | `4.73%` | `89.21%` | `48.75%` |
| Schedule-aware router decomposed factor tensor | `4.73%` | `89.21%` | `48.75%` |

## Interpretation

The explicit router reproduces the Phase 9 schedule-aware result. That is useful:
it proves the schedule-derived pieces can be represented as a concrete router
with full-wrapper dispatch and auditable half-open interval semantics.

It does not yet improve the numerical clone. The remaining errors are more
likely caused by the low-order level/slope/curvature factor projection, local
piece model resolution, and finite-difference derivative estimation than by the
absence of an explicit router object.

## Decision

1. Keep the Phase 10 router example-local.
2. Do not add a generic automatic kink-detection API from this evidence.
3. Treat a minimal schedule-aware piecewise-router API as a follow-up design
   topic only after a stronger piece model improves the residual sensitivity
   errors.
4. Use one-sided maturity diagnostics in future reports whenever validation
   points are near schedule split points.

## Verification

Fresh checks run during Phase 10 implementation:

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0 --configuration Release --filter "FullyQualifiedName~FixedRateBondScheduleAwareRouterTests" --verbosity minimal
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --schedule-aware-router
```

Results:

- focused Phase 10 tests passed `8` tests with `0` failures;
- focused coverage found no missing or partial lines in
  `ScheduleAwareRouterBenchmark.cs`;
- the fixed-rate bond slice passed `77` tests with `0` failures;
- full Release tests passed `1,727` tests with `0` failures;
- `dotnet format --verify-no-changes --verbosity minimal` passed;
- `docfx docs/docfx.json` passed with `0` warnings and `0` errors;
- `git diff --check` passed;
- the Phase 10 example completed successfully and printed the results above.

## Sources

- Phase 9 report: [Maturity Special Points](phase-9-maturity-special-points.md)
- Phase 8 report: [Analytic Coupon Decomposition](phase-8-analytic-coupon-decomposition.md)
