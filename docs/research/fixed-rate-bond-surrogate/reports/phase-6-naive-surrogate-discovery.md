# Phase 6 Report: Naive Dense-Baseline Surrogate Discovery

## Objective

Phase 6 starts from the viewpoint of a new user trying to accelerate a
fixed-rate bond pricer with Chebyshev tensors. It does not assume the
previous modelling hypotheses are correct. The phase asks what happens if the
QLNet-backed bond price is treated naively as one full-PV function of curve
bumps, coupon, and maturity.

This phase intentionally does **not** use analytic coupon decomposition,
maturity buckets, adaptive splitting, or portfolio-specific models. Those
belong to later phases after the failure modes are measured.

## Baseline and Input Counting

The baseline remains the QLNet-backed reference pricer from Phase 5. The dense
fixture contains 60 semiannual zero-rate market points from 0.5Y to 30Y, plus a
valuation-date anchor used internally by the QLNet zero curve.

Conceptually, the product input blocks are:

```text
curve, coupon, maturity, notional
```

For Chebyshev interpolation, however, each scalar curve bump is one dimension.
Holding notional fixed at `100.0`, the naive scalar surrogate is:

```text
(curve bumps[60], coupon, maturity) -> dirty PV
```

That is a 62-dimensional function before adding notional. Including notional
would make it 63-dimensional, but Phase 5 already verifies notional scaling is
linear.

## Dense Tensor Infeasibility

Even very small dense Chebyshev grids are infeasible:

| Nodes per dimension | Dense node count |
| ---: | ---: |
| `3^62` | `381,520,424,476,945,831,628,649,898,809` |
| `5^62` | `21,684,043,449,710,088,680,149,056,017,398,834,228,515,625` |

So the phase does not attempt a dense full tensor. It uses TensorTrain and
Slider as limited naive probes.

## Limited Naive Probe

The probe keeps the full-PV structure but restricts curve inputs to selected
pillars:

```text
f(b1, b5, b10, b20, b30, c, T) = dirty PV
```

where rate bumps are in basis points, `c` is coupon, and `T` is maturity in
years from the valuation date.

Domain:

| Dimension | Range |
| --- | --- |
| 1Y, 5Y, 10Y, 20Y, 30Y zero-rate bumps | `[-150, 150]` bp |
| coupon | `[0.00, 0.12]` |
| maturity | `[2.0, 30.0]` years |

Both models use the same QLNet baseline, validation points, and
finite-difference metrics. The Slider partition is deliberately naive:

```text
[[all selected curve bumps], [coupon], [maturity]]
```

This tests whether separating curve, coupon, and maturity destroys mixed-term
information.

## Compared Quantities

The report compares PV and finite-difference sensitivities:

```text
PV
dPV / dr_i
dPV / dc
dPV / dT
d2PV / dr_i dc
d2PV / dr_i dT
d2PV / dc dT
```

Rate coordinates are basis-point bump coordinates, so the reported
zero-pillar derivative is directly price change per 1 bp coordinate step.
Maturity uses a seven-day finite-difference step. Relative errors use a small
floor; extremely large relative errors often mean the baseline quantity is
near zero, so max absolute error must be read alongside max relative error.

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --naive-surrogate-discovery
```

Measured on 12 deterministic validation points:

| Model | Build evals | Max PV rel. error | Max maturity-slope rel. error | Max coupon-maturity mixed rel. error |
| --- | ---: | ---: | ---: | ---: |
| TensorTrain | 948 | 2.98% | 115.74% | 16.24% |
| Slider | 3135 | 93.87% | 288.39% | 100.00% |

Selected TensorTrain errors:

| Metric | Max abs. error | Max rel. error | Worst point |
| --- | ---: | ---: | --- |
| PV | `4.790271E+000` | 2.98% | `n10` |
| 10Y zero-pillar DV01 | `6.073550E-002` | very large, baseline near zero at worst point | `n12` |
| 30Y zero-pillar DV01 | `6.056521E-002` | 100.00% | `n4` |
| maturity slope | `1.888824E+001` | 115.74% | `n10` |
| 10Y rate-maturity mixed | `1.309789E-001` | very large, baseline near zero at worst point | `n11` |
| coupon-maturity mixed | `1.194684E+001` | 16.24% | `n12` |

Selected Slider errors:

| Metric | Max abs. error | Max rel. error | Worst point |
| --- | ---: | ---: | --- |
| PV | `6.757798E+001` | 93.87% | `n5` |
| coupon derivative | `9.010148E+002` | 423.05% | `n5` |
| maturity slope | `1.712245E+001` | 288.39% | `n10` |
| 10Y rate-coupon mixed | `3.289965E-002` | 100.00% | `n3` |
| 10Y rate-maturity mixed | `1.311016E-001` | 100.00% | `n11` |
| coupon-maturity mixed | `9.184798E+001` | 100.00% | `n5` |

## Maturity Smoothness Evidence

The baseline maturity scan checks one-day left and right slopes around
semiannual schedule-boundary candidates from 2Y to 30Y. The largest observed
second-difference candidates are:

| Maturity date | Offset | Cashflows | Second difference | Left slope/year | Right slope/year |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2038-05-15 | 0d | 25 | `6.116018E-003` | `-1.953303E+000` | `2.790432E-001` |
| 2036-11-15 | 0d | 22 | `5.901720E-003` | `-1.878257E+000` | `2.758709E-001` |
| 2042-11-15 | 0d | 34 | `5.656000E-003` | `-1.784292E+000` | `2.801483E-001` |
| 2038-11-13 | -2d | 26 | `5.506722E-003` | `-1.990271E+000` | `1.968216E-002` |
| 2039-05-14 | -1d | 27 | `5.471144E-003` | `-1.975006E+000` | `2.196122E-002` |

This supports the working hypothesis that maturity is not globally smooth in
the same way as coupon or small curve bumps. Changing maturity changes the
cashflow schedule and can flip local one-day slopes around schedule boundaries.

## Interpretation

The naive dense tensor is not a viable baseline because the scalar dimension
count is too high. The limited naive TensorTrain is useful enough to show the
trade-off: PV can look moderately acceptable on a small validation set, but
maturity slope, DV01 at some points, and mixed terms are much weaker. The naive
Slider partition is a useful negative control: separating curve, coupon, and
maturity destroys cross-group mixed-term behavior and can also damage PV on the
wide maturity domain.

This is enough evidence to justify testing more structured modelling next, but
not enough to pick the final approach. The next phase should compare fixes
against these measured failure modes rather than assume a solution in advance.
