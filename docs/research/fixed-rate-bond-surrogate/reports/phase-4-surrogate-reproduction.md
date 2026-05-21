# Phase 4 Report: Full-PV Surrogate Reproduction

## Objective

Phase 4 tests the failure mode before implementing fixes: train direct full-PV
Chebyshev surrogates for a restricted fixed-rate bond example, then compare PV
and finite-difference sensitivities against the QLNet reference pricer.

This phase intentionally does **not** use analytic coupon decomposition,
maturity splitting, or adaptive splitting. Those changes belong to later phases;
the point here is to capture the baseline problem with a clean public example.

## Data and Baseline

- Data fixture: `examples/FixedRateBondSurrogate/Data/fed-nominal-yield-curve-2026-05-15.json`.
- Source: Federal Reserve nominal yield curve data. The Federal Reserve page
  describes the data as daily fitted nominal yield-curve parameters and smoothed
  yields from 1961 to present, and notes that the model is a staff research
  product rather than an official statistical release.
- Fields used: `SVENY01`, `SVENY02`, `SVENY03`, `SVENY05`, `SVENY07`,
  `SVENY10`, `SVENY20`, and `SVENY30`. The Federal Reserve table labels
  `SVENYXX` as continuously compounded zero-coupon yields.
- Reference pricer: QLNet `FixedRateBond` priced with `DiscountingBondEngine`,
  matching QuantLib's documented bond discounting-engine pattern.

## Experiment Setup

The full-PV function is

$$
f(b_1,b_5,b_{10},c,T) = \operatorname{DirtyPrice}
$$

where $b_1$, $b_5$, and $b_{10}$ are zero-rate pillar bumps in basis points,
$c$ is the annual coupon, and $T$ is maturity in years from valuation date.

The domain is:

| Dimension | Range |
| --- | --- |
| 1Y zero-rate bump | `[-150, 150]` bp |
| 5Y zero-rate bump | `[-150, 150]` bp |
| 10Y zero-rate bump | `[-150, 150]` bp |
| coupon | `[0.00, 0.12]` |
| maturity | `[8.0, 12.0]` years |

The 20Y and 30Y pillars are deliberately excluded in this compact first
surrogate because Phase 3 showed they have zero direct DV01 support for the
regular ten-year example under the current interpolation setup.

## Compared Quantities

PV error is reported as

$$
e_{\mathrm{PV}} = |\hat f(x) - f(x)|.
$$

Relative error uses a small floor to avoid dividing by zero:

$$
e_{\mathrm{rel}} = \frac{|\hat q(x)-q(x)|}{\max(|q(x)|,10^{-10})}.
$$

Zero-pillar DV01 is computed as a finite difference in the basis-point bump
coordinate:

$$
\operatorname{DV01}_i(x) \approx
\frac{f(x+h_i e_i)-f(x-h_i e_i)}{2h_i},
\qquad h_i=1\text{ bp}.
$$

Coupon and maturity sensitivities use the same central-difference pattern:

$$
\frac{\partial f}{\partial c} \approx
\frac{f(x+h_c e_c)-f(x-h_c e_c)}{2h_c},
$$

$$
\frac{\partial f}{\partial T} \approx
\frac{f(x+h_T e_T)-f(x-h_T e_T)}{2h_T}.
$$

Mixed terms use

$$
\frac{\partial^2 f}{\partial a\,\partial b} \approx
\frac{f_{++}-f_{+-}-f_{-+}+f_{--}}{4h_a h_b}.
$$

## Results

Command:

```bash
dotnet run --project examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj -- --surrogate-reproduction
```

Measured on 9 deterministic validation points:

| Model | Build evals | Max PV rel. error | Max maturity-sensitivity rel. error | Max rate-coupon mixed rel. error | Max rate-maturity mixed rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorTrain | 408 | 0.35% | 398.88% | 23.62% | 150.84% |
| Slider | 135 | 6.94% | 381.98% | 100.00% | 100.00% |

Additional TensorTrain sensitivity errors:

| Metric | Max abs. error | Max rel. error |
| --- | ---: | ---: |
| 1Y zero-pillar DV01 | `8.109793E-006` | 11.42% |
| 5Y zero-pillar DV01 | `1.989684E-005` | 1.24% |
| 10Y zero-pillar DV01 | `3.615102E-003` | 8.56% |
| coupon derivative | `2.130192E+000` | 0.30% |

## Interpretation

The compact TensorTrain result reproduces the central concern: the direct
full-PV surrogate can produce a low PV error while some derivative-like
quantities, especially maturity sensitivity and mixed terms, remain weak. This does
not prove TensorTrain is unsuitable; it shows that the direct full-PV tensor is
not enough evidence for risk accuracy.

The Slider result is useful as a contrast case. Its partition is
`[[curve pillars], [coupon], [maturity]]`, so cross-group mixed partials are not
represented by construction. The measured 100% mixed-term relative errors are
therefore an expected consequence of the model structure, not a surprising
implementation bug.

## Next Phase

Phase 5 should test the mathematical decomposition

$$
PV = N\left(P(\mathrm{curve},T) + c A(\mathrm{curve},T)\right),
$$

where $P$ is principal PV and $A$ is fixed-leg annuity. That removes coupon as a
tensor dimension and turns the rate-coupon mixed term into a first derivative
of the annuity surface.

## Verified Sources

- Federal Reserve nominal yield curve page: <https://www.federalreserve.gov/data/nominal-yield-curve.htm>
- Federal Reserve nominal yield curve table: <https://www.federalreserve.gov/data/yield-curve-tables/feds200628_1.html>
- QuantLib-Python bond pricing engines documentation: <https://quantlib-python-docs.readthedocs.io/en/latest/pricing_engines/bonds.html>
- Chebfun guide on Chebyshev interpolation and piecewise smooth functions: <https://www.chebfun.org/docs/guide/guide01.html>
- Local ChebyshevSharp TT guide: [`docs/docs/tensor-train.md`](../../../docs/tensor-train.md)
- Local ChebyshevSharp Slider guide: [`docs/docs/slider.md`](../../../docs/slider.md)
