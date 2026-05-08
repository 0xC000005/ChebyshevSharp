---
title: Which Class Should I Use?
---

# Which Class Should I Use?

ChebyshevSharp has four approximation families. Choose by the shape of your
function first, then tune node counts, knots, groups, or TT rank.

The main questions are:

1. Is the function smooth on one rectangular domain?
2. Is the full grid size, `prod(n_i)`, feasible to evaluate and store?
3. If the full grid is too large, are variable interactions weak, grouped, or
   fully coupled?

## Decision Map

```text
Can you afford the dense grid prod(n_i)?
  yes:
    smooth on one box -> ChebyshevApproximation
    known kinks, jumps, barriers, or regime boundaries -> ChebyshevSpline
  no:
    variables are additive or can be grouped -> ChebyshevSlider
    variables have general cross-coupling -> ChebyshevTT with method: "cross"

Do you already have values on a Chebyshev grid?
  dense full-grid values -> ChebyshevApproximation.FromValues
  piecewise full-grid values -> ChebyshevSpline.FromValues
  high-dimensional dense values -> ChebyshevTT.FromValues only if the dense
    tensor is intentionally feasible
```

## Class Comparison

| Class | Choose it when | Avoid it when | Derivatives | Next guide |
|-------|----------------|---------------|-------------|------------|
| `ChebyshevApproximation` | The function is smooth and the dense grid `prod(n_i)` is feasible. | Dimension or node counts make the dense tensor too large. | Analytical spectral derivatives. | [Getting Started](getting-started.md), [Error-Driven Construction](error-driven-construction.md) |
| `ChebyshevSpline` | Smoothness breaks at known knots: strikes, barriers, jumps, or regime boundaries. | Trouble locations are unknown or too numerous to split cleanly. | Analytical inside each piece; nonzero derivatives at knots are undefined. | [Piecewise Chebyshev Interpolation](spline.md), [Special Points](special-points.md), [Adaptive Refinement](adaptive-refinement.md) |
| `ChebyshevSlider` | High dimensions can be partitioned into weakly interacting groups. | Cross-group interactions or cross-group Greeks are important. | Spectral inside each slide; cross-group mixed derivatives are zero by construction. | [Sliding Technique](slider.md), [Performance](performance.md) |
| `ChebyshevTT` | High-dimensional variables are coupled and a dense grid is infeasible. | You need exact spectral derivatives or frequent dense tensor materialization. | Finite differences on the TT interpolant. | [Tensor Train Interpolation](tensor-train.md), [Testing & Validation](testing-and-validation.md) |

## Cost Rules

The dense classes build from a tensor grid. With `n_i` nodes in each dimension,
the function-evaluation count is:

$$
\prod_i n_i
$$

This is why a 4D grid with 15 nodes per dimension is usually manageable
(`50,625` values), while a 7D grid with 35 nodes per dimension is not.

Splines multiply that dense-grid cost by the number of pieces:

$$
\prod_i (k_i + 1) \prod_i n_i
$$

where `k_i` is the number of interior knots in dimension `i`.

Slider replaces one full grid with a sum of smaller group grids:

$$
1 + \sum_g \prod_{i \in g} n_i
$$

The leading `1` is the pivot evaluation. The public `TotalBuildEvals` property
reports only the slide-grid sum.

TT-Cross avoids dense-grid materialization and targets approximately
`O(d * n * r^2)` function evaluations for bounded rank `r`, but rank growth,
seeds, and stopping tolerances still affect accuracy and reproducibility.

## Practical Defaults

- Start with `ChebyshevApproximation` for smooth functions when the dense grid is
  affordable. Dimension count is only a proxy; the real limit is `prod(n_i)`.
- Use `ChebyshevSpline` when global error stays high near a known kink, jump, or
  barrier. If the location is not known, first use held-out samples and the
  [error-driven workflow](error-driven-construction.md) to find where the error
  concentrates.
- Use `ChebyshevSlider` when you can explain a grouping of variables. Validate
  it with points far from the pivot, because the reported interpolation error
  does not include cross-group decomposition error.
- Use `ChebyshevTT` when cross-variable coupling matters and the dense grid is
  too large. Prefer `Build(method: "cross")` for high-dimensional work.
- Avoid `ToDense()`, TT-SVD, and `ChebyshevTT.FromValues()` unless the full
  tensor is intentionally small enough to materialize.
- If you need precomputed values, use [Pre-computed Values](from-values.md) and
  confirm that your node ordering matches ChebyshevSharp's Type-I root nodes.

## Validation Checklist

After choosing a class:

1. Compare held-out true function values against interpolated values.
2. Increase node counts, split knots, regroup variables, or raise TT rank and
   check whether the error decreases for the expected reason.
3. For splines, inspect each piece separately; one bad interval can dominate
   the global error.
4. For sliders, test points that move several groups away from the pivot.
5. For TT-Cross, start with a fixed seed, then retry multiple seeds when rank is
   near the cap or error is unstable.
6. Save and reload one model in tests when the interpolant is used outside the
   build process.

See [Error Estimation](error-estimation.md), [Testing & Validation](testing-and-validation.md),
and [Performance](performance.md) before treating a class choice as final.
