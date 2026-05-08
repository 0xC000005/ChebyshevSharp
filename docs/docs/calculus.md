---
title: Calculus
---

# Calculus

ChebyshevSharp supports numerical integration, root-finding, and optimization directly on the interpolant, without re-evaluating the original function. All operations exploit the structure of the Chebyshev representation for high accuracy.

## Integration

Integrate the interpolant over one or more dimensions using **Fejer-1 quadrature** — a Chebyshev-node quadrature rule that integrates polynomials of degree n-1 exactly using n nodes [1, Ch. 19].

```csharp
// Integrate over all dimensions (returns a scalar)
double integral = (double)cheb.Integrate();

// Integrate over specific dimensions (returns a lower-dimensional interpolant)
var marginal = (ChebyshevApproximation)cheb.Integrate(dims: new[] { 2 });

// Integrate with sub-interval bounds
double partial = (double)cheb.Integrate(
    dims: new[] { 0, 1 },
    bounds: new[] { (90.0, 110.0), (0.15, 0.35) }
);
```

When integrating over a subset of dimensions, the result is a `ChebyshevApproximation` with reduced dimensionality that can be evaluated, differentiated, or further integrated. When integrating over all dimensions, the result is a `double`.

The return type is `object` — cast to `double` or `ChebyshevApproximation` depending on whether all dimensions are integrated.

**How it works:** Integration is performed as a tensor contraction — the tensor of function values is contracted along the integrated dimensions with the Fejer-1 quadrature weight vector. The weights are computed from the Chebyshev coefficients via DCT-III. For sub-interval integration, modified weights are computed that account for the restricted bounds.

On the reference interval, the implementation uses Chebyshev moments

$$
m_k = \int_{-1}^{1} T_k(t)\,dt =
\begin{cases}
2/(1-k^2), & k \text{ even}, \\
0, & k \text{ odd}.
\end{cases}
$$

Fejer-1 weights are then the DCT-III of these moments, reversed into ChebyshevSharp's ascending node order. For sub-intervals $[t_{\mathrm{lo}}, t_{\mathrm{hi}}]$, the moments become

$$
m_0 = t_{\mathrm{hi}} - t_{\mathrm{lo}}, \qquad
m_1 = \frac{t_{\mathrm{hi}}^2 - t_{\mathrm{lo}}^2}{2},
$$

and for $k \ge 2$:

$$
m_k = \frac{1}{2}\left[
\frac{T_{k+1}(t_{\mathrm{hi}})-T_{k+1}(t_{\mathrm{lo}})}{k+1}
-
\frac{T_{k-1}(t_{\mathrm{hi}})-T_{k-1}(t_{\mathrm{lo}})}{k-1}
\right].
$$

**Accuracy:** Since Fejer-1 weights integrate degree n-1 polynomials exactly, and the Chebyshev interpolant is a polynomial of degree n-1, the quadrature is exact for the interpolant. The only error comes from the interpolation itself. For well-resolved functions, integration accuracy matches the interpolation accuracy.

For `ChebyshevSpline`, integration sums across pieces with automatic bound clipping to each piece's sub-domain. See [Piecewise Chebyshev Interpolation](spline.md) for details.

## Root-Finding

Find all roots (zero crossings) of the interpolant along a single dimension using the **colleague matrix** eigenvalue method [2]:

```csharp
// For a 1D interpolant
double[] roots = cheb1d.Roots();

// For a multi-dimensional interpolant, fix other dimensions
double[] roots = cheb3d.Roots(
    dim: 0,
    fixedDims: new Dictionary<int, double> { { 1, 0.25 }, { 2, 1.0 } }
);
```

The `dim` parameter specifies which dimension to search along. All other dimensions must be fixed to specific values via `fixedDims`. For 1D interpolants, both parameters are optional.

Roots are returned as an array of values within the domain bounds, sorted in ascending order.

**How it works:** The colleague matrix is the Chebyshev analogue of the companion matrix for monomial polynomials [2]. Its eigenvalues are the roots of the Chebyshev expansion. The method:

1. Extracts Chebyshev coefficients via DCT-II
2. Constructs the colleague matrix (a tridiagonal-plus-rank-1 matrix)
3. Computes all eigenvalues
4. Filters to real eigenvalues within the domain bounds

This finds all roots simultaneously (no initial guess needed) and is numerically stable for polynomials of moderate degree (up to ~100 nodes).

For `ChebyshevSpline`, roots are found per-piece and merged with deduplication near knot boundaries. A knot where adjacent pieces have opposite signs is also reported as a zero crossing, matching the piecewise rootfinding convention used by Chebfun. See [Piecewise Chebyshev Interpolation](spline.md) for details.

## Minimization and Maximization

Find the minimum or maximum of the interpolant along a single dimension:

```csharp
// Minimize a 1D interpolant
var (minValue, minLocation) = cheb1d.Minimize();

// Maximize along dim 0, fixing other dimensions
var (maxValue, maxLocation) = cheb3d.Maximize(
    dim: 0,
    fixedDims: new Dictionary<int, double> { { 1, 0.25 }, { 2, 1.0 } }
);
```

Both methods return a tuple of `(double value, double location)`.

**How it works:** The global optimum of a polynomial on a closed interval must occur at either a critical point (where the derivative is zero) or an endpoint. The method:

1. Computes the derivative interpolant via the differentiation matrix
2. Finds all roots of the derivative using the colleague matrix (these are the critical points)
3. Evaluates the interpolant at all critical points and both domain endpoints
4. Returns the best value and its location

This is guaranteed to find the global optimum of the interpolant (not a local one), since all critical points are found via the eigenvalue method.

For `ChebyshevSpline`, optimization searches each piece independently and returns the global optimum across all pieces. See [Piecewise Chebyshev Interpolation](spline.md) for details.

## Class Support

| Operation | `ChebyshevApproximation` | `ChebyshevSpline` | `ChebyshevSlider` | `ChebyshevTT` |
|-----------|:---:|:---:|:---:|:---:|
| Integrate | Yes | Yes | Yes | Yes |
| Roots | Yes | Yes | Yes | Yes |
| Minimize / Maximize | Yes | Yes | Yes | Yes |

For `ChebyshevSlider` and `ChebyshevTT`, roots and optimization reduce the requested dimension to a 1-D interpolant by slicing all other dimensions, then delegate to `ChebyshevApproximation`. For sliders this evaluates a 1-D proxy of the sliding approximation. For TT this materializes only the reduced 1-D slice, not the original dense grid.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
2. Good, I. J. (1961). "The Colleague Matrix, a Chebyshev Analogue of the Companion Matrix." *The Quarterly Journal of Mathematics* 12(1):61-68.
3. Waldvogel, J. (2006). "Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules." *BIT Numerical Mathematics* 46(1):195-202.
4. Trefethen, L. N. (2009, revised 2019). "Rootfinding and Minima and Maxima." *Chebfun Guide*. https://www.chebfun.org/docs/guide/guide03.html

## Slider Integration (v0.9.0)

`ChebyshevSlider.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` integrates over one or more dimensions using the closed-form sliding-decomposition. Returns a scalar (boxed in `object`) when every dim is integrated; otherwise returns a new `ChebyshevSlider` over surviving dims.

```csharp
var slider = new ChebyshevSlider(
    (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 10, 10 },
    partition: new[] { new[] { 0 }, new[] { 1 } },
    pivotPoint: new[] { 0.0, 0.0 });
slider.Build();

// Full integration: ∫∫ (sin(x) + cos(y)) dx dy = 4 sin(1)
double result = (double)slider.Integrate();

// Partial integration: ∫_{-1}^{1} (sin(x) + cos(y)) dy → slider over dim 0 only
var partial = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
```

The integration is exact for the spectrally-resolved part of each slide and
integrates the sliding approximation, not any cross-group coupling missed by the
decomposition. Per-slide classification: a slide whose group is fully covered by
`dims` collapses into the new pivot value; a slide whose group is partially
covered is reduced via `ChebyshevApproximation.Integrate`; a slide whose group
is disjoint from `dims` passes through with a partition-of-unity shift.

## TT Integration (v0.9.0)

`ChebyshevTT.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` integrates over one or more dimensions using Fejér-1 quadrature contracted into each integrated core's node axis. Returns a scalar (boxed in `object`) when every dim is integrated; otherwise returns a new `ChebyshevTT` over surviving dims. Works for all three build methods (`cross`, `svd`, `als`).

```csharp
var tt = new ChebyshevTT(
    x => Math.Sin(x[0]) * Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 12, 12 });
tt.Build();

// Full integration
double total = (double)tt.Integrate();

// Partial: integrate over dim 0, returns a 1D TT in y
var marginal = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });

// Sub-domain bounds
double partial = (double)tt.Integrate(
    dims: new[] { 0, 1 },
    bounds: new[] { (-0.5, 0.5), (0.0, 1.0) });
```

`Roots`, `Minimize`, and `Maximize` are available on `ChebyshevSlider` and `ChebyshevTT` by slicing to a 1-D proxy and applying the same colleague-matrix calculus path used by `ChebyshevApproximation`.
