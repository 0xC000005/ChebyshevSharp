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

**Accuracy:** Since Fejer-1 weights integrate degree n-1 polynomials exactly, and the Chebyshev interpolant is a polynomial of degree n-1, the quadrature is exact for the interpolant. The only error comes from the interpolation itself. For well-resolved functions, integration accuracy matches the interpolation accuracy.

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

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
2. Good, I. J. (1961). "The Colleague Matrix, a Chebyshev Analogue of the Companion Matrix." *The Quarterly Journal of Mathematics* 12(1):61-68.
