# Introduction

ChebyshevSharp provides multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET applications. It replaces expensive function evaluations with fast polynomial lookups while preserving derivatives to arbitrary order.

## Why Chebyshev Interpolation?

Polynomial interpolation with equally-spaced points suffers from **Runge's phenomenon** — oscillations near interval endpoints that grow exponentially with degree. Chebyshev nodes solve this by clustering near boundaries, achieving **spectral convergence**: for smooth functions, the interpolation error decreases exponentially with the number of nodes.

Specifically, if a function f is analytic inside the Bernstein ellipse with parameter rho > 1, then the Chebyshev interpolant of degree n satisfies:

$$
\| f - p_n \|_\infty \leq C \, \rho^{-n}
$$

where $C$ depends on $f$ but not on $n$. This exponential convergence means that 10-20 nodes per dimension typically suffice for 10-12 digit accuracy on smooth functions [1, Ch. 8].

## How It Works

ChebyshevSharp evaluates the Chebyshev interpolant using the **barycentric interpolation formula** [2]:

$$
p(x) = \frac{\displaystyle\sum_j \frac{w_j}{x - x_j} f_j}{\displaystyle\sum_j \frac{w_j}{x - x_j}}
$$

where $x_j$ are the Chebyshev nodes, $f_j$ are the function values at those nodes, and $w_j$ are the barycentric weights. This formula is numerically stable and evaluates in $O(n)$ time.

For multi-dimensional problems, the function values are stored as an N-dimensional tensor and contracted one axis at a time — each contraction is a matrix-vector multiply routed through BLAS. Derivatives are computed analytically via spectral differentiation matrices [2], not finite differences.

## Classes

| Class | Purpose | Status |
|-------|---------|--------|
| `ChebyshevApproximation` | Core multi-dimensional Chebyshev interpolation with analytical derivatives | Available |
| `ChebyshevSpline` | Piecewise Chebyshev interpolation with knots at singularities | Available |
| `ChebyshevSlider` | High-dimensional approximation via the Sliding Technique | Available |
| `ChebyshevTT` | Tensor Train Chebyshev interpolation for 5+ dimensions | Planned |

## Typical Use Case: Option Pricing

A common application is replacing slow pricing models with fast Chebyshev interpolants. For example, a 3D Black-Scholes pricer (spot, volatility, maturity) with 15 x 12 x 10 = 1,800 nodes:

```csharp
double BsPrice(double[] x, object? data)
{
    // Your Black-Scholes pricing model
    double S = x[0], sigma = x[1], T = x[2];
    // ... compute call price ...
    return price;
}

var cheb = new ChebyshevApproximation(
    function: BsPrice,
    numDimensions: 3,
    domain: new[] {
        new[] { 80.0, 120.0 },   // spot
        new[] { 0.1, 0.5 },      // volatility
        new[] { 0.25, 2.0 }      // maturity
    },
    nNodes: new[] { 15, 12, 10 }
);
cheb.Build();

// Price and all Greeks in one call
double[] results = cheb.VectorizedEvalMulti(
    new[] { 100.0, 0.2, 1.0 },
    new[] {
        new[] { 0, 0, 0 },  // price
        new[] { 1, 0, 0 },  // delta
        new[] { 2, 0, 0 },  // gamma
        new[] { 0, 1, 0 },  // vega
    }
);
```

After a one-time build cost of 1,800 function evaluations, each subsequent evaluation takes ~500 ns with all Greeks — orders of magnitude faster than the original model.

For functions with discontinuities or singularities (e.g., digital options, barrier payoffs), use `ChebyshevSpline` to place knots at the trouble points and achieve spectral convergence on each smooth piece. See [Piecewise Chebyshev Interpolation](spline.md) for details.

For high-dimensional problems where a full tensor grid is infeasible, `ChebyshevSlider` partitions the dimensions into small groups and builds a separate interpolant per group around a pivot point, reducing the cost from exponential to additive. See [Sliding Technique](slider.md) for details.

## Relationship to PyChebyshev

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev). The Python reference implementation is included as a git submodule at `ref/PyChebyshev/` for cross-validation. Both libraries produce numerically identical results within floating-point tolerance. On low-dimensional problems (1-3D), the C# implementation is 17-42x faster than Python due to eliminated interpreter overhead; see [Performance](performance.md) for benchmarks.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
2. Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
3. Ruiz, I. & Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View.* Wiley Finance.
4. Good, I. J. (1961). "The Colleague Matrix, a Chebyshev Analogue of the Companion Matrix." *The Quarterly Journal of Mathematics* 12(1):61-68.
