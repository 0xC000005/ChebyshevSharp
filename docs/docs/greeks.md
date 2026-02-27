---
title: Computing Greeks
---

# Computing Greeks

ChebyshevSharp computes option Greeks (and any partial derivatives) analytically
using spectral differentiation matrices -- no finite differences needed.

## Derivative Specification

Derivatives are specified as an array of integers, one per dimension. Each integer
is the derivative order with respect to that dimension.

For a 5D function $V(S, K, T, \sigma, r)$:

| Greek | `derivativeOrder` | Mathematical |
|-------|-------------------|--------------|
| Price | `[0, 0, 0, 0, 0]` | $V$ |
| Delta | `[1, 0, 0, 0, 0]` | $\partial V / \partial S$ |
| Gamma | `[2, 0, 0, 0, 0]` | $\partial^2 V / \partial S^2$ |
| Vega  | `[0, 0, 0, 1, 0]` | $\partial V / \partial \sigma$ |
| Rho   | `[0, 0, 0, 0, 1]` | $\partial V / \partial r$ |

Any combination is valid up to `MaxDerivativeOrder`. For example, cross-gamma
$\partial^2 V / \partial S \, \partial \sigma$ would be `[1, 0, 0, 1, 0]`:

```csharp
// Cross-gamma: d²V / dS dsigma
double crossGamma = cheb.VectorizedEval(point, new[] { 1, 0, 0, 1, 0 });
```

## Example: Black-Scholes Greeks

```csharp
using ChebyshevSharp;

double BlackScholesCall(double[] x, object? data)
{
    double S = x[0], K = x[1], T = x[2], sigma = x[3], r = x[4];
    // ... your pricing function here
    return price;
}

var cheb = new ChebyshevApproximation(
    function: BlackScholesCall,
    numDimensions: 5,
    domain: new[]
    {
        new[] { 80.0, 120.0 },   // Spot
        new[] { 90.0, 110.0 },   // Strike
        new[] { 0.25, 1.0 },     // Maturity
        new[] { 0.15, 0.35 },    // Volatility
        new[] { 0.01, 0.08 }     // Rate
    },
    nNodes: new[] { 11, 11, 11, 11, 11 }
);
cheb.Build();

double[] point = { 100.0, 100.0, 1.0, 0.25, 0.05 };

// All Greeks at once (most efficient)
double[] results = cheb.VectorizedEvalMulti(point, new[]
{
    new[] { 0, 0, 0, 0, 0 },  // Price
    new[] { 1, 0, 0, 0, 0 },  // Delta
    new[] { 2, 0, 0, 0, 0 },  // Gamma
    new[] { 0, 0, 0, 1, 0 },  // Vega
    new[] { 0, 0, 0, 0, 1 },  // Rho
});

double price = results[0];
double delta = results[1];
double gamma = results[2];
double vega  = results[3];
double rho   = results[4];
```

`VectorizedEvalMulti` shares intermediate barycentric weight computation across all
requested derivative orders, making it significantly faster than calling
`VectorizedEval` five separate times.

## How It Works

1. **At build time**: Pre-compute the spectral differentiation matrix $D$ from the
   Chebyshev node positions. This matrix maps function values at nodes to derivative
   values at those same nodes.
2. **At query time**: Apply $D$ to the function value tensor before barycentric
   interpolation. For dimension $k$ with derivative order $m_k$, the tensor is
   contracted against $D^{m_k}$ along axis $k$.
3. **For second derivatives**: Apply $D$ twice ($D^2 \mathbf{f}$). This computes
   second-derivative values at the nodes, which are then interpolated to the query
   point.
4. **Interpolate** the resulting derivative values to the query point using the
   standard barycentric formula.

This provides **exact derivatives of the interpolating polynomial**. Because the
differentiation matrix computes derivatives of the degree-$n$ polynomial $p(x)$
exactly (within machine precision), and $p(x)$ converges spectrally to $f(x)$,
the derivative $p'(x)$ also converges spectrally to $f'(x)$
(Trefethen 2013, Ch. 11).

Mathematically, the spectral differentiation matrix $D$ satisfies:

$$
(D \mathbf{v})_i = p'(x_i), \quad \text{where } p \text{ interpolates } \mathbf{v} \text{ at nodes } x_0, \ldots, x_{n-1}
$$

For the barycentric differentiation matrix formulas, see Berrut & Trefethen (2004), Section 9.

## MaxDerivativeOrder

The `maxDerivativeOrder` constructor parameter (default 2) controls how many powers
of $D$ are pre-computed at build time. If you need third-order derivatives, set it to 3:

```csharp
var cheb = new ChebyshevApproximation(
    function: MyFunction,
    numDimensions: 3,
    domain: new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } },
    nNodes: new[] { 15, 15, 15 },
    maxDerivativeOrder: 3
);
cheb.Build();

// Third derivative is now available
double d3f = cheb.VectorizedEval(point, new[] { 3, 0, 0 });
```

Higher derivative orders require more nodes for accurate results. As a rule of thumb,
use at least $n \geq 2k + 5$ nodes per dimension for derivative order $k$.

> **Runtime validation.** Evaluating a derivative order higher than `maxDerivativeOrder`
> throws `InvalidOperationException`. If you need third-order Greeks, set
> `maxDerivativeOrder: 3` in the constructor.

## Accuracy

With 11 nodes per dimension on a 5D Black-Scholes test:

| Greek | Max Relative Error |
|-------|--------------------|
| Delta | < 0.01% |
| Gamma | < 0.01% |
| Vega  | ~1.98% |
| Rho   | < 0.01% |

Vega has slightly higher error because the volatility sensitivity involves a product
of multiple terms in the Black-Scholes formula, but remains well within practical
tolerance for most applications. Increasing the node count to 13--15 per dimension
reduces Vega error below 0.1%.

> **Tensor Train derivatives.** `ChebyshevTT` (not yet implemented -- Phase 4)
> will use **finite differences** instead of analytical derivatives, because the
> spectral differentiation matrix requires the full tensor (which TT avoids
> storing).

> **Slider cross-group derivatives.** In `ChebyshevSlider`, the additive decomposition
> means that derivatives with respect to variables in **different groups** are exactly
> zero. For example, if dimensions 0--1 form one group and dimensions 2--4 form another,
> then $\partial^2 V / \partial x_0 \, \partial x_2 = 0$ identically. Only derivatives
> within the same group are non-trivial.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation."
  *SIAM Review* 46(3):501--517.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM. Chapter 11.
