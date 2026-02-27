---
title: Chebyshev Algebra
---

# Chebyshev Algebra

## Motivation -- Portfolio Combination

In counterparty credit risk (CCR), thousands of trades sharing common risk factors
must be priced at millions of Monte Carlo scenarios. Building a Chebyshev proxy per
trade reduces pricing cost, but evaluating 1,000 separate proxies at each scenario is
still $O(\text{numTrades})$.

Algebraic combination lets you **pre-combine** trade proxies into a single
netting-set-level proxy:

```csharp
var portfolio = w1 * trade1 + w2 * trade2 + ... + wN * tradeN;
```

One evaluation of `portfolio` gives the netting set price -- $O(1)$ regardless of
the number of trades.

> **When to use algebra.**
> Use algebraic combination when multiple Chebyshev interpolants share the **same
> grid** (same domain, node counts, derivative order) and you want to combine them
> into a single interpolant for faster evaluation.
> If your interpolants live on **different** sets of dimensions (e.g., Trade A
> depends on (spot, rate) while Trade B depends on (spot, vol)), use
> [extrusion and slicing](extrude-slice.md) to bring them onto a common grid first.

## Mathematical Basis

The [barycentric interpolation formula](concepts.md#barycentric-interpolation-formula) evaluates a Chebyshev Tensor (CT) at any point
$\mathbf{x}$:

$$
p_n(\mathbf{x}) = \sum_{i_1, \ldots, i_d} v_{i_1, \ldots, i_d} \prod_{k=1}^{d} \ell^{(k)}_{i_k}(x_k)
$$

where $\ell^{(k)}_{i_k}$ are the barycentric basis functions (Berrut &
Trefethen 2004). This is **linear in the values** $v_{i_1, \ldots, i_d}$.

**Theorem (Linearity of CT operations).** Let $T_f$ and $T_g$ be two CTs on the
same grid. Then:

1. **Addition**: $T_f + T_g$ (element-wise on grid values) is the CT for $f + g$
2. **Scalar multiplication**: $c \cdot T_f$ is the CT for $c \cdot f$
3. **Subtraction**: $T_f - T_g$ is the CT for $f - g$

*Proof.* Direct from linearity of the barycentric formula.

**Corollary (Derivatives).** Since the spectral differentiation matrix
$\mathcal{D}_k$ depends only on grid points (Berrut & Trefethen 2004, Section 9):

$$
\mathcal{D}_k (\mathbf{v}_f + \mathbf{v}_g) = \mathcal{D}_k \mathbf{v}_f + \mathcal{D}_k \mathbf{v}_g
$$

Derivatives of a combined CT equal the combined derivatives.

**Error bound.** By the triangle inequality:

$$
\|(f + g) - (p_f + p_g)\|_\infty \leq \epsilon_f + \epsilon_g
$$

For scalar multiplication: $\|cf - cp_f\|_\infty = |c| \cdot \epsilon_f$.

The linearity of Chebyshev Tensor operations is described in Section 3.9 of Ruiz
& Zeron (2021), *Machine Learning for Risk Calculations*, Wiley Finance.

## Quick Start

```csharp
using ChebyshevSharp;

// Two functions on the same grid
double F(double[] x, object? data) => Math.Sin(x[0]) + Math.Sin(x[1]);
double G(double[] x, object? data) => Math.Cos(x[0]) * Math.Cos(x[1]);

var a = new ChebyshevApproximation(F, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 11, 11 });

var b = new ChebyshevApproximation(G, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 11, 11 });

a.Build(verbose: false);
b.Build(verbose: false);

// Combine into a portfolio proxy
var portfolio = 0.6 * a + 0.4 * b;

// Evaluate price and Greeks at any point
double[] point = { 0.5, 0.3 };
double price = portfolio.VectorizedEval(point, new[] { 0, 0 });
double delta = portfolio.VectorizedEval(point, new[] { 1, 0 });
double gamma = portfolio.VectorizedEval(point, new[] { 2, 0 });
```

The combined `portfolio` is a regular `ChebyshevApproximation` -- all existing
evaluation methods (`Eval`, `VectorizedEval`, `VectorizedEvalMulti`,
`VectorizedEvalBatch`) work unchanged.

## Supported Operations

| Operator | C# Example | Result |
|----------|------------|--------|
| `+` | `a + b` | Element-wise add tensor values |
| `-` | `a - b` | Element-wise subtract |
| `*` scalar | `3.0 * cheb` or `cheb * 3.0` | Scale all tensor values |
| `/` scalar | `cheb / 2.0` | Divide all tensor values |
| unary `-` | `-cheb` | Negate all tensor values |

Compound assignment is supported via the standard C# operators:

```csharp
a += b;    // In-place add (allocates new tensor, reassigns)
a -= b;    // In-place subtract
a *= 3.0;  // In-place scale
a /= 2.0;  // In-place divide
```

## Compatibility Requirements

Both operands must share:

- **Same type** -- both `ChebyshevApproximation`, both `ChebyshevSpline`, etc.
- **Same `NumDimensions`** -- number of interpolation dimensions
- **Same `Domain`** -- identical domain bounds in every dimension
- **Same `NNodes`** -- same node counts in every dimension
- **Same `MaxDerivativeOrder`** -- same spectral differentiation depth
- **Both must be built** -- `Build()` must have been called on each operand

Additional requirements for specific classes:

- **`ChebyshevSpline`**: same `Knots` in every dimension
- **`ChebyshevSlider`**: same `Partition` and same `PivotPoint`

An exception is thrown if any of these conditions are not met:

- `InvalidOperationException` -- type mismatch (e.g., adding a `ChebyshevApproximation` to a `ChebyshevSpline`) or operand not built
- `ArgumentException` -- dimension, domain, node count, derivative order, knot, or partition mismatch

```csharp
var a = new ChebyshevApproximation(F, 2,
    new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } },
    new[] { 10, 10 });

var b = new ChebyshevApproximation(G, 2,
    new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 2.0 } },  // different domain
    new[] { 10, 10 });

a.Build(verbose: false);
b.Build(verbose: false);

var c = a + b;  // throws ArgumentException: domain mismatch
```

## Derivatives

Derivatives propagate automatically through algebraic operations. The combined
interpolant inherits the spectral differentiation matrices from its operands, so
no re-computation is needed:

```csharp
// Build two interpolants for call and put
call.Build(verbose: false);
put.Build(verbose: false);

// Combine
var portfolio = 0.6 * call + 0.4 * put;

// Delta of the portfolio = 0.6 * delta_call + 0.4 * delta_put
double delta = portfolio.VectorizedEval(point, new[] { 1, 0, 0 });

// Gamma works too
double gamma = portfolio.VectorizedEval(point, new[] { 2, 0, 0 });
```

This follows directly from the linearity corollary: the spectral differentiation
matrices $\mathcal{D}_k$ depend only on grid positions, not on function values.
See [Computing Greeks](greeks.md) for more on analytical derivatives.

## Error Estimation

`ErrorEstimate()` recomputes from the combined Chebyshev coefficients (DCT of the
combined tensor values). In practice this may give a **tighter bound** than the
triangle inequality $\epsilon_f + \epsilon_g$, because cancellation between the
high-order coefficients of $f$ and $g$ can reduce the estimated tail:

```csharp
var portfolio = 0.6 * call + 0.4 * put;
double err = portfolio.ErrorEstimate();
Console.WriteLine($"Portfolio error estimate: {err:E2}");
```

## Serialization

Combined interpolants support `Save()` and `Load()` just like any other built
interpolant. The underlying function reference is lost (`Function = null`), but all
tensor values, grid data, and differentiation matrices are preserved:

```csharp
var portfolio = 0.6 * call + 0.4 * put;
portfolio.Save("portfolio.json");

var loaded = ChebyshevApproximation.Load("portfolio.json");
double value = loaded.VectorizedEval(point, new[] { 0, 0 });  // works identically
```

See [Serialization & Construction](serialization.md) for the full save/load API.

## Spline and Slider Examples

### ChebyshevSpline Addition

Two splines with the **same knots** can be combined:

```csharp
var splineA = new ChebyshevSpline(
    F, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
    new[] { 15, 15 },
    knots: new[] { new[] { 100.0 }, Array.Empty<double>() }
);

var splineB = new ChebyshevSpline(
    G, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
    new[] { 15, 15 },
    knots: new[] { new[] { 100.0 }, Array.Empty<double>() }
);

splineA.Build(verbose: false);
splineB.Build(verbose: false);

var combined = splineA + splineB;
double price = combined.Eval(new[] { 110.0, 0.5 }, new[] { 0, 0 });
```

Each piece is combined independently -- the combined spline has the same knot
structure as its operands.

### ChebyshevSlider Addition

Two sliders with the **same partition and pivot point** can be combined:

```csharp
var sliderA = new ChebyshevSlider(
    F, 5, domain, new[] { 11, 11, 11, 11, 11 },
    partition: new[] { new[] { 0, 1 }, new[] { 2, 3, 4 } },
    pivotPoint: new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 }
);

var sliderB = new ChebyshevSlider(
    G, 5, domain, new[] { 11, 11, 11, 11, 11 },
    partition: new[] { new[] { 0, 1 }, new[] { 2, 3, 4 } },
    pivotPoint: new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 }
);

sliderA.Build(verbose: false);
sliderB.Build(verbose: false);

var combined = sliderA + sliderB;
double val = combined.Eval(new[] { 0.5, 0.5, 0.5, 0.5, 0.5 }, new[] { 0, 0, 0, 0, 0 });
```

Each slide is combined independently, preserving the additive decomposition structure.

## Why Pointwise Products are NOT Supported

The product $f \cdot g$ is **not** $\mathbf{v}_f \odot \mathbf{v}_g$ (element-wise product of grid
values). The product of two polynomials of degree $n$ has degree $2n$, which
cannot be represented on the same $n$-point grid.

Only **linear combinations** (addition, subtraction, scalar multiplication) are exact
on the same grid. Pointwise multiplication of two Chebyshev interpolants requires a
grid refinement step and is not supported.

> **Workaround for products.**
> If you need to approximate $f \cdot g$, build a single Chebyshev interpolant
> for the product function directly: define `h(x, data) => f(x) * g(x)` and call
> `new ChebyshevApproximation(h, ...).Build()`.

## Limitations

- **No `ChebyshevTT` operators** -- TT addition requires rank control (rank of
  $T_f + T_g$ is $r_f + r_g$) and is not currently implemented.
- **No cross-type operations** -- you cannot add a `ChebyshevApproximation` to a
  `ChebyshevSpline` or a `ChebyshevSlider`.
- **Operands must share exact grid parameters** -- domain, node counts, derivative
  order, and (where applicable) knots or partition must be identical.
- **Result has `Function = null`** -- the combined interpolant cannot call `Build()`
  again, since it has no underlying function reference.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation."
  *SIAM Review* 46(3):501--517.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.*
  Wiley Finance. Section 3.9.
