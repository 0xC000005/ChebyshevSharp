---
title: Extrusion & Slicing
---

# Extrusion & Slicing

## Motivation -- Portfolio Combination

In practice, trades depend on different subsets of risk factors. Trade A
might depend on (spot, rate) while Trade B depends on (spot, vol). The
algebra operators require operands on the **same grid**, so these
two proxies cannot be added directly.

**Extrusion** solves this by adding new dimensions where the function is
constant. After extruding both trades to a common 3D grid (spot, rate, vol),
they can be combined with the standard `+`, `-`, `*`, `/` operators:

```csharp
var portfolio = tradeA3d + tradeB3d;
```

**Slicing** is the inverse: it fixes a dimension at a specific value, reducing
dimensionality via barycentric interpolation. Together, extrusion and slicing
form the bridge between Chebyshev proxies on heterogeneous risk-factor sets.

> **When to use extrude/slice:**
> Use extrusion when you need to combine Chebyshev interpolants that
> live on **different** sets of dimensions. Use slicing to project a
> high-dimensional interpolant onto a lower-dimensional subspace (e.g.,
> fixing a parameter at its current market value).

## Mathematical Basis

### Partition of Unity

The barycentric basis functions satisfy a fundamental identity:

$$
\sum_{j=0}^{n} \ell_j(x) = 1
$$

for all $x$ in the domain. This is because any polynomial interpolation scheme
reproduces constant functions exactly -- the constant $1$ is interpolated by
$\sum_j 1 \cdot \ell_j(x) = 1$.

> **Reference:** Berrut & Trefethen (2004), "Barycentric Lagrange Interpolation",
> *SIAM Review* 46(3):501--517, Section 2.

### Extrusion Proof

Given a $d$-dimensional CT with values $v_{i_1,\ldots,i_d}$, the extruded
$(d+1)$-dimensional CT inserts a new axis at position $k$ with $M$ Chebyshev
nodes, replicating values:

$$
\hat{v}_{i_1,\ldots,i_{k-1},\,j,\,i_k,\ldots,i_d} = v_{i_1,\ldots,i_d}
\quad \forall\; j = 0,\ldots,M-1
$$

Evaluating at any point $(x_1,\ldots,x_{k-1},x^*,x_k,\ldots,x_d)$:

$$
p(\mathbf{x}) = \sum_{\text{all indices}} v_{i_1,\ldots,i_d}
\cdot \ell_j^{(k)}(x^*) \cdot \prod_{m \neq k} \ell_{i_m}^{(m)}(x_m)
$$

Since the values do not depend on $j$, the $j$-sum factors out:

$$
= \underbrace{\left(\sum_{j=0}^{M-1} \ell_j^{(k)}(x^*)\right)}_{= 1
\;\text{(partition of unity)}} \cdot \sum_{i_1,\ldots,i_d} v_{i_1,\ldots,i_d}
\prod_{m \neq k} \ell_{i_m}^{(m)}(x_m) = p_{\text{orig}}(\mathbf{x}_{\text{orig}})
$$

**Result**: The extruded CT evaluates to the same value as the original,
regardless of the new coordinate. Extrusion is exact.

### Slicing Proof

Given a $d$-dimensional CT, fixing dimension $k$ at value $x^*$:

$$
p(x_1,\ldots,x_{k-1},x^*,x_{k+1},\ldots,x_d) = \sum_{i_1,\ldots,i_d}
v_{i_1,\ldots,i_d} \prod_{m=1}^{d} \ell_{i_m}^{(m)}(x_m) \bigg|_{x_k = x^*}
$$

Factoring out the $k$-th dimension:

$$
= \sum_{i_1,\ldots,i_{k-1},i_{k+1},\ldots,i_d}
\underbrace{\left(\sum_{i_k} v_{i_1,\ldots,i_d}
\cdot \ell_{i_k}^{(k)}(x^*)\right)}_{\hat{v}_{i_1,\ldots,i_{k-1},i_{k+1},\ldots,i_d}}
\prod_{m \neq k} \ell_{i_m}^{(m)}(x_m)
$$

The contracted values $\hat{v}$ define a valid $(d-1)$-dimensional CT.

**Fast path**: When $x^*$ coincides with a Chebyshev node $x_m^{(k)}$
(within tolerance $10^{-14}$), the basis function simplifies to
$\ell_{i_k}^{(k)}(x_m) = \delta_{i_k,m}$, so
$\hat{v} = v_{\ldots,m,\ldots}$ -- a simple array extraction (no arithmetic
needed).

### Extrude-then-Slice = Identity

If we extrude along dimension $k$ and then slice at any value $x^*$
along $k$:

$$
\text{Slice}(\text{Extrude}(T, k), k, x^*) = T
$$

*Proof.* Extrusion replicates values along $k$, then slicing contracts via
$\sum_j v \cdot \ell_j(x^*) = v \cdot 1 = v$.

### Error Bounds

- **Extrusion**: No approximation error introduced (exact operation), following
  directly from the partition of unity (Berrut & Trefethen 2004).
- **Slicing**: The sliced CT evaluates the polynomial interpolant at $x_k = x^*$.
  No additional error beyond the original approximation error
  (Trefethen 2013, Ch. 8):
  if $\|f - p\|_\infty \leq \epsilon$, then
  $\|f(\cdot, x^*) - p(\cdot, x^*)\|_\infty \leq \epsilon$.

> **Book reference:** Extrusion and slicing of Chebyshev Tensors is described
> in Section 24.2.1, Listing 24.15 (slice) and Listing 24.16 (extrude) of
> Ruiz & Zeron (2021), *Machine Learning for Risk Calculations*, Wiley Finance.

## Quick Start

```csharp
using ChebyshevSharp;

double F(double[] x, object? _) => Math.Sin(x[0]) + x[1];
double G(double[] x, object? _) => Math.Cos(x[0]) * x[1];

// Trade A depends on (spot, rate)
var tradeA = new ChebyshevApproximation(F, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.01, 0.08 } },
    new[] { 11, 11 });
tradeA.Build();

// Trade B depends on (spot, vol)
var tradeB = new ChebyshevApproximation(G, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.15, 0.35 } },
    new[] { 11, 11 });
tradeB.Build();

// Extrude both to 3D: (spot, rate, vol)
var tradeA3d = tradeA.Extrude((2, new[] { 0.15, 0.35 }, 11));  // add vol dim
var tradeB3d = tradeB.Extrude((1, new[] { 0.01, 0.08 }, 11));  // add rate dim

// Combine into portfolio
var portfolio = tradeA3d + tradeB3d;
double price = portfolio.VectorizedEval(
    new[] { 100.0, 0.05, 0.25 }, new[] { 0, 0, 0 });
```

## Supported Operations

| Class | `Extrude()` | `Slice()` |
|-------|-------------|-----------|
| `ChebyshevApproximation` | Yes | Yes |
| `ChebyshevSpline` | Yes | Yes |
| `ChebyshevSlider` | Yes | Yes |
| `ChebyshevTT` | No | No |

## Extrude API

```csharp
var result = cheb.Extrude(params);
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | `(int dimIndex, double[] bounds, int nNodes)` or array of tuples | Position, domain, and node count for each new dimension |

- `dimIndex` -- position in the **output** space (0 = prepend, `d` = append)
- `bounds` -- domain bounds `new[] { lo, hi }` for the new dimension
- `nNodes` -- number of Chebyshev nodes (must match other CTs for later algebra)

**Returns**: A new interpolant of the same type, already built, with
`Function = null`.

**Errors**:

- `InvalidOperationException` if the interpolant has not been built
- `ArgumentException` if `dimIndex` is out of range, duplicated, `lo >= hi`, or `nNodes < 2`

### Multi-Extrude: 1D to 3D

```csharp
double F(double[] x, object? _) => Math.Sin(x[0]);

var cheb1d = new ChebyshevApproximation(F, 1,
    new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
cheb1d.Build();

// Add two dimensions at once
var cheb3d = cheb1d.Extrude(
    (1, new[] { 0.0, 5.0 }, 11),
    (2, new[] { -2.0, 2.0 }, 9)
);
```

## Slice API

```csharp
var result = cheb.Slice(params);
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | `(int dimIndex, double value)` or array of tuples | Dimension index and value at which to fix |

- `dimIndex` -- dimension to fix (0-indexed in the current object)
- `value` -- point at which to fix (must be within the domain)

**Returns**: A new interpolant of the same type, already built, with
`Function = null`.

**Errors**:

- `InvalidOperationException` if the interpolant has not been built
- `ArgumentException` if `value` is outside the domain, `dimIndex` is out of range,
  duplicated, or if slicing all dimensions

### Slice 3D to 1D

```csharp
double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]) + x[2];

var cheb3d = new ChebyshevApproximation(F, 3,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 11, 11, 11 });
cheb3d.Build();

// Fix dim 1 and dim 2
var cheb1d = cheb3d.Slice((1, 0.5), (2, -0.3));
```

> **Fast path at exact nodes:**
> When the slice value coincides with a Chebyshev node (within $10^{-14}$),
> the contraction reduces to a simple array extraction -- no floating-point
> arithmetic is required.

## Compatibility with Algebra

Extrusion is the key enabler for the algebra operators. After
extruding two CTs to a common grid, all standard operators work:

```csharp
// Different risk factors
var ctA = new ChebyshevApproximation(F, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.01, 0.08 } },
    new[] { 11, 11 });
var ctB = new ChebyshevApproximation(G, 2,
    new[] { new[] { 80.0, 120.0 }, new[] { 0.15, 0.35 } },
    new[] { 11, 11 });
ctA.Build();
ctB.Build();

// Extrude to common 3D: (spot, rate, vol)
var ctA3d = ctA.Extrude((2, new[] { 0.15, 0.35 }, 11));
var ctB3d = ctB.Extrude((1, new[] { 0.01, 0.08 }, 11));

// Now all algebra operators work
var portfolio = 0.6 * ctA3d + 0.4 * ctB3d;
var hedged    = ctA3d - ctB3d;
var scaled    = 2.0 * ctA3d;
```

The compatibility requirements from [Chebyshev Algebra](algebra.md) apply
to the extruded results: same domain, node counts, derivative order, and
number of dimensions.

## Derivatives

**Extrusion**: Derivatives in the original dimensions are preserved.
Derivatives with respect to the new dimension are zero (the function is
constant along the new axis). This follows from
$\mathcal{D}_k \cdot [c, c, \ldots, c]^T = \mathbf{0}$.

**Slicing**: Derivatives in the remaining dimensions are preserved. The
sliced CT has valid spectral differentiation matrices for all surviving
dimensions.

```csharp
// Extrude: derivative w.r.t. new dim is zero
var ct3d = ct2d.Extrude((2, new[] { 0.0, 1.0 }, 11));
double dNew = ct3d.VectorizedEval(
    new[] { 0.5, 0.3, 0.7 }, new[] { 0, 0, 1 });
// dNew ~ 0.0 (within ~1e-12)

// Slice: derivative in remaining dim preserved
var ct1d = ct2d.Slice((1, 0.5));
double dDx = ct1d.VectorizedEval(new[] { 0.3 }, new[] { 1 });
```

## Serialization

Extruded and sliced interpolants support `Save()` and `Load()` just like
any other built interpolant:

```csharp
var ct3d = ct2d.Extrude((2, new[] { 0.0, 1.0 }, 11));
ct3d.Save("extruded.json");

var loaded = ChebyshevApproximation.Load("extruded.json");
loaded.VectorizedEval(
    new[] { 0.5, 0.3, 0.7 }, new[] { 0, 0, 0 });  // works identically
```

See [Serialization & Construction](serialization.md) for details.

## Class-Specific Notes

### ChebyshevSpline

When extruding a spline, each piece is extruded independently. The new
dimension gets `Knots = Array.Empty<double>()` (no interior knots) and a single interval.

When slicing a spline, only the pieces whose interval along the sliced
dimension contains the slice value survive. Each surviving piece is then
sliced via its underlying `ChebyshevApproximation.Slice()`.

```csharp
// Extrude a 1D spline to 2D
var spline2d = spline1d.Extrude((1, new[] { 0.0, 5.0 }, 11));

// Slice a 2D spline to 1D
var spline1dSliced = spline2d.Slice((1, 2.5));
```

See [Piecewise Chebyshev Interpolation](spline.md) for details.

### ChebyshevSlider

When extruding a slider, the new dimension becomes its own single-dim
slide group with `TensorValues` filled with `PivotValue`, so the slide
contributes zero: $s_{\text{new}}(x) - pv = 0$ for all $x$. The
partition indices for existing dimensions are remapped accordingly.

When slicing a slider, two cases arise:

- **Multi-dim group**: The slide's `ChebyshevApproximation` is sliced at
  the local dimension index within the group.
- **Single-dim group**: The slide is evaluated at the slice value, giving
  a constant $s_{g^*}(v)$. The shift $\delta = s_{g^*}(v) - pv$ is
  absorbed into `PivotValue` and each remaining slide's tensor values.

See [Sliding Technique](slider.md) for details.

## Limitations

- **No `ChebyshevTT` support** -- extrusion and slicing for Tensor Train
  interpolants are not currently implemented.
- **Operand must be built** -- `Build()` must have been called before
  calling `Extrude()` or `Slice()`.
- **No cross-type operations** -- you cannot extrude a `ChebyshevSpline`
  and then add it to a `ChebyshevApproximation`.
- **Result has `Function = null`** -- the extruded/sliced interpolant cannot
  call `Build()` again, since it has no underlying function reference.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation."
  *SIAM Review* 46(3):501--517.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.*
  Wiley Finance. Section 24.2.1.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM. Chapter 8.
