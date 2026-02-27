---
title: Advanced Usage
---

# Advanced Usage

## Batch Evaluation

Evaluate the interpolant at multiple points in a single call. Supported by `ChebyshevApproximation` (`VectorizedEvalBatch`), `ChebyshevSpline` (`EvalBatch`), and `ChebyshevTT` (`EvalBatch`). `ChebyshevSlider` does not support batch evaluation.

```csharp
double[][] points = new[]
{
    new[] { 100.0, 0.2, 1.0 },
    new[] { 105.0, 0.3, 0.5 },
    new[] { 95.0,  0.15, 1.5 }
};

double[] values = cheb.VectorizedEvalBatch(points, new[] { 0, 0, 0 });
```

`VectorizedEvalBatch` loops over points internally, calling `VectorizedEval` for each. It is 23x faster than Python's equivalent on 3D problems because the loop runs in JIT-compiled native code rather than the Python interpreter (see [Performance](performance.md)).

## Evaluation Methods

`ChebyshevApproximation` provides several evaluation methods for different use cases:

| Method | Use case |
|--------|----------|
| `VectorizedEval` | Single point, single derivative order. BLAS-optimized; recommended for most uses. |
| `Eval` | Same as `VectorizedEval` but uses loop-based contraction. Matches the Python `eval()` method. |
| `VectorizedEvalBatch` | Multiple points, same derivative order. Loops over points with JIT-compiled code. |
| `VectorizedEvalMulti` | Single point, multiple derivative orders (e.g., price + all Greeks). Shares barycentric weight computation across outputs. |

`ChebyshevSpline` and `ChebyshevSlider` provide `Eval` and `EvalMulti` with the same signatures. `ChebyshevSpline` also supports `EvalBatch`.

`ChebyshevTT` provides `Eval`, `EvalBatch`, and `EvalMulti`. Note that `EvalMulti` computes derivatives via finite differences (not spectral differentiation), supporting derivative orders 0, 1, and 2 per dimension. See [Tensor Train Interpolation](tensor-train.md) for details.

## Multi-Output Evaluation

Compute the function value and multiple derivatives at the same point in a single call using `VectorizedEvalMulti`. This shares intermediate barycentric weight computation across all requested outputs:

```csharp
// Black-Scholes 3D: (spot, volatility, maturity)
// Compute price and all Greeks at one point
int[][] derivativeOrders = new[]
{
    new[] { 0, 0, 0 },  // value (price)
    new[] { 1, 0, 0 },  // delta (df/dS)
    new[] { 2, 0, 0 },  // gamma (d²f/dS²)
    new[] { 0, 1, 0 },  // vega  (df/dsigma)
    new[] { 0, 0, 1 },  // theta (df/dT)
};

double[] results = cheb.VectorizedEvalMulti(
    new[] { 100.0, 0.2, 1.0 },
    derivativeOrders
);

double price = results[0];
double delta = results[1];
double gamma = results[2];
double vega  = results[3];
double theta = results[4];
```

`VectorizedEvalMulti` is faster than calling `VectorizedEval` in a loop because it computes the barycentric weights and exact-node detection once and reuses them across all derivative orders. For 3D with 5 outputs, this gives a ~3x speedup over separate calls.

## Error Estimation

`ErrorEstimate()` measures how well the interpolant has resolved the underlying function, without requiring knowledge of the true function values at test points.

```csharp
double error = cheb.ErrorEstimate();
Console.WriteLine($"Estimated max error: {error:E2}");
```

**How it works:** The method extracts Chebyshev expansion coefficients along each dimension using the Discrete Cosine Transform (DCT-II). For a well-resolved function, the Chebyshev coefficients decay geometrically — the rate of decay is governed by the analyticity region (Bernstein ellipse) of the function [1, Ch. 8]. The magnitude of the last coefficient provides an upper bound on the truncation error.

The estimate is computed as the maximum of the last Chebyshev coefficient across all 1D slices and all dimensions. This is a conservative (pessimistic) heuristic — the actual interpolation error is typically smaller.

The result is cached — repeated calls return the stored value without recomputation.

For `ChebyshevSpline`, the error estimate is the maximum across all pieces — reflecting the worst-case error. For `ChebyshevSlider`, the error estimate is the **sum** across all slides — reflecting the additive nature of the approximation. See [Piecewise Chebyshev Interpolation](spline.md) and [Sliding Technique](slider.md) for details.

## Extrusion

Add new dimensions to an existing interpolant. The function value is constant along the new dimensions:

```csharp
// Start with a 2D interpolant f(x, y)
var cheb2d = new ChebyshevApproximation(...);
cheb2d.Build();

// Add a third dimension at index 2: g(x, y, z) = f(x, y)
var cheb3d = cheb2d.Extrude((2, new[] { 0.0, 1.0 }, 10));
```

The extruded interpolant has one more dimension. You can extrude multiple dimensions at once by passing multiple tuples. Each tuple specifies `(dimensionIndex, bounds, numberOfNodes)`.

This is useful for building up multi-dimensional interpolants incrementally, or for combining interpolants that depend on different subsets of variables via arithmetic operators.

`ChebyshevSpline` supports extrusion, which extrudes each piece independently. See [Piecewise Chebyshev Interpolation](spline.md) for details.

`ChebyshevSlider` supports extrusion, which adds a new singleton-group slide for the new dimension. The new dimension is constant at the pivot value. See [Sliding Technique](slider.md) for details.

## Slicing

Fix one or more dimensions at specific values, reducing the dimensionality:

```csharp
// Fix dimension 1 (sigma) at 0.25: f(S, T) = g(S, 0.25, T)
var cheb2d = cheb3d.Slice((1, 0.25));

// Fix multiple dimensions
var cheb1d = cheb3d.Slice((1, 0.25), (2, 1.0));
```

Slicing performs barycentric interpolation along the fixed dimensions and returns a new `ChebyshevApproximation` with reduced dimensionality. When the slice value coincides with a Chebyshev node, the interpolation reduces to direct extraction with no approximation error.

Slicing is used internally by `Roots`, `Minimize`, and `Maximize` to reduce multi-dimensional problems to 1D before applying root-finding or optimization (see [Calculus](calculus.md)).

`ChebyshevSpline` supports slicing, selecting the piece containing the slice value. See [Piecewise Chebyshev Interpolation](spline.md) for details.

`ChebyshevSlider` supports slicing. If the sliced dimension belongs to a multi-dimension group, the slide is sliced within that group. If it is a singleton group, the slide is evaluated and its contribution is absorbed into the remaining slides. See [Sliding Technique](slider.md) for details.

`ChebyshevTT` does **not** support extrusion or slicing.

## Arithmetic Operators

Interpolants defined on the same grid support pointwise arithmetic:

```csharp
var sum    = cheb1 + cheb2;     // pointwise addition
var diff   = cheb1 - cheb2;     // pointwise subtraction
var neg    = -cheb1;            // pointwise negation
var scaled = cheb1 * 2.0;       // scalar multiplication
var halved = cheb1 / 2.0;       // scalar division
```

Both operands must have the same number of dimensions, domain bounds, and node counts. The result is a new `ChebyshevApproximation` with tensor values computed pointwise.

Arithmetic on interpolants is exact at the nodes. Between nodes, the result is the Chebyshev interpolant of the pointwise operation — which differs from the true pointwise result by the interpolation error. For well-resolved interpolants, this difference is negligible.

`ChebyshevSpline` also supports arithmetic operators. Both operands must have the same knots, domain bounds, and node counts. See [Piecewise Chebyshev Interpolation](spline.md) for details.

`ChebyshevSlider` also supports arithmetic operators. Both operands must have the same partition and pivot point in addition to domain and node counts. See [Sliding Technique](slider.md) for details.

`ChebyshevTT` does **not** support arithmetic operators.

## Derivative Orders

The `derivativeOrder` parameter controls which partial derivative to compute. Each entry corresponds to the derivative order along that dimension:

```csharp
// f(x, y, z) — 3D interpolant
cheb.VectorizedEval(point, new[] { 0, 0, 0 });  // f
cheb.VectorizedEval(point, new[] { 1, 0, 0 });  // df/dx
cheb.VectorizedEval(point, new[] { 0, 1, 0 });  // df/dy
cheb.VectorizedEval(point, new[] { 2, 0, 0 });  // d²f/dx²
cheb.VectorizedEval(point, new[] { 1, 1, 0 });  // d²f/dxdy (cross-derivative)
```

**How derivatives work:** Derivatives are computed analytically using spectral differentiation matrices D [2]. The matrix D maps function values at Chebyshev nodes to derivative values at the same nodes. For the k-th derivative, the differentiation matrix D is applied k times via matrix multiplication. This is done as part of the tensor contraction — before contracting a dimension, the tensor is multiplied by D^k along that axis if the derivative order is k > 0.

Spectral derivatives converge at the same exponential rate as the function values, unlike finite differences which lose accuracy proportional to the step size.

The maximum supported derivative order is controlled by `MaxDerivativeOrder` (default: 2). Set it in the constructor if you need higher-order derivatives.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
2. Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
