# Getting Started

## Installation

Install ChebyshevSharp from NuGet:

```bash
dotnet add package ChebyshevSharp
```

Or add to your `.csproj`:

```xml
<PackageReference Include="ChebyshevSharp" Version="0.13.0" />
```

The [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas) package is included as a transitive dependency and provides pre-built OpenBLAS binaries for all platforms (Windows, Linux, macOS). No system BLAS installation is required.

## Quick Start

### 1. Define a function

ChebyshevSharp interpolates functions of the form `f(x, data) -> double`, where `x` is a point in multi-dimensional space:

```csharp
using ChebyshevSharp;

// A 2D function: f(x, y) = sin(x) * cos(y)
double MyFunction(double[] x, object? data)
{
    return Math.Sin(x[0]) * Math.Cos(x[1]);
}
```

The second parameter `data` is an optional pass-through for user context (set to `null` if unused).

### 2. Build the interpolant

Specify the number of dimensions, domain bounds, and polynomial orders:

```csharp
var cheb = new ChebyshevApproximation(
    function: MyFunction,
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11 }
);
cheb.Build();
```

`Build()` evaluates the function at all 11 x 11 = 121 Chebyshev node combinations and pre-computes the barycentric weights and differentiation matrices. This is a one-time cost.

**Choosing node counts:** 10-15 nodes per dimension is typical for smooth functions. The interpolation error decreases exponentially with node count (spectral convergence), so adding a few nodes can gain several digits of accuracy. Use `ErrorEstimate()` (below) to verify.

### 3. Evaluate

```csharp
// Function value
double value = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });

// Partial derivative df/dx0
double dfdx0 = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });

// Second derivative d²f/dx1²
double d2fdx1 = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 2 });
```

The second argument specifies the derivative order along each dimension. `{0, 0}` means the function value, `{1, 0}` means first derivative with respect to dimension 0, `{0, 2}` means second derivative with respect to dimension 1.

Evaluation points must be inside the declared domain. For the example above, both coordinates must be between `-1.0` and `1.0`; out-of-domain points throw `ArgumentOutOfRangeException` rather than extrapolating silently.

Derivatives are computed analytically using spectral differentiation matrices — they converge at the same rate as the function values, unlike finite differences which lose accuracy.

### 4. Check accuracy

```csharp
double error = cheb.ErrorEstimate();
Console.WriteLine($"Estimated max error: {error:E2}");
```

`ErrorEstimate()` extracts Chebyshev coefficients via DCT-II and uses the magnitude of the last coefficient as a proxy for interpolation error. For well-resolved functions, the actual error is typically smaller than this estimate. If the error is too large, increase the node counts.

### 5. Save for later

```csharp
cheb.Save("my_interpolant.json");

// Later, in another process:
var restored = ChebyshevApproximation.Load("my_interpolant.json");
double val = restored.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });
```

See [Serialization & Construction](serialization.md) for details on `Save`, `Load`, `FromValues`, and `Nodes`.

The same workflow is available as a runnable console project:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
```

## Choosing the Right Class

| Scenario | Class |
|----------|-------|
| Smooth function on a single domain | `ChebyshevApproximation` |
| Function with known discontinuities or singularities | `ChebyshevSpline` — place knots at trouble points for spectral convergence on each piece |
| High-dimensional function (6+ dims), additively separable or nearly so | `ChebyshevSlider` — partition dimensions into groups, build cost is sum instead of product |
| High-dimensional function (5+ dims), general cross-variable coupling | `ChebyshevTT` — Tensor Train decomposition, build cost is $O(d \cdot n \cdot r^2)$ |

`ChebyshevSpline` supports the same API shape as `ChebyshevApproximation` (`Eval`, derivatives, batch evaluation, multi-output evaluation, `Save`, `Load`, arithmetic, extrusion, slicing, integration, roots, and optimization). See [Piecewise Chebyshev Interpolation](spline.md) for a full guide.

`ChebyshevSlider` supports `Eval`, `EvalMulti`, error estimation, `Save`, `Load`, integration, roots, optimization, extrusion, slicing, and arithmetic operators. It does not support batch evaluation or the `Nodes()`/`FromValues()` workflow. See [Sliding Technique](slider.md) for a full guide.

`ChebyshevTT` supports `Eval`, `EvalBatch`, `EvalMulti`, finite-difference derivatives, error estimation, `Save`, `Load`, integration, roots, optimization, algebra, extrusion, slicing, rank refinement, and dimension reordering. It intentionally guards dense materialization paths because a full grid can exceed memory even when TT-Cross is feasible. See [Tensor Train Interpolation](tensor-train.md) for a full guide.

For a more detailed decision map, see [Which Class Should I Use?](which-class.md).

## Next Steps

- [Which Class Should I Use?](which-class.md) — decision guide for dense, spline, sliding, and TT approximations
- [Mathematical Concepts](concepts.md) — theory behind Chebyshev interpolation, Bernstein ellipse, spectral convergence
- [Error-Driven Construction](error-driven-construction.md) — select node counts based on measured error
- [Piecewise Chebyshev Interpolation](spline.md) — handling discontinuities with ChebyshevSpline
- [Sliding Technique](slider.md) — high-dimensional approximation with ChebyshevSlider
- [Tensor Train Interpolation](tensor-train.md) — high-dimensional approximation with general coupling via ChebyshevTT
- [Computing Greeks](greeks.md) — analytical derivatives for option Greeks
- [Calculus](calculus.md) — integration, root-finding, minimization, maximization
- [Advanced Usage](advanced-usage.md) — batch/multi eval, extrusion, slicing, arithmetic operators
- [Error Estimation](error-estimation.md) — measuring interpolation accuracy via DCT-II
- [Serialization & Construction](serialization.md) — `Save`, `Load`, `FromValues`, `Nodes`
- [Performance](performance.md) — BLAS integration and benchmark results
- [Testing & Validation](testing-and-validation.md) — local quality gates, property tests, and mutation testing
- [Citations](citations.md) — references used by the algorithms and documentation
- [API Reference](../api/ChebyshevSharp.yml) — full class and method documentation
