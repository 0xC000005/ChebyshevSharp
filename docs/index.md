---
_layout: landing
---

# ChebyshevSharp

Multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET.

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev), providing fast evaluation of smooth multi-dimensional functions via barycentric interpolation with pre-computed weights.

## Key Features

- **Multi-dimensional Chebyshev interpolation** with spectral convergence
- **Analytical derivatives** via spectral differentiation matrices
- **Vectorized evaluation** routing N-D tensor contractions through BLAS (via [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas))
- **Piecewise Chebyshev splines** with user-specified knots at singularities
- **Sliding technique** for high-dimensional approximation
- **Tensor Train** decomposition for 5+ dimensional functions
- **Chebyshev algebra** — combine interpolants via `+`, `-`, `*`, `/`
- **Spectral calculus** — integration, rootfinding, and optimization
- Targets **.NET 8** and **.NET 10**

## Installation

```bash
dotnet add package ChebyshevSharp
```

## Quick Start

```csharp
using ChebyshevSharp;

// Define a function to interpolate
double MyFunction(double[] x, object? data) => Math.Sin(x[0]) + Math.Sin(x[1]);

// Build a 2D Chebyshev interpolant
var cheb = new ChebyshevApproximation(
    function: MyFunction,
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11 }
);
cheb.Build();

// Evaluate at a point
double value = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });

// Compute partial derivative df/dx1
double dfdx = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });
```

## Where to Go Next

| Need | Start here |
|------|------------|
| First working interpolant | [Getting Started](docs/getting-started.md) |
| Choose between dense, spline, slider, and TT | [Which Class Should I Use?](docs/which-class.md) |
| Understand convergence and derivatives | [Mathematical Concepts](docs/concepts.md) |
| Build high-dimensional coupled approximations | [Tensor Train Interpolation](docs/tensor-train.md) |
| Validate changes locally | [Testing & Validation](docs/testing-and-validation.md) |
| Report bugs or ask for help | [Support & Reporting](docs/support.md) |
| Contribute code or docs | [Contributing](docs/contributing.md) |
| Check algorithm sources | [Citations](docs/citations.md) |

## API Reference

See the [API documentation](api/ChebyshevSharp.yml) for full class and method reference, auto-generated from XML documentation comments.
