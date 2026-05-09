---
_layout: landing
---

# ChebyshevSharp

Multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET.

ChebyshevSharp builds fast, reusable polynomial surrogates for smooth multi-dimensional functions. Use it when direct model evaluations are expensive but repeated values, derivatives, integrals, roots, or optimizers are needed inside .NET applications.

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

| Path | Start here |
|------|------------|
| Build your first interpolant | [Getting Started](docs/getting-started.md) |
| Choose an approximation family | [Which Class Should I Use?](docs/which-class.md) |
| Understand the numerical assumptions | [Mathematical Concepts](docs/concepts.md) |
| Improve accuracy or handle kinks | [Error-Driven Construction](docs/error-driven-construction.md), then [Piecewise Chebyshev Interpolation](docs/spline.md) |
| Work in higher dimensions | [Sliding Technique](docs/slider.md) or [Tensor Train Interpolation](docs/tensor-train.md) |
| Use saved or external function values | [Serialization & Construction](docs/serialization.md) and [Pre-computed Values](docs/from-values.md) |
| Validate or contribute changes | [Testing & Validation](docs/testing-and-validation.md), [Support & Reporting](docs/support.md), and [Contributing](docs/contributing.md) |
| Check mathematical sources | [Citations](docs/citations.md) |
| Check release history | [Release Notes](docs/changelog.md) |

## Project Links

- [GitHub repository](https://github.com/0xC000005/ChebyshevSharp) - source code,
  pull requests, issues, and CI history.
- [NuGet package](https://www.nuget.org/packages/ChebyshevSharp) - installable
  package and published versions.
- [Project links](docs/project.md) - releases, support, reporting, and
  contribution entry points.

## API Reference

See the [API documentation](api/ChebyshevSharp.yml) for full class and method reference, auto-generated from XML documentation comments.
