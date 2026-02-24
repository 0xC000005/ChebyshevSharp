# ChebyshevSharp

[![Tests](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml/badge.svg)](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml)
[![NuGet](https://img.shields.io/nuget/v/ChebyshevSharp.svg)](https://www.nuget.org/packages/ChebyshevSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET.

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev), providing fast evaluation of smooth multi-dimensional functions via barycentric interpolation with pre-computed weights.

## Features

- **Multi-dimensional Chebyshev interpolation** with spectral convergence
- **Analytical derivatives** via spectral differentiation matrices
- **Vectorized evaluation** routing N-D tensor contractions through BLAS (via [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas))
- **Piecewise Chebyshev splines** with user-specified knots at singularities
- **Sliding technique** for high-dimensional approximation
- **Tensor Train** decomposition for 5+ dimensional functions
- **Chebyshev algebra** — combine interpolants via `+`, `-`, `*`, `/`
- **Extrusion and slicing** — add/fix dimensions for portfolio aggregation
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

// Compute partial derivative ∂f/∂x₁
double dfdx = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });
```

## Status

**Phase 1 complete** — `ChebyshevApproximation` with 212 passing tests covering construction, evaluation, derivatives, algebra, extrusion/slicing, calculus, and serialization.

Phases 2-4 (ChebyshevSpline, ChebyshevSlider, ChebyshevTT) are not yet implemented. See [skip_csharp.txt](skip_csharp.txt) for current feature parity with PyChebyshev.

## License

MIT
