# ChebyshevSharp

[![Tests](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml/badge.svg)](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/0xC000005/ChebyshevSharp/graph/badge.svg)](https://codecov.io/gh/0xC000005/ChebyshevSharp)
[![NuGet](https://img.shields.io/nuget/v/ChebyshevSharp.svg)](https://www.nuget.org/packages/ChebyshevSharp)
[![NuGet Downloads](https://img.shields.io/nuget/dt/ChebyshevSharp.svg)](https://www.nuget.org/packages/ChebyshevSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-blue.svg)](https://dotnet.microsoft.com/)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-blue.svg)](https://dotnet.microsoft.com/)

Multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET.

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev), providing fast polynomial evaluation of smooth multi-dimensional functions via barycentric interpolation with pre-computed weights. On low-dimensional problems (1-3D), C# is **17-42x faster** than the Python reference; see [Performance](https://0xc000005.github.io/ChebyshevSharp/docs/performance.html).

## Features

| Feature | Description |
|---------|-------------|
| **Chebyshev interpolation** | Multi-dimensional tensor interpolation with spectral convergence |
| **Analytical derivatives** | Spectral differentiation matrices — no finite differences |
| **BLAS acceleration** | N-D tensor contractions routed through OpenBLAS via [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas) |
| **Piecewise splines** | `ChebyshevSpline` — place knots at singularities for spectral convergence on each piece |
| **Sliding technique** | `ChebyshevSlider` — partition dimensions into groups for high-dimensional approximation |
| **Algebra** | Combine interpolants via `+`, `-`, `*`, `/` |
| **Extrusion & slicing** | Add or fix dimensions for portfolio aggregation |
| **Spectral calculus** | Integration (Fejer-1), root-finding (colleague matrix), minimization, maximization |
| **Serialization** | Save/load interpolants as JSON for deployment without the original function |

## Installation

```bash
dotnet add package ChebyshevSharp
```

No system BLAS installation required — cross-platform OpenBLAS binaries are included.

## Quick Start

```csharp
using ChebyshevSharp;

// 1. Define a function
double MyFunc(double[] x, object? data)
    => Math.Sin(x[0]) * Math.Cos(x[1]);

// 2. Build the interpolant
var cheb = new ChebyshevApproximation(
    function: MyFunc,
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11 }
);
cheb.Build();

// 3. Evaluate — function value and derivatives
double value = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });
double dfdx  = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });
double d2fdy = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 2 });

// 4. Check accuracy
double error = cheb.ErrorEstimate();  // ~1e-15 for this function

// 5. Save for deployment
cheb.Save("interpolant.json");
```

## Classes

| Class | Use case | Build cost |
|-------|----------|-----------|
| `ChebyshevApproximation` | Smooth functions on a single domain | $\prod_i n_i$ |
| `ChebyshevSpline` | Functions with discontinuities or singularities | pieces $\times \prod_i n_i$ |
| `ChebyshevSlider` | High-dimensional, additively separable functions | $\sum_g \prod_{i \in g} n_i$ |
| `ChebyshevTT` | High-dimensional, general coupling | Planned |

## Example: Option Pricing

Replace a slow Black-Scholes pricer with a fast Chebyshev interpolant that returns price and all Greeks in ~500 ns:

```csharp
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
cheb.Build();  // 1,800 function evaluations (one-time cost)

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

## Status

| Phase | Class | Tests | Status |
|-------|-------|------:|--------|
| 1 | `ChebyshevApproximation` | 233 | Done |
| 2 | `ChebyshevSpline` | 128 | Done |
| 3 | `ChebyshevSlider` | 122 | Done |
| 4 | `ChebyshevTT` | — | Planned |
| **Total** | | **483** | |

See [skip_csharp.txt](skip_csharp.txt) for detailed feature parity with PyChebyshev.

## Documentation

Full documentation is available at **[0xc000005.github.io/ChebyshevSharp](https://0xc000005.github.io/ChebyshevSharp/)**.

- [Getting Started](https://0xc000005.github.io/ChebyshevSharp/docs/getting-started.html)
- [Mathematical Concepts](https://0xc000005.github.io/ChebyshevSharp/docs/concepts.html)
- [Piecewise Chebyshev (Splines)](https://0xc000005.github.io/ChebyshevSharp/docs/spline.html)
- [Sliding Technique](https://0xc000005.github.io/ChebyshevSharp/docs/slider.html)
- [Computing Greeks](https://0xc000005.github.io/ChebyshevSharp/docs/greeks.html)
- [Chebyshev Algebra](https://0xc000005.github.io/ChebyshevSharp/docs/algebra.html)
- [Advanced Usage](https://0xc000005.github.io/ChebyshevSharp/docs/advanced-usage.html)
- [Calculus](https://0xc000005.github.io/ChebyshevSharp/docs/calculus.html)
- [Error Estimation](https://0xc000005.github.io/ChebyshevSharp/docs/error-estimation.html)
- [Serialization](https://0xc000005.github.io/ChebyshevSharp/docs/serialization.html)
- [Performance](https://0xc000005.github.io/ChebyshevSharp/docs/performance.html)
- [API Reference](https://0xc000005.github.io/ChebyshevSharp/api/ChebyshevSharp.html)

## License

MIT
