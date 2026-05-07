# ChebyshevSharp

[![Tests](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml/badge.svg)](https://github.com/0xC000005/ChebyshevSharp/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/0xC000005/ChebyshevSharp/graph/badge.svg)](https://codecov.io/gh/0xC000005/ChebyshevSharp)
[![NuGet](https://img.shields.io/nuget/v/ChebyshevSharp.svg)](https://www.nuget.org/packages/ChebyshevSharp)
[![NuGet Downloads](https://img.shields.io/nuget/dt/ChebyshevSharp.svg)](https://www.nuget.org/packages/ChebyshevSharp)
[![PyChebyshev parity](https://img.shields.io/badge/PyChebyshev_parity-v0.21.1-blue)](https://github.com/0xC000005/PyChebyshev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-blue.svg)](https://dotnet.microsoft.com/)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-blue.svg)](https://dotnet.microsoft.com/)

Multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET.

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev), providing fast polynomial evaluation of smooth multi-dimensional functions via barycentric interpolation with pre-computed weights. On low-dimensional problems (1-3D), C# is **17-42x faster** than the Python reference; see [Performance](https://0xc000005.github.io/ChebyshevSharp/docs/performance.html).

## Project Links

- [Documentation](https://0xc000005.github.io/ChebyshevSharp/)
- [API Reference](https://0xc000005.github.io/ChebyshevSharp/api/ChebyshevSharp.html)
- [NuGet Package](https://www.nuget.org/packages/ChebyshevSharp)
- [Changelog](https://0xc000005.github.io/ChebyshevSharp/docs/changelog.html)
- [Contributing](https://github.com/0xC000005/ChebyshevSharp/blob/main/CONTRIBUTING.md)
- [Security Policy](https://github.com/0xC000005/ChebyshevSharp/security/policy)

## Features

| Feature | Description |
|---------|-------------|
| **Chebyshev interpolation** | Multi-dimensional tensor interpolation with spectral convergence |
| **Analytical derivatives** | Spectral differentiation matrices — no finite differences |
| **BLAS acceleration** | N-D tensor contractions routed through OpenBLAS via [BlasSharp.OpenBlas](https://www.nuget.org/packages/BlasSharp.OpenBlas) |
| **Piecewise splines** | `ChebyshevSpline` — place knots at singularities for spectral convergence on each piece |
| **Sliding technique** | `ChebyshevSlider` — partition dimensions into groups for high-dimensional approximation |
| **Tensor Train interpolation** | `ChebyshevTT` — approximate high-dimensional coupled functions without materializing the dense grid |
| **Algebra** | Combine interpolants via `+`, `-`, `*`, `/` |
| **Extrusion & slicing** | Add or fix dimensions for portfolio aggregation |
| **Spectral calculus** | Integration (Fejer-1), root-finding (colleague matrix), minimization, maximization |
| **Serialization** | Save/load interpolants as JSON, plus portable `.pcb` for dense approximations and compatible splines |

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

// Query points must be inside the declared domain; out-of-domain points throw.

// 4. Check accuracy
double error = cheb.ErrorEstimate();  // ~1e-15 for this function

// 5. Save for deployment
cheb.Save("interpolant.json");
```

Runnable example projects are available in `examples/`:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

## Classes

| Class | Use case | Build cost |
|-------|----------|-----------|
| `ChebyshevApproximation` | Smooth functions on a single domain | $\prod_i n_i$ |
| `ChebyshevSpline` | Functions with discontinuities or singularities | pieces $\times \prod_i n_i$ |
| `ChebyshevSlider` | High-dimensional, additively separable functions | $\sum_g \prod_{i \in g} n_i$ |
| `ChebyshevTT` | High-dimensional functions with general coupling | $O(d \cdot n \cdot r^2)$ for TT-Cross |

## Example: Option Pricing

Replace a slow Black-Scholes pricer with a fast Chebyshev interpolant that returns price in ~500 ns, or price plus delta and gamma in ~2 us:

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

// Price, delta, and gamma in one call
double[] results = cheb.VectorizedEvalMulti(
    new[] { 100.0, 0.2, 1.0 },
    new[] {
        new[] { 0, 0, 0 },  // price
        new[] { 1, 0, 0 },  // delta
        new[] { 2, 0, 0 },  // gamma
    }
);
```

## Status

| Area | Status |
|------|--------|
| `ChebyshevApproximation` | Dense Chebyshev interpolation with analytical derivatives, algebra, calculus, serialization, and Sobol indices |
| `ChebyshevSpline` | Piecewise interpolation with knots, algebra, calculus, serialization, and automatic knot helpers |
| `ChebyshevSlider` | Partitioned high-dimensional approximation with parallel build, progress reporting, and sensitivity diagnostics |
| `ChebyshevTT` | Tensor Train build/eval, finite-difference derivatives, integration, roots, optimization, algebra, slicing/extrusion, reordering, and guarded dense materialization |
| Validation | xUnit regression suite, deterministic FsCheck properties, Codecov patch gate, package validation, DocFX build, and scheduled/manual mutation testing |

See the [changelog](https://0xc000005.github.io/ChebyshevSharp/docs/changelog.html) for per-release feature parity with PyChebyshev.

## Documentation

Full documentation is available at **[0xc000005.github.io/ChebyshevSharp](https://0xc000005.github.io/ChebyshevSharp/)**.

- [Getting Started](https://0xc000005.github.io/ChebyshevSharp/docs/getting-started.html)
- [Which Class Should I Use?](https://0xc000005.github.io/ChebyshevSharp/docs/which-class.html)
- [Mathematical Concepts](https://0xc000005.github.io/ChebyshevSharp/docs/concepts.html)
- [Piecewise Chebyshev (Splines)](https://0xc000005.github.io/ChebyshevSharp/docs/spline.html)
- [Sliding Technique](https://0xc000005.github.io/ChebyshevSharp/docs/slider.html)
- [Tensor Train](https://0xc000005.github.io/ChebyshevSharp/docs/tensor-train.html)
- [Computing Greeks](https://0xc000005.github.io/ChebyshevSharp/docs/greeks.html)
- [Chebyshev Algebra](https://0xc000005.github.io/ChebyshevSharp/docs/algebra.html)
- [Advanced Usage](https://0xc000005.github.io/ChebyshevSharp/docs/advanced-usage.html)
- [Calculus](https://0xc000005.github.io/ChebyshevSharp/docs/calculus.html)
- [Error Estimation](https://0xc000005.github.io/ChebyshevSharp/docs/error-estimation.html)
- [Serialization](https://0xc000005.github.io/ChebyshevSharp/docs/serialization.html)
- [Performance](https://0xc000005.github.io/ChebyshevSharp/docs/performance.html)
- [Testing & Validation](https://0xc000005.github.io/ChebyshevSharp/docs/testing-and-validation.html)
- [Support & Reporting](https://0xc000005.github.io/ChebyshevSharp/docs/support.html)
- [Citations](https://0xc000005.github.io/ChebyshevSharp/docs/citations.html)
- [API Reference](https://0xc000005.github.io/ChebyshevSharp/api/ChebyshevSharp.html)

## Support and Reporting

- Questions and usage problems: start a GitHub Discussion with a small runnable
  example.
- Bugs: use the bug report template and include OS, .NET SDK version,
  ChebyshevSharp version, expected behavior, and actual behavior.
- Numerical accuracy concerns: use the numerical accuracy template and include
  the function, domain, node counts, construction method, tolerance, and a
  reference value or independent check.
- Security issues: do not open a public issue; follow the
  [security policy](https://github.com/0xC000005/ChebyshevSharp/security/policy).

## Contributing

See [CONTRIBUTING.md](https://github.com/0xC000005/ChebyshevSharp/blob/main/CONTRIBUTING.md) for the development workflow, required
checks, Codecov policy, Stryker.NET mutation-testing expectations, and PR
checklist. For a shorter site version, see
[Contributing](https://0xc000005.github.io/ChebyshevSharp/docs/contributing.html).

This project follows [CODE_OF_CONDUCT.md](https://github.com/0xC000005/ChebyshevSharp/blob/main/CODE_OF_CONDUCT.md). Keep issue and PR
discussions focused, reproducible, and respectful.

## License

MIT
