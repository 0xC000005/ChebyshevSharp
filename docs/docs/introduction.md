# Introduction

ChebyshevSharp provides multi-dimensional Chebyshev tensor interpolation with analytical derivatives for .NET applications.

## Why Chebyshev Interpolation?

Polynomial interpolation with equally-spaced points suffers from **Runge's phenomenon** — wild oscillations near interval endpoints. Chebyshev nodes solve this by clustering near boundaries, achieving **spectral convergence**: for smooth functions, the interpolation error decreases exponentially with the number of nodes.

## Classes

| Class | Purpose |
|-------|---------|
| `ChebyshevApproximation` | Core multi-dimensional Chebyshev interpolation with analytical derivatives |
| `ChebyshevSpline` | Piecewise Chebyshev interpolation with knots at singularities |
| `ChebyshevSlider` | High-dimensional approximation via the Sliding Technique |
| `ChebyshevTT` | Tensor Train Chebyshev interpolation for 5+ dimensions |

## Relationship to PyChebyshev

ChebyshevSharp is a C# port of [PyChebyshev](https://github.com/0xC000005/PyChebyshev). The Python reference implementation is included as a git submodule at `ref/PyChebyshev/` for cross-validation. Both libraries produce numerically identical results within floating-point tolerance.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.* Wiley Finance.
