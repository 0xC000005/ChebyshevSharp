---
title: Which Class Should I Use?
---

# Which Class Should I Use?

ChebyshevSharp has four approximation families. Pick the simplest class that matches the function shape and dimensionality before tuning node counts or ranks.

## Decision Map

```text
Is the function smooth on one rectangular domain?
  yes, and dimension is small or moderate -> ChebyshevApproximation
  no, but singularities or kinks are known -> ChebyshevSpline
  no, and the function changes by regime -> split the domain, then use ChebyshevSpline

Is the function high-dimensional?
  mostly additive or grouped by variables -> ChebyshevSlider
  coupled across many variables -> ChebyshevTT
```

## Class Comparison

| Class | Best fit | Build cost | Derivatives | Main trade-off |
|-------|----------|------------|-------------|----------------|
| `ChebyshevApproximation` | Smooth 1D-5D functions on one box | Dense grid, `prod(n_i)` | Analytical spectral derivatives | Dense grid grows exponentially |
| `ChebyshevSpline` | Kinks, jumps, barriers, or piecewise-smooth functions | Pieces times dense grid | Analytical within each piece | You must choose useful knots |
| `ChebyshevSlider` | High-dimensional functions with weak cross-group coupling | Sum of group grids | Uses each group interpolant | Accuracy depends on grouping and pivot |
| `ChebyshevTT` | High-dimensional functions with general coupling | TT-Cross, about `O(d * n * r^2)` | Finite-difference derivatives | Rank and seed affect compression and accuracy |

## Practical Defaults

- Start with `ChebyshevApproximation` for smooth functions up to about five dimensions.
- Use `ChebyshevSpline` when the error estimate stays high near a known kink or discontinuity.
- Use `ChebyshevSlider` when variables can be partitioned into nearly independent groups.
- Use `ChebyshevTT` when cross-variable coupling matters and the dense grid is too large.
- For TT work, avoid `ToDense()` unless the dense grid is intentionally small; dense-grid materialization is guarded and will throw for oversized tensors.

## Validation Checklist

After choosing a class:

1. Compare a held-out sample of true function values against interpolated values.
2. Increase node counts or TT rank and check whether the error decreases.
3. For splines, inspect each piece separately; one bad interval can dominate the global error.
4. For TT, run with a fixed seed first, then try a few seeds when rank is near the cap.
5. Save and reload one model in tests when the interpolant is used outside the build process.
