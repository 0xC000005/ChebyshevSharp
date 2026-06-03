---
title: How-To Guides
---

# How-To Guides

Use these pages when you already know the task and need the concrete API
pattern.

## Build And Update Interpolants

- [Pre-computed Values](from-values.md) builds an approximation from an existing
  Chebyshev grid.
- [Error-Driven Construction](error-driven-construction.md) increases node
  counts until a target error is reached.
- [Adaptive Refinement](adaptive-refinement.md) explains coefficient-tail error
  checks, Sobol indices, and refinement helpers.
- [Serialization & Construction](serialization.md) saves, loads, and rebuilds
  approximation objects.
- [Parallel Build & Progress](parallel-build.md) controls multi-threaded build
  execution and progress reporting.

## Model Difficult Functions

- [Piecewise Chebyshev (Splines)](spline.md) places knots at known kinks or
  singularities.
- [Special Points](special-points.md) routes known trouble points into
  piecewise construction.
- [Sliding Technique](slider.md) uses grouped low-dimensional slices for
  high-dimensional models with limited cross-group interaction.
- [Tensor Train](tensor-train.md) uses TT-Cross for high-dimensional models with
  broader variable coupling.
- [Continuous-State Dynamic Programming](continuous-state-dynamic-programming.md)
  shows the finite-horizon Bellman-collocation pattern.

## Compute Derived Quantities

- [Computing Greeks](greeks.md) evaluates derivatives for sensitivity analysis.
- [Calculus](calculus.md) covers differentiation and integration operations.
- [Chebyshev Algebra](algebra.md) composes, adds, and multiplies approximations.
- [Extrusion & Slicing](extrude-slice.md) fixes or adds coordinates to reshape a
  model.
- [Advanced Usage](advanced-usage.md) collects lower-level usage patterns and
  caveats.
