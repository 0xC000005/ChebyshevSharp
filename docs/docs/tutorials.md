---
title: Tutorials
---

# Tutorials

Use these pages when you are learning the library or following a worked case
study end to end.

## Start Here

- [Introduction](introduction.md) explains why Chebyshev interpolation is useful
  and how the main classes fit together.
- [Getting Started](getting-started.md) walks through the first dense
  interpolation workflow.
- [Examples](examples.md) lists the runnable console projects in the repository.
- [Which Class Should I Use?](which-class.md) maps common modelling problems to
  `ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`, and
  `ChebyshevTT`.

## Technical Case Studies

- [Fixed-Rate Bond Case Study](fixed-rate-bond-surrogate.md) shows how a
  request-level fixed-rate bond wrapper can be decomposed into smooth discount
  kernels instead of cloned as one global high-dimensional tensor.
- [Callable Bond Case Study](callable-bond-surrogate.md) studies callable-bond
  risk acceleration while preserving the non-smooth exercise logic.
- [American Option Case Study](american-option-dynamic-chebyshev.md) compares
  regression, reinforcement-learning-style simulation, and dynamic Chebyshev
  continuation approximation for American options.
