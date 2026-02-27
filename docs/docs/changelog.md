---
title: Changelog
---

# Changelog

All notable changes to ChebyshevSharp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-26

### Added

- `ChebyshevSlider` — Phase 3 sliding technique for high-dimensional approximation via additive decomposition around a pivot point
- Slider API: eval, eval_multi, error_estimate, save/load, extrude, slice, arithmetic operators (+, -, *, /, unary -)
- Slider tests ported from PyChebyshev `test_slider.py` (40 tests) plus cross-type algebra (19 tests) and extrude/slice (10 tests)
- C#-specific slider tests: validation, serialization, eval, properties, error estimation, arithmetic edge cases, extrude/slice edge cases (53 tests)
- Sliding Technique documentation page with algorithm, LaTeX math, examples, and cross-references
- ChebyshevSlider references added to all existing documentation pages (introduction, getting-started, advanced-usage, calculus, serialization, performance)
- ChebyshevSlider API reference auto-generated via DocFX

## [0.2.0] - 2026-02-26

### Added

- `ChebyshevSpline` — Phase 2 piecewise Chebyshev interpolation with user-specified knots at singularities
- Full spline API: eval, eval_multi, eval_batch, error_estimate, save/load, nodes, from_values, extrude, slice, integrate, roots, minimize, maximize, arithmetic operators
- Spline tests ported from PyChebyshev `test_spline.py` (55 tests)
- C#-specific spline tests: validation, serialization, knot routing, immutability, concurrent eval (73 tests)
- BLAS-backed GEMV/GEMM via `BlasSharp.OpenBlas` NuGet package for tensor contraction
- Pre-transposed differentiation matrices (computed once, stored as flat arrays)
- FFT-based DCT-II (O(n log n)) for n > 32
- Piecewise Chebyshev documentation page with Gibbs phenomenon, Bernstein ellipse math, and examples
- 5 additional Phase 1 tests (207 → 212): `Test_verbose_build`, `Test_load_wrong_type_raises`, `Test_integrate_cross_validate_scipy`, `Test_scipy_cross_validation`, `Test_bounds_length_mismatch_raises`

### Changed

- 2–4x speedup for 1D–3D eval, 1.5–2.5x for 5D via BLAS and pre-transposed matrices
- Shape allocation elimination in `VectorizedEval`
- Simplified `PrecomputeTransposedDiffMatrices` to single-pass transpose+flatten

### Removed

- `NativeBlas` wrapper class (use `BlasSharp` directly)
- Dead code: old shape-based `MatmulLastAxis`/`MatmulLastAxisMatrix` overloads
- Unnecessary `DiffMatricesT` intermediate (transpose directly to flat arrays)

### Dependencies

- Added `BlasSharp.OpenBlas` 0.3.0 for cross-platform OpenBLAS

## [0.1.0] - 2026-02-24

### Added

- Project scaffold: solution, library (net8.0;net10.0), xUnit test project
- PyChebyshev reference submodule at `ref/PyChebyshev/`
- CI/CD: GitHub Actions for testing (.NET 8 + 10), NuGet publishing, Dependabot auto-merge
- Branch protection ruleset on main
- Codecov integration
- `ChebyshevApproximation` — Phase 1 implementation with 207 passing tests
- DocFX documentation site with GitHub Pages deployment
