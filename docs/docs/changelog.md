---
title: Changelog
---

# Changelog

All notable changes to ChebyshevSharp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-04-28

### PyChebyshev parity: v0.18.0 (binary format fill-in)

Phase 3 of the v0.20.1 phased port. Adds a portable little-endian binary
serialization format (`.pcb`) so cross-language consumers (C, Rust, Julia) can
read ChebyshevSharp interpolants without the .NET runtime. Bit-for-bit
byte-compatible with PyChebyshev v0.14's format.

#### Added

- `ChebyshevApproximation.Save(string path, string format = "json")` and
  `ChebyshevSpline.Save(string path, string format = "json")` — `format` accepts
  `"json"` (existing default) or `"binary"` (the portable `.pcb` format).
- `ChebyshevApproximation.Load(string path)` and `ChebyshevSpline.Load(string path)`
  now auto-detect JSON vs binary by sniffing the first 4 bytes for the
  `b"PCB\x00"` magic header.
- `static int ChebyshevApproximation.PeekFormatVersion(string path)` and
  `static int ChebyshevSpline.PeekFormatVersion(string path)` — read the major
  version byte without parsing the body.
- New `Internal/PcbFormat.cs` holding all binary read/write logic with explicit
  little-endian assertions.

#### Changed

- `ChebyshevSpline.Save(path, format="binary")` throws `NotSupportedException`
  for splines built with nested per-piece `nNodes` (the `int[][]` form from
  Phase 1's special-points work). Use `format="json"` for those.
- `ChebyshevSlider` and `ChebyshevTT` remain JSON-only in v0.7.0.

#### Test count: 794 → 797 (+3 fixture-based tests)

See `docs/docs/binary-format.md` for the full format documentation and the
cross-language round-trip guide.

## [0.6.0] - 2026-04-28

### PyChebyshev parity: v0.18.0

#### Added — TT canonicalization + ALS

- `ChebyshevTT.OrthLeft(int position)` and `OrthRight(int position)` —
  in-place QR/LQ canonicalization of the TT chain.
- `ChebyshevTT.InnerProduct(ChebyshevTT)` — Frobenius inner product of
  two TTs' Chebyshev coefficient tensors.
- `Build(method: "als")` — rank-adaptive alternating least-squares build
  mode. Starts at rank 1 and grows by 1 per outer iteration until the
  grid residual falls below `tolerance` or rank reaches `maxRank`.
- `ChebyshevTT.RunCompletion(double tolerance, int maxIter)` — refine an
  already-built TT in place via fixed-rank ALS sweeps.
- New `Method` property mirroring Python's `tt.method` attribute.
- New `BuildWarning` property — set when ALS hits `maxRank` before
  tolerance is satisfied (replaces Python's `RuntimeWarning`).

#### Added — TT factories, materialization, slicing, algebra

- `ChebyshevTT.Nodes(numDim, domain, nNodes)` — static factory matching
  `ChebyshevApproximation.Nodes`.
- `ChebyshevTT.FromValues(tensorValues, ...)` — TT-SVD compress a
  precomputed dense tensor into a TT (skips TT-Cross).
- `ChebyshevTT.ToDense()` — materialize the TT chain into a flat row-major
  dense tensor. Throws `OverflowException` on huge shapes.
- `ChebyshevTT.Slice(int dim, double value)` — fix a dimension at a value,
  returning a lower-dim TT.
- `ChebyshevTT.Extrude(int dim, (double, double), int)` — insert a new
  dimension where the function is constant.
- Operators `+`, `-`, scalar `*` and `/`, unary `-`. Binary operators
  round to `max(maxRank_a, maxRank_b)` at default tolerance `1e-12`.
- In-place methods `AddInPlace`, `SubInPlace`, `ScalarMulInPlace`,
  `ScalarDivInPlace`, `NegateInPlace`, `RoundInPlace(tolerance)`.

#### Changed

- TT JSON serialization format bumped to `"0.6.0"` to persist `Method`.
  Loading 0.5.0 files is supported (Method backfills to null).
- `Internal/TensorTrainKernel.cs` split into kernel + algebra + extrude
  modules. Public API and behavior unchanged.

#### Skipped

- Multi-dim variadic forms of `Extrude`/`Slice` (Python accepts a list
  of tuples). C# surface is single-dim per call; chain calls for
  multi-dim. No information lost; cleaner overload set.

## [0.5.0] - 2026-04-27

### PyChebyshev parity: v0.12.0

#### Added — Error-Driven Construction

- `ChebyshevApproximation` and `ChebyshevSpline` constructors accept
  `errorThreshold` and `maxN` parameters. `nNodes` may be `int?[]` with
  `null` per dim signalling auto-N for that dimension.
- New `static int ChebyshevApproximation.GetOptimalN1(...)` 1-D capacity
  estimator.
- New `double? GetErrorThreshold()` accessor on Approximation and Spline.
- New `string? BuildWarning` property — set when `maxN` is reached before
  `errorThreshold` is satisfied (replaces Python's `RuntimeWarning`).
- New internal `Internal/AdaptiveBuild.cs` runs the doubling loop.
- New `double[] ErrorEstimatePerDim()` public method (was internal in
  Python; exposed here for symmetry with the new auto-N infrastructure).

#### Added — Special Points

- `ChebyshevSpline.WithSpecialPoints(...)` static factory: kink
  declaration with `specialPoints` (mirrors Python's `special_points` kwarg
  via a C#-idiomatic factory, since constructors cannot return a different
  type).
- `ChebyshevSpline` accepts a nested `int[][]` form for `nNodes`,
  per-sub-interval. `nNodesNested[d][i]` is the node count for piece `i`
  along dimension `d`.

#### Changed

- `Build()` is now a public dispatcher; the original behavior moved to a
  private `BuildFixedGrid()`. No behavior change for fixed-N callers.
- `ErrorEstimate()` is now backed by `ErrorEstimatePerDim().Sum()`. Cache
  semantics unchanged.
- JSON `Save`/`Load` format version bumped to "0.5.0". `Load` backfills
  `OriginalNNodes`, `ErrorThreshold`, `MaxN` for older files (default
  values: `OriginalNNodes` ← `NNodes`, `ErrorThreshold` ← null, `MaxN`
  ← 64).

#### Skipped

- Python's `ChebyshevApproximation(special_points=...)` constructor
  dispatch to `ChebyshevSpline` is not mirrored. C# constructors cannot
  return a different type. Use `ChebyshevSpline.WithSpecialPoints(...)`.

## [0.4.0] - 2026-02-27

### Added

- `ChebyshevTT` -- Phase 4 Tensor Train Chebyshev interpolation for high-dimensional functions (5+ dimensions)
- TT-Cross decomposition with maxvol pivoting, adaptive rank selection, and evaluation caching
- TT-SVD decomposition for deterministic, optimal rank-$r$ approximation on small grids
- TT API: `Eval`, `EvalBatch` (vectorized), `EvalMulti` (finite-difference derivatives), `ErrorEstimate`, `Save`/`Load`
- Finite-difference derivatives: central differences for order 1 and 2, mixed partials via 4-point stencil, boundary nudging
- TT-specific properties: `TtRanks`, `CompressionRatio`, `TotalBuildEvals`
- Cross-language validation tests: 9 tests comparing C# TT-SVD output against hardcoded Python reference values
- Correctness tests: 16 additional tests covering mathematical invariants, edge cases, and robustness
- TT tests ported from PyChebyshev `test_tensor_train.py` (35 tests)
- Tensor Train Interpolation documentation page with theory, examples, and parameter guidance
- ChebyshevTT references added to all existing documentation pages

### Summary

All four phases complete. Feature parity with PyChebyshev v0.10.1 achieved. 613 tests passing.

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
