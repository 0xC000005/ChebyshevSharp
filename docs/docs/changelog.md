---
title: Changelog
---

# Changelog

All notable changes to ChebyshevSharp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This page is release history, not a tutorial. Older entries may mention port
phases, parity tags, PyChebyshev, or MoCaX when those details were part of a
specific release. Current user guidance is organized in the tutorials, concepts,
how-to guides, and API reference.

## [Unreleased]

### Documentation
- Added a callable fixed-rate bond case study that keeps a 65-dimensional
  request-level wrapper, uses QLNet's Hull-White tree callable-bond pricer as
  the reference, records naive TT/Slider failure evidence, and compares
  structured factor and embedded-option surrogate trials
  ([#207](https://github.com/0xC000005/ChebyshevSharp/issues/207)).

### Examples
- Added `examples/CallableBondSurrogate`, including the QLNet-backed callable
  baseline, the full-dimensional public wrapper, naive surrogate discovery, and
  structured-alternatives evidence modes
  ([#207](https://github.com/0xC000005/ChebyshevSharp/issues/207)).

## [0.13.3] - 2026-05-13 — High-dimensional documentation examples

This patch release packages high-dimensional documentation improvements for
`ChebyshevSlider` and `ChebyshevTT`. It does not intentionally change public API
or numerical behavior.

### Documentation
- Added `examples/SliderPartitionValidation`, a runnable 8D Slider example that
  compares a correct pairwise partition against an intentionally weak singleton
  partition and shows why held-out validation matters
  ([#185](https://github.com/0xC000005/ChebyshevSharp/issues/185),
  [#188](https://github.com/0xC000005/ChebyshevSharp/pull/188)).
- Expanded the Slider guide with a pivot/slide mental model, a practical
  partition-selection workflow, and held-out validation code for detecting
  missing cross-group interactions.
- Improved the Tensor Train guide with a core-chain mental model, a complete
  first example that builds, evaluates, validates, and prints ranks/compression,
  plus a TT-Cross acceptance checklist and troubleshooting table.

### CI
- Runs the .NET 10 xUnit suite with VSTest parallelization disabled because
  existing verbose-output tests capture `Console.Out`; this avoids
  nondeterministic writer-disposal failures in CI while preserving full test
  coverage.

### Release
- Advanced package metadata to `0.13.3` and updated package validation to
  compare this release against `0.13.2`.

## [0.13.2] - 2026-05-11 — Slider high-dimensional diagnostic overflow fix

This patch release fixes a `ChebyshevSlider` overflow in high-dimensional
partitioned builds and publishes the documentation navigation improvements
merged after `0.13.1`.

### Fixed
- `ChebyshevSlider.Build()` no longer overflows full-grid diagnostic arithmetic
  for high-dimensional partitioned sliders whose per-slide grids are tractable
  ([#182](https://github.com/0xC000005/ChebyshevSharp/issues/182),
  [#183](https://github.com/0xC000005/ChebyshevSharp/pull/183)).
- `ChebyshevSlider.ToString()` now reports oversized full-tensor comparisons
  without throwing for valid slider decompositions
  ([#183](https://github.com/0xC000005/ChebyshevSharp/pull/183)).

### Documentation
- Improved documentation-site navigation by promoting release notes, adding a
  project-links page, exposing a top-level GitHub repository link, and adding
  footer links for GitHub, NuGet, issues, release notes, and contributing
  ([#181](https://github.com/0xC000005/ChebyshevSharp/pull/181)).

### Release
- Advanced package metadata to `0.13.2` and updated package validation to
  compare this release against `0.13.1`.

## [0.13.1] - 2026-05-08 — Documentation audit completion

This patch release packages the post-`0.13.0` documentation audit. It does not
intentionally change public APIs or numerical behavior.

### Documentation
- Completed the post-`0.13.0` public documentation audit across identity and
  provenance framing, citation support, navigation, class selection,
  dense/spline/slider/Tensor Train workflows, algebra and calculus, persistence,
  validation, contributing, API XML comments, and runnable examples
  ([#151](https://github.com/0xC000005/ChebyshevSharp/pull/151),
  [#153](https://github.com/0xC000005/ChebyshevSharp/pull/153),
  [#155](https://github.com/0xC000005/ChebyshevSharp/pull/155),
  [#157](https://github.com/0xC000005/ChebyshevSharp/pull/157),
  [#159](https://github.com/0xC000005/ChebyshevSharp/pull/159),
  [#161](https://github.com/0xC000005/ChebyshevSharp/pull/161),
  [#163](https://github.com/0xC000005/ChebyshevSharp/pull/163),
  [#165](https://github.com/0xC000005/ChebyshevSharp/pull/165),
  [#167](https://github.com/0xC000005/ChebyshevSharp/pull/167),
  [#169](https://github.com/0xC000005/ChebyshevSharp/pull/169),
  [#171](https://github.com/0xC000005/ChebyshevSharp/pull/171),
  [#173](https://github.com/0xC000005/ChebyshevSharp/pull/173),
  [#175](https://github.com/0xC000005/ChebyshevSharp/pull/175)).
- Added a docs-site Examples page and updated the runnable console examples to
  print accuracy checks against their original functions
  ([#175](https://github.com/0xC000005/ChebyshevSharp/pull/175)).
- Added a repository-root changelog entry point so GitHub users can find release
  history without first opening the documentation site
  ([#176](https://github.com/0xC000005/ChebyshevSharp/issues/176)).
- Finalized the persistent documentation-audit ledger so future audit cycles
  know that the current public documentation queue has been covered and should
  open a new issue for newly added or materially changed docs surfaces
  ([#179](https://github.com/0xC000005/ChebyshevSharp/pull/179)).

### Release
- Advanced package metadata to `0.13.1` and updated package validation to compare
  this release against `0.13.0`.

## [0.13.0] - 2026-05-08 — Audit-driven correctness and validation hardening

This release collects the post-0.12.0 audit fixes. It is primarily a
backward-compatible correctness release, but users may see stricter exceptions
for invalid inputs that were previously accepted, extrapolated, or surfaced as
implementation-level exceptions.

The summary sections below describe user-visible impact. The traceability
inventories at the end of this release section cover the 83 audit pull requests
merged after `v0.12.0`, this release-preparation pull request, and the 35 issues
closed in the post-0.12.0 release window.

### Upgrade Notes
- No public APIs were intentionally removed in this release.
- Invalid domains, dimensions, scalar controls, derivative orders, malformed
  load data, and out-of-domain evaluation points may now throw earlier with
  public argument exceptions instead of producing extrapolated values,
  inconsistent state, or lower-level runtime exceptions.

### Changed
- Public evaluation now rejects out-of-domain points across dense, spline,
  slider, and Tensor Train interpolants instead of silently extrapolating
  ([#74](https://github.com/0xC000005/ChebyshevSharp/pull/74),
  [#76](https://github.com/0xC000005/ChebyshevSharp/pull/76)).
- Public scalar algebra now rejects `NaN`, infinity, zero divisors, and null
  operands consistently before producing invalid interpolant state
  ([#129](https://github.com/0xC000005/ChebyshevSharp/pull/129),
  [#131](https://github.com/0xC000005/ChebyshevSharp/pull/131)).
- Runtime evaluation, derivative registries, and serialized derivative metadata
  now enforce configured `maxDerivativeOrder` bounds
  ([#123](https://github.com/0xC000005/ChebyshevSharp/pull/123),
  [#125](https://github.com/0xC000005/ChebyshevSharp/pull/125)).
- NuGet package validation now compares this release against `0.12.0`.

### Fixed
- Adaptive auto-N construction now validates off-grid convergence more robustly,
  rejects non-finite validation samples, and commits rebuilt approximation state
  only after the full adaptive loop succeeds
  ([#41](https://github.com/0xC000005/ChebyshevSharp/pull/41),
  [#147](https://github.com/0xC000005/ChebyshevSharp/pull/147)).
- Dense, spline, slider, and Tensor Train builders now reject non-finite build
  values and invalid numeric controls earlier, with public argument exceptions
  instead of delayed internal failures.
- `ChebyshevSpline.Build()`, `ChebyshevSlider.Build()`, and adaptive
  `ChebyshevApproximation.Build()` now publish replacement state atomically, so
  failed rebuilds preserve the previous built interpolant
  ([#143](https://github.com/0xC000005/ChebyshevSharp/pull/143),
  [#145](https://github.com/0xC000005/ChebyshevSharp/pull/145),
  [#147](https://github.com/0xC000005/ChebyshevSharp/pull/147)).
- Public state accessors now return snapshots rather than mutable live arrays,
  including domains, node counts, tensor values, special points, cached
  evaluation points, and result wrapper records
  ([#75](https://github.com/0xC000005/ChebyshevSharp/pull/75),
  [#77](https://github.com/0xC000005/ChebyshevSharp/pull/77),
  [#78](https://github.com/0xC000005/ChebyshevSharp/pull/78),
  [#79](https://github.com/0xC000005/ChebyshevSharp/pull/79),
  [#80](https://github.com/0xC000005/ChebyshevSharp/pull/80),
  [#81](https://github.com/0xC000005/ChebyshevSharp/pull/81),
  [#82](https://github.com/0xC000005/ChebyshevSharp/pull/82)).
- JSON and `.pcb` loaders now reject malformed dimensions, tensor lengths,
  oversized binary declarations, invalid spline/slider/TT state, and legacy TT
  dimension metadata before materializing inconsistent objects
  ([#72](https://github.com/0xC000005/ChebyshevSharp/pull/72),
  [#117](https://github.com/0xC000005/ChebyshevSharp/pull/117),
  [#141](https://github.com/0xC000005/ChebyshevSharp/pull/141)).
- `ChebyshevSpline.SobolIndices()` now includes between-piece variance and
  interval-membership interactions instead of only aggregating independent
  per-piece Sobol energies ([#84](https://github.com/0xC000005/ChebyshevSharp/pull/84)).
- `ChebyshevSpline.AutoKnots()` now validates public inputs before scanning the
  user function ([#86](https://github.com/0xC000005/ChebyshevSharp/pull/86)).
- Spline calculus and batch evaluation now reject derivative requests at knot
  boundaries and handle jump root detection more consistently.
- Tensor Train validation now rejects zero-dimensional grids, invalid auto-order
  inputs, invalid round tolerances, non-finite build values, and build calls
  without a source function at the public boundary
  ([#121](https://github.com/0xC000005/ChebyshevSharp/pull/121),
  [#127](https://github.com/0xC000005/ChebyshevSharp/pull/127),
  [#135](https://github.com/0xC000005/ChebyshevSharp/pull/135)).
- Tensor Train lifecycle and ownership fixes preserve unbuilt clone state,
  isolate extrusion/slicing result cores from source cores, and repair reduced
  QR handling for wide matrices
  ([#137](https://github.com/0xC000005/ChebyshevSharp/pull/137),
  [#139](https://github.com/0xC000005/ChebyshevSharp/pull/139)).
- Deferred construction, integration results, slider derived results, fixture
  generation, and legacy JSON migration fixtures now preserve their intended
  construction and metadata contracts.

### Documentation
- README and docs now distinguish value-only evaluation benchmarks from
  price-plus-Greeks benchmarks and record benchmark provenance
  ([#107](https://github.com/0xC000005/ChebyshevSharp/pull/107),
  [#109](https://github.com/0xC000005/ChebyshevSharp/pull/109),
  [#111](https://github.com/0xC000005/ChebyshevSharp/pull/111)).
- Package snippets and current guides now use C# API names and document JSON
  plus portable `.pcb` serialization ([#113](https://github.com/0xC000005/ChebyshevSharp/pull/113)).
- DocFX API metadata is generated into ignored build output instead of tracked
  source files, preventing local source-path churn during docs verification
  ([#88](https://github.com/0xC000005/ChebyshevSharp/pull/88)).

### CI and Security
- GitHub Actions are pinned to full-length commit SHAs, DocFX and Stryker tool
  versions are pinned, and the SDK matrix explicitly selects the requested SDK
  ([#90](https://github.com/0xC000005/ChebyshevSharp/pull/90),
  [#103](https://github.com/0xC000005/ChebyshevSharp/pull/103),
  [#105](https://github.com/0xC000005/ChebyshevSharp/pull/105)).
- CI now audits transitive NuGet dependencies across target frameworks and
  keeps main branch merges behind required test and Codecov patch gates
  ([#115](https://github.com/0xC000005/ChebyshevSharp/pull/115)).

### Complete PR Inventory
- Release metadata and changelog:
  [#148](https://github.com/0xC000005/ChebyshevSharp/pull/148).
- Documentation, community, and benchmarks:
  [#30](https://github.com/0xC000005/ChebyshevSharp/pull/30),
  [#31](https://github.com/0xC000005/ChebyshevSharp/pull/31),
  [#32](https://github.com/0xC000005/ChebyshevSharp/pull/32),
  [#44](https://github.com/0xC000005/ChebyshevSharp/pull/44),
  [#88](https://github.com/0xC000005/ChebyshevSharp/pull/88),
  [#101](https://github.com/0xC000005/ChebyshevSharp/pull/101),
  [#107](https://github.com/0xC000005/ChebyshevSharp/pull/107),
  [#109](https://github.com/0xC000005/ChebyshevSharp/pull/109),
  [#111](https://github.com/0xC000005/ChebyshevSharp/pull/111),
  [#113](https://github.com/0xC000005/ChebyshevSharp/pull/113).
- CI, dependency, package, and security gates:
  [#33](https://github.com/0xC000005/ChebyshevSharp/pull/33),
  [#34](https://github.com/0xC000005/ChebyshevSharp/pull/34),
  [#90](https://github.com/0xC000005/ChebyshevSharp/pull/90),
  [#103](https://github.com/0xC000005/ChebyshevSharp/pull/103),
  [#105](https://github.com/0xC000005/ChebyshevSharp/pull/105),
  [#115](https://github.com/0xC000005/ChebyshevSharp/pull/115),
  [#117](https://github.com/0xC000005/ChebyshevSharp/pull/117).
- `ChebyshevApproximation`, adaptive build, and evaluation validation:
  [#35](https://github.com/0xC000005/ChebyshevSharp/pull/35),
  [#38](https://github.com/0xC000005/ChebyshevSharp/pull/38),
  [#41](https://github.com/0xC000005/ChebyshevSharp/pull/41),
  [#43](https://github.com/0xC000005/ChebyshevSharp/pull/43),
  [#45](https://github.com/0xC000005/ChebyshevSharp/pull/45),
  [#46](https://github.com/0xC000005/ChebyshevSharp/pull/46),
  [#62](https://github.com/0xC000005/ChebyshevSharp/pull/62),
  [#63](https://github.com/0xC000005/ChebyshevSharp/pull/63),
  [#64](https://github.com/0xC000005/ChebyshevSharp/pull/64),
  [#71](https://github.com/0xC000005/ChebyshevSharp/pull/71),
  [#74](https://github.com/0xC000005/ChebyshevSharp/pull/74),
  [#76](https://github.com/0xC000005/ChebyshevSharp/pull/76),
  [#77](https://github.com/0xC000005/ChebyshevSharp/pull/77),
  [#81](https://github.com/0xC000005/ChebyshevSharp/pull/81),
  [#147](https://github.com/0xC000005/ChebyshevSharp/pull/147).
- Core numerical tests and fixture provenance:
  [#36](https://github.com/0xC000005/ChebyshevSharp/pull/36),
  [#37](https://github.com/0xC000005/ChebyshevSharp/pull/37),
  [#39](https://github.com/0xC000005/ChebyshevSharp/pull/39),
  [#40](https://github.com/0xC000005/ChebyshevSharp/pull/40),
  [#93](https://github.com/0xC000005/ChebyshevSharp/pull/93),
  [#95](https://github.com/0xC000005/ChebyshevSharp/pull/95),
  [#97](https://github.com/0xC000005/ChebyshevSharp/pull/97),
  [#99](https://github.com/0xC000005/ChebyshevSharp/pull/99).
- `ChebyshevSpline` validation, calculus, Sobol, and rebuild lifecycle:
  [#47](https://github.com/0xC000005/ChebyshevSharp/pull/47),
  [#52](https://github.com/0xC000005/ChebyshevSharp/pull/52),
  [#53](https://github.com/0xC000005/ChebyshevSharp/pull/53),
  [#54](https://github.com/0xC000005/ChebyshevSharp/pull/54),
  [#55](https://github.com/0xC000005/ChebyshevSharp/pull/55),
  [#56](https://github.com/0xC000005/ChebyshevSharp/pull/56),
  [#57](https://github.com/0xC000005/ChebyshevSharp/pull/57),
  [#59](https://github.com/0xC000005/ChebyshevSharp/pull/59),
  [#60](https://github.com/0xC000005/ChebyshevSharp/pull/60),
  [#78](https://github.com/0xC000005/ChebyshevSharp/pull/78),
  [#84](https://github.com/0xC000005/ChebyshevSharp/pull/84),
  [#86](https://github.com/0xC000005/ChebyshevSharp/pull/86),
  [#143](https://github.com/0xC000005/ChebyshevSharp/pull/143).
- `ChebyshevSlider` validation, derived results, ownership, and rebuild lifecycle:
  [#48](https://github.com/0xC000005/ChebyshevSharp/pull/48),
  [#51](https://github.com/0xC000005/ChebyshevSharp/pull/51),
  [#68](https://github.com/0xC000005/ChebyshevSharp/pull/68),
  [#69](https://github.com/0xC000005/ChebyshevSharp/pull/69),
  [#70](https://github.com/0xC000005/ChebyshevSharp/pull/70),
  [#79](https://github.com/0xC000005/ChebyshevSharp/pull/79),
  [#145](https://github.com/0xC000005/ChebyshevSharp/pull/145).
- `ChebyshevTT` validation, rank controls, load state, clone state, and core ownership:
  [#42](https://github.com/0xC000005/ChebyshevSharp/pull/42),
  [#49](https://github.com/0xC000005/ChebyshevSharp/pull/49),
  [#50](https://github.com/0xC000005/ChebyshevSharp/pull/50),
  [#65](https://github.com/0xC000005/ChebyshevSharp/pull/65),
  [#66](https://github.com/0xC000005/ChebyshevSharp/pull/66),
  [#67](https://github.com/0xC000005/ChebyshevSharp/pull/67),
  [#75](https://github.com/0xC000005/ChebyshevSharp/pull/75),
  [#121](https://github.com/0xC000005/ChebyshevSharp/pull/121),
  [#127](https://github.com/0xC000005/ChebyshevSharp/pull/127),
  [#135](https://github.com/0xC000005/ChebyshevSharp/pull/135),
  [#137](https://github.com/0xC000005/ChebyshevSharp/pull/137),
  [#139](https://github.com/0xC000005/ChebyshevSharp/pull/139),
  [#141](https://github.com/0xC000005/ChebyshevSharp/pull/141).
- Shared public validation, file boundaries, result wrappers, and algebra contracts:
  [#58](https://github.com/0xC000005/ChebyshevSharp/pull/58),
  [#72](https://github.com/0xC000005/ChebyshevSharp/pull/72),
  [#80](https://github.com/0xC000005/ChebyshevSharp/pull/80),
  [#82](https://github.com/0xC000005/ChebyshevSharp/pull/82),
  [#119](https://github.com/0xC000005/ChebyshevSharp/pull/119),
  [#123](https://github.com/0xC000005/ChebyshevSharp/pull/123),
  [#125](https://github.com/0xC000005/ChebyshevSharp/pull/125),
  [#129](https://github.com/0xC000005/ChebyshevSharp/pull/129),
  [#131](https://github.com/0xC000005/ChebyshevSharp/pull/131),
  [#133](https://github.com/0xC000005/ChebyshevSharp/pull/133).

### Resolved Issue Inventory
- Public evaluation and ownership audit:
  [#61](https://github.com/0xC000005/ChebyshevSharp/issues/61),
  [#73](https://github.com/0xC000005/ChebyshevSharp/issues/73).
- Spline correctness and validation:
  [#83](https://github.com/0xC000005/ChebyshevSharp/issues/83),
  [#85](https://github.com/0xC000005/ChebyshevSharp/issues/85),
  [#142](https://github.com/0xC000005/ChebyshevSharp/issues/142).
- Documentation, CI, security, package, and branch-protection work:
  [#87](https://github.com/0xC000005/ChebyshevSharp/issues/87),
  [#89](https://github.com/0xC000005/ChebyshevSharp/issues/89),
  [#91](https://github.com/0xC000005/ChebyshevSharp/issues/91),
  [#102](https://github.com/0xC000005/ChebyshevSharp/issues/102),
  [#104](https://github.com/0xC000005/ChebyshevSharp/issues/104),
  [#106](https://github.com/0xC000005/ChebyshevSharp/issues/106),
  [#108](https://github.com/0xC000005/ChebyshevSharp/issues/108),
  [#110](https://github.com/0xC000005/ChebyshevSharp/issues/110),
  [#112](https://github.com/0xC000005/ChebyshevSharp/issues/112),
  [#114](https://github.com/0xC000005/ChebyshevSharp/issues/114),
  [#116](https://github.com/0xC000005/ChebyshevSharp/issues/116).
- Test precision, fixture provenance, and stale tooling:
  [#92](https://github.com/0xC000005/ChebyshevSharp/issues/92),
  [#94](https://github.com/0xC000005/ChebyshevSharp/issues/94),
  [#96](https://github.com/0xC000005/ChebyshevSharp/issues/96),
  [#98](https://github.com/0xC000005/ChebyshevSharp/issues/98),
  [#100](https://github.com/0xC000005/ChebyshevSharp/issues/100).
- Shared validation, file boundaries, derivative bounds, and algebra contracts:
  [#118](https://github.com/0xC000005/ChebyshevSharp/issues/118),
  [#122](https://github.com/0xC000005/ChebyshevSharp/issues/122),
  [#124](https://github.com/0xC000005/ChebyshevSharp/issues/124),
  [#128](https://github.com/0xC000005/ChebyshevSharp/issues/128),
  [#130](https://github.com/0xC000005/ChebyshevSharp/issues/130),
  [#132](https://github.com/0xC000005/ChebyshevSharp/issues/132).
- Tensor Train validation, rounding, lifecycle, load-state, and ownership:
  [#120](https://github.com/0xC000005/ChebyshevSharp/issues/120),
  [#126](https://github.com/0xC000005/ChebyshevSharp/issues/126),
  [#134](https://github.com/0xC000005/ChebyshevSharp/issues/134),
  [#136](https://github.com/0xC000005/ChebyshevSharp/issues/136),
  [#138](https://github.com/0xC000005/ChebyshevSharp/issues/138),
  [#140](https://github.com/0xC000005/ChebyshevSharp/issues/140).
- Exception-safe rebuild lifecycle:
  [#144](https://github.com/0xC000005/ChebyshevSharp/issues/144),
  [#146](https://github.com/0xC000005/ChebyshevSharp/issues/146).

## [0.12.0] - 2026-05-02 — Quality and documentation hardening ([#25](https://github.com/0xC000005/ChebyshevSharp/pull/25), [#26](https://github.com/0xC000005/ChebyshevSharp/pull/26), [#27](https://github.com/0xC000005/ChebyshevSharp/pull/27), [#28](https://github.com/0xC000005/ChebyshevSharp/pull/28))

### Added
- Shared tensor-shape arithmetic guards now cover dense construction, serialization, slicing/extrusion, axis kernels, slider diagnostics, and Tensor Train materialization paths.
- Deterministic FsCheck property tests cover batch-vs-loop evaluation, save/load preservation, extrude/slice recovery, algebra identities, and TT reorder preservation.
- Stryker.NET mutation testing configuration plus a scheduled/manual mutation workflow target high-risk numerical files.
- Runnable examples: `examples/QuickStart` and `examples/TensorTrainHighDim`.
- Documentation pages for class selection, testing and validation, and citations.
- Root `CONTRIBUTING.md`, docs-site contributor guide, and GitHub PR template with coverage and mutation-testing expectations.

### Changed
- CI now validates formatting, analyzer build, package validation, tests with coverage, Codecov patch coverage, and DocFX documentation.
- User documentation is organized around tutorials, how-to guides, concepts, and reference material.
- README and package metadata now reflect PyChebyshev v0.21.1 parity and available `ChebyshevTT` functionality.
- Mathematical docs now state the Type I node convention, explain how it differs from MoCaX/Ruiz--Zeron Lobatto grids, and clarify the Chebyshev-weighted meaning of `SobolIndices()`.
- `SobolIndices()` now uses a scale-relative numerical-noise floor instead of a fixed absolute variance cutoff, preserving legitimate low-amplitude sensitivity signals.
- `ChebyshevTT.SobolIndices()` now contracts nonconstant coefficient energy directly instead of subtracting nearly equal total and constant energies, avoiding cancellation for functions such as `1 + eps * x`.

### Fixed
- Tensor-size and byte-size products now fail fast with clear overflow errors instead of wrapping fixed-width arithmetic.
- Tensor Train documentation now warns that TT-Cross can be feasible while dense-grid materialization is intentionally guarded.
- Stale documentation claims about Slider calculus and TT extrusion/slicing/arithmetic availability were corrected.

## [0.11.1] - 2026-05-01 — TT high-dimensional overflow fix ([#23](https://github.com/0xC000005/ChebyshevSharp/issues/23))

### Fixed
- `ChebyshevTT.Build(method: "cross")` no longer overflows rank-cap arithmetic or cache keys for high-dimensional grids such as `N=35`, `n=7`.
- `ChebyshevTT` full-grid diagnostics no longer wrap fixed-width integer products for large tensor shapes.
- Full-grid materialization paths now fail with clear overflow errors when the dense Chebyshev grid is too large.

### CI
- Added Codecov patch coverage policy for PRs.

## [0.11.0] - 2026-04-29 — Phase 7: catch-up to PyChebyshev v0.21.1

Bundles upstream PyChebyshev v0.21.0 + v0.21.1 into one cohesive post-port maintenance
release. Parity tag advances 0.20.1 → 0.21.1.

### Added
- `ChebyshevSlider.Roots(dim, fixedDims)` / `Minimize` / `Maximize`
- `ChebyshevTT.Roots(dim, fixedDims)` / `Minimize` / `Maximize`
- `ChebyshevTT.SobolIndices()` — TT-native O(d·n·r²), no dense materialization

### Fixed
- `ChebyshevTT.GetEvaluationPoints` returns user-frame columns (was storage-frame)
- `ChebyshevTT.EvalMulti` race condition: refactored to use non-mutating `EvalStorageFrame` helper
- `ChebyshevTT.InnerProduct` now throws `ArgumentException` on `_dimOrder` mismatch (was silent wrong result)
- `ChebyshevTT.Integrate` error messages reference user-frame dim indices (was storage-frame)
- `Algebra.CheckCompatible` uses numerical tolerance (`np.allclose`-style) on `Domain[d]` (was exact `SequenceEqual`)

### Performance
- `VectorizedEvalBatch` hoists differentiation-matrix matmul outside per-point loop for non-zero derivative orders
- `Calculus.Optimize1D` uses single vectorized barycentric evaluation over all candidates

### Test count: 1103 → ~1113 (+10)

Phase 7 coverage gap fillers: 10 tests in `Phase7CoverageGapTests.cs` (defensive paths, edge cases).

## [0.10.0] - 2026-04-29 — PyChebyshev parity v0.20.1

### Build performance (from PyChebyshev v0.19.0)

- `nWorkers` ctor kwarg on all four classes (`int?`): `null` (default) = sequential; `-1` = `Environment.ProcessorCount`; positive int = pool size.
- `IProgress<int>` ctor kwarg on all four classes: per-evaluation in Approx/Spline/Slider; per-sweep in TT.
- Thread-safety contract: when `nWorkers` is non-null, the user-supplied function may be invoked concurrently from multiple threads. Functions that capture mutable state must use locks or external synchronization, or pass `nWorkers: null`.
- TT accepts `nWorkers` for API symmetry but ignores it: TT-Cross is adaptive sampling, not pre-grid evaluation.

### Adaptive refinement (from PyChebyshev v0.20.0 + v0.20.1)

- `ChebyshevSpline.AutoKnots(...)` static factory: auto-place knots at function kinks via curvature-spike scan. Defaults: `thresholdFactor=5.0`, `maxKnotsPerDim=5`, `nScanPoints=200`.
- `SobolIndices()` instance method on `ChebyshevApproximation` and `ChebyshevSpline`: variance decomposition from spectral Chebyshev coefficients. Returns new `SobolResult` record (`FirstOrder`, `TotalOrder`, `Variance`). No Monte Carlo, no extra evaluations.
- `ChebyshevTT.WithAutoOrder(...)` static factory: heuristic dim ordering to minimize TT rank. Methods: `"greedy_swap"` (deterministic) and `"random"` (seeded).
- `ChebyshevTT.Reorder(newOrder, maxRank?, tolerance?)` instance method: TT-swap-based realignment via adjacent-axis SVDs in coefficient space.
- `ChebyshevTT.DimOrder` read-only property: storage permutation, identity by default.
- All TT public methods now thread `_dimOrder` correctly: `Eval`, `EvalBatch`, `EvalMulti`, `Slice`, `Extrude`, `ToDense`, partial `Integrate`, unary algebra. Binary algebra (`+`, `-`) requires matching `_dimOrder` and throws `ArgumentException` with a hint at `Reorder` on mismatch.

### JSON migration

- `ChebyshevTT` save format bumped to `"jsonVersion": 2` with new `"dimOrder"` field.
- v0.9.0 and earlier files load with identity `dimOrder` backfilled.

### Skipped (Python-only ergonomic features)

- `plot_convergence`, `plot_1d`, `plot_2d_surface`, `plot_2d_contour` (matplotlib helpers). Documented under "Python-only ergonomic features."

### Internal

- New `Internal/ParallelBuild.cs` — `NormalizeNWorkers`, `EvaluateInParallel`.
- New `Internal/Sensitivity.cs` — `ChebyshevCoefficientsND`, `ComputeSobolFromCoeffs`.
- `Internal/TensorTrainAlgebra.cs` extended with `TtSwapAdjacent` (coefficient-space adjacent-axis swap via SVD).

### Submodule and parity

- Submodule `ref/PyChebyshev` bumped from v0.18.0 to v0.20.1 (15 commits).
- `<PyChebyshevParity>` advances 0.17.0 → 0.20.1 (Phase 5's deliberate non-monotonic drop is now corrected forward; Phase 6 ships everything between v0.18 and v0.20.1).
- This is the **final phase** of the v0.20.1 port: ChebyshevSharp is feature-complete against PyChebyshev v0.20.1 modulo the deliberately skipped matplotlib plotting helpers.

### Test count: 946 → 1018 (+72)

Phase 6 fan-out: 17 tests in `BuildPerfTests.cs` (nWorkers normalization + parallel parity + progress reporting), 12 tests in `SobolIndicesTests.cs` (variance decomposition), 10 tests in `AutoKnotsTests.cs` (knot discovery), 18 tests in `TtAutoOrderTests.cs` (heuristic dim reordering + reorder + JSON migration), 11 tests in `TtDimOrderTests.cs` (full _dimOrder threading surface + binary algebra type checking + identity shortcut); 4 regression tests for hardened integration bounds + other edge cases.

See [PR #21](https://github.com/0xC000005/ChebyshevSharp/pull/21) for the full diff.

## [0.9.0] - 2026-04-28

### Added — Integrate Everywhere

- `ChebyshevSlider.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial integration via the closed-form sliding-decomposition. Returns a scalar (boxed in `object`) on full integration; a `ChebyshevSlider` over surviving dims on partial integration.
- `ChebyshevTT.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial integration via Fejér-1 quadrature contraction into each integrated core's node axis (after coefficient→value core conversion). Works for all three TT build methods (`cross`, `svd`, `als`).
- New internal helpers in `Internal/Calculus.cs`: `SliderPartitionIntersect`, `IntegrateTtAlongDim`.

After v0.9.0, all four ChebyshevSharp classes support integration — matching PyChebyshev v0.17.0 parity. Roots / Min / Max on Slider and TT remain deferred (Python defers to v0.21).

### PyChebyshev parity tracking

- `<PyChebyshevParity>` tag drops 0.18.0 → 0.17.0. This looks like a regression but is not — the parity tag is a non-monotonic indicator of "the most recent feature batch we ported." Phase 4 (v0.8.0) filled in v0.15+v0.16 features behind the v0.18.0 binary format that was already shipped, so the tag stayed at 0.18.0. Phase 5 (this release) ports v0.17.0 calculus completion features; the tag drops to 0.17.0 to indicate which batch was just delivered. Phase 6 will advance to v0.20.1.
- The `<Version>` (release-engineering version) advances monotonically: 0.8.0 → 0.9.0.

### Test count: 902 → 946 (+44)

Phase 5 fan-out: 17 tests in `SliderIntegrateTests.cs` (full + partial + validation + ergonomics), 22 tests in `TtIntegrateTests.cs` (full + partial + validation + cross-class build-mode preservation), 5 helper tests appended to `CalculusTests.cs`.

See [PR #20](https://github.com/0xC000005/ChebyshevSharp/pull/20) for the full diff.

## [0.8.0] - 2026-04-28 — Ergonomics polish (PyChebyshev v0.15+v0.16 fill-in)

> **PyChebyshev parity stays at v0.18.0.** Phase 4 of the v0.20.1 phased port —
> backfills the v0.15+v0.16 ergonomics layer that was skipped during the initial
> port. Same pattern as Phase 3's binary `.pcb` format fill-in.

### Added — descriptor, additional data, registry, introspection

- `SetDescriptor(string)` / `GetDescriptor()` on all four classes — free-form
  text labels that survive Save/Load.
- `additionalData` constructor kwarg (`object?`) on all four classes. Threads
  user-supplied context through every `f(point, data)` call during
  `ChebyshevApproximation`/`ChebyshevSpline`/`ChebyshevSlider` build. Stored
  for introspection on `ChebyshevTT` (its function signature has no data arg).
- `GetDerivativeId(int[] orders)` registry on all four classes — returns a
  stable session-local int per registered orders tuple. New
  `Eval(double[] point, int derivativeId)` overload looks up the orders.
- `IsConstructionFinished()`, `GetConstructorType()`, `GetUsedNs()` on all
  four classes — runtime introspection of build state.

### Added — clone, accessors

- Typed `Clone()` per class (`ChebyshevApproximation Clone()`,
  `ChebyshevSpline Clone()`, etc.). Returns deep copy with `Function = null`
  (matches Save/Load convention).
- `GetMaxDerivativeOrder()` on all four classes.
- `GetErrorThreshold()`, `GetSpecialPoints()` on Approximation + Spline.
- `GetEvaluationPoints()`, `GetNumEvaluationPoints()` on all four classes.

### Added — deferred construction, typed records

- `deferBuild` constructor kwarg (`bool`) + `SetOriginalFunctionValues(double[] values)`
  instance mutator on Approximation + Spline. Construct shell now, populate
  values later. Bit-identical to the `FromValues()` factory.
- `ChebyshevTT` constructor: new `maxDerivativeOrder = 2` keyword-only kwarg.
- New public records `Domain`, `Ns`, `SpecialPoints` with implicit conversions
  to/from raw arrays. Optional ergonomic wrappers.

### JSON migration

- `Load()` for all four classes tolerates missing v0.8.0 fields in pre-v0.8.0
  JSON files. Defaults: `Descriptor = null`, `MaxDerivativeOrder = 2`,
  `SpecialPoints = null`, empty derivative-id registry.
- 4 committed fixture files in `tests/fixtures/json-pre-v080/` prove the
  migration path.
- `additionalData` is **not** serialized.
- Derivative-id registry now persists through Save/Load on all four classes.

### Test count: 812 → 884 (+72)

Phase 4 fan-out across 8 new test files plus appended cross-class tests. See
[PR #19](https://github.com/0xC000005/ChebyshevSharp/pull/19) for the full diff.

Phase 5 (integrate everywhere — Slider/TT integration on calculus,
PyChebyshev parity bump to v0.17.0) is next.

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
- TT-SVD decomposition for deterministic Frobenius-controlled approximation on small grids
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
