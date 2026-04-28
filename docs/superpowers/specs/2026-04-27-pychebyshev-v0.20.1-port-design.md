# ChebyshevSharp v0.10.1 → v0.20.1 Phased Port — Design

**Date:** 2026-04-27
**Author:** Max Zhang (with Claude)
**Status:** Spec — pending user approval before implementation plan

---

## Context

ChebyshevSharp v0.4.0 is feature-complete against PyChebyshev v0.10.1. Upstream PyChebyshev has since shipped 10 minor releases (v0.11.0 → v0.20.1), 144 commits ahead, comprising error-driven construction, TT feature parity, binary serialization, ergonomics, integration on Slider/TT, parallel build, plotting helpers, adaptive refinement, and cross-language interop.

This document specifies a phased port to bring ChebyshevSharp to v0.20.1 parity.

## Decisions

### 1. Scope filter (Q1: option C)

| Python feature | C# treatment |
|---|---|
| `n_workers=` parallel build (`ProcessPoolExecutor`) | **Port** as `Parallel.For` over the Cartesian grid (no pickling, shared memory) |
| `verbose=2` tqdm progress bars | **Port** as `IProgress<int>` callback (idiomatic .NET) |
| `plot_convergence`, `plot_1d`, `plot_2d_surface`, `plot_2d_contour` (matplotlib) | **Skip.** Not idiomatic for a numerical .NET library; users compose ScottPlot/OxyPlot directly with raw arrays |

The C# API surface diverges from Python here. Document under "Python-only ergonomic features" in the changelog.

### 2. Parity testing strategy (Q2: option B — T1 + T2)

Three tiers exist; we adopt the first two:

| Tier | Mechanism | Status |
|---|---|---|
| **T1: Inline reference values** | Hand-curated Python output literals embedded in C# `[Theory]` data | **Active** (current pattern in `TensorTrainCrossValidationTests.cs`); expand each phase |
| **T2: Shared `.pcb` fixtures** | C# tests load the same `tests/fixtures/*.pcb` files as Python/Rust/Julia readers | **Adopted from phase 4**, when the v0.14 reader lands |
| **T3: MoCaX C cross-validation** | P/Invoke MoCaX 4.3.1, side-by-side comparison | **Skipped.** Transitive parity (PyChebyshev↔MoCaX upstream + ChebyshevSharp↔PyChebyshev here) is sufficient evidence; documented in `MOCAX_PARITY.md` |

### 3. Phasing strategy (Q3: thematic 6-phase)

Six thematic phases. Bundling rationale per phase below — each bundle groups Python tags that touch the same code or are conceptually inseparable:

- **Phase 1 = v0.11 + v0.12** — both modify the `ChebyshevApproximation`/`ChebyshevSpline` constructor signatures around adaptive node placement (`error_threshold`, `max_n`, `special_points`, per-piece `n_nodes`). Touching the same ctor twice is wasteful.
- **Phase 2 = v0.13 + v0.18** — TT feature parity. v0.18's algebra reuses the canonicalization helpers introduced in v0.13's `OrthLeft`/`OrthRight`; splitting forces redundant scaffolding or a v0.13 release with code v0.18 will rewrite.
- **Phase 3 = v0.14** — Binary `.pcb` format. Standalone, big enough alone, and unlocks T2 fixture infrastructure for all subsequent phases.
- **Phase 4 = v0.15 + v0.16** — Ergonomics. Both add small additive accessors/getters across all four classes; share JSON migration plumbing.
- **Phase 5 = v0.17** — Slider/TT integration. Clean conceptual unit, no bundling needed.
- **Phase 6 = v0.19 + v0.20.0 + v0.20.1** — Post-parity polish. v0.20.0 ships `NotImplementedError` stubs that v0.20.1 fixes (must bundle); v0.19 (parallel build/progress) joins because it's the same "post-feature-completeness performance/UX" theme and lands upstream in the same window. Plotting from v0.19 is filtered out per Q1.

(Python v0.13.1 is a pure dependency bump and is not ported.)

### 4. Release cadence + version tracking (Q4: option A)

**NuGet release per phase, natural C# version increments. Parity is surfaced as metadata.**

- ChebyshevSharp keeps its own version line, continuing naturally from current `v0.4.0`. Each phase bumps the next minor: phase 1 → `v0.5.0`, phase 2 → `v0.6.0`, …, phase 6 → `v0.10.0`.
- PyChebyshev parity is tracked through a new MSBuild property in `src/ChebyshevSharp/ChebyshevSharp.csproj`:
  ```xml
  <PyChebyshevParity>0.11.0</PyChebyshevParity>
  ```
  Surfaced in:
  - **README badge** — `![PyChebyshev parity](https://img.shields.io/badge/PyChebyshev_parity-0.11.0-blue)` (regenerated per release).
  - **Package `<Description>`** — auto-generated to include "feature parity with PyChebyshev v0.X.Y".
  - **Assembly `InformationalVersion`** — `0.5.0+pychebyshev.0.11.0` (visible in `nuget package details` and runtime reflection; does not affect SemVer ordering since `+build_metadata` is ignored by version comparators).
  - **`docs/docs/changelog.md`** — every entry leads with the parity claim.
- No `v1.0.0` reservation in this spec. The major bump happens whenever the API is declared stable (likely once PyChebyshev v1.0.0 lands upstream and we mirror it). Phase 2's TT parity completion is a changelog note only.

Rationale: existing NuGet consumers (`v0.1.0`–`v0.4.0`) see continuous SemVer history; ports/bindings conventionally track upstream via metadata, not the package version itself (e.g., NumSharp ↔ NumPy).

### 5. Target version (Q5: option A)

Chase upstream `main`. PyChebyshev v1.0.0 is in progress upstream — when it lands, evaluate as a separate effort (additive features land as phase 7; breaking changes restart the planning loop).

---

## Phase Overview

| # | C# Version | PyChebyshev Parity | Theme | Tests Added (est.) |
|---|---|---|---|---|
| 1 | 0.5.0 | v0.11.0 + v0.12.0 | **Adaptive constructor** — `error_threshold`, `max_n`, auto-N doubling loop, `GetOptimalN1()`, `special_points` dispatch, per-sub-interval nested `n_nodes` | ~45 |
| 2 | 0.6.0 | v0.13.0 + v0.18.0 | **TT feature parity** — ALS + completion + `InnerProduct` + `OrthLeft`/`OrthRight`; `Nodes()`, `FromValues()`, `Extrude()`, `Slice()`, `ToDense()`; full TT algebra (`+ - * /` scalar + in-place) | ~75 |
| 3 | 0.7.0 | v0.14.0 | **Binary `.pcb` format** + reader/writer + `Save(format=...)` + `PeekFormatVersion`; **T2 fixture infrastructure unlocked** | ~30 |
| 4 | 0.8.0 | v0.15.0 + v0.16.0 | **Ergonomics** — `additional_data`, `Set/GetDescriptor`, `GetDerivativeId` registry, introspection getters, `Clone()`, `defer_build`, typed helpers (`Domain`/`Ns`/`SpecialPoints` records) | ~50 |
| 5 | 0.9.0 | v0.17.0 | **Integrate everywhere** — `Slider.Integrate()` + `TT.Integrate()` (full + partial) | ~25 |
| 6 | 0.10.0 | v0.19.0 (filtered) + v0.20.0 + v0.20.1 | **Build perf + adaptive refinement** — `Parallel.For` build, `IProgress<int>` progress; `Spline.AutoKnots`, `SobolIndices`, TT `WithAutoOrder` + `Reorder` + full `_dim_order` threading. **Plotting skipped.** | ~65 |

**Test growth:** 613 → ~900 (Python-parity additions only; C#-specific gap tests carry over).

**No `v1.0.0` reserved in this spec.** The major bump happens later, naturally — likely when PyChebyshev itself ships v1.0.0 upstream and we mirror that API stability claim.

## Plan–Spec Boundary

This spec covers all 6 phases at the design level. Each phase will get its own `writing-plans` pass producing a concrete implementation plan. The expected workflow:

1. Brainstorm produces this spec (6-phase design).
2. `writing-plans` produces an implementation plan for **phase 1 only**.
3. Phase 1 is implemented, tested, released.
4. Loop: re-run `writing-plans` for phase 2, implement, ship, repeat.

The spec is the durable contract; per-phase plans are short-lived execution artifacts. If a phase's actual implementation diverges from this spec, update this spec rather than letting the plan and spec drift.

## Per-Phase Deliverable Shape

Each phase ships with all of the following:

1. **Source port** — internal helpers under `src/ChebyshevSharp/Internal/` matching Python `_*.py` naming where reasonable.
2. **Public API** — XML `<summary>` doc on every new method; correct nullable annotations.
3. **T1 parity tests** — port the Python tests for that version 1:1 (same inputs, same tolerances).
4. **T2 fixture tests** (phase 4 onward) — load fixtures from `tests/fixtures/`, verify exact value match.
5. **Build green** — `dotnet build` with zero warnings; `dotnet test` 100% pass; no `[Skip]` added.
6. **Tracker update** — `skip_csharp.txt` updated with new test counts and which Python file they came from.
7. **CHANGELOG entry** — `docs/docs/changelog.md` documents what shipped, with Python parity mapping.
8. **Submodule advance** — `git -C ref/PyChebyshev checkout v0.X.Y` committed as the phase's first commit.
9. **Parity metadata bump** — `<PyChebyshevParity>0.X.Y</PyChebyshevParity>` updated in `src/ChebyshevSharp/ChebyshevSharp.csproj`; README badge regenerated; package `<Description>` rebuilt to include parity claim.
10. **NuGet release** — `<Version>` bump in `src/ChebyshevSharp/ChebyshevSharp.csproj`, git tag, GitHub release; `publish.yml` ships the package.

## Per-Phase Detail

### Phase 1 — Adaptive Constructor (Python v0.11.0 + v0.12.0)

**C# version:** 0.5.0 (PyChebyshev parity: v0.12.0; bundles upstream v0.11.0 + v0.12.0)
**Python source:** `barycentric.py`, `spline.py`
**New internal:** `Internal/AdaptiveBuild.cs` — doubling loop driver.

**Public API additions:**
- `ChebyshevApproximation` ctor: optional `errorThreshold` (`double?`) and `maxN` (`int?`); `nNodes` accepts `int?[]` (null per dim = auto-N); optional `specialPoints` kwarg (`double[][]?`).
- `static int ChebyshevApproximation.GetOptimalN1(Func<double, double> f, (double a, double b) domain, double errorThreshold, int maxN = 64)`.
- `double? GetErrorThreshold()` accessor (Approx + Spline).
- `ChebyshevSpline` ctor: same `errorThreshold` / `maxN`; `Knots` becomes optional (defaults to empty per dim); `nNodes` accepts nested `int[][]` (per-dim, per-piece).

**Internal restructuring:**
- `Build()` becomes a thin public dispatcher; original logic moves to `BuildFixedGrid()`. New `BuildWithThreshold()` runs the doubling loop.
- New private field `_originalNNodes` (preserves `null` sentinels for re-build).
- `_ErrorEstimatePerDim()` internal helper.

**JSON migration:** `Load()` populates `_originalNNodes` from `nNodes` if absent (pre-v0.5.0 C# JSON files).

**Special-points dispatch — open decision for the implementation plan:** Python's `ChebyshevApproximation(special_points=[[...]])` returns a `ChebyshevSpline`. C# constructors can't return a different type. Two C# idioms to pick from in `writing-plans`:
- `static ChebyshevApproximation.Create(...)` factory that returns the appropriate class via a common interface or `object`-typed return.
- Explicit `ChebyshevSpline.WithSpecialPoints(...)` factory; the Approximation ctor rejects non-empty `specialPoints` with a `throw new ArgumentException` pointing the user at the Spline factory.

**Tests:** ~45 — port `test_error_threshold.py` (~27) + `test_special_points.py` (~15) + `TestErrorEstimatePerDim` (~3). T1 only.

### Phase 2 — TT Feature Parity (Python v0.13.0 + v0.18.0)

**C# version:** 0.6.0 (PyChebyshev parity: v0.18.0; bundles upstream v0.13.0 + v0.18.0)
**Python source:** `tensor_train.py`, `_algebra.py`, `_extrude_slice.py`

**Public API additions on `ChebyshevTT`:**
- `Build(method: "cross" | "svd" | "als", ...)` — extend dispatcher with rank-adaptive ALS.
- `RunCompletion(double tolerance, int maxIter)` — refine existing TT via ALS sweeps.
- `double InnerProduct(ChebyshevTT other)` — exact TT inner product via core-by-core contraction.
- `void OrthLeft(int position)`, `void OrthRight(int position)` — in-place QR/LQ canonicalization.
- `static TtNodes Nodes(int numDim, double[][] domain, int[] nNodes)`.
- `static ChebyshevTT FromValues(double[] tensorValues, int numDim, double[][] domain, int[] nNodes, int? maxRank = null, double tolerance = 1e-6, ...)` — TT-SVD compression skipping TT-Cross.
- `ChebyshevTT Extrude(...)`, `ChebyshevTT Slice(...)`.
- `double[] ToDense()` — materialize via einsum chain.
- Algebra operators: `+`, `-`, `*` (scalar), `/` (scalar), unary `-`; in-place equivalents.

**Internal restructuring:**
- ALS solver added to `Internal/TensorTrainKernel.cs`.
- Split `Internal/TensorTrainKernel.cs` into:
  - `TensorTrainKernel.cs` (build cores: cross/svd/als)
  - `TensorTrainAlgebra.cs` (`+`/`-`/round/inner_product/orth)
  - `TensorTrainExtrude.cs` (extrude/slice/to_dense/from_values via TT-SVD)
- New helpers: `_TtAddCores`, `_TtRoundCores`, `_TtSvdFromTensor`.

**Why bundle v0.13 + v0.18:** v0.18's algebra rounding step relies on left/right canonicalization, which is exactly what v0.13's `OrthLeft`/`OrthRight` provides. Splitting forces v0.13's `OrthLeft`/`OrthRight` to be a public-API hook with no public consumer in v0.13 itself, and v0.18 then has to refactor the canonicalization into algebra-internal helpers anyway. Bundling avoids a round-trip refactor.

**Tests:** ~75 — port `test_tensor_train.py` v0.13 + v0.18 additions; cross-feature tests (algebra+extrude+save/load combos); tolerance-based assertions for stochastic ALS.

### Phase 3 — Binary `.pcb` Format (Python v0.14.0)

**C# version:** 0.7.0 (PyChebyshev parity: v0.14.0)
**Python source:** `barycentric.py`, `spline.py` (Save/Load), `docs/user-guide/binary-format.md` (format spec)

**Public API additions:**
- `Save(string path, string format = "json")` — `format` accepts `"json"` (existing default for backward compat) or `"binary"`.
- `Load(string path)` — auto-detect by 4-byte magic header `b"PCB\x00"` vs JSON.
- `static int PeekFormatVersion(string path)` — read major version byte without full deserialize.

**New internal:** `Internal/PcbFormat.cs` — read/write per the v0.14 spec; uses `BinaryReader`/`BinaryWriter` and explicit little-endian.

**Restrictions (mirror Python):**
- `format="binary"` requires flat (non-nested) `nNodes` for `ChebyshevSpline`. Nested form falls back to JSON with a warning.
- `ChebyshevSlider`/`ChebyshevTT` remain JSON-only in v0.7.0 (matches Python — pickle-only there).

**T2 infrastructure (the unlock for phases 4–6):**
- New `tests/fixtures/` directory; copy the three Python-shipped files (`approx_2d_simple.pcb`, `approx_5d_bs.pcb`, `spline_1d_kink.pcb`).
- New `tests/ChebyshevSharp.Tests/Helpers/PcbFixtures.cs` — fixture loader.
- New `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` — round-trip tests + cross-language fixture-load tests.

**Tests:** ~30. T1 (inline) + T2 (fixture-based).

**CI:** consider new `.github/workflows/parity.yml` that pulls the submodule and re-validates fixtures against Python's test suite (catches drift if Python regenerates fixtures upstream).

### Phase 4 — Ergonomics (Python v0.15.0 + v0.16.0)

**C# version:** 0.8.0 (PyChebyshev parity: v0.16.0; bundles upstream v0.15.0 + v0.16.0)
**Python source:** all four class files

**Public API additions (across all four classes unless noted):**
- `additionalData` ctor kwarg (`object?`) — threaded through every `f(point, data)` call during build.
- `void SetDescriptor(string)`, `string? GetDescriptor()`.
- `int GetDerivativeId(int[] orders)` — stable session-local int per registered orders tuple. Backed by `Dictionary<TupleKey, int>`.
- `Eval(double[] point, int derivativeId)` overload — looks up registered orders by id.
- Introspection: `bool IsConstructionFinished()`, `string GetConstructorType()`, `int[] GetUsedNs()`.
- `T Clone<T>()` — deep copy; `Function` is null on the clone (matches Python).
- `int GetMaxDerivativeOrder()` (all four), `double? GetErrorThreshold()` (Approx/Spline only), `double[][]? GetSpecialPoints()` (Approx/Spline), `double[] GetEvaluationPoints()`, `int GetNumEvaluationPoints()`.
- `bool DeferBuild` ctor kwarg + `void SetOriginalFunctionValues(double[] values)` instance mutator (Approx + Spline). Bit-identical to `FromValues()` factory output.
- Optional typed helpers: `record Domain(double[][] Bounds)`, `record Ns(int[] Counts)`, `record SpecialPoints(double[][] Points)`. Constructor overloads accept either raw arrays or these records.
- `ChebyshevTT` ctor: `maxDerivativeOrder = 2` kwarg.

**JSON migration:** `Load()` backfill for descriptor, additional_data (omitted from binary `.pcb` per Python convention), derivative_id, special_points, max_derivative_order. Pre-v0.8.0 C# JSON files load with sensible defaults.

**Tests:** ~50 — fan-out across class-specific test files; ergonomics-heavy.

### Phase 5 — Integrate Everywhere (Python v0.17.0)

**C# version:** 0.9.0 (PyChebyshev parity: v0.17.0)
**Python source:** `slider.py`, `tensor_train.py`, `_calculus.py`

**Public API additions:**
- `ChebyshevSlider.Integrate(int[]? dims = null, double[][]? bounds = null)` — full (scalar) or partial (returns `ChebyshevSlider`).
- `ChebyshevTT.Integrate(int[]? dims = null, double[][]? bounds = null)` — full (scalar) or partial (returns `ChebyshevTT`). Works for cross/svd/als builds.

**New internal helpers in `Internal/Calculus.cs`:** `_SliderPartitionIntersect()`, `_IntegrateTtAlongDim()`.

**Tests:** ~25.

### Phase 6 — Build Perf + Adaptive Refinement (Python v0.19.0 + v0.20.0 + v0.20.1)

**C# version:** 0.10.0 (PyChebyshev parity: v0.20.1; bundles upstream v0.19.0 + v0.20.0 + v0.20.1; matplotlib plotting filtered out)
**Python source:** `barycentric.py`, `spline.py`, `slider.py`, `tensor_train.py`, `_algebra.py`, `_sensitivity.py`

**Public API additions — build perf (from v0.19):**
- `nWorkers` ctor kwarg (`int?`): null = sequential (default), positive int = `Parallel.For` parallelism level, -1 = `Environment.ProcessorCount`.
- `IProgress<int>?` ctor kwarg (`progress`): when supplied, fires per-evaluation in Approx/Spline/Slider; per-sweep in TT. Replaces Python's `verbose=2` tqdm.

**Public API additions — adaptive refinement (from v0.20.0 + v0.20.1):**
- `static ChebyshevSpline ChebyshevSpline.AutoKnots(Func<double[], double> f, int numDim, double[][] domain, ..., double thresholdFactor = ..., int maxKnotsPerDim = ..., int nScanPoints = ...)` — auto-place knots at function kinks via curvature-spike scan.
- `(double[] firstOrder, double[] totalOrder) SobolIndices()` on `ChebyshevApproximation` and `ChebyshevSpline` — spectral-coefficient sensitivity (no Monte Carlo).
- `static ChebyshevTT WithAutoOrder(Func<double[], double> f, ..., int[]? initialOrder = null, string method = "greedy_swap" | "random")`.
- `ChebyshevTT Reorder(int[] newOrder, int? maxRank = null, double? tolerance = null)` — TT-swap-based realignment.
- `int[] DimOrder { get; }` property on `ChebyshevTT`.
- All TT public methods (`EvalMulti`, `Slice`, `Extrude`, `ToDense`, partial `Integrate`, algebra) thread `_dimOrder` correctly (matches Python's v0.20.1 fix; v0.20.0's `NotImplementedException` stubs go straight to working code in C#).

**Skipped (option C):** `plot_convergence`, `plot_1d`, `plot_2d_surface`, `plot_2d_contour`. Documented under "Python-only ergonomic features" in the changelog.

**Internal:**
- `Internal/Sensitivity.cs` — `_ComputeSobolFromCoeffs()`.
- `_TtSwapAdjacent()` helper in `Internal/TensorTrainAlgebra.cs`.
- TT JSON migration: backfill `dimOrder = [0, 1, ..., n-1]` for pre-v0.10.0 C# JSON files.

**Why bundle v0.19 + v0.20.0 + v0.20.1:** all three are post-feature-completeness work. v0.20.0 ships `NotImplementedError` stubs for TT dim-threading that v0.20.1 fixes — porting v0.20.0 alone would mean shipping a C# release with `NotImplementedException` placeholders that v0.20.1 immediately rewrites. v0.19 (perf + UX) lands in the same upstream window and shares no infrastructure with anything earlier; bundling it here cuts two small releases (~15 + ~50) into one cohesive "post-parity polish" release (~65).

**Cross-language readers note:** v0.20.0 ships Rust+Julia readers consuming the same `tests/fixtures/*.pcb` files. C# already became the third such consumer in phase 3 — no new work, but the changelog should mention "ChebyshevSharp joins Rust+Julia in the cross-language reader club."

**Tests:** ~65 — parallel build (parity-with-sequential), `IProgress` callback counts, `auto_knots`, Sobol indices polynomial-exactness, TT dim_order threading across the full surface.

## Internal Restructuring Summary

| Phase | Files added | Files split |
|---|---|---|
| 1 | `Internal/AdaptiveBuild.cs` | — |
| 2 | `Internal/TensorTrainAlgebra.cs`, `Internal/TensorTrainExtrude.cs` | `TensorTrainKernel.cs` shrinks to build-only |
| 3 | `Internal/PcbFormat.cs`, `tests/Helpers/PcbFixtures.cs`, `tests/BinaryFormatTests.cs`, `tests/fixtures/` | — |
| 4 | `Domain.cs`, `Ns.cs`, `SpecialPoints.cs` (record types) | — |
| 6 | `Internal/Sensitivity.cs` | — |

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Random seed divergence (NumPy `default_rng` vs `System.Random`) blows up bit-exact tests for stochastic outputs (TT-Cross, ALS, `WithAutoOrder` random method) | Continue current pattern: tolerance-based assertions for seeded-stochastic outputs; never inline-literal them. Documented in CLAUDE.md already. |
| `.pcb` reader endianness/alignment bugs | T2 round-trip across all three reference readers (Python, Rust, Julia) plus C# is a strong correctness oracle. Format spec is authoritative. |
| Phase 4 ergonomics JSON migration complexity | Each phase that adds persisted state ships a `Load()` backfill for older C# JSON versions (defaults populate missing fields). We bridge our own format history forward; we do NOT attempt to read Python pickle files. |
| MathNet `Matrix<double>` regression risk creeping back in for new TT algebra paths | `BlasThreshold = 256` rule applies; benchmark every kernel change against `benchmark-results/COMPARISON.md`. Memory captures three failed managed-only attempts. |
| Upstream PyChebyshev moves while we port (v0.21+) | Each phase advances submodule to a pinned tag; new upstream features get tracked in changelog but not chased mid-phase. |
| API surface drift between Python's `ChebyshevApproximation(special_points=...)` returning `ChebyshevSpline` vs C# constructor return-type rules | Resolve in phase 1 implementation plan: factory method `ChebyshevApproximation.Create(...)` vs explicit `ChebyshevSpline.WithSpecialPoints(...)`. |
| Phase 2 splits `TensorTrainKernel.cs` into three files — risk of breaking InternalsVisibleTo consumers (tests, benchmarks) | All tests/benchmarks live in this repo; refactor is local. CI catches breakage. |
| Phase 2 is the largest single phase (~75 tests, two upstream tags' worth of TT work) | If implementation friction is high, split into 2a (v0.13: ALS+orth+inner_product) and 2b (v0.18: extrude/slice/algebra) at writing-plans time, shipping both as v0.6.0 (one combined release, two PR-sized work units). |

## CI/CD Impact

- `test.yml` — no change; T2 fixtures live in repo, not generated in CI.
- `publish.yml` — already triggered by GitHub release; per-phase releases reuse this.
- New `.github/workflows/parity.yml` (optional, phase 3+) — pulls submodule, runs Python's binary-format tests against the same `tests/fixtures/` files we ship. Catches drift if upstream regenerates fixtures.
- `dependabot-automerge.yml` — no change.
- `docs.yml` — no change; new user-guide pages get added to `docs/docs/toc.yml` per phase.

## Documentation Plan

Per phase, add a user-guide page under `docs/docs/`:

| Phase | New / extended doc page |
|---|---|
| 1 | `error-driven-construction.md` (new) + `special-points.md` (new) |
| 2 | extend existing `tensor-train.md` (ALS, orth, algebra, extrude/slice, from_values, to_dense) |
| 3 | `binary-format.md` (new — port from Python's format spec) |
| 4 | `ergonomics.md` (new — descriptor, additional_data, clone, defer_build, typed helpers) |
| 5 | extend existing `calculus.md` (Slider/TT integration) |
| 6 | `parallel-build.md` (new) + `adaptive-refinement.md` (new — auto_knots, Sobol, TT WithAutoOrder) |

Plus a one-time `MOCAX_PARITY.md` documenting our T3 stance.

`docs/docs/toc.yml` updated each phase. API reference is auto-regenerated from XML doc comments.

## What We Explicitly Do NOT Do

- No matplotlib bridge (no `Plot1d`/`Plot2dSurface`/`Plot2dContour`/`PlotConvergence`).
- No MoCaX C P/Invoke (T3).
- No backwards-compat shim for pre-v0.4.0 JSON files.
- No GUI / web / console projects.
- No `fast_eval` / `_jit.py` port (deprecated upstream).

## Definition of Done (whole-port)

- [ ] All 6 phases shipped, NuGet releases v0.5.0 → v0.10.0
- [ ] `dotnet test` ~900 passing across net8.0 + net10.0
- [ ] Submodule pinned at `v0.20.1` or later stable upstream tag
- [ ] `tests/fixtures/*.pcb` cross-loadable by Python, Rust, Julia, C# readers
- [ ] CHANGELOG documents each phase with Python-version mapping
- [ ] `MOCAX_PARITY.md` explains the transitive-parity stance
- [ ] CLAUDE.md updated to reflect new public API surface and any new internal helpers

## Per-Phase Definition of Done

- [ ] Source ported with XML docs and correct nullability
- [ ] T1 parity tests ported 1:1 from Python with same tolerances
- [ ] T2 fixture tests added (phase 3 onward)
- [ ] `dotnet build` zero warnings
- [ ] `dotnet test` 100% pass on net8.0 + net10.0
- [ ] `skip_csharp.txt` updated
- [ ] `docs/docs/changelog.md` entry leads with PyChebyshev parity claim
- [ ] Submodule advanced to phase's Python tag
- [ ] `<Version>` and `<PyChebyshevParity>` bumped in `.csproj`
- [ ] README parity badge regenerated
- [ ] git tag + GitHub release → NuGet ships
