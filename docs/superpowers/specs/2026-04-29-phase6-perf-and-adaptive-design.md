# Phase 6 Design — Build Perf + Adaptive Refinement (ChebyshevSharp v0.10.0)

**Status:** approved (auto-mode `lgtm`'d through Sections 1–6, 2026-04-29)
**Master spec section:** `docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md` lines 229–257
**Upstream parity target:** PyChebyshev v0.20.1 (bundles upstream v0.19.0 + v0.20.0 + v0.20.1)
**Worktree (planned):** `.worktrees/phase6-perf-and-adaptive`
**Test count progression:** 950 → 1017 (+67)

---

## 1. Goal and Scope

Port PyChebyshev v0.19.0 + v0.20.0 + v0.20.1 to ChebyshevSharp as a single
"post-parity polish" release. Two subsystems:

- **Build performance** (from upstream v0.19): `nWorkers` ctor kwarg, `IProgress<int>` ctor kwarg.
- **Adaptive refinement** (from upstream v0.20.0 + v0.20.1): `ChebyshevSpline.AutoKnots`,
  `SobolIndices` on Approx and Spline, `ChebyshevTT.WithAutoOrder` + `Reorder` + `DimOrder`,
  full `_dimOrder` threading through every public TT method.

**Skipped (option C from master spec):** matplotlib plotting helpers
(`plot_convergence`, `plot_1d`, `plot_2d_surface`, `plot_2d_contour`).
Documented in changelog under "Python-only ergonomic features."

**Why bundle v0.19 + v0.20.0 + v0.20.1:** All three are upstream post-feature-completeness
work. Upstream v0.20.0 ships `NotImplementedError` stubs for TT `_dim_order` threading
that v0.20.1 fixes; porting v0.20.0 alone would mean shipping a C# release with
`NotImplementedException` placeholders that the very next release rewrites. v0.19 lands
in the same upstream window and shares no infrastructure with anything earlier — bundling
all three cuts what would be three small releases (~17 + ~50 + ~0) into one cohesive
release (~67 tests).

This is **the final phase** of the v0.20.1 port. After v0.10.0 ships, ChebyshevSharp is
feature-complete against PyChebyshev v0.20.1.

---

## 2. Architecture

### 2.1 New types and files

| File | Type | Purpose |
|---|---|---|
| `src/ChebyshevSharp/SobolResult.cs` | new public record | Return type of `SobolIndices()` |
| `src/ChebyshevSharp/Internal/Sensitivity.cs` | new internal | `ChebyshevCoefficientsND`, `ComputeSobolFromCoeffs` |
| `src/ChebyshevSharp/Internal/ParallelBuild.cs` | new internal | `NormalizeNWorkers`, `EvaluateInParallel` |
| `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` | extended | gains `TtSwapAdjacent` static helper |

### 2.2 Modified public surface (delta vs v0.9.0)

All four ctors gain two ctor-time kwargs:
- `int? nWorkers = null`
- `IProgress<int>? progress = null`

`ChebyshevApproximation` gains:
- instance method `SobolResult SobolIndices()`

`ChebyshevSpline` gains:
- instance method `SobolResult SobolIndices()`
- static factory `AutoKnots(...)`

`ChebyshevTT` gains:
- static factory `WithAutoOrder(...)`
- instance method `Reorder(int[] newOrder, int? maxRank, double? tolerance)`
- read-only property `int[] DimOrder { get; }`
- private field `_dimOrder` (initialized to identity in every existing ctor and factory; backfilled to identity by JSON Load when reading v1 files)
- threading of `_dimOrder` through every public method that takes a coordinate or returns a sub-TT

### 2.3 New `SobolResult` record

```csharp
namespace ChebyshevSharp;

public sealed record SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance);
```

Three fields chosen over a 2-tuple because (a) the variance is genuinely useful for
detecting near-constant functions where Sobol ratios are meaningless, and (b) Phase 4
already established the record-for-result convention (`Domain`, `Ns`, `SpecialPoints`).

### 2.4 What does NOT change

- All existing tests pass unchanged. Identity `_dimOrder` is the default everywhere.
- All existing ctors keep their full v0.9.0 surface — the new kwargs are appended at the end.
- The Phase 5 `Integrate` API is untouched on the surface; partial integrate gains
  `_dimOrder` threading internally (test added for non-identity-order partial integrate).
- The Phase 4 `Clone`, `GetUsedNs`, `GetEvaluationPoints`, descriptor, and ergonomics
  layer all carry through transparently.

---

## 3. Build Performance (Section 2 of dialog)

### 3.1 `nWorkers` semantics (D1)

Python uses `concurrent.futures.ProcessPoolExecutor` with picklability. C# uses
`Parallel.For` with thread-safety contract — true parallelism via thread pool, no GIL,
no IPC overhead. The constraint shifts from "function must be picklable" to "function
must be thread-safe."

| Value | Effect |
|---|---|
| `null` (default) | Sequential evaluation; existing build paths unchanged |
| `-1` | Resolves to `Environment.ProcessorCount` |
| Positive int | `Parallel.For` with `MaxDegreeOfParallelism = that int` |
| `0` | `ArgumentException` |
| `< -1` | `ArgumentException` |

Validation is performed once via `ParallelBuild.NormalizeNWorkers`, which returns
`null` for sequential or a positive int for the effective worker count. Consumers
(`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`) call this once at
ctor time and store the result.

**Thread-safety contract** (XML doc, surfaced on each ctor and on `AutoKnots`):

> When `nWorkers` is non-null, `function` may be invoked concurrently from multiple
> threads. Functions that capture mutable state must use locks or external
> synchronization, or pass `nWorkers: null` (the default).

### 3.2 `IProgress<int>` semantics (D2)

Cumulative count via `Interlocked.Increment`. Caller computes the total upfront
when desired (the Phase 4 `GetNumEvaluationPoints()` getter exposes the total
for Approx/Spline/Slider; for TT, the total equals `maxSweeps`).

```csharp
int done = 0;
Parallel.For(0, points.Length, opts, i => {
    results[i] = function(points[i], additionalData);
    progress?.Report(Interlocked.Increment(ref done));
});
```

`progress: null` is a true no-op — the `?.` skips both the increment and the
report call entirely in the sequential path.

### 3.3 `ParallelBuild` API

```csharp
internal static class ParallelBuild
{
    internal static int? NormalizeNWorkers(int? nWorkers);

    internal static double[] EvaluateInParallel(
        Func<double[], object?, double> function,
        double[][] points,
        object? additionalData,
        int? effectiveWorkers,
        IProgress<int>? progress);
}
```

### 3.4 Per-class wiring

| Class | nWorkers behavior | IProgress granularity |
|---|---|---|
| `ChebyshevApproximation` | `_BuildFixedGrid` flattens tensor grid → `points[]` → `EvaluateInParallel` → reshape | Per grid evaluation |
| `ChebyshevSpline` | Each piece is an Approx; pass `nWorkers`/`progress` to each piece's ctor | Per piece evaluation (sums across pieces) |
| `ChebyshevSlider` | Each slide is an Approx; pass `nWorkers`/`progress` to each slide's ctor | Per slide evaluation (sums across slides) |
| `ChebyshevTT` | `nWorkers` ignored (TT-Cross is adaptive, not on a precomputed grid; documented limitation) | Per sweep (`maxSweeps` increments after a full Cross build) |

For `ChebyshevTT.from_values` / SVD mode, `nWorkers` is also ignored (no per-grid
evaluation phase; the tensor is already supplied by the caller).

---

## 4. Sobol Indices (Section 3 of dialog)

### 4.1 Theory

For a multi-D Chebyshev expansion `f(x) = Σ_α c_α T_α(x)` with multi-index
α = (α₁, …, α_d):

- `⟨T_n, T_n⟩_w = π/2` if n ≥ 1, `π` if n = 0 under weight `w(x) = 1/√(1-x²)` on [-1, 1]
- Multi-D inner product norm² = `∏_i ⟨T_{α_i}, T_{α_i}⟩`
- Total variance = `Σ_{α ≠ 0} c_α² · ‖T_α‖²`
- First-order index for dim d = (variance from terms with `α_d > 0` and all other `α_i = 0`) / total variance
- Total-order index for dim d = (variance from all terms with `α_d > 0`) / total variance

This is exact (no Monte Carlo, no extra evaluations) for any function exactly
representable on the Chebyshev grid.

### 4.2 `Internal/Sensitivity.cs`

```csharp
internal static class Sensitivity
{
    internal static double[] ChebyshevCoefficientsND(double[] tensorValues, int[] shape);
    internal static SobolResult ComputeSobolFromCoeffs(double[] coeffs, int[] shape);

    private static double ChebyshevNormSquared(int n) => n == 0 ? Math.PI : Math.PI / 2;
    private static double MultiIndexNormSquared(int[] alpha);
    private static int[] UnravelIndex(long flat, int[] shape);  // row-major
}
```

`ChebyshevCoefficientsND` reuses Phase 1's `BarycentricKernel.ChebyshevCoefficients1D`
applied along each axis (axis-DCT-II), with the standard halving of `c_0` along
every axis (matches PyChebyshev convention).

`ComputeSobolFromCoeffs` validates that all coefficients are finite (throws
`ArgumentException` otherwise — matches Python's `_compute_sobol_from_coeffs`
NaN/Inf guard); iterates every multi-index via row-major unravel; accumulates
energy into per-dim first-order and total-order buckets; normalizes by total
variance.

For zero-variance (constant function), early-returns `SobolResult` with all
indices = 0 and `Variance = 0`. Caller inspects `Variance` to detect this case.

### 4.3 Public API

```csharp
// On ChebyshevApproximation
public SobolResult SobolIndices()
{
    if (!IsConstructionFinished())
        throw new InvalidOperationException("SobolIndices requires a built ChebyshevApproximation");
    var coeffs = Sensitivity.ChebyshevCoefficientsND(TensorValues, Shape);
    return Sensitivity.ComputeSobolFromCoeffs(coeffs, Shape);
}
```

For `ChebyshevSpline`, aggregation across pieces matches Python `spline.py:735`
exactly. For each piece, compute `vol = ∏ (hi_d - lo_d)` over piece-local domain
and `piece_result = _compute_sobol_from_coeffs(piece_coeffs)`; accumulate
`global_variance += vol × piece_variance` and per-dim
`global_first_order_energy[d] += vol × piece_first_order[d] × piece_variance`
(same for `total_order`). After all pieces, normalize: `FirstOrder[d] = global_first_order_energy[d] / global_variance`,
`TotalOrder[d] = global_total_order_energy[d] / global_variance`. Zero global
variance returns `SobolResult` with all-zero arrays and `Variance = 0` (matches
Python's early return).

This is a **hybrid measure**: per-piece Chebyshev-weighted variance × per-piece
domain volume. For a single-piece (no-knot) Spline, the rule reduces to
the `ChebyshevApproximation` case. Documented as such in `adaptive-refinement.md`.
Tested against analytical references via abs(x) piecewise (1D), abs(x) + abs(y)
(2D), and a single-piece Spline matching the Approximation result on the same
function.

---

## 5. AutoKnots (Section 4 of dialog)

### 5.1 Algorithm

For each dim d:
1. Generate `nScanPoints` evenly-spaced sample points along dim d, with all other dims fixed at the midpoint of their respective domains.
2. Evaluate `f` at each sample point. Reject (throw `ArgumentException`) if any value is non-finite.
3. Compute discrete second derivative `d2[i] = (y[i+1] - 2·y[i] + y[i-1]) / h²` for interior i; pad boundaries with 0.
4. Compute `mean_d2 = mean(|d2|)`. Mark spikes where `|d2[i]| > thresholdFactor × mean_d2`.
5. Cluster spikes within `cluster_radius = max(1, nScanPoints / (maxKnotsPerDim × 4))` indices; keep the index with max `|d2|` per cluster.
6. Sort clusters by `|d2|` descending; cap at `maxKnotsPerDim`.
7. Convert sample-indices to domain coordinates → list of knots for dim d.

After scanning all dims, pass the discovered knots into the existing `ChebyshevSpline`
ctor's `specialPoints` pipeline (Phase 1 already handles knot insertion).

### 5.2 Public signature

```csharp
public static ChebyshevSpline AutoKnots(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] numNodes,
    int maxOrderDerivative = 2,
    object? additionalData = null,
    string? descriptor = null,
    double thresholdFactor = 5.0,
    int maxKnotsPerDim = 5,
    int nScanPoints = 200,
    int? nWorkers = null,
    IProgress<int>? progress = null,
    bool verbose = false)
```

Defaults match Python exactly: `thresholdFactor = 5.0`, `maxKnotsPerDim = 5`,
`nScanPoints = 200`.

### 5.3 Validation

- `thresholdFactor > 0` (else throw)
- `maxKnotsPerDim >= 0` (0 = no auto-knots, degenerate but valid; matches Python)
- `nScanPoints >= 3` (else can't compute 2nd-derivative finite difference)
- function returning non-finite anywhere in scan → `ArgumentException` with message
  "AutoKnots requires a finite-valued function over the entire domain"

### 5.4 Internal scan

The scan phase itself is `nDim × nScanPoints` evaluations, routed through
`ParallelBuild.EvaluateInParallel` with the user-supplied `nWorkers`/`progress`.
For typical inputs (`nDim ≤ 5`, `nScanPoints = 200`) the scan is ≤ 1000 evaluations
— often dwarfed by the subsequent Spline build, so progress reporting tracks both
phases cumulatively.

---

## 6. TT WithAutoOrder + Reorder + DimOrder + Threading (Section 5 of dialog)

### 6.1 `_dimOrder` storage

Add private field to `ChebyshevTT`:

```csharp
private int[] _dimOrder;  // _dimOrder[k] = original-dim index stored at TT position k
```

- Identity `[0, 1, …, n-1]` initialization at **construction entry points** —
  the public `ChebyshevTT(...)` ctor, the `Nodes(...)` factory, and the
  `FromValues(...)` factory. These are the only places a TT is built from
  scratch with no source TT to inherit from.
- **Factory bypasses derive `_dimOrder` from the source**, not from identity:
  `Slice`, `Extrude`, partial `Integrate`, and algebra results apply the
  per-method threading rule from §6.5; `Reorder` sets `_dimOrder = newOrder`;
  `WithAutoOrder` sets `_dimOrder = chosen permutation`; `Clone` (Phase 4)
  copies the source `_dimOrder` directly.
- JSON `Load` backfills identity for v1 files; new saves write `"jsonVersion": 2`
  with the array.

Public read-only property:

```csharp
public int[] DimOrder => (int[])_dimOrder.Clone();
```

Defensive copy — caller can't mutate internal state.

### 6.2 `_TtSwapAdjacent` helper

Direct port from Python `_algebra.py:177` (`_tt_swap_adjacent`):

```csharp
internal static TtCore[] TtSwapAdjacent(
    TtCore[] cores, int i, int maxRank, double tolerance = 1e-12);
```

- Operates on coefficient cores (the swap is linear; coefficient-space and value-space
  results agree because each axis transforms independently under the DCT-II basis).
- Forms joint `M[r_l, n_a, n_b, r_r] = Σ_r_m A · B`.
- Transposes middle axes: `(r_l, n_a, n_b, r_r) → (r_l, n_b, n_a, r_r)`.
- Reshapes to matrix, SVDs (MathNet, already loaded by Phase 4).
- Truncates by `maxRank` and relative-tolerance cutoff (`s_max × tolerance`).
- Repacks into two new cores.

### 6.3 `Reorder`

```csharp
public ChebyshevTT Reorder(int[] newOrder, int? maxRank = null, double? tolerance = null);
```

- Validates `newOrder` is a permutation of `[0, n-1]` (length matches numDimensions, all distinct, all in range; else `ArgumentException`).
- `maxRank ?? _maxRank`, `tolerance ?? _tolerance`.
- Bubble-sort the storage frame from `_dimOrder` to `newOrder`, applying
  `TtSwapAdjacent` at each transposition. O(n²) swaps worst-case.
- Result built via existing `BuildResultFromCores` helper (Phase 2); result inherits
  `_maxRank`, `_tolerance`, `_maxSweeps`, `Method`, `_descriptor`, `_additionalData`,
  `_maxDerivativeOrder` from the source. Sets `result._dimOrder = newOrder.ToArray()`.

### 6.4 `WithAutoOrder` static factory

```csharp
public static ChebyshevTT WithAutoOrder(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] numNodes,
    int maxRank = 10,
    double tolerance = 1e-6,
    int maxSweeps = 10,
    object? additionalData = null,
    int nTrials = 5,
    string method = "greedy_swap",
    int? seed = null,           // ignored for "greedy_swap"
    IProgress<int>? progress = null,
    bool verbose = false);
```

For each candidate permutation `order`:
- Build a permuted f: `permF(point) = function(unpermute(point))`
- Build a TT-Cross at `maxRank/tolerance/maxSweeps` with `seed`
- Score = sum of all bond ranks
- Pick lowest-rank result; set `_dimOrder = order` on the result

Two strategies:
- `"greedy_swap"`: start from canonical order; try every adjacent transposition;
  if any reduces total rank, accept and continue from there; loop up to `nTrials`
  outer iterations (terminate early if no improving swap exists).
- `"random"`: build canonical + `nTrials` random permutations; return lowest-rank.
  Uses `seed ?? Environment.TickCount`. Tests with explicit seed = 42 for reproducibility.

`method` values other than these two → `ArgumentException("unknown method...")`.

### 6.5 Threading `_dimOrder` through every public method

Mechanical port from Python v0.20.1:

| Method | Threading rule |
|---|---|
| `Eval(point)` | If `_dimOrder ≠ identity`, remap: `permPoint[k] = point[_dimOrder[k]]`. Same for `EvalBatch`. |
| `EvalMulti(point, derivOrders[])` | Permute `point` AND `derivOrders` axes. Run eval against permuted; result indexing matches user's original-dim order. |
| `Slice(dim, value)` | Map user `dim` → storage position `_dimOrder.IndexOf(dim)`. Drop that index from `_dimOrder` and renumber remaining: `newDimOrder[k] = newDimIndex[oldDimOrder[storagePos]]` where `newDimIndex` shifts post-removed indices down. |
| `Extrude(dim, …)` | Insert at storage position; new `_dimOrder` appends `numDimensions` (the new dim becomes "highest original-dim index"). |
| `ToDense()` | If `_dimOrder ≠ identity`, transpose result tensor so output axes match user's original-dim order. |
| Partial `Integrate(dims, bounds)` | Same drop-and-renumber rule as `Slice`, applied to each integrated dim. |
| Unary algebra (`-`, scalar `*`, scalar `/`, in-place variants) | Result inherits `_dimOrder`. |
| Binary algebra (`+`, `-` between TTs) | Both operands must have matching `_dimOrder`. Mismatched → `ArgumentException("dim_order mismatch; call Reorder() to align before adding/subtracting")`. Result inherits the shared `_dimOrder`. |

Helper methods land in `ChebyshevTT.cs` private static:

```csharp
private static double[] PermutePointToStorage(double[] point, int[] dimOrder);
private static int[] DropDimFromOrder(int[] dimOrder, int storagePos);
private static int[] AppendDimToOrder(int[] dimOrder);
```

### 6.6 JSON migration

```csharp
// In Save(): writes
"jsonVersion": 2,
"dimOrder": _dimOrder,

// In Load():
int jsonVersion = doc.RootElement.TryGetProperty("jsonVersion", out var v) ? v.GetInt32() : 1;
int[] dimOrder = jsonVersion >= 2
    ? doc.RootElement.GetProperty("dimOrder").EnumerateArray().Select(e => e.GetInt32()).ToArray()
    : Enumerable.Range(0, numDimensions).ToArray();  // backfill identity
```

A v0.9.0-saved fixture file will be committed to `tests/fixtures/` to verify the
backfill path works against an actual pre-Phase-6 save.

---

## 7. Test Plan

### 7.1 Test count progression

| Step | New tests | Cumulative |
|---|---|---|
| 0. baseline (Phase 5 shipped) | — | 950 |
| 1. `Internal/ParallelBuild.cs` + helper unit tests | +3 | 953 |
| 2. `nWorkers` + `IProgress` wiring on Approx | +5 | 958 |
| 3. Same wiring on Spline + Slider + TT | +6 | 964 |
| 4. Validation tests for `nWorkers` (0, -2) | +3 | 967 |
| 5. `Internal/Sensitivity.cs` + Approx `SobolIndices` | +8 | 975 |
| 6. `Spline.SobolIndices` aggregation | +4 | 979 |
| 7. `ChebyshevSpline.AutoKnots` | +10 | 989 |
| 8. `_TtSwapAdjacent` helper unit tests | +3 | 992 |
| 9. `Reorder` instance method | +5 | 997 |
| 10. `WithAutoOrder` + `DimOrder` property + JSON migration | +10 | 1007 |
| 11. `_dimOrder` threading across every public TT method | +10 | 1017 |
| **Phase 6 total** | **+67** | **1017** |

±5 noise budget per Phase 4/5 precedent. Larger drift requires investigation.

### 7.2 New test files

| File | Tests |
|---|---|
| `tests/ChebyshevSharp.Tests/BuildPerfTests.cs` | 17 |
| `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs` | 12 |
| `tests/ChebyshevSharp.Tests/AutoKnotsTests.cs` | 10 |
| `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs` | 10 |
| `tests/ChebyshevSharp.Tests/TtDimOrderTests.cs` | 15 |
| `tests/ChebyshevSharp.Tests/CalculusTests.cs` | +3 (new tests for `_TtSwapAdjacent` helper appended to existing file) |
| **Total new tests** | **67** |

### 7.3 Test patterns

- **Numerical parity**: parallel build vs sequential build → `AssertClose` on TensorValues.
- **Thread-safety smoke**: pure non-capturing function under `nWorkers: 4` produces identical TensorValues to sequential.
- **Stochastic outputs (TT-Cross / ALS / `WithAutoOrder` random)**: use `AssertClose` (tolerance-based), never bit-exact (matches Phase 2 TT-Cross precedent).
- **Deterministic outputs (`greedy_swap`)**: bit-exact `Equals` on `DimOrder` permutation arrays.
- **Reorder round-trip**: `Reorder(perm).Reorder(invPerm).Eval(point)` matches original `Eval(point)` within 1e-6 — uses higher-than-default `maxRank` to keep SVD truncation error tiny.
- **JSON v1 → v2 migration**: load a v0.9.0-saved fixture file from `tests/fixtures/`; verify identity-order behavior; verify save/load round-trip in v2 format preserves `_dimOrder`.

---

## 8. Release Prep

### 8.1 Version metadata

`src/ChebyshevSharp/ChebyshevSharp.csproj`:
- `<Version>0.10.0</Version>`
- `<PyChebyshevParity>0.20.1</PyChebyshevParity>` (advances vs Phase 5's `0.17.0` because Phase 6 ships everything between v0.18 and v0.20.1)
- `<InformationalVersion>0.10.0+pychebyshev.0.20.1</InformationalVersion>`

### 8.2 Submodule bump

`ref/PyChebyshev` → `v0.20.1` (single hop, 15 commits across v0.18.0 → v0.19.0 → v0.20.0 → v0.20.1). Task 1 includes a brief diff review to flag anything beyond v0.19/v0.20.0/v0.20.1 in the bump.

### 8.3 Changelog

`docs/docs/changelog.md` v0.10.0 entry, two-tier (PyChebyshev parity + ChebyshevSharp internal):

```
## [0.10.0] - 2026-04-?? — PyChebyshev parity v0.20.1

### Build performance (from PyChebyshev v0.19.0)
- nWorkers ctor kwarg on all four classes (null, -1, positive int)
- IProgress<int> ctor kwarg on all four classes (per-evaluation in Approx/Spline/Slider; per-sweep in TT)
- Thread-safety contract documented; functions used with nWorkers must be thread-safe

### Adaptive refinement (from PyChebyshev v0.20.0 + v0.20.1)
- ChebyshevSpline.AutoKnots(...) — auto-place knots at function kinks via curvature-spike scan
- SobolIndices() on ChebyshevApproximation and ChebyshevSpline (returns SobolResult record)
- ChebyshevTT.WithAutoOrder(...) — heuristic dim ordering to minimize TT rank
- ChebyshevTT.Reorder(newOrder, maxRank?, tolerance?) — TT-swap-based realignment
- ChebyshevTT.DimOrder property — read-only access to current storage permutation
- All TT public methods now thread _dimOrder correctly (eval, slice, extrude, toDense, partial integrate, algebra)

### JSON migration
- ChebyshevTT save format bumped to "jsonVersion": 2 with new "dimOrder" field
- v0.9.0-and-earlier files load with identity dimOrder backfilled

### Skipped (Python-only ergonomic features)
- plot_convergence, plot_1d, plot_2d_surface, plot_2d_contour (matplotlib helpers)
```

### 8.4 Documentation

- `docs/docs/parallel-build.md` (new) — `nWorkers` + `IProgress` with thread-safety examples
- `docs/docs/adaptive-refinement.md` (new) — `AutoKnots` + `SobolIndices` + `WithAutoOrder` worked examples
- `docs/docs/toc.yml` — adds both pages

### 8.5 Skip list

`skip_csharp.txt` — add Phase 6 line, mark Phase 5 line as resolved.

### 8.6 CLAUDE.md status

Update Status block:
- Test count `950 → 1017`
- Phase list: add Phase 6 (Build Perf + Adaptive Refinement, v0.10.0)
- Parity tag: `0.18.0 → 0.20.1`
- Mark **6 of 6 phases complete — port complete**

---

## 9. Implementation Phasing (Plan Preview)

The writing-plans skill will dispatch 12 tasks (10 TDD + 2 housekeeping). Suggested grouping:

1. Submodule bump v0.18.0 → v0.20.1 + csproj prep + scaffolding (no code, no new tests)
2. `Internal/ParallelBuild.cs` + helper unit tests (3 tests)
3. `nWorkers` + `IProgress` wiring on `ChebyshevApproximation` (5 tests)
4. Same wiring on `ChebyshevSpline` + `ChebyshevSlider` + `ChebyshevTT` (6 tests)
5. `nWorkers` validation edge cases on all four classes (3 tests)
6. `Internal/Sensitivity.cs` + `SobolResult` + `ChebyshevApproximation.SobolIndices` (8 tests)
7. `ChebyshevSpline.SobolIndices` aggregation (4 tests)
8. `ChebyshevSpline.AutoKnots` (10 tests)
9. `Internal/TensorTrainAlgebra.TtSwapAdjacent` + `ChebyshevTT.Reorder` + `DimOrder` (8 tests)
10. `WithAutoOrder` + JSON migration (10 tests)
11. `_dimOrder` threading across every public TT method (10 tests)
12. Docs + changelog + skip_csharp + CLAUDE.md + parity tags (no new tests)

(Tasks 1 and 12 carry no test count; the +67 lands across Tasks 2–11.)

Each task is sequential. Each completes with `dotnet test` verification matching the cumulative count from §7.1.

---

## 10. Design Decisions Log

| ID | Decision | Why | Source |
|---|---|---|---|
| D1 | `Parallel.For` over thread pool, document thread-safety contract | C# has no GIL; process isolation in C# would require a research project (no pickle equivalent for `Func<…>`). User confirmation: Section 2 dialog. | §3.1 |
| D2 | Cumulative `int` reporting via `Interlocked.Increment`; caller computes total via existing Phase 4 `GetNumEvaluationPoints()` | Standard .NET `IProgress<int>` idiom; total already exposed for three of four classes; minimal API surface bloat. User confirmation: Section 2 dialog. | §3.2 |
| D3 | `SobolResult` is a record with `FirstOrder`, `TotalOrder`, `Variance` | Matches Phase 4 record-for-result convention; `Variance` is genuinely useful for detecting near-constant functions (where Sobol ratios are meaningless). User confirmation: Q1. | §2.3, §4.3 |
| D4 | `AutoKnots` defaults `5.0 / 5 / 200` (Python-exact) | Faithful port of Python defaults from `spline.py:2117-2119`. Master spec hand-wave (`0.1 / 5 / 1000`) was incorrect. | §5.2 |
| D5 | `WithAutoOrder` takes optional `int? seed`, ignored for `greedy_swap` | C# convention: stochastic factories take seed for reproducibility (matches Phase 2 TT-Cross, Phase 4 ALS). User confirmation: Q5. | §6.4 |
| D6 | `_dimOrder` field initialized to identity in every existing ctor and factory; backfilled by JSON Load v1 → v2 | All existing TTs behave unchanged; new field adds no behavioral surprise to v0.9.0 callers. | §6.1, §6.6 |
| D7 | Binary algebra requires matching `_dimOrder`; mismatch throws `ArgumentException` pointing at `Reorder` | Matches Python v0.20.1 exactly. Avoids silent wrong-results bugs. The `Reorder` method is the explicit alignment escape hatch. | §6.5 |
| D8 | JSON schema bumped to `"jsonVersion": 2`; v1 backfills identity `dimOrder` on Load | Phase 4 backfill pattern (Load populates missing fields with defaults). v0.9.0-saved fixture file committed to verify. | §6.6 |
| D9 | Single PR (~67 tests); single submodule hop v0.18 → v0.20.1 | Phase 4 (50 tests, single hop) precedent; bundling avoids shipping `NotImplementedException` placeholders that the next release rewrites. | §1, §8.2 |
| D10 | TT skips `nWorkers` (TT-Cross is adaptive sampling, not pre-grid evaluation); progress fires per-sweep | Documented limitation; matches Python `n_workers` constraint (Python's TT-Cross also doesn't parallelize). | §3.4 |

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `Parallel.For` with non-thread-safe user function silently produces wrong values | Document the contract loudly in XML doc + `parallel-build.md`. Thread-unsafe smoke test demonstrates lock-wrapping pattern. We don't try to detect non-thread-safe functions — that's the user's responsibility, same as Python's "must be picklable." |
| Sobol from spline coefficients (per-piece aggregation) is subtly wrong | Three tests against analytical references (Ishigami, multiplicative, additive) on both Approx and Spline forms catch this. Plus 1D / constant / NaN edge cases. |
| `Reorder` accumulates SVD truncation error → eval drifts | Test `Reorder(perm).Reorder(invPerm)` round-trip against original within 1e-6. Use higher-than-default `maxRank` to keep error tiny. |
| `_dimOrder` threading misses a method, silently produces wrong result for non-identity-order TTs | One "permuted vs canonical equivalence" test per public TT method. The 15 tests in `TtDimOrderTests.cs` cover the full surface. |
| `WithAutoOrder` `random` mode tests use `System.Random` — different stream than NumPy | Tolerance assertions on TT values (matches Phase 2 TT-Cross precedent); bit-exact only on `DimOrder` permutation arrays. |
| JSON v1 → v2 migration breaks existing v0.9.0 saves | Backfill test loads a v0.9.0-saved fixture file from `tests/fixtures/` and verifies identity-order behavior. |
| Submodule jump v0.18 → v0.20.1 brings unexpected upstream changes | Task 1 includes a "submodule bump diff review" sub-step. The 15-commit window is bounded by the v0.19/v0.20.0/v0.20.1 entries we're explicitly porting; anything outside that window is a flag. |

---

## 12. After Phase 6

Phase 6 is **the last phase** of the v0.20.1 port. After v0.10.0 ships:
- ChebyshevSharp is feature-complete against PyChebyshev v0.20.1.
- All four interpolant classes have full surface parity with their Python counterparts (modulo deliberately skipped matplotlib helpers).
- 6 of 6 phases complete.
- Future upstream PyChebyshev releases (v0.21+) get tracked in changelog but not pursued mid-port.

The release flow on Phase 6 completion follows the established pattern (Phases 3, 4, 5):
- User explicit "create PR" → `gh pr create`
- User explicit "merge and release" → `gh pr merge --squash` → `gh release create v0.10.0` → NuGet auto-publish via existing workflow
- User explicit "clean up worktree" → `git worktree remove` + branch cleanup
