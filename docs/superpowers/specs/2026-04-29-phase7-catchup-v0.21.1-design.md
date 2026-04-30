# Phase 7 Design — Catch-up to PyChebyshev v0.21.1 (ChebyshevSharp v0.11.0)

**Status:** approved (auto-mode bundled approval, 2026-04-29)
**Master spec section:** `docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md` (parity tag advances 0.20.1 → 0.21.1)
**Upstream parity target:** PyChebyshev v0.21.1 (bundles upstream v0.21.0 + v0.21.1)
**Worktree (planned):** `.worktrees/phase7-catchup-v0.21.1`
**Test count progression:** 1018 → ~1100 (+82 expected, ±2 drift per task)
**Predecessor:** Phase 6 / v0.10.0 (shipped 2026-04-29; commit `cd2195a`)

---

## 1. Goal and Scope

Port PyChebyshev v0.21.0 + v0.21.1 to ChebyshevSharp as a single "post-port maintenance"
release. Two subsystems:

- **Calculus parity** (from upstream v0.21.0): Slider and TT gain `Roots`, `Minimize`,
  `Maximize`. Closes the gap promised since Phase 5 / v0.17. After Phase 7, all four
  classes have the full calculus surface.
- **TT `_dimOrder` cluster + cross-cutting fixes + perf** (from upstream v0.21.1): one
  new TT method (`SobolIndices`), six bug fixes (most apply verbatim to C# — the bugs
  are direct ports of the Python originals), and two perf improvements
  (`VectorizedEvalBatch` hoist, `_optimize_1d` vectorization).

**Why bundle v0.21.0 + v0.21.1:** Upstream released them back-to-back on 2026-04-27
(one day after v0.20.1). v0.21.1 explicitly cleans up the v0.20+v0.20.1+v0.21.0
`_dim_order` cluster — bundling matches the upstream's mental model. Splitting would
mean shipping `0.11.0` (calculus only, with the `_dim_order` cluster bugs latent in
TT.Roots/Min/Max) and then `0.11.1` immediately after — same churn as upstream's
v0.21.0 → v0.21.1 cycle, no upside. Bundle size (~85–95 C# tests) is comparable to
Phase 6 (67 tests) and stays within the "single PR is fine" envelope.

**Skipped:** nothing. v0.21.0 + v0.21.1 have no plotting helpers (Option C from master
spec was Phase 6's scope).

After v0.11.0 ships, ChebyshevSharp is feature-complete against PyChebyshev v0.21.1.

---

## 2. Architecture

### 2.1 Modified files (no new public types)

| File | Change | Purpose |
|---|---|---|
| `src/ChebyshevSharp/ChebyshevSlider.cs` | extended | gains `Roots`/`Minimize`/`Maximize` + private `To1DChebyshev` helper |
| `src/ChebyshevSharp/ChebyshevTT.cs` | extended | gains `Roots`/`Minimize`/`Maximize`, `SobolIndices`, `EvalStorageFrame` private helper. Bug fixes: `GetEvaluationPoints`, `EvalMulti`, `InnerProduct`, `Integrate` error messages. |
| `src/ChebyshevSharp/Internal/Algebra.cs` | extended | gains `DoublesAllClose` helper; `CheckCompatible` switches to numerical comparison on `Domain[d]` |
| `src/ChebyshevSharp/Internal/BarycentricKernel.cs` | extended | `VectorizedEvalBatch` hoists diff-matrix matmul outside per-point loop |
| `src/ChebyshevSharp/Internal/Calculus.cs` | extended | `_optimize_1d` analog uses single vectorized barycentric eval over critical points + endpoints |

### 2.2 Public API additions (delta vs v0.10.0)

`ChebyshevSlider` gains:
- `double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)`
- `(double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)`
- `(double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)`

`ChebyshevTT` gains:
- `double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)`
- `(double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)`
- `(double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)`
- `SobolResult SobolIndices()`

Signatures match `ChebyshevApproximation.Roots/Minimize/Maximize` and Phase 6
`SobolIndices()` exactly (same return types, same parameter names, same defaults).
Cross-class consistency requirement.

### 2.3 No new public types

Phase 7 introduces zero new public records, enums, or interfaces. `SobolResult` already
exists from Phase 6.

---

## 3. Design Decisions

### D1 — `TT.SobolIndices` ports the TT-native O(d·n·r²) coefficient-core algorithm directly

**Decision.** Implement TT.SobolIndices via direct contraction through the coefficient cores
(`_coeffCores`), as in upstream `_sensitivity._compute_sobol_from_tt_coeffs`. **Do not**
materialize the dense tensor and reuse Phase 6 `Sensitivity.ComputeSobolFromCoeffs`.

**Why.** The TT-native implementation is upstream's headline reason for the v0.21.1
addition. Materializing the dense tensor would defeat the purpose (memory) and produces
identical numerical results, but with a `ToDense()` allocation that scales as ∏ nNodes.
Phase 6 `Sensitivity.cs` already operates in coefficient space, so the port is a direct
algorithmic translation, not a paradigm switch.

**How to apply.** New private static method
`Internal.Sensitivity.ComputeSobolFromTtCoeffs(TtCore[] coeffCores, int[] nNodes, int[] dimOrder)`:
- For each user-frame dim `d`:
  - Map to storage-frame dim `s = Array.IndexOf(dimOrder, d)`.
  - Contract along all storage dims except `s` into a 1-D coefficient slice.
  - Use Chebyshev T_n inner-product norms (π for n=0, π/2 for n≥1) to compute first-order variance contribution.
- Compute total-order via complement contraction (variance over points where dim `d` is held vs. varying).
- Output keys are user-frame dim indices.

Test parity: outputs must match `to_dense() + ComputeSobolFromCoeffs` to within `1e-12` on
small TTs (separable + coupled fixtures).

### D2 — `EvalMulti` race fix via `EvalStorageFrame` private helper

**Decision.** Refactor `ChebyshevTT.EvalMulti` to call a new private helper
`EvalStorageFrame(double[] storagePoint, int[][] storageOrders)` that **always operates
in storage frame** (no mutation of `_dimOrder`). Eliminate the existing `try/finally`
mutation pattern (race condition under concurrent calls).

**Why.** The existing C# code (line 463 in `ChebyshevTT.cs`) mirrors Python pre-v0.21.1's
race-bug pattern verbatim: "save `_dimOrder` → set to identity → call inner methods → restore
in `finally`." Two concurrent threads on the same TT will trample each other's saved state.
Issue #19 in upstream tracks this; v0.21.1 fixes via a non-mutating helper. Phase 6 already
established `EvalCore` as a non-mutating inner-call boundary; this extends the pattern to
the multi-derivative case.

**How to apply.**
- Public `EvalMulti(point, derivativeOrders)` does the user-frame → storage-frame
  permutation locally (in stack-allocated arrays), then calls
  `EvalStorageFrame(storagePoint, storageOrders)`. For identity `_dimOrder`, the
  permutation is a no-op `Array.Copy` (fast path).
- Private `EvalStorageFrame(double[] storagePoint, int[][] storageOrders)`:
  - Always operates in storage frame internally.
  - Calls `EvalCore` (Phase 6) and `FdDerivative` (existing private) directly with
    storage-frame coordinates. **No mutation of `_dimOrder`.**
- Regression test: spin up 4 threads each calling `EvalMulti` 1000 times concurrently
  with different `derivativeOrders` on a TT built via `WithAutoOrder` (non-identity
  `_dimOrder`); assert no thrown exceptions and assert each thread's results match a
  single-threaded baseline.

### D3 — `InnerProduct` mismatch throws `ArgumentException` with reorder hint

**Decision.** When `self._dimOrder != other._dimOrder`, `ChebyshevTT.InnerProduct(other)`
throws `ArgumentException` with message:
`"InnerProduct requires matching _dimOrder; got [a, b, c] vs [c, a, b]. Call other.Reorder(self.DimOrder) first."`.

**Why.** Two TTs with different `_dimOrder` represent the same underlying interpolant
under different storage permutations. The Frobenius product on the storage tensors is
not the inner product of the interpolants. Pre-Phase-7 code returns the wrong number
silently (no validation). Throwing matches v0.10.0 binary-algebra `_check_compatible`
convention (which already throws `ArgumentException` with a `Reorder` hint).

**How to apply.** Add `_dimOrder` SequenceEqual check to existing `InnerProduct`
validation block (line 990). Re-use Phase 6 `IsIdentityDimOrder` pattern. Test fixtures:
two TTs from same data, one reordered via `Reorder([2, 0, 1])`; assert
`a.InnerProduct(b)` throws `ArgumentException`; assert `a.InnerProduct(b.Reorder(a.DimOrder))`
returns the correct (matching) inner product.

### D4 — `Algebra.CheckCompatible` numerical tolerance via `DoublesAllClose` helper

**Decision.** Replace `Domain[d].SequenceEqual(other.Domain[d])` with numerical comparison
using a new helper `DoublesAllClose(double[] a, double[] b, double rtol = 1e-5, double atol = 1e-8)`.
Match Python `np.allclose` defaults. **Keep** node-count check as exact `SequenceEqual`
(it's an int array; floating-point tolerance is meaningless).

**Why.** Issue #22 upstream: when one TT was constructed with `domain=[(a, b), (c, d)]`
and another with `domain=[[a, b], [c, d]]`, both numerically identical, the post-construction
algebra raised "Domain mismatch" because Python's exact-equality comparison failed on the
tuple-vs-list type difference. C# has the analogous bug: `SequenceEqual` on `double[]`
is exact equality, and operations like `a * 1.0` may produce a domain that's been through
`new double[]` allocation (different reference, same values). Tolerance comparison fixes
the false-positive without weakening the genuine-mismatch case.

**How to apply.** Add `internal static bool DoublesAllClose(double[] a, double[] b, double rtol = 1e-5, double atol = 1e-8)`
to `Algebra.cs`. Update `CheckCompatible` line 33 from `SequenceEqual` to `DoublesAllClose`.
Five new tests in `AlgebraTupleListTests.cs` verify mixed `double[][]` arrays with
genuinely identical (within tolerance) bounds compose under `+`/`-`; tests verifying
that genuinely different bounds (e.g., `1.0` vs `1.5`) still throw remain green.

### D5 — Single PR / single phase / single release

**Decision.** Phase 7 is one cohesive release. One worktree, one PR, one git tag,
one NuGet publish.

**Why.** Phase 6 set the precedent (67 tests, three upstream releases bundled, single PR).
Phase 7 is comparable in size (~85 tests). The `v0.21.0 → v0.21.1` upstream split was
a pragmatic upstream choice; for downstream maintenance work, the optimization is single
review, single ship, single rollback target if anything goes sideways.

---

## 4. Per-task subsystem detail

### 4.1 Slider calculus (Tasks 2–3)

Slider's calculus reduces to its 1D specialization. Upstream's strategy:

1. **Reduce to 1D Slider** via existing `Slice(...)` with `fixedDims`.
2. **Build a 1D ChebyshevApproximation** by evaluating the 1D Slider at Chebyshev nodes.
3. **Delegate** to `ChebyshevApproximation.Roots/Minimize/Maximize`.

C# implementation:

```csharp
// ChebyshevSlider.cs (private helper)
private ChebyshevApproximation To1DChebyshev(int dim)
{
    // Precondition: this is a 1-D slider (NumDimensions == 1, Partition has one group)
    if (NumDimensions != 1)
        throw new InvalidOperationException("To1DChebyshev requires a 1-D slider");

    int n = _slides[0].NNodes[0];
    double[] domain = _slides[0].Domain[0];

    // Evaluate at Chebyshev T1 nodes (canonical helper from Phase 1)
    double[] nodes = BarycentricKernel.MakeNodesForDim(domain[0], domain[1], n);
    double[] values = new double[n];
    for (int i = 0; i < n; i++)
        values[i] = Eval(new[] { nodes[i] });

    return ChebyshevApproximation.FromValues(
        numDim: 1,
        domain: new[] { domain },
        nNodes: new[] { n },
        tensorValues: values);
}
```

Public `Roots/Minimize/Maximize` then:
1. Validate `dim` and `fixedDims` per existing `ChebyshevApproximation` convention.
2. Call `Slice(fixedDims)` to reduce.
3. Call `To1DChebyshev(dim)` on the 1D result.
4. Delegate to the 1D `ChebyshevApproximation.{method}()`.

Slider has no `_dimOrder`, so user-frame and storage-frame coincide. No special handling.

### 4.2 TT calculus (Task 4)

TT calculus reduces via `ToDense() → ChebyshevApproximation.FromValues → delegate`. Upstream uses:

1. **Reduce to 1D TT** via successive `Slice(dim, value)` calls for each `fixedDims` entry.
2. **Materialize to dense** via `ToDense()` on the 1D TT (size `nNodes[storage_d]`, cheap).
3. **Build 1D ChebyshevApproximation** via `FromValues`.
4. **Delegate**.

`fixedDims` validation is the v0.21.1 fix: validate against **user-frame** physical domain,
not storage-frame domain. Implementation:

```csharp
public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    CheckBuilt();

    // Validate fixedDims against user-frame physical domain
    // (each TT method's "user-frame domain at user-dim d" is _domain[_dimOrder.IndexOf(d)],
    // i.e. the storage-frame domain at the storage-position holding original-dim d)
    if (fixedDims != null)
    {
        foreach (var (userDim, value) in fixedDims)
        {
            int storageDim = Array.IndexOf(_dimOrder, userDim);
            if (storageDim < 0)
                throw new ArgumentException($"fixedDims key {userDim} is not a valid user-frame dim");
            double lo = _domain[storageDim][0], hi = _domain[storageDim][1];
            if (value < lo || value > hi)
                throw new ArgumentOutOfRangeException(
                    $"fixedDims[{userDim}] = {value} is outside user-frame domain [{lo}, {hi}]");
        }
    }

    // Reduce: successive Slice() calls peel off each fixedDim entry. Slice()
    // already handles user-frame → storage-frame translation (Phase 6).
    var reduced = ReduceByFixedDims(fixedDims);  // returns 1-D ChebyshevTT

    // Materialize the 1-D TT to dense (cheap: nNodes[remaining] entries).
    double[] values = reduced.ToDense();

    // Build a 1-D ChebyshevApproximation over the surviving dim's user-frame domain.
    // Since reduced is now 1-D, _dimOrder is [0] (single element); user-frame and
    // storage-frame coincide. Reading reduced.Domain[0] gives the correct domain.
    var approx = ChebyshevApproximation.FromValues(
        numDim: 1,
        domain: new[] { reduced.Domain[0] },
        nNodes: new[] { reduced.NNodes[0] },
        tensorValues: values);

    return approx.Roots();
}
```

`Minimize` and `Maximize` follow the same pattern. The reducer must handle the user-frame
→ storage-frame translation correctly: `Slice(userDim, value)` already does this in Phase 6.

### 4.3 TT.SobolIndices (Task 5) — TT-native algorithm

Direct port of upstream `_sensitivity._compute_sobol_from_tt_coeffs`. Skeleton:

```csharp
public SobolResult SobolIndices()
{
    CheckBuilt();
    int d = _numDimensions;
    var firstOrder = new double[d];
    var totalOrder = new double[d];

    // Compute total variance once: contract all dims except none (full sum-of-squares
    // over coefficient cores weighted by Chebyshev T_n inner-product norms).
    double variance = TtCoeffSquaredNorm(_coeffCores!, _nNodes, excludingConstantTerm: true);

    if (variance < 1e-20)
        return new SobolResult(new double[d], new double[d], variance);

    for (int userDim = 0; userDim < d; userDim++)
    {
        int storageDim = Array.IndexOf(_dimOrder, userDim);

        // First-order: contract along all storage dims except storageDim,
        // skip the n=0 component on storageDim.
        firstOrder[userDim] = TtCoeffSquaredNorm_OnlyDim(_coeffCores!, _nNodes, storageDim) / variance;

        // Total-order: contract along all storage dims except storageDim,
        // include the n=0 component on storageDim, then subtract the
        // contribution where storageDim is held constant (n=0 only).
        totalOrder[userDim] = TtCoeffSquaredNorm_IncludingDim(_coeffCores!, _nNodes, storageDim) / variance;
    }

    return new SobolResult(firstOrder, totalOrder, variance);
}
```

Helper methods `TtCoeffSquaredNorm`, `TtCoeffSquaredNorm_OnlyDim`,
`TtCoeffSquaredNorm_IncludingDim` go in `Internal/Sensitivity.cs` (extended). Each
performs a left-to-right sweep contracting cores with appropriate per-mode weights:
- Constant (n=0) only: zero out all `n>=1` slices, keep `n=0`.
- Variable (n>=1) only: zero out `n=0`, keep `n>=1`.
- Weighted: each mode contributes `weight[n] = (n == 0) ? π : π/2` (Chebyshev T_n L²
  norm² over `[-1, 1]` with weight `1/sqrt(1-x²)`).

Cross-validation test: build a small (d=3, n=8) TT from a known `f`, compute SobolIndices
via TT-native and via `ToDense() + Sensitivity.ComputeSobolFromCoeffs`, assert agreement
to `1e-12`. Run for both separable (`f = x + y + z`) and coupled (`f = exp(x*y)·z`) fixtures.

### 4.4 GetEvaluationPoints user-frame fix (Task 6)

Currently (`ChebyshevTT.cs:1797`) returns columns in storage-frame order: column `k` is
the storage-frame `k`-th coordinate. After `WithAutoOrder`, this means
`Eval(GetEvaluationPoints()[i])` does **not** round-trip — `Eval` expects user-frame, but
`GetEvaluationPoints` returns storage-frame.

Fix: compute the inverse permutation `inv[_dimOrder[k]] = k`, then permute columns by
`inv` before returning. Result: column `inv[k]` of the returned array is the user-frame
`k`-th coordinate.

```csharp
public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;
    int num = GetNumEvaluationPoints();
    int ndim = _numDimensions;

    // ... (existing storage-frame grid generation) ...

    // Permute columns by inverse _dimOrder so column k is the user-frame k-th coord.
    if (!IsIdentityDimOrder())
    {
        var inv = new int[ndim];
        for (int k = 0; k < ndim; k++) inv[_dimOrder[k]] = k;
        var permuted = new double[num * ndim];
        for (int flat = 0; flat < num; flat++)
            for (int k = 0; k < ndim; k++)
                permuted[flat * ndim + k] = points[flat * ndim + inv[k]];
        points = permuted;
    }

    _evaluationPointsCache = points;
    return points;
}
```

Regression test: build TT with `WithAutoOrder` to force non-identity `_dimOrder`, call
`Eval(GetEvaluationPoints()[i])` for several `i`, compare to direct `Eval` of the
same coordinates in user-frame ordering.

### 4.5 Integrate user-frame error messages (Task 8 second half)

Phase 6 Task 11 already split the bounds remap so validation happens against the
user-frame domain (the bug there was a *positional* mismatch, fixed during review).
The remaining v0.21.1 issue (#20) is the **error message text**: when bounds are
out-of-domain, the message currently identifies the dim by its storage-frame index,
not the user-frame index the caller passed. The validation logic itself is correct.

Fix is text-only — change the error format string to use the user-frame `dims[i]`
that the caller supplied, plus the user-frame domain interval read from
`_domain[Array.IndexOf(_dimOrder, dims[i])]`:

```csharp
// In ValidateBounds (already runs in user-frame after Phase 6 fix):
int userDim = dims[i];
int storageDim = Array.IndexOf(_dimOrder, userDim);
double lo = _domain[storageDim][0], hi = _domain[storageDim][1];
if (bounds[i].lower < lo || bounds[i].upper > hi)
    throw new ArgumentOutOfRangeException(
        $"bounds[{i}] = [{bounds[i].lower}, {bounds[i].upper}] " +
        $"is outside user-frame domain[{userDim}] = [{lo}, {hi}]");
```

Test fixture: build TT with `WithAutoOrder([2, 0, 1])`, call `Integrate(dims=[0],
bounds=[(out-of-range)])`, assert error message contains `"domain[0]"` not `"domain[1]"`
or `"domain[2]"`.

### 4.6 Perf — VectorizedEvalBatch hoist (Task 10)

Currently each batch point does:
```
diff_matrix @ values  →  evaluate barycentric on result
```
For `nDerivatives > 0` and a batch of `M` points, the diff-matrix matmul is repeated `M`
times even though it's the same result. Hoist:
```
intermediate = diff_matrix @ values  (once)
for each point in batch:
    evaluate barycentric on intermediate  (M times)
```

**Hoist condition:** when `derivativeOrder` is non-zero in some dim. When all-zero, no
matmul is needed (fast path stays unchanged).

Test: parity-with-old-impl on a 3D BS interpolant with `derivativeOrder = [1, 0, 0]`,
batch size 1000. Assert numerical agreement to `1e-13`.

### 4.7 Perf — `_optimize_1d` vectorized barycentric (Task 10)

Currently `Calculus.Optimize1D` evaluates the 1D Chebyshev at critical points + endpoints
via a Python-style list comprehension (in C#: a `for` loop calling
`BarycentricInterpolate` repeatedly). Replace with a single vectorized call: evaluate at
the full array of critical points + endpoints in one pass.

C# already has `VectorizedEvalBatch` (now hoisted, see 4.6). Reuse it: pack critical
points + endpoints into a `double[][]`, one batch call, one pass. Same numerical results.

---

## 5. Test Plan

### 5.1 New test files (~85 tests)

| File | Tests | Coverage |
|---|---|---|
| `SliderRootsTests.cs` | ~10 | Slider 1D/2D Roots; `fixedDims` reduction; multi-D requires fixed; fixed includes target raises; no roots returns empty |
| `SliderOptimizeTests.cs` | ~10 | Slider Min/Max parity-with-brute-force; `(value, location)` tuple ordering; fixedDims path |
| `TtCalculusTests.cs` | ~15 | TT 1D/2D Roots/Min/Max; user-frame `dim` under WithAutoOrder; user-frame `fixedDims` validation; fixed includes target raises |
| `TtSobolIndicesTests.cs` | ~12 | TT.SobolIndices polynomial-exactness; sum-to-one for additive functions; matches `ToDense + ComputeSobolFromCoeffs` to 1e-12; works under non-identity `_dimOrder`; constant-function edge case |
| `TtDimOrderClusterTests.cs` | ~15 | GetEvaluationPoints user-frame; EvalMulti race-safe (concurrent invocation regression test); InnerProduct mismatch raises ArgumentException; Integrate error message references user-frame dim; Roots/Min/Max validate `fixedDims` against user-frame domain |
| `AlgebraTupleListTests.cs` | ~5 | Mixed-allocation `double[][]` arrays with numerically identical bounds compose under `+`/`-`/`*`/`/`; genuinely different bounds still throw |
| `VectorizedEvalBatchPerfTests.cs` | ~3 | Parity-with-old-impl after diff-matrix hoist; correctness on large batch (1000 points); fast path unchanged when derivativeOrder is all-zero |
| `OptimizeVectorizedTests.cs` | ~5 | _optimize_1d vectorized path produces same numerical results as old loop for known critical-point fixtures |
| `Phase7CoverageGapTests.cs` | ~10 | Defensive paths: Roots/Min/Max validation order, edge cases (single-node TT, etc.) |

**Estimated total:** 82 tests (matches per-task summation in §5.2). Allow drift of ±5
across the phase from defensive tests added during review or test consolidations
(precedent: Phase 4 +50 plan / +51 actual; Phase 5 +50 plan / +47 actual).

### 5.2 Test count progression per task (target totals after each task)

| Task | Description | Δ tests | Total |
|---|---|---|---|
| 1 | Submodule bump + scaffold + version bump | 0 | 1018 |
| 2 | Slider.To1DChebyshev + Slider.Roots | +6 | 1024 |
| 3 | Slider.Minimize + Slider.Maximize | +10 | 1034 |
| 4 | TT.Roots + TT.Minimize + TT.Maximize | +15 | 1049 |
| 5 | TT.SobolIndices | +12 | 1061 |
| 6 | TT.GetEvaluationPoints user-frame fix | +5 | 1066 |
| 7 | TT.EvalMulti race fix | +5 | 1071 |
| 8 | TT.InnerProduct mismatch + Integrate user-frame error | +6 | 1077 |
| 9 | Algebra.CheckCompatible numerical tolerance | +5 | 1082 |
| 10 | Perf: VectorizedEvalBatch hoist + _optimize_1d vectorized | +8 | 1090 |
| 11 | Coverage gap fillers + docs | +10 | 1100 |

Allow ±2 drift per task per Phase 4–6 precedent.

---

## 6. Migration and Compatibility

**API additions only — no breaking changes.** The public API delta is purely additive:
six new methods on Slider/TT, one new method on TT.

**Behavior changes (failure-path only, not silent):**
- `TT.InnerProduct(other)` previously returned wrong number on `_dimOrder` mismatch;
  now throws `ArgumentException`. This is a behavior change for callers who relied on
  the old (incorrect) behavior — none, since the old result was meaningless.
- `TT.GetEvaluationPoints()` previously returned columns in storage-frame order; now
  returns in user-frame order. **Behavior change for callers who used the result on
  TTs built with `WithAutoOrder`.** For identity `_dimOrder`, no change. We document
  this in the changelog as a fix; the old behavior was a bug.
- `TT.Integrate(dims, bounds)` error messages now reference user-frame indices.
  No change to non-error paths.

**JSON migration:** None. Phase 6 introduced `JsonVersion = 2`; Phase 7 makes no
schema change.

**Version markers:**
- `<Version>0.11.0</Version>` (was 0.10.0)
- `<PyChebyshevParity>0.21.1</PyChebyshevParity>` (was 0.20.1)
- `<InformationalVersion>0.11.0+pychebyshev.0.21.1</InformationalVersion>`

---

## 7. Implementation Plan Outline (Tasks 1–11)

Sequential. Each task gets its own subagent. Two-stage review (spec compliance +
code quality) between every task. Worktree enforcement on every task subagent.

| # | Task | Mode | Notes |
|---|---|---|---|
| 1 | Submodule bump v0.20.1 → v0.21.1; scaffold test files; bump csproj 0.10.0 → 0.11.0; parity 0.20.1 → 0.21.1 | housekeeping | Mechanical. 0 tests added. |
| 2 | Slider.To1DChebyshev private helper + Slider.Roots | TDD | Test: Slider.Roots on 1D and 2D fixtures; reduces via Slice; delegates to ChebyshevApproximation.Roots |
| 3 | Slider.Minimize + Slider.Maximize | TDD | Same reduction pattern; returns `(value, location)` tuple |
| 4 | TT.Roots + TT.Minimize + TT.Maximize | TDD | User-frame dim and fixedDims; reduces via Slice → ToDense → ChebyshevApproximation.FromValues; user-frame domain validation |
| 5 | TT.SobolIndices TT-native algorithm | TDD | Helper functions in Sensitivity.cs; cross-validation with ToDense + ComputeSobolFromCoeffs; works under non-identity _dimOrder |
| 6 | TT.GetEvaluationPoints user-frame fix | TDD | Inverse permutation by _dimOrder; cache invalidation if _dimOrder changes (it doesn't, but document) |
| 7 | TT.EvalMulti race fix | TDD | EvalStorageFrame helper; concurrent-invocation regression test (4 threads × 1000 calls) |
| 8 | TT.InnerProduct dim_order mismatch + Integrate user-frame error messages | TDD | Two related fixes; both touch ChebyshevTT.cs |
| 9 | Algebra.CheckCompatible numerical tolerance | TDD | DoublesAllClose helper; replace SequenceEqual on Domain[d]; node-count check stays exact |
| 10 | Perf: VectorizedEvalBatch hoist + _optimize_1d vectorized | TDD | Both perf changes have parity-with-old regression tests; no behavior change |
| 11 | Coverage gap fillers + docs/changelog/parity tags + skip_csharp.txt | housekeeping | Lift codecov to ≥96% on Phase 7 additions |

---

## 8. Risks and Mitigations

### 8.1 TT.SobolIndices algorithm — easiest place for off-by-one

The TT-native O(d·n·r²) algorithm has multi-axis bookkeeping; off-by-one in the
storage-frame ↔ user-frame mapping is the primary risk. Mitigation:
- Cross-validation test against `ToDense + Sensitivity.ComputeSobolFromCoeffs` (Phase 6,
  known-correct) to `1e-12` on small TTs.
- Test fixtures cover both identity `_dimOrder` (sanity) and non-identity (via
  `WithAutoOrder`).
- If the cross-validation fails, fallback path is `ToDense() + ComputeSobolFromCoeffs`
  (D1 alternative). Ship-stop only if direct port also fails on small TTs.

### 8.2 EvalMulti race fix — concurrency test reliability

Concurrent regression tests are flaky if not designed carefully. Mitigation:
- Use `Task.WhenAll` with deterministic per-thread inputs.
- Assert on result values, not timing.
- Run 1000 iterations per thread to amplify race window.

### 8.3 Numerical tolerance regression — false positives in CheckCompatible

`DoublesAllClose` is more permissive than `SequenceEqual`. Test that genuinely-different
domains (e.g., `[0, 1]` vs `[0, 1.5]`) still throw (the existing `AlgebraTests` will
catch this regression if `DoublesAllClose` is too permissive).

### 8.4 Perf regression — VectorizedEvalBatch hoist correctness

The hoist changes when the diff-matrix matmul runs (once vs. M times). Numerical
results must match the old impl exactly (no order-of-operations rounding difference).
Mitigation: parity-with-old test in `VectorizedEvalBatchPerfTests.cs` on a 1000-point
batch; if numerical drift exceeds `1e-13`, investigate before merging.

---

## 9. Out of Scope (Explicitly Deferred)

- **Plotting helpers** — Option C from master spec; deferred indefinitely.
- **TT.Roots fast-path** — could short-circuit when `fixedDims` covers all but one dim
  without going through `ToDense`. Premature; only worth it if benchmarks flag it.
- **Sobol with confidence intervals** — Python doesn't have this; out of scope.
- **AlgebraTupleListTests for non-Algebra paths** — covers `_check_compatible` only;
  if other comparison sites have the same bug, they're separate fixes (none discovered
  during pre-spec scan).

---

## 10. Definition of Done

- All 11 tasks complete with two-stage review approval.
- `dotnet test` passes 1098–1102 (target 1100, ±2 drift permitted).
- Codecov patch coverage on Phase 7 additions ≥ 96% (Phase 6 baseline).
- `dotnet build` zero warnings.
- CLAUDE.md status updated with v0.11.0 + parity 0.21.1.
- `skip_csharp.txt` reflects Phase 7 additions (calculus parity rows removed for
  Slider/TT).
- Single PR opened, CI green, squash-merged to main.
- Tag `v0.11.0` created; `publish.yml` triggers NuGet publish.
- GitHub release v0.11.0 with detailed notes.

---

## 11. Glossary

- **User-frame dim** — `d` ∈ `[0, NumDimensions)` representing the original (pre-reorder)
  dimension index. The frame the caller thinks in.
- **Storage-frame dim** — `s` ∈ `[0, NumDimensions)` representing where the data is
  actually stored after reordering. Related to user-frame by `s = Array.IndexOf(_dimOrder, d)`,
  i.e. `_dimOrder[s] = d`.
- **`_dimOrder`** — Permutation array of length `NumDimensions`. `_dimOrder[s] = d` means
  storage position `s` holds original dim `d`. Identity (`[0, 1, ..., n-1]`) for TTs not
  built via `WithAutoOrder`/`Reorder`.
- **TT-native** — Algorithm operates directly on the TT cores without materializing
  `ToDense()`. Memory-cheap; numerically identical to dense path on small TTs.

---

**End of design.**
