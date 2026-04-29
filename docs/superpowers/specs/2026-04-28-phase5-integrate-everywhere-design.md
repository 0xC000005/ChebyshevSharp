# Phase 5 Design: Integrate Everywhere (ChebyshevSharp v0.9.0)

**Master spec section:** `docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md` lines 216-227.

**Date:** 2026-04-28
**Author:** brainstorming session, auto mode
**Phase:** 5 of 6 in the PyChebyshev v0.20.1 port
**Target version:** ChebyshevSharp v0.9.0 (PyChebyshev parity tag: v0.17.0)
**Submodule bump:** none — `ref/PyChebyshev` stays at v0.18.0 (already contains v0.17 features)
**Test delta:** 902 → 946 (+44)
**PR strategy:** single PR

## 1. Goal

Port PyChebyshev v0.17.0's "integrate everywhere" feature: add `Integrate()` on
`ChebyshevSlider` and `ChebyshevTT`, completing calculus integration coverage
across all four interpolant classes. Phase 1 added it to `ChebyshevApproximation`;
Phase 2 added it to `ChebyshevSpline`. After Phase 5, every C# class has the
identical `Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)`
signature.

Roots / Minimize / Maximize on Slider and TT remain deferred — Python defers
those to v0.21 and we follow the same staging.

## 2. Scope

### In scope

- `ChebyshevSlider.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial.
- `ChebyshevTT.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial, works for `cross`/`svd`/`als` build modes.
- New internal helpers in `Internal/Calculus.cs`: `SliderPartitionIntersect`, `IntegrateTtAlongDim`.
- 44 new tests across 3 files (2 new, 1 extended).
- Documentation: extend `docs/docs/calculus.md` with Slider/TT sections; update changelog.
- Version bump: 0.8.0 → 0.9.0; parity tag 0.18.0 → 0.17.0 (non-monotonic batch tracker, see D4).

### Out of scope (deferred to Phase 6 / future)

- Roots / Minimize / Maximize on Slider and TT (Python defers to v0.21).
- TT JSON migration — Phase 5 adds no new persisted state on TT.
- Submodule bump — already at v0.18.0.
- Build perf, adaptive refinement, Sobol indices — all Phase 6 (v0.10.0).

## 3. Approach (locked: Approach 1)

**Faithful Python port.** Single signature on Slider and TT exactly matching the
existing `ChebyshevApproximation.Integrate` and `ChebyshevSpline.Integrate`
signatures — `object` return type for full-or-partial polymorphism, `int[]?`
dims, `(double lo, double hi)[]?` bounds. Algorithms ported line-for-line from
`ref/PyChebyshev/src/pychebyshev/slider.py:877-1132` and
`ref/PyChebyshev/src/pychebyshev/tensor_train.py:1487-1635`.

Considered and rejected:
- **Two-method API** (`double IntegrateFull()` + typed-return `IntegratePartial`):
  diverges from established Approx/Spline pattern; cross-class API drift.
- **Fluent builder** (`tt.Integrate().Over(...).Result()`): no precedent in
  the codebase.

## 4. Public API

### 4.1 Method signatures

```csharp
// ChebyshevSlider.cs (new method)
public object Integrate(
    int[]? dims = null,
    (double lo, double hi)[]? bounds = null)

// ChebyshevTT.cs (new method)
public object Integrate(
    int[]? dims = null,
    (double lo, double hi)[]? bounds = null)
```

### 4.2 Return-type contract

| Input | Returns |
|---|---|
| `dims = null` (or `dims` covers every dim) | `(object)(double)` — boxed scalar |
| `dims` covers some dims | `(object)(ChebyshevSlider)` or `(object)(ChebyshevTT)` |

Caller casts: `(double)slider.Integrate()` for full; `(ChebyshevSlider)slider.Integrate(dims: new[] {0})` for partial. Matches `ChebyshevApproximation.Integrate` and `ChebyshevSpline.Integrate` exactly.

### 4.3 Parameter normalization

- `dims = null` → integrate over every dimension.
- `dims` is sorted + deduplicated internally.
- `bounds = null` → integrate each requested dim over its full domain.
- `bounds` is positional with sorted-`dims`. Each entry is `(lo, hi)` or
  uses `(double.NaN, double.NaN)` (or could be skipped via per-element
  null — but C# value-tuples don't nullably nest, so we re-use the existing
  C# convention from Phase 1 where each entry is required). See D9.

### 4.4 Validation

| Failure | Exception |
|---|---|
| `Built` is false | `InvalidOperationException` |
| `dims` index < 0 or ≥ NumDimensions | `ArgumentException` |
| `dims` has duplicated indices | `ArgumentException` |
| `bounds.Length != dims.Length` | `ArgumentException` |
| `bounds[i].lo` < domain[d].lo or `bounds[i].hi` > domain[d].hi | `ArgumentException` |
| `bounds[i].lo >= bounds[i].hi` | `ArgumentException` |

All failure messages match Python wording where the Python message is
informative, otherwise C#-idiomatic ("dim N out-of-range [0, M]" etc.).

## 5. Algorithms

### 5.1 ChebyshevSlider.Integrate (Python `slider.py:877-1132`)

**Identity.** A `ChebyshevSlider` represents
$$f(x) \approx \mathrm{pv} + \sum_i \big[s_i(x_{G_i}) - \mathrm{pv}\big]$$
where pv = pivot value, G_i = i-th partition group, s_i = i-th slide
(a `ChebyshevApproximation` over the dims in G_i).

**Integrate over T (a sub-domain across some or all dims):**
$$\int_T f \, dx = \mathrm{pv} \cdot \mathrm{vol}(T) + \sum_i \int_T \big[s_i(x_{G_i}) - \mathrm{pv}\big] \, dx$$

**Per-slide classification via `SliderPartitionIntersect(group, dims)`:**

| kind | meaning | action |
|---|---|---|
| `"full"` | every dim of G_i is in T | slide collapses to scalar I_i = `s_i.Integrate(localDims, localBounds)`; contribution `vol(T \ G_i) · (I_i − pv · vol(G_i ∩ T))` folds into pv_new |
| `"partial"` | some dims of G_i are in T | reduce slide via `s_i.Integrate(localDims, localBounds)` → smaller `ChebyshevApproximation` over kept dims; apply unified rule below |
| `"none"` | no dims of G_i are in T | slide passes through; apply unified rule below |

**Unified rule for surviving slide tensors** (Python `slider.py:1040-1041, 1067-1071`):
```
new_tensor = scale * source_tensor + (pv_new - pv * vol_T)
```
where:
- `scale = vol_T` for "none" slides
- `scale = vol_outside = ∏ widths over T \ G_i` for "partial" slides

New slide is built via `ChebyshevApproximation._FromGrid(reduced, new_tensor)`.

**pv_new accumulator (the new pivot value):**
```
pv_new = pv * vol_T  +  Σ_{i: full} vol(T \ G_i) * (I_i − pv * vol(G_i ∩ T))
```

**Full integration (every dim integrated)** → returns `(object)(double)pv_new`.

**Partial integration** → constructs a new `ChebyshevSlider` via factory bypass with:
- `Domain`, `NumDimensions`, `NumNodes`, `PivotPoint` projected onto surviving dims.
- `Partition` reindexed via old→new dim map; "full" slides dropped, "none"/"partial" slides reindexed.
- `Slides` = list of new `ChebyshevApproximation` instances (one per surviving slide).
- `PivotValue = pv_new`.
- Inherits `Descriptor`, `_additionalData`, `_maxDerivativeOrder`, `_constructorType`.
- **Resets** `_derivativeIdRegistry = {}` and `_registeredDerivativeOrders = []` (Python `slider.py:1130-1131`).
- `Built = true`, `_cachedErrorEstimate = null`.

### 5.2 ChebyshevTT.Integrate (Python `tensor_train.py:1487-1635`)

**Per-dim quadrature weights:**

| bounds[d] | weights |
|---|---|
| `null` (full domain) | `Calculus.ComputeFejer1Weights(n) * (b - a) / 2` |
| `(b_lo, b_hi)` sub-interval | map to reference: `t_lo = 2(b_lo-a)/(b-a) − 1`, `t_hi = 2(b_hi-a)/(b-a) − 1`; then `Calculus.ComputeSubIntervalWeights(n, t_lo, t_hi) * (b - a) / 2` |

**Per-integrated-dim contraction** (Python `tensor_train.py:1571-1573`):

```csharp
var valueCore = TensorTrainKernel.CoeffCoreToValueCore(_coeffCores[d]);
contracted[d] = Calculus.IntegrateTtAlongDim(valueCore, weightsPerDim[d]);
// shape: (rLeft, rRight) — collapsed scalar at the node axis
```

Critical: TT-cores live in Chebyshev coefficient space. Must convert to value
space before applying Fejér-1 weights (which are defined for value-space samples
at Type-I Chebyshev nodes).

**Full integration** (every dim integrated) → chain-multiply all M_k matrices:
```
result = contracted[dims[0]]
for d in dims[1:]: result = result @ contracted[d]
return (object)(double)result[0, 0]
```

**Partial integration** (Python `tensor_train.py:1582-1608`) → walk the TT chain,
absorbing each contracted matrix into a neighboring kept core's left rank dim:

```csharp
double[,]? pending = null;
var newCores = new List<TtCore>();
for (int k = 0; k < numDimensions; k++)
{
    if (integratedSet.Contains(k))
    {
        var M = contracted[k];
        if (pending != null) M = MatMul(pending, M);
        pending = M;
    }
    else  // kept dim
    {
        var core = _coeffCores[k].Copy();
        if (pending != null) { core = AbsorbLeft(pending, core); pending = null; }
        newCores.Add(core);
    }
}
// trailing pending: absorb into last kept core's right rank
if (pending != null && newCores.Count > 0)
    newCores[^1] = AbsorbRight(newCores[^1], pending);
```

`AbsorbLeft(M, core)` and `AbsorbRight(core, M)` are matrix-times-core einsums:
- `AbsorbLeft`: `core[l, j, s] = Σ_r M[l, r] * core[r, j, s]`
- `AbsorbRight`: `core[l, j, r] = Σ_s core[l, j, s] * M[s, r]`

(These are not new — `TensorTrainAlgebra` already has equivalent helpers from
Phase 2 algebra. We'll reuse them or add minimal new ones if needed.)

**Construct result TT** via factory bypass:
- `Domain`, `NumDimensions`, `NumNodes` projected onto surviving dims.
- `_coeffCores = newCores`, `_ttRanks` recomputed from new cores' rank dims.
- `_built = true`, `_buildTime = 0`, `_totalBuildEvals = 0`, `_cachedErrorEstimate = null`.
- Inherits `Descriptor`, `_additionalData`, `_maxRank`, `_tolerance`, `_maxSweeps`,
  `_maxDerivativeOrder`, `Method`.
- `_constructorType` is **not** explicitly set; `GetConstructorType()` falls back
  to `Method ?? "function"`, so it preserves the original build mode (D3).

## 6. Internal Architecture

### 6.1 New helpers in `Internal/Calculus.cs`

```csharp
internal static (string kind, int[] kept) SliderPartitionIntersect(
    int[] groupDims, int[] integrateDims)
{
    var groupSet = new HashSet<int>(groupDims);
    var integrateSet = new HashSet<int>(integrateDims);
    var overlap = new HashSet<int>(groupSet);
    overlap.IntersectWith(integrateSet);
    if (overlap.Count == 0) return ("none", (int[])groupDims.Clone());
    if (overlap.SetEquals(groupSet)) return ("full", Array.Empty<int>());
    return ("partial", groupDims.Where(d => !integrateSet.Contains(d)).ToArray());
}

internal static double[,] IntegrateTtAlongDim(
    TensorTrainKernel.TtCore core, double[] weights)
{
    var result = new double[core.RLeft, core.RRight];
    for (int r = 0; r < core.RLeft; r++)
        for (int s = 0; s < core.RRight; s++)
        {
            double acc = 0.0;
            for (int j = 0; j < core.N; j++) acc += core[r, j, s] * weights[j];
            result[r, s] = acc;
        }
    return result;
}
```

Both `internal static` for unit-testability from the test project (D5).

### 6.2 No new files

Phase 5 is purely additive on existing classes. No new public types, no new
internal classes. Calculus.cs gets two new methods; Slider.cs and TT.cs each
get one new public method plus any private factory-bypass helpers.

## 7. Tests (44 total)

### 7.1 New: `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs` (~17 tests)

Mirror of Python `TestSliderFullIntegrate` (4) + `TestSliderPartialIntegrate` (7)
+ relevant cross-class consistency (3) + validation (3):

- Full integration accuracy on 1-group slider (analytical reference).
- Full integration on multi-group slider matching closed-form expression.
- Full integration on 5D slider matching numerical reference quadrature.
- Full integration over sub-domain bounds.
- Partial integration single dim — surviving slider Eval matches expected reduction.
- Partial integration multi-dim within one group.
- Partial integration crossing partition boundaries (multiple groups affected).
- Partial integration where one group becomes "full" (group dropped from result).
- Partial integration where one group is "none" (passes through with shift).
- Sub-domain bounds on partial integration.
- Result type — full returns `double` (boxed); partial returns `ChebyshevSlider`.
- Descriptor passthrough on partial result.
- AdditionalData passthrough on partial result.
- Derivative-id registry **reset** on partial result (was non-empty on input → empty on output).
- Validation: out-of-range dim → `ArgumentException`.
- Validation: duplicated dim → `ArgumentException`.
- Validation: bounds outside domain → `ArgumentException`.

### 7.2 New: `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs` (~22 tests)

Mirror of Python `TestTTFullIntegrate` (6) + `TestTTPartialIntegrate` (7)
+ `TestTTIntegrateBoundsAndValidation` (6) + cross-class (3):

- Full integration accuracy on TT-cross 3D vs reference.
- Full integration accuracy on TT-svd 3D vs reference.
- Full integration accuracy on TT-als 3D vs reference (parity across build modes).
- Full integration on 5D TT (TtBs5D fixture).
- Full integration over sub-domain bounds.
- Cross-mode consistency: cross/svd/als builds of same function give same integral.
- Partial integration single dim — leading position (dim 0).
- Partial integration single dim — middle position.
- Partial integration single dim — trailing position (last dim).
- Partial integration multi-dim — adjacent dims.
- Partial integration multi-dim — non-adjacent dims (gap in integrated set).
- Partial integration with sub-domain bounds.
- Result-type test — full returns `double`; partial returns `ChebyshevTT`.
- Build mode preserved: cross-built TT integrated partially → result.GetConstructorType() == "cross".
- Same for svd → "svd"; als → "als".
- Descriptor + additionalData passthrough on partial result.
- Inherited build params (max_rank, tolerance, max_sweeps, max_derivative_order).
- Result is a fully working TT — partial result.Eval, .EvalBatch, .Integrate (recursive) all work.
- Validation: out-of-range dim, duplicated dim, bounds outside domain, lo≥hi, build-required guard.

### 7.3 Appended to existing `tests/ChebyshevSharp.Tests/CalculusTests.cs` (~5 tests)

- `SliderPartitionIntersect`: full overlap returns `("full", [])`.
- `SliderPartitionIntersect`: no overlap returns `("none", group)`.
- `SliderPartitionIntersect`: partial overlap returns `("partial", kept)` with kept-dim ordering preserved.
- `SliderPartitionIntersect`: empty integrate-dims returns `("none", group)`.
- `IntegrateTtAlongDim`: numerical accuracy on a hand-rolled (rLeft=2, n=4, rRight=3) core with known integrand.

### 7.4 Test count progression

| After task | Total tests |
|---|---|
| Baseline (Phase 4 complete) | 902 |
| Task 1 (helpers + helper tests) | 907 |
| Task 2 (Slider Integrate full) | ~916 |
| Task 3 (Slider Integrate partial) | ~924 |
| Task 4 (TT Integrate full) | ~934 |
| Task 5 (TT Integrate partial) | ~942 |
| Task 6 (Slider validation + cross-class consistency) | ~944 |
| Task 7 (TT validation + cross-class consistency) | ~946 |
| Task 8 (release prep — no new tests) | 946 |

Final: 946. (Drift of ±2 acceptable; documented forward.)

## 8. Documentation Updates

- `docs/docs/calculus.md` — add Slider Integration and TT Integration subsections.
  Brief examples with code blocks. Note that Roots/Min/Max remain deferred (Phase 6/v0.21).
- `docs/docs/changelog.md` — v0.9.0 entry following two-tier convention used in
  Phases 1–4. Top tier: "Slider/TT integration completes calculus parity across
  all four classes." Bottom tier: API additions list, design decisions summary.
- `docs/docs/toc.yml` — no change (calculus page already exists; only adds subsections).
- `skip_csharp.txt` — append Phase 5 entry showing the Python tests now ported
  (full v0.17 calculus completion).
- `CLAUDE.md` — bump Status block (902 → 946; mark Phase 5 complete; phase list now 1+2+3+4+5 of 6).

## 9. Release Prep

- `src/ChebyshevSharp/ChebyshevSharp.csproj`:
  - `<Version>0.9.0</Version>`
  - `<PyChebyshevParity>0.17.0</PyChebyshevParity>` (down from 0.18.0 — see D4)
  - `<InformationalVersion>0.9.0+pychebyshev.0.17.0</InformationalVersion>`
- Submodule: stays at v0.18.0. Confirm with `git -C ref/PyChebyshev rev-parse HEAD` matching the v0.18.0 tag.

## 10. Design Decisions (locked)

### D1 — Single signature, `object` return type (Approach 1)

Both `ChebyshevSlider.Integrate` and `ChebyshevTT.Integrate` use the identical
signature already used by `ChebyshevApproximation.Integrate` and
`ChebyshevSpline.Integrate`:

```csharp
public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
```

**Why:** cross-class API consistency. Caller pattern is uniform — `(double)x.Integrate()` for full, `(T)x.Integrate(dims: ...)` for partial.

**Considered:** strongly-typed split (two methods), fluent builder. Both rejected as inconsistent with existing pattern.

### D2 — Port all 44 Python tests (full parity)

**Why:** Phases 1–4 all hit Python parity (Phase 4 even slightly exceeded with the reflection-based clone audit). Going under feels like deliberate parity drift. Master spec's "~25" estimate was a hand-wave.

### D3 — Partial integrate result preserves original build mode

`result.GetConstructorType()` returns the original's build method (TT: `"cross"` / `"svd"` / `"als"`; Slider: original `_constructorType`).

**Why:** matches Python (`tensor_train.py:1634`: `result_tt.method = self.method`).
The semantic "this TT originated from a cross build" is preserved through derived
operations — same as `Slice()`, `Extrude()`, `Clone()` already do per Phase 4.

**Considered:** setting `_constructorType = "integrate"` to flag derivation. Rejected — no Python precedent; user can detect derivation by checking dimensions.

### D4 — Parity tag drops 0.18.0 → 0.17.0 (non-monotonic batch tracker)

**Why:** the parity tag is established (Phase 4 spec D9, CLAUDE.md note) as a
non-monotonic indicator of "the most recent feature batch we ported." Phase 4
filled in v0.15+v0.16 *behind* the v0.18.0 binary format that was already
shipped, so the tag stayed at 0.18.0. Phase 5 ports v0.17.0 features (calculus
completion), so the tag drops to 0.17.0 to indicate which batch was just
delivered. Phase 6 will advance to v0.20.1.

**Documentation:** the changelog entry calls this out explicitly so cross-language consumers don't read the tag drop as a regression.

**Considered:** skip-ahead to 0.18.0 (we're at parity with v0.18 features now anyway). Rejected because it conflates "feature-complete with v0.18" (true after Phase 4) with "this release adds v0.17 features" (true now) — the tag tracks the latter.

### D5 — Internal helpers exposed as `internal` (not `private`)

**Why:** Unit tests in `tests/ChebyshevSharp.Tests/CalculusTests.cs` need to
call `SliderPartitionIntersect` and `IntegrateTtAlongDim` directly. Existing
convention from Phase 1: `BarycentricKernel`, `Algebra`, `ExtrudeSlice`,
`Calculus` are all `internal` classes whose methods are unit-tested directly.

`tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` already has
`<InternalsVisibleTo>` configured.

### D6 — TT result inherits all build params

`max_rank`, `tolerance`, `max_sweeps`, `max_derivative_order`, `descriptor`,
`additional_data`, `Method` all carry through to the partial-integrate result.

**Why:** matches Python (`tensor_train.py:1628-1634`). The result is "the same TT minus integrated dims" semantically; build-quality params describe how the original was built and stay relevant context.

### D7 — Slider result resets derivative-id registry

`_derivativeIdRegistry = {}` and `_registeredDerivativeOrders = []` on the new Slider.

**Why:** matches Python (`slider.py:1130-1131`: `obj._derivative_id_registry = {}; obj._derivative_id_to_orders = []`). The dim space changed; previously-registered derivative-id mappings refer to old dim indices that no longer exist.

**Asymmetry note:** TT's partial-integrate result does NOT reset its registry in Python (the integrate method on TT doesn't touch the registry). C# matches this asymmetry — TT result inherits its registry as-is. A separate followup could address whether this is correct (re-indexing might be needed in some edge cases), but for v0.17 parity we match Python.

### D8 — TT cores converted coeff→value before quadrature

`TensorTrainKernel.CoeffCoreToValueCore(_coeffCores[d])` is invoked on each
integrated core before `IntegrateTtAlongDim` is applied.

**Why:** TT cores are stored in Chebyshev coefficient space (Phase 4 design,
`ValueToCoeffCores` in TT constructor). Fejér-1 weights are defined for value
samples at Type-I Chebyshev nodes. Applying the weights to coefficient cores
directly would compute the wrong integral.

Python source: `tensor_train.py:1572` `val_core = _coeff_core_to_value_core(self._coeff_cores[d])`.

### D9 — Bounds API: `(double lo, double hi)[]?` not `double[][]?`

Master spec line 222-223 hand-waved the bounds shape as `double[][]?`. The
established C# convention (Phase 1 Approx, Phase 2 Spline) is the value-tuple
array `(double lo, double hi)[]?` because:

- Stronger typing — caller writes `new[] { (0.0, 1.0), (2.0, 3.0) }` not nested
  arrays where the inner length must be 2 (validated at runtime only).
- Existing `Calculus.NormalizeBounds` already accepts this shape.

Phase 5 follows this established pattern.

### D10 — Single PR

Phase 5 is small (~44 tests, 2 new methods on existing classes, 2 new helpers).
Single PR with all tasks. Same as Phase 5's "clean conceptual unit" framing in
master spec line 45. Same approach worked well in Phase 3 (single PR for binary
format, ~110 net tests) and Phase 4 (single PR for ergonomics, ~90 net tests).

## 11. Open Risks (small)

| Risk | Mitigation |
|---|---|
| TT-als build mode has stochastic cores; partial-integrate result preservation of `Method = "als"` is correct but tests must use tolerance assertions, not bit-exact, when comparing across runs | Established pattern from Phase 4 (Phase 2 TT tests already do this). Reuse `AssertClose` helper. |
| Slider partial-integrate's "partition-of-unity shift" math (the unified rule) is subtle; implementer may get the `pv_new - pv * vol_T` term wrong | Python source comments at `slider.py:1085-1097` explicitly describe the math. Plan must replicate that comment block in the C# implementation for future readers. Tests verify Eval match against numerical-reference quadrature. |
| Result of partial-integrate TT must be a fully-functional TT — Eval, EvalBatch, Integrate (recursive) must all work | Test 7.2 includes "result is a fully working TT — partial result.Eval, .EvalBatch, .Integrate (recursive) all work" specifically to catch any half-built result. |

No major risks identified. Phase 5 is the smallest of the 6 phases by a clear margin.

## 12. Worktree

Branch: `phase5-integrate-everywhere`.
Path: `.worktrees/phase5-integrate-everywhere` (project-local; `.worktrees/` already gitignored).
Baseline: `main` at `39223fe`, 902/902 tests passing, submodule `ref/PyChebyshev` at v0.18.0.

Created via `superpowers:using-git-worktrees` skill once the spec is approved.
