# Phase 4 — Ergonomics Polish Design

**Status:** Approved (2026-04-28)
**ChebyshevSharp version:** v0.8.0
**PyChebyshev parity tag:** stays at v0.18.0 (Phase 4 fills v0.15 + v0.16-era ergonomics)
**Branch:** `phase4-ergonomics`
**Predecessor:** Phase 3 (v0.7.0, shipped 2026-04-28)

---

## 1. Goal

Backfill the ergonomics layer that PyChebyshev shipped in v0.15.0 + v0.16.0 across
all four public classes (`ChebyshevApproximation`, `ChebyshevSpline`,
`ChebyshevSlider`, `ChebyshevTT`). These are strictly additive accessors,
introspection getters, deep-copy, and constructor-sugar features — no breaking
changes to the existing API surface.

This is the fourth of six phases in the v0.20.1 port. Phase 4 is the most
fan-out-heavy phase by class count (every feature lands on every class) but
each individual feature is small.

## 2. Non-goals

- Calculus/integration on Slider/TT (Phase 5 work).
- Adaptive refinement, parallel build, Sobol indices (Phase 6 work).
- `is_dimensionality_allowed(num_dim)` static — Python ships it as a forward-
  hook returning `True` for any positive int. We skip it; consumers can add
  per-class capability caps when they're actually needed.
- Reading Python pickle files. ChebyshevSharp continues to bridge only its
  own JSON/`.pcb` format history forward.
- Async I/O variants. YAGNI; ergonomics getters are fast paths.

## 3. Design decisions (chronological brainstorm record)

The following decisions were settled during the brainstorm dialog. Each is
load-bearing for the implementation plan.

### D1: Parity tag stays at v0.18.0

The csproj `<PyChebyshevParity>` tag does not regress to 0.16.0 even though
Phase 4's features come from upstream v0.15 + v0.16. Precedent: Phase 3 stayed
at v0.18.0 while filling the v0.14 `.pcb`-format gap. Treating parity as
*highest contiguous-or-not version we claim parity with* keeps the metadata
monotonic and avoids surprising downstream consumers reading the package version.

`<InformationalVersion>` becomes `0.8.0+pychebyshev.0.18.0`. Changelog and
README badge text frame Phase 4 as "v0.15 + v0.16 ergonomics fill-in."

### D2: Single PR for all four classes

All ~50 tests and ~15 file changes land in one PR. Same precedent as Phases
2 and 3. Splitting at the v0.15/v0.16 boundary or by class would introduce
ordering constraints — `Clone()` needs descriptor + introspection state
already merged; `DeferBuild` needs the descriptor field already there.

### D3: JSON migration via absence-check pattern

When `Load()` encounters a pre-v0.8.0 JSON file, missing fields (`Descriptor`,
`MaxDerivativeOrder`, `SpecialPoints`, `RegisteredDerivativeIds`) deserialize to
defaults via `JsonElement.TryGetProperty(...)`. No schema-version field is
introduced. Files saved by v0.5.0 → v0.7.0 load silently with the new fields
populated as `null` / `1` / `null` / empty-dict respectively. Future breaking
schema changes (none currently planned) can introduce a `SchemaVersion` field
at *that* point with the same absence-check fallback for v0.5.0 → v0.10.0
files.

### D4: `Clone()` is typed per class, not generic

Each class declares `public ChebyshevApproximation Clone()` (or its own
type) returning the concrete type. No generic `T Clone<T>()`, no shared
`ICloneable<T>` interface. The four classes don't share a common base, and a
shared interface would add plumbing for no payoff. Caller writes
`var c = approx.Clone()` and gets the right type back.

### D5: Implicit conversions from raw arrays for typed records

`Domain`, `Ns`, `SpecialPoints` are added as record types with
`public static implicit operator` conversions from `double[][]` (or `int[]`).
The four constructors keep their existing raw-array signatures. Users who
want type-tag safety wrap with `new Domain(...)`; users who don't pass
`double[][]` and the conversion happens silently. **Zero new constructor
overloads** — the records become purely additive type-tag wrappers.

### D6: Method-style accessors, not C# properties

Every new accessor is a method (`GetDescriptor()`, `SetDescriptor(string)`,
`GetMaxDerivativeOrder()`, etc.) not a property. Reasons: (a) name-for-name
parity with the Python source, (b) consistency with existing methods like
`ErrorEstimate()` and `IsConstructionFinished()`, (c) some accessors do
non-trivial work (`GetEvaluationPoints()` lazy-builds and caches), and a
property that hides allocation is misleading.

### D7: `additionalData` is build-time only, instance-stored, not serialized

`additionalData` is a constructor kwarg of type `object?`. During `Build()`,
every `f(point, data)` call passes the stored value as `data`. After build,
`Eval()` doesn't call the function at all (uses precomputed values), so
`additionalData` becomes inert. Stored in a private field for
`GetAdditionalData()` introspection. **Both JSON and `.pcb` omit it** —
same convention as `Function` itself, which is also non-serializable.
Loaded interpolants have `AdditionalData = null`; caller re-supplies if
rebuilding.

### D8: `DeferBuild` and `FromValues` both stay; share an internal code path

`FromValues` is the static factory for "I have values now" workflows.
`DeferBuild=true` + `SetOriginalFunctionValues(values)` is the instance-based
"construct shell, populate when ready" workflow (e.g., async values from a
network fetch). Internally they share the same `_FromGrid`-style code path,
guaranteeing bit-identical output. While `DeferBuild=true` and
`SetOriginalFunctionValues` has not yet been called, `IsConstructionFinished()`
returns `false` and `Eval`/`Save`/etc throw `InvalidOperationException`.

### D9: `GetEvaluationPoints` is lazy + cached

Returns `double[]` of length `numPoints × ndim` in C-order (one row per
evaluation point, dimensions packed contiguously per row). Generated on first
call from precomputed per-dim nodes via Cartesian product, stored in
`_evaluationPoints` field. Subsequent calls return the cached array.
`GetNumEvaluationPoints()` returns `nNodes.Aggregate(1, (a,b) => a*b)` —
no caching needed (cheap arithmetic).

## 4. Public API surface (additions only)

All additions are strictly additive — no existing signatures change.

### 4.1 New record types (top-level namespace `ChebyshevSharp`)

```csharp
public sealed record Domain(double[][] Bounds)
{
    public static implicit operator Domain(double[][] bounds) => new(bounds);
    public static implicit operator double[][](Domain d) => d.Bounds;
}

public sealed record Ns(int[] Counts)
{
    public static implicit operator Ns(int[] counts) => new(counts);
    public static implicit operator int[](Ns n) => n.Counts;
}

public sealed record SpecialPoints(double[][] Points)
{
    public static implicit operator SpecialPoints(double[][] points) => new(points);
    public static implicit operator double[][](SpecialPoints sp) => sp.Points;
}
```

### 4.2 Per-class additions

| Member | Approx | Spline | Slider | TT |
|---|:-:|:-:|:-:|:-:|
| Ctor: `additionalData = null` (`object?`) | ✅ | ✅ | ✅ | ✅ |
| Ctor: `deferBuild = false` (`bool`) | ✅ | ✅ | — | — |
| Ctor: `maxDerivativeOrder = 2` (`int`) | — | — | — | ✅ |
| `void SetDescriptor(string)` | ✅ | ✅ | ✅ | ✅ |
| `string? GetDescriptor()` | ✅ | ✅ | ✅ | ✅ |
| `object? GetAdditionalData()` | ✅ | ✅ | ✅ | ✅ |
| `int GetDerivativeId(int[] orders)` | ✅ | ✅ | ✅ | ✅ |
| `double Eval(double[] point, int derivativeId)` overload | ✅ | ✅ | ✅ | ✅ |
| `bool IsConstructionFinished()` | ✅ | ✅ | ✅ | ✅ |
| `string GetConstructorType()` | ✅ | ✅ | ✅ | ✅ |
| `int[] GetUsedNs()` | ✅ | ✅ | ✅ | ✅ |
| Typed `Clone()` | ✅ | ✅ | ✅ | ✅ |
| `int GetMaxDerivativeOrder()` | ✅ | ✅ | ✅ | ✅ |
| `double? GetErrorThreshold()` | ✅ | ✅ | — | — |
| `double[][]? GetSpecialPoints()` | ✅ | ✅ | — | — |
| `double[] GetEvaluationPoints()` | ✅ | ✅ | ✅ | ✅ |
| `int GetNumEvaluationPoints()` | ✅ | ✅ | ✅ | ✅ |
| `void SetOriginalFunctionValues(double[] values)` | ✅ | ✅ | — | — |

### 4.3 `GetConstructorType()` return values

| Constructor path | Returned tag |
|---|---|
| Function-based ctor + Build | `"function"` |
| `FromValues(...)` factory | `"from_values"` |
| `DeferBuild` + `SetOriginalFunctionValues` | `"from_values"` (bit-identical end state) |
| `Load(...)` (any format) | `"load"` |
| `Clone()` (any source) | `"clone"` |
| `ChebyshevTT.Build(method="cross")` | `"cross"` |
| `ChebyshevTT.Build(method="svd")` | `"svd"` |
| `ChebyshevTT.Build(method="als")` | `"als"` |

## 5. Internal architecture

### 5.1 Shared state added to each class

```csharp
private string? _descriptor;
private object? _additionalData;
private readonly Dictionary<TupleKey, int> _derivativeIdRegistry = new();
private List<int[]> _registeredDerivativeOrders = new();  // index = derivative_id
private string _constructorType = "function";
private int _maxDerivativeOrder;       // Approx/Spline: from ErrorEstimate; TT: from ctor kwarg; Slider: max across slides
private bool _isConstructionFinished;
private double[]? _evaluationPointsCache;  // lazy
```

`TupleKey` is a value-equality wrapper struct around `int[]` (since C# arrays
have reference equality by default). Implements `IEquatable<TupleKey>` and
`GetHashCode` from element-wise hashing. Internal helper in
`Internal/TupleKey.cs`.

### 5.2 New internal file: `Internal/CloneHelpers.cs`

Houses deep-copy primitives shared across all four `Clone()` implementations:

```csharp
internal static class CloneHelpers
{
    public static double[] DeepCopy(double[] src) => (double[])src.Clone();
    public static double[][] DeepCopy(double[][] src) { /* row-by-row copy */ }
    public static double[,] DeepCopy(double[,] src) => (double[,])src.Clone();
    public static int[] DeepCopy(int[] src) => (int[])src.Clone();
    public static int[][] DeepCopy(int[][] src) { /* row-by-row copy */ }
    public static Dictionary<TupleKey, int> DeepCopy(Dictionary<TupleKey, int> src);
    public static List<int[]> DeepCopyOrders(List<int[]> src);
}
```

Each class's `Clone()` allocates a new instance via private "from-precomputed-
state" ctor (already exists for Load), copies all arrays via `CloneHelpers`,
and explicitly sets `Function = null` and `_constructorType = "clone"`.

### 5.3 `additionalData` threading

Existing build code calls `function(point, null)` (or similar). Phase 4
threads the stored `_additionalData` through every such call. The change is
mechanical — replace the `null` arg with `_additionalData` in `Build()`,
`FromValues` private path, and `SetOriginalFunctionValues` paths.

`ChebyshevSpline` propagates `additionalData` to each piece's
`ChebyshevApproximation` constructor. `ChebyshevSlider` propagates it to each
slide. `ChebyshevTT` threads it through `TtCross` / `TtSvd` / `Als` builders.

### 5.4 `GetDerivativeId` registry semantics

Session-local; not serialized. First call to `GetDerivativeId([1, 0])`
registers the orders tuple, returns `0`. Second call to `GetDerivativeId([0, 1])`
returns `1`. Third call to `GetDerivativeId([1, 0])` returns `0` again
(stable per orders-tuple).

`Eval(point, derivativeId)` overload: looks up
`_registeredDerivativeOrders[derivativeId]`, calls existing
`Eval(point, orders)`. Throws `ArgumentOutOfRangeException` if id is unknown.

### 5.5 `GetEvaluationPoints` row-major layout

For `ndim=2`, `nNodes=[3, 2]`, total `numPoints = 6`. Layout in returned
`double[]` of length 12:

```
[ x_0,0 , x_0,1 ,    // row 0: point 0's coords
  x_1,0 , x_1,1 ,    // row 1: point 1's coords
  ...
  x_5,0 , x_5,1 ]
```

Iteration order matches Python: outer index walks the first dim's nodes,
inner index walks the last dim's nodes. Cache is invalidated and rebuilt if
the interpolant is mutated post-construction (none of the public mutators do
this; `SetOriginalFunctionValues` invalidates the cache as a defensive
measure).

### 5.6 JSON serialization changes (per class)

Save writes the new fields after the existing fields:

```jsonc
{
  // ... existing fields (Domain, Ns, TensorValues, etc.) ...
  "Descriptor": "my interpolant",
  "MaxDerivativeOrder": 2,
  "SpecialPoints": [[0.5], [0.7]],          // Approx/Spline only; null otherwise
  "RegisteredDerivativeOrders": [[1, 0], [0, 1]],
  "ConstructorType": "from_values"
}
```

Load uses `JsonElement.TryGetProperty(...)`:

```csharp
_descriptor = root.TryGetProperty("Descriptor", out var d)
    ? d.GetString() : null;
_maxDerivativeOrder = root.TryGetProperty("MaxDerivativeOrder", out var m)
    ? m.GetInt32() : 1;  // Pre-Phase-4 default
// ... etc.
```

`additionalData` is **not** serialized (Python convention; arbitrary `object?`
isn't safely round-trippable through JSON without type info).

### 5.7 `.pcb` format changes

**None.** Phase 4 features are JSON-only. Saving a Phase 4 interpolant via
`Save(path, "binary")` is allowed but the new fields are dropped. Loading a
`.pcb` produces an interpolant with `Descriptor = null`, `MaxDerivativeOrder = 1`,
`SpecialPoints = null` (Approx/Spline), empty derivative-id registry. This
matches Python's behavior at v0.15+v0.16 (binary skips ergonomics).

## 6. Testing

Target: 812 → ~862 passing (+~50). Fan-out across class-specific test files
(or new sibling files for clarity).

### 6.1 New tests by file

| File | New tests | Coverage |
|---|---:|---|
| `ApproxErgonomicsTests.cs` (new) | 12 | descriptor get/set, additionalData threading + introspection, deriv-id registry, Eval-by-id, IsConstructionFinished, GetConstructorType, GetUsedNs, GetMaxDerivativeOrder, GetErrorThreshold, GetSpecialPoints round-trip, GetEvaluationPoints layout |
| `SplineErgonomicsTests.cs` (new) | 12 | same as Approx + per-piece additionalData propagation + per-piece special-points + multi-piece evaluation-points |
| `SliderErgonomicsTests.cs` (new) | 9 | descriptor, additionalData (per-slide propagation), deriv-id, Eval-by-id, introspection, evaluation-points (across all slides), GetMaxDerivativeOrder |
| `TtErgonomicsTests.cs` (new) | 7 | descriptor, additionalData, deriv-id (FD-derivatives respect id lookup), introspection, maxDerivativeOrder ctor kwarg, evaluation-points, GetUsedNs |
| `CloneTests.cs` (new) | 5 | typed return on each class; deep-copy verification (mutating clone doesn't affect source); Function=null on clone; descriptor preserved; serialization round-trips equal |
| `DeferBuildTests.cs` (new) | 4 | Approx unbuilt → SetValues → built; Spline unbuilt → SetValues → built; bit-identical to FromValues; Eval-on-unbuilt throws |
| `JsonMigrationTests.cs` (new) | 4 | pre-v0.8.0 JSON files load with sensible defaults (one per class) |
| `RecordTypesTests.cs` (new) | 3 | Domain/Ns/SpecialPoints implicit conversions both directions; ctor accepts both raw and record forms; record equality |

**Total new: 56.** Some may consolidate during writing-plans (e.g.,
`CloneTests.cs` may absorb the `JsonMigrationTests.cs` clone-vs-load
comparison) — final count target ~50–55.

### 6.2 What we don't test

- Bit-exact JSON byte equality across versions. JSON is human-readable; we
  only assert round-trip equivalence (Save → Load → equal precomputed state).
- `additionalData` survival across Save/Load — by design it's lost.
- `GetEvaluationPoints` for very high-dim cases (memory pressure). Cap test
  cases at `ndim=3` so total points stay under 10⁴.
- Property-based tests for `Clone` deep-copy completeness. Manual coverage
  of the documented mutable fields is sufficient at this surface.

## 7. Migration impact on existing code

- All existing tests continue to pass with no changes — every Phase 4
  addition is strictly additive.
- Existing `Save()` calls produce JSON files with the new fields populated
  (descriptor=null, etc.). Files are larger by a small constant (~50 bytes
  for empty new fields).
- Existing `.pcb` files unchanged.
- Pre-v0.8.0 JSON files load via the absence-check pattern with sensible
  defaults (descriptor=null, max_derivative_order=1, special_points=null).

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `Clone()` deep-copy completeness drift over future phases (Phase 5/6 add new mutable state, Clone forgets to copy it) | Phase 4 adds a `[Theory]` parameterized test that walks every `private` field via reflection, asserting the clone's value is reference-not-equal. Catches missed copies in future phases. |
| `GetEvaluationPoints` cache invalidation bug — cache returned after a `SetOriginalFunctionValues` call | Cache invalidated explicitly in `SetOriginalFunctionValues` (`_evaluationPointsCache = null`). Test asserts post-set value reflects new state. |
| `additionalData` causes silent test breakage in existing function fixtures that use the second `data` arg | Existing fixtures pass `null` as `additionalData` (the new ctor kwarg defaults to null), so behavior is unchanged. Audited during writing-plans. |
| Implicit conversions from `double[][]` make ambiguous overload resolution if we later add a record-typed overload | We commit to NOT adding record-typed constructor overloads (D5). If future phases need them, the implicit conversion goes away or becomes explicit. |
| `GetDerivativeId` registry grows unbounded across long-lived sessions | Registry is per-instance, not per-session. Lifecycle bounded by interpolant lifecycle. Documented in XML doc as "session-local per instance." |
| JSON migration regressions silently produce wrong defaults | Phase 4 commits 4 fixture files (one per class) generated by v0.7.0 into `tests/fixtures/json-pre-v080/`. Migration tests load them and assert the defaults match expected. |

## 9. Definition of done

- [ ] All ~50 new tests passing; existing 812 unchanged.
- [ ] `dotnet build` succeeds with zero warnings.
- [ ] `dotnet test` reports total ≥ 862.
- [ ] XML doc on every new public method/ctor kwarg.
- [ ] JSON migration verified via 4 committed pre-v0.8.0 fixture files.
- [ ] `Clone()` deep-copy completeness test (reflection-based) passes.
- [ ] csproj `<Version>0.8.0`, `<PyChebyshevParity>0.18.0` (unchanged),
  `<InformationalVersion>0.8.0+pychebyshev.0.18.0`.
- [ ] `docs/docs/changelog.md` v0.8.0 entry following two-tier convention
  (PyChebyshev parity stays at v0.18.0).
- [ ] `docs/docs/ergonomics.md` user-guide page covering descriptor,
  additionalData, derivative-id registry, clone, defer-build, typed records.
- [ ] `skip_csharp.txt` updated.
- [ ] `CLAUDE.md` Status block: `812/812` → `862/862` (or whatever the final
  count is); phase list mentions Phase 4 complete.

## 10. Out-of-scope but adjacent

- **`is_dimensionality_allowed(num_dim)`** — Python's forward-hook static
  returning `True` for any positive int. Skipped per non-goal §2. If a
  future phase needs per-class capability caps, add it then.
- **Property-based shape testing** for `GetEvaluationPoints` row-major
  layout. Not worth the FsCheck dependency at this scope.
- **Migration tests for `.pcb` v0.5.0 → v0.7.0 → v0.8.0**. The `.pcb` format
  is untouched in Phase 4. The existing Phase 3 fixture tests already cover
  cross-version `.pcb` reads.
- **A `derivative_id` MRU eviction policy**. Registries stay small in
  practice (≤ ndim²); unbounded growth isn't a real concern.

## 11. Phase 4 summary table

| Aspect | Value |
|---|---|
| ChebyshevSharp version | 0.8.0 |
| PyChebyshev parity tag | 0.18.0 (unchanged from Phase 3) |
| Upstream features ported | v0.15.0 + v0.16.0 (bundled) |
| New tests | ~50 |
| Final test count target | ~862 |
| New public types | `Domain`, `Ns`, `SpecialPoints` records |
| New internal files | `Internal/CloneHelpers.cs`, `Internal/TupleKey.cs` |
| New test files | 8 (see §6.1) |
| Files modified | All 4 public class files + their JSON Save/Load paths |
| PR shape | Single PR, single branch `phase4-ergonomics` |
| Breaking changes | None |
| `.pcb` format changes | None |
| JSON schema migration | Absence-check pattern; no schema-version field |
