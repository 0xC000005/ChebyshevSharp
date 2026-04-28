# Phase 2 — TT Feature Parity (ChebyshevSharp v0.6.0) — Design

**Date:** 2026-04-28
**Author:** Max Zhang (with Claude)
**Status:** Spec — pending user approval before implementation plan
**Parent spec:** [`docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md`](2026-04-27-pychebyshev-v0.20.1-port-design.md) (Phase 2, lines 142–168)

---

## Context

Phase 1 (ChebyshevSharp v0.5.0, PyChebyshev parity v0.12.0) shipped on 2026-04-28: adaptive constructor (`errorThreshold` / `maxN` doubling loop) plus `ChebyshevSpline.WithSpecialPoints(...)`. Phase 2 closes the upstream gap on `ChebyshevTT` by porting PyChebyshev v0.13.0's TT canonicalization + ALS build mode and v0.18.0's TT factories, materialization, slicing, and full algebra. ChebyshevSharp ships as v0.6.0 claiming PyChebyshev parity v0.18.0.

## Decisions

### 1. Work-unit decomposition: single PR

Phase 2 ships as one `phase2-tt-parity` branch / one PR / one v0.6.0 release. Spec line 280 pre-authorized a 2a/2b split (v0.13 then v0.18 as separate work units), but the user elected single-bundle: simpler git topology, one review cycle, one ship event. The internal commit sequence still mirrors the v0.13→v0.18 dependency arrow (orth primitives ship before the algebra rounding that consumes them).

### 2. In-place algebra idiom: void instance methods

C# cannot overload `+=` independently of `+` — the compiler synthesizes `a += b` as `a = a + b`. So "in-place equivalents" become explicit named methods. We adopt the .NET BCL idiom (`Span<T>.Sort()`, `List<T>.Add()`):

```csharp
// Functional (allocates)
var c = a + b;          // ChebyshevTT + ChebyshevTT
var d = 2.0 * a;        // double * ChebyshevTT
var e = -a;             // unary minus

// In-place (mutates receiver, returns void)
a.AddInPlace(b);
a.SubInPlace(b);
a.ScalarMulInPlace(2.0);
a.ScalarDivInPlace(2.0);
a.NegateInPlace();
a.RoundInPlace(1e-10);  // TT-SVD rounding to tolerance
```

No fluent chaining (returning `this`) — atypical for numeric .NET libraries (MathNet, TensorPrimitives don't use it).

### 3. Doc convention: collapse to two version tiers in user-facing docs

Per parent spec §4, ChebyshevSharp keeps its own version line and tracks PyChebyshev parity as metadata. Phase 1's changelog stacked three tiers (CS version + parity + per-feature `(Python vX.Y)` attributions) which proved noisy. Going forward:

| Surface | Tiers shown |
|---|---|
| README badge | parity only (1) |
| Changelog top-of-section | CS version + single parity line (2) |
| Changelog subsections | feature-themed names; **no per-upstream-tag parentheticals** (1) |
| `CLAUDE.md` Status | parity + phase number (2) |
| Spec / plan docs | full granularity — phase number + CS version + per-tag mapping (3) |

Retroactive fix to v0.5.0 changelog applied in commit `830deb6`. Phase 2 changelog follows the new convention.

### 4. Test fixture strategy for stochastic outputs

ALS (like TT-Cross) is seeded-stochastic. `System.Random(seed)` produces a different stream than NumPy's `default_rng(seed)`, so any inline-literal expected values from Python tests will diverge bit-for-bit. We continue the existing pattern (documented in `CLAUDE.md` and used in `TensorTrainCorrectnessTests`): tolerance-based assertions only for ALS-touched outputs, never inline-literal expected values for stochastic results.

Deterministic outputs (orth, inner_product, extrude, slice, to_dense, scalar algebra) are inline-literal-safe and ported 1:1 from Python tests.

---

## Architecture

### Public API additions on `ChebyshevTT`

**v0.13 (canonicalization + ALS):**

| Member | Signature | Notes |
|---|---|---|
| `Build` extension | `Build(method: "cross" \| "svd" \| "als", ...)` | Dispatcher gains `"als"` branch |
| `OrthLeft` | `void OrthLeft(int position)` | In-place left-canonicalization up to `position` |
| `OrthRight` | `void OrthRight(int position)` | In-place right-canonicalization down to `position` |
| `InnerProduct` | `double InnerProduct(ChebyshevTT other)` | Throws `ArgumentException` on grid-mismatch |
| `RunCompletion` | `void RunCompletion(double tolerance, int maxIter)` | Refines existing TT via ALS sweeps; throws if `Function == null` |

**v0.18 (factories, materialization, slicing, algebra):**

| Member | Signature | Notes |
|---|---|---|
| `Nodes` | `static (double[][] Nodes, int[] Shape) Nodes(int numDim, double[][] domain, int[] nNodes)` | Static factory matching Python contract |
| `FromValues` | `static ChebyshevTT FromValues(double[] tensorValues, int numDim, double[][] domain, int[] nNodes, int? maxRank = null, double tolerance = 1e-6)` | TT-SVD compress, skips TT-Cross |
| `Extrude` | `ChebyshevTT Extrude(int dim, (double, double) newDomain, int newN)` | Returns new TT |
| `Slice` | `ChebyshevTT Slice(int dim, double value)` | Returns new TT; `value` must be in domain |
| `ToDense` | `double[] ToDense()` | Row-major flat; throws if `Π nNodes > int.MaxValue / 8` |
| Operators | `+`, `-` (binary), `*` and `/` (scalar both sides), unary `-` | Allocate new TT |
| `AddInPlace` | `void AddInPlace(ChebyshevTT other)` | + TT-SVD round to default tolerance |
| `SubInPlace` | `void SubInPlace(ChebyshevTT other)` | + TT-SVD round |
| `ScalarMulInPlace` | `void ScalarMulInPlace(double scalar)` | |
| `ScalarDivInPlace` | `void ScalarDivInPlace(double scalar)` | Throws `DivideByZeroException` on `0` |
| `NegateInPlace` | `void NegateInPlace()` | |
| `RoundInPlace` | `void RoundInPlace(double tolerance)` | TT-SVD truncation; rank-shrinks |

### Internal restructuring

Existing `Internal/TensorTrainKernel.cs` (901 lines) splits into three files. **The split is a no-op refactor commit landed first**, before any new logic, so subsequent commits add semantics on a clean substrate and reviewers can verify zero behavior change.

| File | Contents after split |
|---|---|
| `Internal/TensorTrainKernel.cs` | `TtCore` struct, column-pivoted QR (Householder), `Maxvol`, `TtCross`, `TtSvd`, `ValueToCoeffCores`. **Adds:** `OrthLeftSweep`, `OrthRightSweep`, `AlsFixedRankSweep`, `AlsAdaptiveRank` |
| `Internal/TensorTrainAlgebra.cs` | **New.** Core-level `AddCores`, `ScalarMulCores`, `NegateCores`, `RoundCores` (TT-SVD truncation), `InnerProductCores` |
| `Internal/TensorTrainExtrude.cs` | **New.** `ExtrudeCores`, `SliceCores`, `ToDenseEinsumChain`, `FromValuesTtSvd` |

`InternalsVisibleTo` already exposes `Internal/*` to test and benchmark assemblies.

### Data flow

```
v0.13 path:
  Build("als")
    └─ AlsAdaptiveRank
        ├─ AlsFixedRankSweep
        │   ├─ OrthLeftSweep
        │   │   └─ Manual Householder QR (existing)
        │   └─ OrthRightSweep
        └─ residual via InnerProductCores

  RunCompletion(tol, maxIter)
    └─ AlsFixedRankSweep loops until residual < tol or maxIter

v0.18 path:
  a + b  ──► AddCores ──► RoundCores ──► new ChebyshevTT
                 │
                 └─ uses OrthLeftSweep / OrthRightSweep (from v0.13)

  a.AddInPlace(b)  ──► AddCores then RoundCores in-place on a's cores

  Slice(d, v)    ──► SliceCores (Chebyshev recurrence eval at v on dim d)
  Extrude(d, …)  ──► ExtrudeCores (insert rank-1 core)
  FromValues(T)  ──► FromValuesTtSvd (TT-SVD on dense tensor)
  ToDense()      ──► ToDenseEinsumChain (sequential core contraction)
```

The v0.18 algebra-rounding step genuinely depends on v0.13's orth primitives — that's why these two upstream tags are bundled.

### Error handling

| Condition | Behavior |
|---|---|
| Grid-mismatch on `InnerProduct`, `+`, `-`, `AddInPlace`, `SubInPlace` | `ArgumentException` with `(numDim, nNodes, domain)` diff. Format mirrors existing `Internal/Algebra.cs` checks. |
| `Slice(dim, value)` with `value` outside domain | `ArgumentOutOfRangeException` |
| `Slice(dim, …)` with `dim < 0` or `dim >= numDim` | `ArgumentOutOfRangeException` |
| `Extrude(dim, …)` with `dim` out of range | `ArgumentOutOfRangeException` |
| `ToDense` with `Π nNodes * 8 > int.MaxValue` | `OverflowException` (allocation guard) |
| `ScalarDivInPlace(0.0)` or `a / 0.0` | `DivideByZeroException` |
| ALS non-convergence (max sweeps reached, residual still above tolerance) | Set `BuildWarning` (same pattern as Phase 1's `maxN`-not-met). Build still completes. |
| `RunCompletion` on TT loaded from disk (`Function == null`) | `InvalidOperationException` — matches Python |
| `FromValues` with `tensorValues.Length != Π nNodes` | `ArgumentException` |

### Testing strategy

- **T1 inline parity:** port Python's `test_tensor_train.py` v0.13 additions + `test_v018_tt_parity.py` 1:1, ~75 new tests as estimated by parent spec.
- **Stochastic ALS:** tolerance-based assertions (`Assert.True(error < bound)`) — never inline-literal expected outputs from Python. Documented precedent in CLAUDE.md.
- **Cross-feature combinations:**
  - `(a + b).Eval(x) ≈ a.Eval(x) + b.Eval(x)` (linearity)
  - `Slice(d, v).Eval(x_minus_d) ≈ a.Eval(x with x[d]=v)` (slice consistency)
  - `ToDense` round-trip → `FromValues` reproduces (within rank tolerance)
  - `(a.AddInPlace(b); a)` and `var c = a + b;` produce equal results
- **Operator symmetry:** `2.0 * a` and `a * 2.0` produce equal results; `a / 2.0` and `0.5 * a` produce equal results.
- **`Save`/`Load` round-trip** at format version 0.6.0; backfill from 0.5.0 files.

### Build sequencing within the single PR

Internal commit ordering preserves the v0.13 → v0.18 dependency arrow:

1. Advance submodule to `v0.18.0` + add empty test stubs (Phase 1 cadence).
2. **Refactor (no behavior change):** split `TensorTrainKernel.cs` into `kernel + algebra + extrude`. All 666 existing tests pass.
3. v0.13.a: `OrthLeftSweep` / `OrthRightSweep` + tests.
4. v0.13.b: `InnerProduct` + tests.
5. v0.13.c: `AlsFixedRankSweep` + `Build(method="als")` adaptive driver + tests.
6. v0.13.d: `RunCompletion` + tests.
7. v0.18.a: `Nodes()` + `FromValues()` factories + tests.
8. v0.18.b: `Extrude` + `Slice` + `ToDense` + tests.
9. v0.18.c: scalar algebra (`*`, `/`, `ScalarMulInPlace`, `ScalarDivInPlace`, unary `-`, `NegateInPlace`) + tests.
10. v0.18.d: `+`, `-` binary, `AddInPlace`, `SubInPlace`, `RoundInPlace` + tests.
11. JSON migration: format version `0.5.0` → `0.6.0`. No new persisted state from either v0.13 (orth/ALS/run_completion mutate existing cores) or v0.18 additions (algebra results are cores, already serialized). Version bumps for consistency; `Load` continues to backfill from older files.
12. Docs: extend `docs/docs/tensor-train.md` (ALS section + algebra section); README badge `v0.12.0` → `v0.18.0`; CLAUDE.md Status updated; `<Version>0.6.0`, `<PyChebyshevParity>0.18.0`; `skip_csharp.txt`; changelog entry per the new two-tier convention.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| ALS rank-adaptive driver oscillates / fails to converge on hard targets | Cap `maxIter` and surface non-convergence via `BuildWarning`; tolerance-based tests rather than asserting convergence at fixed sweep counts. |
| TT-SVD rounding tolerance choice in `RoundInPlace` defaults | Match Python's default exactly. Test that `(a + b).RoundInPlace(eps)` and Python's `(a + b).round(eps)` produce TT with same rank profile (within ±1 from numerical noise). |
| Manual Householder QR numerical stability when ALS pushes ranks higher than TT-Cross typically does | Reuse existing column-pivoted QR; benchmark new ALS test cases against MathNet SVD as oracle in unit tests. |
| Operator overload on `ChebyshevTT` interacting with `ChebyshevApproximation` operators by accident | Operators are only defined for `ChebyshevTT` × `ChebyshevTT` and `ChebyshevTT` × `double`; no implicit conversions. C# overload resolution is strict — no cross-type collisions. |
| File-split refactor commit hides behavior changes in code-motion noise | Land split as a single commit with `git diff -M` rename detection; CI must show 666/666 pass on the post-split commit before any new logic lands. |
| `ToDense` on a 5D TT with reasonable shapes can allocate gigabytes | Hard-throw on `Π nNodes * 8 > int.MaxValue`; document in XML doc that this is for low-D inspection / round-trip testing, not production high-D use. |

## CI / CD impact

- `test.yml` — no change; new tests added under `tests/ChebyshevSharp.Tests/`.
- `publish.yml` — triggered by GitHub release `v0.6.0`; reuses existing flow.
- `docs.yml` — re-deploys on docs changes.
- No new workflow files this phase.

## Documentation plan

| File | Change |
|---|---|
| `docs/docs/tensor-train.md` | Extend with: ALS build mode, run_completion, OrthLeft/Right, InnerProduct, factories (Nodes/FromValues), Extrude/Slice/ToDense, full algebra section. |
| `docs/docs/changelog.md` | Add `## [0.6.0]` entry per the new two-tier convention. |
| `docs/docs/toc.yml` | No new pages; tensor-train.md already in toc. |
| `README.md` | Parity badge bump to `v0.18.0`. |
| `CLAUDE.md` | Status block: PyChebyshev parity v0.18.0, Phase 2 of 6 complete, test count update. |
| `skip_csharp.txt` | Add Phase 2 / TT v0.13+v0.18 entries. |

## Definition of done

- [ ] All commits in sequencing list above landed on `phase2-tt-parity`
- [ ] `dotnet build` zero warnings on net8.0 + net10.0
- [ ] `dotnet test` 100% pass on both TFMs (666 + ~75 = ~741 tests)
- [ ] Submodule pinned at `v0.18.0`
- [ ] `<Version>0.6.0`, `<PyChebyshevParity>0.18.0`, `<InformationalVersion>0.6.0+pychebyshev.0.18.0` in csproj
- [ ] README badge regenerated
- [ ] `docs/docs/changelog.md` v0.6.0 entry follows two-tier convention
- [ ] `docs/docs/tensor-train.md` extended
- [ ] `CLAUDE.md` Status block updated
- [ ] `skip_csharp.txt` updated
- [ ] PR opened, code review pass, merge to main
- [ ] `gh release create v0.6.0` → `publish.yml` ships to NuGet
- [ ] NuGet flat-container index shows `0.6.0`

## What we explicitly do NOT do this phase

- No `.pcb` binary serialization (Phase 3).
- No `additional_data` / `Set/GetDescriptor` / `Clone()` (Phase 4).
- No `ChebyshevTT.Integrate` (Phase 5).
- No `Parallel.For` build / `IProgress<int>` / `WithAutoOrder` (Phase 6).
- No matplotlib bridge (filtered out of port entirely per parent spec §1).
- No public exposure of new internal helpers (`OrthLeftSweep`, `RoundCores`, etc.) beyond `InternalsVisibleTo`.
