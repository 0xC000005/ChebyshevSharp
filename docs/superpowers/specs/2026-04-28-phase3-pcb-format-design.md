# Phase 3 — Binary `.pcb` Format Design

**Status:** Approved (2026-04-28)
**ChebyshevSharp version:** v0.7.0
**PyChebyshev parity tag:** stays at v0.18.0 (Phase 3 fills in a v0.14-era feature)
**Branch:** `phase3-pcb-format`
**Predecessor:** Phase 2 (v0.6.0, shipped 2026-04-28)

---

## 1. Goal

Ship a portable little-endian binary serialization format (`.pcb`) that lets
C/Rust/Julia consumers read ChebyshevSharp interpolants without needing the .NET
runtime. Bytes must be byte-for-byte compatible with PyChebyshev v0.14's `.pcb`
format (v1, fixed-header + length-prefixed sections + raw f64/uint32 in C-order).

This is the third of six phases in the v0.20.1 port. Phase 3 is intentionally
the smallest phase — the format is well-bounded by the Python reference at
`ref/PyChebyshev/docs/user-guide/binary-format.md` and `_binary.py`.

## 2. Non-goals

- New format features beyond what Python v0.14 ships. (Phase 4+ may add
  fields; we'd bump `minor`.)
- `ChebyshevSlider.Save(format="binary")` and `ChebyshevTT.Save(format="binary")`.
  Python keeps these pickle-only at v0.14; we keep them JSON-only.
- Async I/O variants (`SaveAsync`/`LoadAsync`). YAGNI; format is small.
- Big-endian platform support. .NET supports BE in theory; no realistic deploy
  target. We assert `BitConverter.IsLittleEndian` at module load.
- Cross-language drift detection in CI. Hermetic committed fixtures are enough
  (see §8).
- Bumping `<PyChebyshevParity>`. Phase 2 already claimed v0.18.0; Phase 3 is
  filling a v0.14-era gap, not advancing the tag.

## 3. Design decisions (chronological brainstorm record)

### 3.1 `PeekFormatVersion` lands in Phase 3 (not Phase 4)

PyChebyshev only added `peek_format_version()` in v0.16.0, but the master spec
already calls for it under Phase 3. We keep it in Phase 3.

**Rationale:** the implementation is ~10 lines and depends only on the binary
header layout. Splitting it from the rest of `PcbFormat.cs` would force a tiny
follow-up PR in Phase 4 with no organizational benefit. The `<PyChebyshevParity>`
tag is a forward-looking promise honored at v0.10.0; small in-phase deviations
are normal in a port. The v0.7.0 changelog notes "+ v0.16 PeekFormatVersion
early arrival" as a transparency signal.

### 3.2 Hermetic committed fixtures (no submodule reads at test time)

Three Python-shipped `.pcb` files (`approx_2d_simple.pcb`, `approx_5d_bs.pcb`,
`spline_1d_kink.pcb`) live in `tests/fixtures/` and are committed to the
ChebyshevSharp repo. Tests load them from there.

**Rationale:** Phase 3's whole point is "files we wrote can be read elsewhere"
— proving that requires our code reading bytes that an external party produced
and trusts. Hermeticity matters more than auto-tracking upstream churn. The
format is at v1 and locked; v1.1 would be a deliberate upstream spec bump that
we'd respond to deliberately. The three fixtures are documented byte-for-byte
in Python's `binary-format.md` — they are effectively part of the format spec.

### 3.3 Hard-throw on nested-nNodes Spline binary save (matches Python)

Python's `_binary.py` line 250 raises `NotImplementedError` with the message:
> `"binary format requires flat n_nodes (shared across pieces); use format='pickle' for nested-n_nodes splines"`

We throw `NotSupportedException` with an analogous message (s/`pickle`/`json`/).
The master spec's earlier "warn and fall back" wording was a drafting error and
is hereby corrected.

**Rationale:** silent format substitution is the worst possible UX for
cross-language workflows. If a user passes `format="binary"` and we save JSON
under a `.pcb` extension, a downstream C/Rust consumer fails confusingly.
Hard-throw with a clear message is honest and gives the caller a deliberate
choice (drop nested nNodes, switch to JSON, or reshape their problem).

### 3.4 No parity CI workflow in Phase 3

No `.github/workflows/parity.yml`. CI stays at `dotnet test` on .NET 8 + 10. No
Python install, no submodule init, no cross-language re-derivation.

**Rationale:** the realistic drift surface is tiny (format is at v1; bumps are
deliberate). The committed fixture bytes are documented in `binary-format.md`
so they are essentially part of the format spec. The first time we manually
bump the submodule (Phase 4 → v0.16.0), we `cmp` the fixture bytes as a
one-line check. If we ever feel the need for automated drift detection, it's a
30-minute add-on PR.

## 4. Public API surface (additions only)

### 4.1 `ChebyshevApproximation`

```csharp
// Existing today: void Save(string path)
public void Save(string path, string format = "json");
//   format ∈ { "json", "binary" }
//   Throws ArgumentException for unknown values.
//   Throws NotSupportedException if format="binary" and the instance has
//     state that the binary format cannot represent. (For Approximation,
//     no such state exists in v0.7.0; reserved for future Ergonomics fields.)

// Existing today: static ChebyshevApproximation Load(string path)
//   Behavior change: now sniffs first 4 bytes.
//   b"PCB\x00" → binary path; otherwise → JSON path (unchanged behavior).
//   Throws InvalidDataException for malformed binary files.

// New:
public static int PeekFormatVersion(string path);
//   Returns the major version byte from a binary file (1 for v1).
//   Throws InvalidDataException if path is not a binary file (no PCB magic).
//   Throws FileNotFoundException if path does not exist.
```

### 4.2 `ChebyshevSpline`

Same three additions as Approximation. `PeekFormatVersion` is shared logic;
`Save(format="binary")` dispatches to the spline body writer; `Load` auto-detect
routes to `_pcb` reader for class_tag=2.

**Spline-specific:** `Save(path, format="binary")` throws `NotSupportedException`
if the spline was built with nested per-piece `nNodes` (the `int[][]` form
introduced in Phase 1 for special points).

### 4.3 `ChebyshevSlider` and `ChebyshevTT`

No API changes in Phase 3. Their existing `Save(string path)` signature stays
unchanged and JSON-only. No `format` parameter is added; there is therefore no
`format="binary"` code path on these classes to either succeed or throw. Users
who want binary serialization of a Slider/TT must wait for Phase 4+ (and only
if Python adds it upstream).

## 5. Internal architecture

### 5.1 New file: `src/ChebyshevSharp/Internal/PcbFormat.cs`

Holds:

- **Constants:** `Magic` (4 bytes `0x50, 0x43, 0x42, 0x00`), `MajorVersion = 1`,
  `MinorVersion = 0`, `ClassTagApproximation = 1`, `ClassTagSpline = 2`,
  `HeaderSize = 12`.
- **Static ctor assertion:** `Debug.Assert(BitConverter.IsLittleEndian)`.
- **Header structures:** internal `record struct PcbHeader(int Major, int Minor, int ClassTag)`.
- **Reader entry points:**
  - `static PcbHeader ReadHeader(BinaryReader r)` — validates magic, version, reserved.
  - `static (double[][] domain, int[] nNodes, double[] tensorValues) ReadApproximationBody(BinaryReader r)`
  - `static (double[][] domain, int[] nNodes, double[][] knotsPerDim, double[][] pieceTensors) ReadSplineBody(BinaryReader r)`
- **Writer entry points:**
  - `static void WriteHeader(BinaryWriter w, int classTag)`
  - `static void WriteApproximation(BinaryWriter w, ChebyshevApproximation cheb)`
  - `static void WriteSpline(BinaryWriter w, ChebyshevSpline spline)`
- **Detection helpers:**
  - `static bool IsBinary(string path)` — returns true if file exists and starts with `Magic`.
  - `static int PeekFormatVersion(string path)` — public-via-class-static-method, reads byte 4.

### 5.2 Save/Load wiring

In `ChebyshevApproximation.cs`:

```csharp
public void Save(string path, string format = "json")
{
    switch (format)
    {
        case "json":
            SaveJson(path);            // existing private method (rename current SaveImpl)
            break;
        case "binary":
            using (var fs = File.Create(path))
            using (var w = new BinaryWriter(fs))
                PcbFormat.WriteApproximation(w, this);
            break;
        default:
            throw new ArgumentException($"Unknown format '{format}'. Expected 'json' or 'binary'.", nameof(format));
    }
}

public static ChebyshevApproximation Load(string path)
{
    if (PcbFormat.IsBinary(path))
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        var header = PcbFormat.ReadHeader(r);
        if (header.ClassTag != PcbFormat.ClassTagApproximation)
            throw new InvalidDataException(
                $"binary file is class_tag={header.ClassTag} (Spline?) — call ChebyshevSpline.Load instead");
        var (domain, nNodes, tensor) = PcbFormat.ReadApproximationBody(r);
        return FromTensor(domain, nNodes, tensor);  // existing factory or new private ctor
    }
    return LoadJson(path);                          // existing private method
}
```

`ChebyshevSpline.Load` mirrors this with `ClassTagSpline`.

## 6. Format spec (verbatim from Python v1)

12-byte header + class-specific body. All multi-byte fields little-endian. Raw
f64 in C-order.

```
Header (12 bytes):
  [0..4)   magic       = 0x50, 0x43, 0x42, 0x00   ("PCB\0")
  [4]      major       = 1
  [5]      minor       = 0
  [6..8)   class_tag   = 1 (Approximation) | 2 (Spline)
  [8..12)  reserved    = 0x00000000

Approximation body (class_tag=1):
  uint32                  num_dimensions d
  f64[d]                  domain_lo
  f64[d]                  domain_hi
  uint32[d]               n_nodes
  f64[prod(n_nodes)]      tensor_values   (C-order, row-major)

Spline body (class_tag=2):
  uint32                  num_dimensions d
  f64[d]                  domain_lo
  f64[d]                  domain_hi
  uint32[d]               n_nodes (flat — shared across pieces)
  uint32[d]               num_knots_per_dim
  f64[sum(num_knots)]     knots_concatenated (dim-by-dim)
  uint32                  num_pieces P = prod(num_knots[i] + 1)
  for p in 0..P-1:
      f64[prod(n_nodes)]  piece_tensor_values (C-order)
```

C-order indexing: for shape `(n_0, n_1, ..., n_{d-1})`, the flat index of
multi-index `(i_0, ..., i_{d-1})` is
`i_{d-1} + n_{d-1} * (i_{d-2} + n_{d-2} * (... + n_1 * i_0))`. This is
.NET's natural row-major layout and matches NumPy's default. No transpose
needed on either side.

## 7. What the binary format drops on save

| Field | C# v0.7.0 behavior | Python v0.14 behavior |
|---|---|---|
| `Function` | Always dropped (also dropped by JSON) | Same |
| `BarycentricWeights`, `DiffMatrices`, `DiffMatricesT`, `DiffMatricesTFlat` | Recomputed on load from `(domain, nNodes)` | Same (only `weights` and `diff_matrices` exist in Py) |
| Cached error estimate | Recomputed lazily on next `ErrorEstimate()` call | Same |
| Build telemetry (`BuildTime`, `NEvaluations`, `Method`, `BuildWarning`) | Not preserved | Same |
| `MaxDerivativeOrder` | Resets to default `2` on load | Same |
| `Descriptor`, `AdditionalData`, derivative-id registry | Not in C# yet (Phase 4) — no-op | Phase 4 will add reject-or-drop logic |

`Method` exists on `ChebyshevTT` but TT stays JSON-only in Phase 3, so this is
moot for v0.7.0.

## 8. Testing

T1 inline tests (in-memory write → read or `Path.GetTempFileName()` paths):

| Group | Count | Coverage |
|---|---:|---|
| Header validation | 5 | magic detect, major version reject, reserved nonzero reject, class tag dispatch, EOF on truncated header |
| Approximation round-trip | 7 | 1D, 2D, 3D sin, 5D BS, n=1 dim, single-node-per-dim edge, tensor-byte-count mismatch reject |
| Spline round-trip | 5 | 1D `abs(x)` (Python worked example), 2D, multi-knot-per-dim, nested-nNodes throws `NotSupportedException`, unbuilt throws `InvalidOperationException` |
| Save/Load API | 4 | format="binary" writes magic, format="json" unchanged, Load auto-detect routes correctly, unknown format throws `ArgumentException` |
| `PeekFormatVersion` | 3 | returns 1 on valid binary, throws on JSON, throws on missing file |
| Corruption rejection | 3 | truncated body, bad magic, bad class tag |

T2 fixture-based (load committed `tests/fixtures/*.pcb`, evaluate, compare to
analytical or known-exact value):

| Group | Count | Coverage |
|---|---:|---|
| Fixture load | 3 | `approx_2d_simple.pcb`, `approx_5d_bs.pcb`, `spline_1d_kink.pcb` |

Cross-feature tests:

| Group | Count | Coverage |
|---|---:|---|
| Cross-feature round-trip | 2 | round-trip after `+`/algebra, round-trip after `Slice`/`Extrude` |

**Total: 32 tests.** Test count: 765 → 797.

### Fixture provenance (one-time setup)

At Phase 3 start, manually generate the three fixtures via PyChebyshev v0.14
(or any later version, since the format is locked) and copy bytes into
`tests/fixtures/`. Document the generation commands in
`tests/fixtures/REGENERATE.md` for reproducibility:

```python
# tests/fixtures/REGENERATE.md (paraphrased)
import numpy as np
from pychebyshev import ChebyshevApproximation, ChebyshevSpline

# approx_2d_simple.pcb: f(x,y) = x + y on [-1,1]^2, n_nodes=[3,3]
ChebyshevApproximation(
    function=lambda pt, _: pt[0] + pt[1],
    num_dimensions=2,
    domain=[(-1.0, 1.0), (-1.0, 1.0)],
    n_nodes=[3, 3],
).build_and_save("tests/fixtures/approx_2d_simple.pcb", format='binary')

# approx_5d_bs.pcb: 5D Black-Scholes
# spline_1d_kink.pcb: |x| on [-1,1], knots=[[0.0]], n_nodes=[3]
```

The committed `.pcb` bytes are the source of truth for the C# tests; the
regeneration script exists for reference and future submodule bumps.

## 9. Stochastic / numerical concerns

None. The format is deterministic byte-level: given the same `(domain, nNodes,
tensor_values, knots)`, the saved bytes are identical across runs and across
languages. No floats are computed during save/load (just stored/read).

The only numerical concern is float bit-representation across .NET and CPython
— both use IEEE 754 binary64, so this is bit-identical by construction.

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Upstream format bumps to v1.1 (or v2) | Low | Manual `cmp` of fixture bytes when bumping submodule (Phase 4); minor-version readers ignore unknown trailing bytes per spec |
| .NET runs on a big-endian platform | Vanishingly low | Static-ctor assertion; tested CI matrix is all little-endian |
| Tensor C-order layout differs from Python's NumPy | Low | Both use row-major by default; no transpose needed; covered by fixture round-trip tests |
| Nested-nNodes Spline silently skipped instead of rejected | Mitigated by §3.3 | Hard-throw matches Python; explicit unit test |
| Phase 1's flat→nested `nNodes` distinction breaks Spline round-trip | Medium | Unit test specifically for the throw; `IsNestedNNodes` helper exists from Phase 1 |

## 11. Definition of done

- [ ] All 32 new tests passing; total **765 → 797**.
- [ ] Three committed `.pcb` fixtures verified once against PyChebyshev v0.14
      (manually `python -c "..."` outside CI; copy bytes in; commit).
- [ ] `<Version>0.7.0</Version>`, `<PyChebyshevParity>0.18.0</PyChebyshevParity>`
      (unchanged), `<InformationalVersion>0.7.0+pychebyshev.0.18.0</InformationalVersion>`.
- [ ] `docs/docs/changelog.md` — v0.7.0 entry following two-tier convention,
      noting "+ v0.16 PeekFormatVersion early arrival".
- [ ] `docs/docs/binary-format.md` — new user-guide page with C# examples.
- [ ] README parity badge unchanged (still v0.18.0).
- [ ] `skip_csharp.txt` — remove binary-format related entries.
- [ ] Submodule **stays at v0.18.0**. No bump needed in Phase 3.
- [ ] Single PR, single merge to main, GitHub release `v0.7.0` triggering
      `publish.yml` → NuGet.

## 12. Out-of-scope but adjacent

These show up nearby but are deferred to Phase 4 or beyond:

- `Save(format="binary")` for `ChebyshevSlider` / `ChebyshevTT` — Python keeps
  these pickle-only at v0.14. Phase 4+ if Python adds it.
- `additional_data` rejection on binary save — Phase 4 (Ergonomics) adds the
  field; reject logic lands then.
- `descriptor` silent-drop on binary save — Phase 4.
- Derivative-id registry preservation across save/load — Phase 4.
- A formal C# example reader (analogue of Python's `examples/binary_reader/reader.c`)
  — out of scope for the library; the user-guide doc covers what consumers need.
- Async I/O — never (out of scope for the library).

---

**Approved by:** Max Zhang (2026-04-28, conversation transcript)
**Implementation plan:** to be drafted next via `superpowers:writing-plans`,
saved to `docs/superpowers/plans/2026-04-28-phase3-pcb-format.md`.
