---
title: Portable Binary Format (.pcb)
---

# Portable Binary Format (`.pcb`)

`.pcb` is ChebyshevSharp's documented binary layout for dense
`ChebyshevApproximation` objects and flat-node `ChebyshevSpline` objects. It
stores the numerical representation needed for evaluation: domain bounds, node
counts, knots for splines, and row-major tensor values.

Use JSON when you need full ChebyshevSharp state. Use `.pcb` when you need a
compact primitive layout that can be read without .NET-specific JSON classes.

## When to use which format

| Format | Use when |
|---|---|
| **JSON** (default) | You need the richest .NET round trip, including build metadata, derivative registries, descriptors, and adaptive-construction metadata where the class supports them. |
| **Binary** (`.pcb`) | You need a small, documented, primitive layout for dense approximations or flat-node splines. |

```csharp
cheb.Save("model.pcb", format: "binary");      // .pcb
cheb.Save("model.json");                       // JSON (default)
cheb.Save("model.json", format: "json");       // explicit

var dense = ChebyshevApproximation.Load("model.pcb");
```

Use the matching class to load the file. `ChebyshevApproximation.Load()` and
`ChebyshevSpline.Load()` sniff the first four bytes: `PCB\0` routes to the
binary reader, anything else routes to the JSON reader. If the file is a valid
`.pcb` for the other class, loading fails with a `class_tag` error instead of
silently converting it.

## Supported Objects

- **`ChebyshevApproximation`** — full support.
- **`ChebyshevSpline`** — supported only when every piece shares one positive
  `nNodes` vector. Splines built with adaptive node counts or nested per-piece
  `nNodes` throw `NotSupportedException`; save those as JSON.
- **`ChebyshevSlider`** and **`ChebyshevTT`** — JSON only.

```csharp
// Adaptive or nested-node splines keep their metadata in JSON.
spline.Save("spline.json");

// Flat-node splines can also be saved as .pcb.
spline.Save("spline.pcb", format: "binary");
var restored = ChebyshevSpline.Load("spline.pcb");
```

## Format specification (v1)

All multi-byte fields are **little-endian**. Numeric arrays are raw `f64`
blobs in C-order (row-major).

### File layout

```
Header (12 bytes)
  [0..3]  magic     : 4 bytes  = 0x50 0x43 0x42 0x00  ("PCB\0")
  [4]     major     : uint8    = 1
  [5]     minor     : uint8    = 0
  [6..7]  class_tag : uint16LE = 1 (Approximation) or 2 (Spline)
  [8..11] reserved  : uint32LE = 0

Body for class_tag=1 (Approximation)
  d          : uint32LE   — number of dimensions
  lo[d]      : f64[d]     — domain lower bounds
  hi[d]      : f64[d]     — domain upper bounds
  n_nodes[d] : uint32LE[d]— node count per dimension
  tensor     : f64[prod(n_nodes)] — tensor values, row-major

Body for class_tag=2 (Spline)
  d              : uint32LE   — number of dimensions
  lo[d]          : f64[d]     — domain lower bounds
  hi[d]          : f64[d]     — domain upper bounds
  n_nodes[d]     : uint32LE[d]— shared node count per dimension
  num_knots[d]   : uint32LE[d]— knot count per dimension
  knots          : f64[sum(num_knots)] — knots concatenated in dim order
  num_pieces     : uint32LE   — = prod(num_knots[i]+1)
  piece_tensors  : f64[prod(n_nodes)] × num_pieces — one tensor per piece
```

### Worked example — `approx_2d_simple.pcb` (128 bytes)

The committed fixture `tests/fixtures/approx_2d_simple.pcb` encodes
f(x,y) = x + y on [-1,1]² with n = [3, 3]:

```
Offset  Size  Value           Field
------  ----  ----------      -----
0       4     50 43 42 00     magic "PCB\0"
4       1     01              major = 1
5       1     00              minor = 0
6       2     01 00           class_tag = 1 (Approximation)
8       4     00 00 00 00     reserved = 0
12      4     02 00 00 00     d = 2
16      8     0000 0000 0000 F0BF   lo[0] = -1.0
24      8     0000 0000 0000 F0BF   lo[1] = -1.0
32      8     0000 0000 0000 F03F   hi[0] =  1.0
40      8     0000 0000 0000 F03F   hi[1] =  1.0
48      4     03 00 00 00     n_nodes[0] = 3
52      4     03 00 00 00     n_nodes[1] = 3
56      72    9 × f64         tensor values (row-major)
```

Total: 12 (header) + 4 (d) + 16 (lo) + 16 (hi) + 8 (nNodes) + 72 (tensor) = **128 bytes**.

### Worked example — `spline_1d_kink.pcb` (100 bytes)

The committed fixture encodes |x| on [-1,1] with one knot at 0:

```
Offset  Size  Value           Field
------  ----  ----------      -----
0       12    (header)        magic + version + class_tag=2
12      4     01 00 00 00     d = 1
16      8     lo[0] = -1.0
24      8     hi[0] =  1.0
32      4     03 00 00 00     n_nodes[0] = 3
36      4     01 00 00 00     num_knots[0] = 1
40      8     0.0             knots[0][0] = 0.0
48      4     02 00 00 00     num_pieces = 2
52      24    2 × 3 × f64     piece tensors
```

Total: 12 + 4 + 8 + 8 + 4 + 4 + 8 + 4 + 48 = **100 bytes**.

## Peeking at the version

To read the major version byte without deserializing the body:

```csharp
int v = ChebyshevApproximation.PeekFormatVersion("model.pcb"); // returns 1
int v = ChebyshevSpline.PeekFormatVersion("model.pcb");        // returns 1
```

This is useful when deciding whether to upgrade files after a format bump.

## What the format does not store

These fields are dropped on `format="binary"`:

| Field | Replacement |
|---|---|
| `Function` | always dropped (also dropped by JSON) |
| `Weights`, `DiffMatrices` | recomputed on load from `(domain, nNodes)` |
| Derivative-id registry | rebuilt lazily as derivative IDs are requested |
| Cached error estimate | recomputed lazily |
| Build telemetry (`BuildTime`, `NEvaluations`) | not preserved |
| Descriptors and special-point labels | not preserved |
| Adaptive or nested-node construction intent | unsupported; use JSON |
| `MaxDerivativeOrder` | resets to default `2` on load |

If you need any of those preserved, use JSON.

## Validation and Error Behavior

The binary reader validates the header and body before constructing the public
object:

- bad magic, unsupported major versions, and nonzero reserved bytes are rejected
- wrong class tags are rejected by the class-specific `Load()` method
- dimensions, domains, node counts, knots, and tensor sizes must be finite and
  internally consistent
- truncated files throw `EndOfStreamException`; malformed records usually throw
  `InvalidDataException`

`PeekFormatVersion(path)` reads only the 12-byte header and returns the major
version byte for `.pcb` files. It throws for missing files, JSON files, and
short headers.

## Compatibility Fixtures

The v1 layout is locked by binary fixtures under `tests/fixtures/`. Those
fixtures validate byte-level compatibility for supported dense and flat-node
spline cases and give contributors a stable way to detect accidental format
changes. Regeneration instructions and fixture provenance are maintained in
`tests/fixtures/REGENERATE.md`.

## Security

ChebyshevSharp reads `.pcb` with `BinaryReader` and explicit primitive fields;
it does not use `BinaryFormatter` or polymorphic object deserialization.
Microsoft's .NET guidance treats `BinaryFormatter`-style deserialization as
unsafe for untrusted input because it can instantiate object graphs and cross a
trust boundary. `.pcb` avoids that object-deserialization model, but it is not a
security sandbox. Treat files from untrusted sources as untrusted input, verify
provenance when possible, and load them with appropriate file-size and memory
limits.

Reference: [Microsoft .NET BinaryFormatter security
guide](https://learn.microsoft.com/en-us/dotnet/standard/serialization/binaryformatter-security-guide).
