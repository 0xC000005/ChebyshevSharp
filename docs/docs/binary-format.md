---
title: Portable Binary Format (.pcb)
---

# Portable Binary Format (`.pcb`)

ChebyshevSharp v0.7.0 introduces a portable binary serialization format
alongside the default JSON format. The goal: let consumers in **C, Rust, Julia,
or any other language** read ChebyshevSharp interpolants without a .NET runtime.

The format is byte-for-byte compatible with PyChebyshev v0.14's `.pcb`. Files
written by either library can be read by the other.

## When to use which format

| Format | Use when |
|---|---|
| **JSON** (default) | .NET-only round-trips; need full fidelity (build telemetry, error threshold metadata) |
| **Binary** (`.pcb`) | Cross-language consumers; sharing models with C/Rust/Julia code; long-term archival |

```csharp
cheb.Save("model.pcb", format: "binary");      // portable
cheb.Save("model.json");                       // JSON (default)
cheb.Save("model.json", format: "json");       // explicit

ChebyshevApproximation.Load("model.pcb");      // auto-detects
```

`Load()` sniffs the first 4 bytes — `b"PCB\x00"` routes to the binary reader,
anything else to the JSON reader.

## Coverage in v0.7.0

- **`ChebyshevApproximation`** — full support.
- **`ChebyshevSpline`** — full support, with one restriction: the spline must
  use **flat** `nNodes` (a single `int` per dim, shared across pieces). Splines
  built with nested per-piece `nNodes` (the `int[][]` form introduced in Phase
  1 for special points) cannot be saved as `.pcb` and throw
  `NotSupportedException`; use JSON for those.
- **`ChebyshevSlider`**, **`ChebyshevTT`** — JSON only in v0.7.0 (matches
  PyChebyshev v0.14's restriction for these).

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
| Cached error estimate | recomputed lazily |
| Build telemetry (`BuildTime`, `NEvaluations`, `Method`) | not preserved (use JSON for full fidelity) |
| `MaxDerivativeOrder` | resets to default `2` on load |

If you need any of those preserved, use JSON.

## Security

The binary reader does no executable deserialization. It can be used to load
files from untrusted sources — it will reject malformed files with
`InvalidDataException`.
