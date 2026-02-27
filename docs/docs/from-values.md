---
title: Pre-computed Values
---

# Pre-computed Values

## Motivation -- Nodes First, Values Later

In production environments -- HPC clusters, distributed pricing engines, cloud
compute -- the function to be interpolated often cannot be called from within
C#. The evaluation may run on a separate process, a remote node, or inside
a proprietary pricing library.

ChebyshevSharp's standard workflow (constructor + `Build()`) requires a C#
delegate. The **pre-computed values** workflow decouples node generation from
function evaluation:

1. **Generate nodes** -- call `Nodes()` to get the Chebyshev grid points.
2. **Evaluate externally** -- feed the points to your own pricing engine.
3. **Load values** -- call `FromValues()` to construct the interpolant.

The resulting object is *fully functional*: evaluation, derivatives,
integration, rootfinding, optimization, algebra, extrusion/slicing, and
serialization all work identically to a `Build()`-based interpolant.

## Workflow

### ChebyshevApproximation

```csharp
using ChebyshevSharp;

// Step 1: Get nodes (no function needed)
NodeInfo info = ChebyshevApproximation.Nodes(
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
    nNodes: new[] { 15, 11 }
);
// info.NodesPerDim  -> double[2][] (15 nodes, 11 nodes)
// info.FullGrid     -> double[165][] (each row is one [x0, x1] point)
// info.Shape        -> int[] { 15, 11 }

// Step 2: Evaluate externally
// Each row of FullGrid is one (x0, x1) point.
// Feed these to your pricing engine, collect results.
double[] values = new double[info.FullGrid.Length];
for (int i = 0; i < info.FullGrid.Length; i++)
    values[i] = MyPricingEngine(info.FullGrid[i]);

// Step 3: Build from values
var cheb = ChebyshevApproximation.FromValues(
    values,
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
    nNodes: new[] { 15, 11 }
);

// Now use it like any other interpolant
double price    = cheb.VectorizedEval(new[] { 0.5, 1.0 }, new[] { 0, 0 });
double delta    = cheb.VectorizedEval(new[] { 0.5, 1.0 }, new[] { 1, 0 });
double integral = (double)cheb.Integrate();
```

### ChebyshevSpline

For piecewise interpolation with knots, the same pattern applies per piece:

```csharp
using ChebyshevSharp;

// Step 1: Get per-piece nodes
SplineNodeInfo info = ChebyshevSpline.Nodes(
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: new[] { 15 },
    knots: new[] { new[] { 0.0 } }
);
// info.Pieces      -> SplinePieceNodeInfo[] (2 pieces)
// info.NumPieces   -> 2
// info.PieceShape  -> int[] { 2 }

// Step 2: Evaluate each piece externally
double[][] pieceValues = new double[info.NumPieces][];
for (int p = 0; p < info.NumPieces; p++)
{
    SplinePieceNodeInfo piece = info.Pieces[p];
    pieceValues[p] = new double[piece.FullGrid.Length];
    for (int i = 0; i < piece.FullGrid.Length; i++)
        pieceValues[p][i] = MyPricingEngine(piece.FullGrid[i]);
}

// Step 3: Build
var spline = ChebyshevSpline.FromValues(
    pieceValues,
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: new[] { 15 },
    knots: new[] { new[] { 0.0 } }
);
```

## Indexing Convention

The tensor entries must satisfy:

$$
\texttt{values}[i_0 \cdot n_1 \cdots n_{d-1} + i_1 \cdot n_2 \cdots n_{d-1} + \cdots + i_{d-1}]
= f\!\bigl(\texttt{NodesPerDim}[0][i_0],\; \texttt{NodesPerDim}[1][i_1],\; \ldots\bigr)
$$

The rows of `FullGrid` follow row-major (C-order) enumeration, so iterating
over `FullGrid` in order and storing the results in a flat `double[]` produces
the correct tensor layout automatically.

> **Warning: Use row-major (C-order) layout.**
> The flat `double[]` passed to `FromValues()` must be in row-major order
> (last index varies fastest). Do **not** use column-major (Fortran-order)
> layout, which would silently produce incorrect tensor entries and lead to
> wrong interpolation results.

For **spline pieces**, the list order follows row-major multi-index enumeration
over `PieceShape`. In 2D with `knots = { { 0.0 }, { 1.0 } }` on domain
`[[-1,1], [0,2]]`, the four pieces are:

| Index | PieceIndex | SubDomain |
|-------|------------|-----------|
| 0 | (0, 0) | [(-1, 0), (0, 1)] |
| 1 | (0, 1) | [(-1, 0), (1, 2)] |
| 2 | (1, 0) | [(0, 1), (0, 1)] |
| 3 | (1, 1) | [(0, 1), (1, 2)] |

## Mathematical Justification

Everything the interpolant needs -- *except* the function values themselves --
depends only on the node positions:

| Pre-computed data | Depends on |
|-------------------|------------|
| Chebyshev nodes | domain, nNodes |
| Barycentric weights | nodes |
| Differentiation matrices | nodes, weights |
| Fejer quadrature weights | nodes |

The function values appear only in `TensorValues`. The Chebyshev nodes are
the zeros of $T_n(x)$, mapped to each dimension's domain (Trefethen 2013,
Ch. 3). Since `FromValues()` computes all of the above from the same node
formula as `Build()`, the resulting interpolant is **bit-identical** to one
built the traditional way.

## Examples

### 1-D: sin(x) on [0, pi]

```csharp
using ChebyshevSharp;

NodeInfo info = ChebyshevApproximation.Nodes(1,
    new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

double[] values = new double[info.FullGrid.Length];
for (int i = 0; i < info.FullGrid.Length; i++)
    values[i] = Math.Sin(info.FullGrid[i][0]);

var cheb = ChebyshevApproximation.FromValues(values, 1,
    new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

// Integral of sin(x) on [0, pi] ~ 2.0
double integral = (double)cheb.Integrate();
Console.WriteLine(integral);  // 1.9999999...
```

### Combine with Algebra

```csharp
// Two proxies built externally
var chebA = ChebyshevApproximation.FromValues(
    valsA, 2, domain, nNodes);
var chebB = ChebyshevApproximation.FromValues(
    valsB, 2, domain, nNodes);

// Portfolio-level proxy
var portfolio = 0.6 * chebA + 0.4 * chebB;
```

### Save and Load

```csharp
cheb.Save("my_proxy.json");
var loaded = ChebyshevApproximation.Load("my_proxy.json");
```

## Error Handling

| Error | Raised when |
|-------|-------------|
| `ArgumentException` (shape) | `tensorValues.Length` does not equal the product of `nNodes` |
| `ArgumentException` (NaN/Inf) | `tensorValues` contains non-finite values |
| `ArgumentException` (dimensions) | `domain.Length != numDimensions` or `nNodes.Length != numDimensions` |
| `ArgumentException` (domain) | `lo >= hi` for any dimension |
| `InvalidOperationException` (build) | Calling `Build()` on a `FromValues` object (`Function` is null) |

> **Calling `Build()` on a `FromValues` result:**
> Objects created via `FromValues()` have `Function = null`. To re-build
> with a different set of values, create a new object via `FromValues()`.
> To re-build from a callable, assign a function first:
> `cheb.Function = MyFunc;`, then `cheb.Build()`.

## Combining with Other Features

Objects created via `FromValues()` support **all** ChebyshevSharp operations:

- **Evaluation & derivatives** -- `VectorizedEval()`, `VectorizedEvalMulti()`
- **Calculus** -- `Integrate()`, `Roots()`, `Minimize()`, `Maximize()`
- **Algebra** -- `+`, `-`, `*`, `/` (including with `Build()`-based objects)
- **Extrusion & slicing** -- `Extrude()`, `Slice()`
- **Serialization** -- `Save()`, `Load()`
- **Error estimation** -- `ErrorEstimate()`

> **Note:** `ChebyshevSlider` does **not** support `Nodes()` or `FromValues()`.
> The sliding technique builds a separate low-dimensional interpolant for each
> partition group, each requiring a different subset of dimensions evaluated at
> pivot values for the remaining dimensions. There is no single "nodes first,
> values later" grid that covers this decomposition. To use pre-computed values
> with a slider-like workflow, build a `ChebyshevApproximation` from values for
> each group independently.

## API Reference

### `ChebyshevApproximation.Nodes()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `numDimensions` | `int` | Number of dimensions |
| `domain` | `double[][]` | Bounds per dimension (`new[] { lo, hi }` each) |
| `nNodes` | `int[]` | Nodes per dimension |

**Returns** `NodeInfo` with properties:

| Property | Type | Description |
|----------|------|-------------|
| `NodesPerDim` | `double[][]` | Chebyshev nodes for each dimension, sorted ascending |
| `FullGrid` | `double[][]` | Full Cartesian product grid, shape (totalPoints, numDimensions) |
| `Shape` | `int[]` | Expected tensor shape (equal to `nNodes`) |

### `ChebyshevApproximation.FromValues()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensorValues` | `double[]` | Function values, flat array of length `Product(nNodes)` |
| `numDimensions` | `int` | Number of dimensions |
| `domain` | `double[][]` | Bounds per dimension |
| `nNodes` | `int[]` | Nodes per dimension |
| `maxDerivativeOrder` | `int` (default 2) | Maximum derivative order |

**Returns** `ChebyshevApproximation` with `Function = null`.

### `ChebyshevSpline.Nodes()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `numDimensions` | `int` | Number of dimensions |
| `domain` | `double[][]` | Bounds per dimension |
| `nNodes` | `int[]` | Nodes per dimension per piece |
| `knots` | `double[][]` | Knot positions per dimension |

**Returns** `SplineNodeInfo` with properties:

| Property | Type | Description |
|----------|------|-------------|
| `Pieces` | `SplinePieceNodeInfo[]` | Per-piece node info |
| `NumPieces` | `int` | Total number of pieces |
| `PieceShape` | `int[]` | Per-dimension piece counts |

Each `SplinePieceNodeInfo` contains:

| Property | Type | Description |
|----------|------|-------------|
| `PieceIndex` | `int[]` | Multi-index of this piece |
| `SubDomain` | `double[][]` | Sub-domain bounds for this piece |
| `NodesPerDim` | `double[][]` | Per-dimension node arrays |
| `FullGrid` | `double[][]` | Full Cartesian product grid |
| `Shape` | `int[]` | Tensor shape |

### `ChebyshevSpline.FromValues()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `pieceValues` | `double[][]` | Per-piece values in row-major order |
| `numDimensions` | `int` | Number of dimensions |
| `domain` | `double[][]` | Bounds per dimension |
| `nNodes` | `int[]` | Nodes per dimension per piece |
| `knots` | `double[][]` | Knot positions per dimension |
| `maxDerivativeOrder` | `int` (default 2) | Maximum derivative order |

**Returns** `ChebyshevSpline` with `Function = null`.

## See Also

- [Advanced Usage](advanced-usage.md) -- arithmetic operators and algebra
- [Calculus](calculus.md) -- integrate, find roots, optimize
- [Extrusion & Slicing](extrude-slice.md) -- add/fix dimensions
- [Serialization & Construction](serialization.md) -- persist FromValues objects

## References

- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM. Chapter 3.
