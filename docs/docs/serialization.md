---
title: Serialization & Construction
---

# Serialization & Construction

ChebyshevSharp provides multiple ways to create and persist interpolants beyond the standard `Build()` workflow.

## Save and Load

A built interpolant can be saved to disk and restored later without the original function:

```csharp
// Save to JSON
cheb.Save("interpolant.json");

// Load from file (no function reference needed)
var restored = ChebyshevApproximation.Load("interpolant.json");
double value = restored.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });
```

JSON is the default format. It saves the pre-computed data needed for full-fidelity .NET round trips: nodes, barycentric weights, differentiation matrices, tensor values, domain bounds, node counts, and build metadata where supported. The loaded interpolant is fully functional for evaluation, derivatives, integration, root-finding, and all other operations.

Loaded interpolants cannot call `Build()` since they do not retain the original function reference. Pre-transposed differentiation matrices (`DiffMatricesTFlat`) are recomputed on load from the stored differentiation matrices.

ChebyshevSharp also supports a portable binary format for dense approximations and compatible splines:

```csharp
cheb.Save("model.pcb", format: "binary");
var portable = ChebyshevApproximation.Load("model.pcb");
```

`Load()` auto-detects JSON versus `.pcb` by checking the binary magic header. Use JSON for .NET-only round trips and rich metadata. Use `.pcb` for cross-language consumers or long-term archival; it is byte-compatible with PyChebyshev's portable binary format but stores only the grid, domain, and tensor values needed to reconstruct the interpolant. See [Portable Binary Format (.pcb)](binary-format.md) for format details and restrictions.

**Format note:** ChebyshevSharp's JSON format is not Python pickle, and it is not the same as PyChebyshev's JSON/pickle serialization. Use `.pcb` where supported for cross-language binary transfer, or use `FromValues` with exported Type I node positions and function values when binary coverage does not fit your model.

`ChebyshevSpline` also supports `Save` and `Load` with the same JSON format. The serialized file includes all pieces and knot positions. `Nodes()` and `FromValues()` are available for `ChebyshevSpline` as well. See [Piecewise Chebyshev Interpolation](spline.md) for details.

`ChebyshevSlider` supports `Save` and `Load`. The serialized file includes the partition, pivot point, pivot value, and all slide states. `Nodes()` and `FromValues()` are not available for `ChebyshevSlider` — use the constructor and `Build()` workflow instead. See [Sliding Technique](slider.md) for details.

`ChebyshevTT` supports `Save` and `Load`. The serialized file includes all coefficient cores, TT ranks, domain, node counts, and build metadata. `Nodes()` and `FromValues()` are available for TT workflows; `FromValues()` compresses the supplied dense tensor via TT-SVD. If the file was saved with a different library version, a `LoadWarning` property is set. See [Tensor Train Interpolation](tensor-train.md) for details.

## FromValues

If you already have function values at Chebyshev nodes, use `FromValues` to construct an interpolant directly without providing a function:

```csharp
// Get the node positions first
var nodeInfo = ChebyshevApproximation.Nodes(
    numDimensions: 2,
    domain: new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } },
    nNodes: new[] { 10, 10 }
);

// Evaluate your function at the nodes (can be parallelized)
double[] values = new double[nodeInfo.Shape[0] * nodeInfo.Shape[1]];
for (int i = 0; i < nodeInfo.FullGrid.Length; i++)
{
    double[] pt = nodeInfo.FullGrid[i];
    values[i] = Math.Sin(pt[0]) * Math.Cos(pt[1]);
}

// Build the interpolant from pre-computed values
var cheb = ChebyshevApproximation.FromValues(
    tensorValues: values,
    numDimensions: 2,
    domain: new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } },
    nNodes: new[] { 10, 10 }
);
```

**When to use FromValues:**
- Function evaluations are expensive and you want to parallelize them externally (e.g., across a cluster)
- Values come from an external source (simulation output, market data, another language)
- You need fine-grained control over the evaluation process (progress reporting, error handling)

The values array must be in **row-major (C-order) layout**: the last dimension varies fastest. This matches the order returned by `Nodes().FullGrid`.

`FromValues` produces a result identical to `Build()` — all pre-computed data (weights, differentiation matrices) depends only on the node positions, not the function.

## Nodes

The static `Nodes` method generates Chebyshev node positions without evaluating any function:

```csharp
var nodeInfo = ChebyshevApproximation.Nodes(
    numDimensions: 3,
    domain: new[] {
        new[] { 80.0, 120.0 },
        new[] { 0.1, 0.5 },
        new[] { 0.25, 2.0 }
    },
    nNodes: new[] { 15, 12, 10 }
);

// nodeInfo.NodesPerDim — Chebyshev nodes for each dimension (double[][])
// nodeInfo.FullGrid    — full Cartesian product grid (double[][])
// nodeInfo.Shape       — tensor shape (int[], e.g., [15, 12, 10])
```

ChebyshevSharp uses **Type I Chebyshev nodes** (roots of the Chebyshev polynomial $T_n$):

$$
x_i = \cos\!\left(\frac{(2i - 1)\,\pi}{2n}\right), \quad i = 1, \ldots, n
$$

These are mapped to the domain $[a, b]$ via the affine transformation $\text{node} = \tfrac{a+b}{2} + \tfrac{b-a}{2}\,x_i$. Nodes are stored in ascending order within each dimension (smallest first).

Type I nodes avoid the endpoints of the interval. This is advantageous when the function has singularities or discontinuities at the boundary [1, Ch. 3].

MoCaX C uses Chebyshev--Lobatto/extrema nodes with endpoints. If your external values come from MoCaX, rebuild them on ChebyshevSharp's Type I nodes or resample before calling `FromValues()`.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
