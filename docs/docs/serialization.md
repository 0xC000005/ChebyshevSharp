---
title: Serialization & Construction
---

# Serialization & Construction

ChebyshevSharp provides multiple ways to create and persist interpolants beyond the standard `Build()` workflow.

## Save and Load

A built interpolant can be saved to disk and restored later without the original function:

```csharp
// Save to file
cheb.Save("interpolant.json");

// Load from file (no function reference needed)
var restored = ChebyshevApproximation.Load("interpolant.json");
double value = restored.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });
```

The serialization format is JSON. All pre-computed data is saved: nodes, barycentric weights, differentiation matrices, tensor values, domain bounds, and node counts. The loaded interpolant is fully functional for evaluation, derivatives, integration, root-finding, and all other operations.

Loaded interpolants cannot call `Build()` since they do not retain the original function reference. Pre-transposed differentiation matrices (`DiffMatricesTFlat`) are recomputed on load from the stored differentiation matrices.

**Format note:** ChebyshevSharp uses its own JSON format, not Python pickle. Files saved by PyChebyshev cannot be loaded by ChebyshevSharp and vice versa. To transfer between languages, use `FromValues` with exported node positions and function values.

`ChebyshevSpline` also supports `Save` and `Load` with the same JSON format. The serialized file includes all pieces and knot positions. `Nodes()` and `FromValues()` are available for `ChebyshevSpline` as well. See [Piecewise Chebyshev Interpolation](spline.md) for details.

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

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
