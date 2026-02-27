---
title: Piecewise Chebyshev Interpolation (Splines)
---

# Piecewise Chebyshev Interpolation (Splines)

## The Gibbs Phenomenon

Chebyshev interpolation converges **exponentially** for smooth (analytic) functions -- this is the spectral advantage that makes the method so powerful. But when the target function has a **discontinuity** or a **kink**, this advantage disappears.

**Jump discontinuities.** For a function $f$ with a jump discontinuity at $c \in (a, b)$, the Chebyshev interpolant converges only as $O(1/n)$ pointwise away from $c$ (Trefethen 2013, Ch. 9). Near $c$, oscillations persist regardless of how many nodes you add -- this is the classical **Gibbs phenomenon**.

**Kinks.** For a function that is continuous but whose derivative is discontinuous at $c$ -- for example $|x|$ at $x = 0$ or a call payoff $\max(S - K, 0)$ at $S = K$ -- the situation is better but still algebraic: convergence is $O(1/n^2)$ instead of exponential. This means that increasing the node count from 15 to 30 only halves the error, rather than reducing it by orders of magnitude as it would for a smooth function.

> **Example: Interpolating $|x|$ on $[-1, 1]$**
>
> With global Chebyshev interpolation, the error near $x = 0$ plateaus at approximately $0.01$ regardless of whether you use 10, 20, or 40 nodes. This is exactly the algebraic $O(1/n^2)$ convergence rate -- the spectral advantage of Chebyshev is lost.

In quantitative finance this problem is ubiquitous: option payoffs have kinks at strike prices, barrier levels, and exercise boundaries. Applying global Chebyshev interpolation to such functions wastes nodes fighting the Gibbs oscillations instead of refining the smooth parts of the function.

## Why Piecewise Chebyshev Restores Spectral Convergence

The key to understanding why piecewise interpolation helps lies in the **Bernstein ellipse theorem** (Trefethen 2013, Ch. 8).

### The Bernstein ellipse

For a function $f$ analytic in the interior of the **Bernstein ellipse** $\mathcal{E}_\rho$ -- the ellipse in the complex plane with foci at $\pm 1$ and semi-axis sum $\rho > 1$ -- the Chebyshev interpolation error on $[-1, 1]$ with $n$ nodes satisfies:

$$
\| f - p_n \|_\infty \leq \frac{2 M}{\rho^n (\rho - 1)}
$$

where $M = \max_{z \in \mathcal{E}_\rho} |f(z)|$. The rate $\rho^{-n}$ is **exponential** in $n$ -- this is spectral convergence.

### How kinks destroy analyticity

A function with a kink at $c$ is **not analytic** at $c$. The Bernstein ellipse cannot extend past the singularity: it collapses to the real interval near $c$, forcing $\rho \to 1$ and reducing convergence to algebraic.

### How splitting restores it

By placing a **knot** at $c$ and interpolating each sub-interval separately, each piece sees a **smooth** function:

- $|x|$ restricted to $[-1, 0]$ is just $-x$, which is entire (analytic everywhere in $\mathbb{C}$).
- $|x|$ restricted to $[0, 1]$ is just $x$, also entire.

Each piece has a large Bernstein ellipse parameter $\rho_k \gg 1$, and the error on piece $k$ with $n$ Chebyshev nodes is:

$$
E_k \leq \frac{2 M_k}{\rho_k^n (\rho_k - 1)}
$$

where $M_k = \max_{z \in \mathcal{E}_{\rho_k}} |f(z)|$ on that piece's Bernstein ellipse. Because pieces cover disjoint sub-domains, the overall interpolation error is:

$$
\| f - \mathcal{S}_n f \|_\infty = \max_k \, E_k
$$

This is exponential in $n$ -- **spectral convergence is restored**.

> **Book reference:** The Chebyshev Spline technique is described in Section 3.8 of Ruiz & Zeron (2022). The book demonstrates that pricing a European call near the strike kink requires 95 nodes with global Chebyshev but only 25 nodes (split into two pieces at $K$) with a Chebyshev spline -- same accuracy, 4x fewer evaluations.

## Quick Start

```csharp
using ChebyshevSharp;

// European call payoff max(S - K, 0) * exp(-rT) with kink at K = 100
double Payoff(double[] x, object? data)
{
    return Math.Max(x[0] - 100.0, 0.0) * Math.Exp(-0.05 * x[1]);
}

// Place a knot at S = 100 (the strike), no knots in the T dimension
var spline = new ChebyshevSpline(
    function: Payoff,
    numDimensions: 2,
    domain: new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
    nNodes: new[] { 15, 15 },
    knots: new[] { new[] { 100.0 }, Array.Empty<double>() }
);
spline.Build(verbose: false);

// Evaluate in-the-money
double priceItm = spline.Eval(new[] { 110.0, 0.5 }, new[] { 0, 0 });

// Evaluate out-of-the-money
double priceOtm = spline.Eval(new[] { 90.0, 0.5 }, new[] { 0, 0 });

// Delta (dV/dS) on the in-the-money side
double delta = spline.Eval(new[] { 110.0, 0.5 }, new[] { 1, 0 });

// Check accuracy
double error = spline.ErrorEstimate();
Console.WriteLine($"Error estimate: {error:E2}");
Console.WriteLine(spline);
```

Output:

```
ChebyshevSpline (2D, built)
  Nodes:       [15, 15] per piece
  Knots:       [[100], []]
  Pieces:      2 (2 x 1)
  Build:       0.003s (450 function evals)
  Domain:      [80, 120] x [0.25, 1]
  Error est:   1.23E-10
```

With global `ChebyshevApproximation` on the same domain, you would need approximately 95 nodes to achieve similar accuracy. The spline uses 2 pieces of 15 nodes each (450 total evaluations vs. 9,025).

## Choosing Knots

Place knots at the locations where the function is **non-smooth**:

| Singularity type | Example | Knot location |
|-----------------|---------|---------------|
| Payoff kink | European call $\max(S - K, 0)$ | $S = K$ |
| Barrier level | Knock-out option | $S = B$ |
| Exercise boundary | American option (if known) | $S = S^*(T)$ |
| Absolute value | $\|x\|$ | $x = 0$ |

**Guidelines:**

- **Only add knots where the function is non-smooth.** For smooth functions, knots provide no benefit -- you pay extra build cost for no accuracy gain.
- **More knots = more pieces = more build evaluations.** Each dimension with $k_d$ interior knots creates $k_d + 1$ sub-intervals. The total number of pieces is the Cartesian product $\prod_d (k_d + 1)$.
- **Knots must be known a priori.** `ChebyshevSpline` does not detect singularities automatically; you must specify where they are.

### Multiple knots in one dimension

```csharp
// Two knots in dimension 0: at x = -1 and x = 1
// This creates 3 pieces in dimension 0
var spline = new ChebyshevSpline(
    function: MyFunction,
    numDimensions: 1,
    domain: new[] { new[] { -3.0, 3.0 } },
    nNodes: new[] { 15 },
    knots: new[] { new[] { -1.0, 1.0 } }
);
```

### Multi-dimensional knots

Each dimension has its own independent list of knots. The total number of pieces is the Cartesian product of per-dimension intervals:

```csharp
// 2D: 2 knots in dim 0, 1 knot in dim 1
// Pieces: (2+1) x (1+1) = 3 x 2 = 6
var spline = new ChebyshevSpline(
    function: MyFunction,
    numDimensions: 2,
    domain: new[] { new[] { 0.0, 10.0 }, new[] { 0.0, 5.0 } },
    nNodes: new[] { 15, 15 },
    knots: new[] { new[] { 3.0, 7.0 }, new[] { 2.5 } }
);
```

### No knots in a dimension

Use an empty array for dimensions where the function is smooth:

```csharp
// Knot at S = 100 in price dimension, none in time or vol
var spline = new ChebyshevSpline(
    function: BsPrice,
    numDimensions: 3,
    domain: new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 }, new[] { 0.15, 0.35 } },
    nNodes: new[] { 15, 15, 15 },
    knots: new[] { new[] { 100.0 }, Array.Empty<double>(), Array.Empty<double>() }
);
```

### Degenerate case: no knots at all

If every dimension has an empty knot list, the spline has exactly one piece and behaves identically to a plain `ChebyshevApproximation`.

## Derivatives

Within each piece, derivatives are computed **analytically** via spectral differentiation matrices -- the same mechanism used by `ChebyshevApproximation`. No finite differences are needed.

```csharp
// First derivative w.r.t. dimension 0
double dfdx0 = spline.Eval(new[] { 110.0, 0.5 }, new[] { 1, 0 });

// Second derivative w.r.t. dimension 0
double d2fdx0 = spline.Eval(new[] { 110.0, 0.5 }, new[] { 2, 0 });

// Multiple derivatives at once (shared barycentric weights)
double[] results = spline.EvalMulti(
    new[] { 110.0, 0.5 },
    new[]
    {
        new[] { 0, 0 },  // function value
        new[] { 1, 0 },  // dV/dS
        new[] { 2, 0 },  // d2V/dS2
        new[] { 0, 1 },  // dV/dT
    }
);
```

`EvalMulti` is faster than calling `Eval` in a loop because it routes to the appropriate piece once and computes the barycentric weights and exact-node detection once, reusing them across all derivative orders.

### Derivatives at knot boundaries

Derivatives are **not defined** at knot boundaries. At a kink, the left and right polynomial pieces have different derivative values. Requesting a derivative at a knot raises `ArgumentException`:

```csharp
// This throws ArgumentException:
// "Derivative w.r.t. dimension 0 is not defined at knot x[0]=100"
spline.Eval(new[] { 100.0, 0.5 }, new[] { 1, 0 });

// Function values are fine at knots:
spline.Eval(new[] { 100.0, 0.5 }, new[] { 0, 0 });  // OK
```

> **Tip:** If you need a derivative near a knot, evaluate slightly to one side: `spline.Eval(new[] { 100.001, 0.5 }, new[] { 1, 0 })` gives the right-side derivative.

## Error Estimation

`ErrorEstimate()` returns the **maximum** error estimate across all pieces:

```csharp
spline.Build(verbose: false);
double error = spline.ErrorEstimate();
Console.WriteLine($"Error estimate: {error:E2}");
```

Since pieces cover disjoint sub-domains, the interpolation error at any point is bounded by the error of the piece containing that point. The worst-case error is therefore the **maximum** over all pieces:

$$
\hat{E} = \max_k \hat{E}_k
$$

This differs from `ChebyshevSlider`, where all slides contribute to every point and the error estimate is the **sum** over slides.

The result is cached -- repeated calls return the stored value without recomputation.

## Integration, Roots, and Optimization

`ChebyshevSpline` supports the same calculus operations as `ChebyshevApproximation`, adapted for piecewise structure.

### Integration

Integration sums the contributions from each piece, with automatic clipping when sub-interval bounds partially overlap a piece:

```csharp
// Integrate over all dimensions (returns a scalar)
double integral = (double)spline.Integrate();

// Integrate over specific dimensions (returns a lower-dimensional spline)
var marginal = (ChebyshevSpline)spline.Integrate(dims: new[] { 1 });

// Integrate with sub-interval bounds
double partial = (double)spline.Integrate(
    dims: new[] { 0, 1 },
    bounds: new[] { (90.0, 110.0), (0.3, 0.8) }
);
```

When bounds span multiple pieces, the integration correctly clips to the overlap region of each piece and sums the results. Pieces with no overlap are skipped entirely.

### Root-finding

Roots are found independently within each piece using the colleague matrix eigenvalue method, then merged and deduplicated near knot boundaries:

```csharp
// Roots of a 1D spline
double[] roots = spline1d.Roots();

// Roots along dimension 0, fixing other dimensions
double[] roots = spline2d.Roots(
    dim: 0,
    fixedDims: new Dictionary<int, double> { { 1, 0.5 } }
);
```

Roots at knot boundaries where two pieces meet are deduplicated to avoid reporting the same root twice.

### Minimization and Maximization

Each piece is searched independently, and the global optimum across all pieces is returned:

```csharp
// Find the minimum along dimension 0
var (minValue, minLocation) = spline.Minimize(
    dim: 0,
    fixedDims: new Dictionary<int, double> { { 1, 0.5 } }
);

// Find the maximum along dimension 0
var (maxValue, maxLocation) = spline.Maximize(
    dim: 0,
    fixedDims: new Dictionary<int, double> { { 1, 0.5 } }
);
```

Both methods evaluate all critical points (derivative roots) and domain/knot endpoints within each piece, then return the best result across all pieces. This guarantees finding the global optimum of the interpolant.

## Batch Evaluation

For evaluating many points at once, `EvalBatch` routes each point to its piece and groups points by piece for efficient evaluation:

```csharp
double[][] points = new double[1000][];
for (int i = 0; i < 1000; i++)
{
    points[i] = new[]
    {
        80.0 + 40.0 * Random.Shared.NextDouble(),   // S in [80, 120]
        0.25 + 0.75 * Random.Shared.NextDouble(),   // T in [0.25, 1.0]
    };
}

double[] values = spline.EvalBatch(points, new[] { 0, 0 });
```

Points that fall in the same piece are batched together, minimizing per-point overhead from piece routing.

## Arithmetic Operators

Splines defined on the same grid **and** with the same knots support pointwise arithmetic:

```csharp
var sum    = splineF + splineG;    // pointwise addition
var diff   = splineF - splineG;    // pointwise subtraction
var neg    = -splineF;             // pointwise negation
var scaled = splineF * 2.0;       // scalar multiplication
var halved = splineF / 2.0;       // scalar division
```

Both operands must have the same number of dimensions, domain bounds, node counts, **and knot positions**. The result is a new `ChebyshevSpline` with tensor values computed pointwise on each piece.

Arithmetic on splines is exact at the nodes. Between nodes, the result is the Chebyshev interpolant of the pointwise operation -- which differs from the true pointwise result by the interpolation error. For well-resolved interpolants, this difference is negligible.

## Extrusion and Slicing

### Extrusion

Add new dimensions to a spline where the function is constant along the new axis:

```csharp
// Start with a 1D spline f(x) with a knot
// Extrude to 2D: g(x, y) = f(x), constant in y
var spline2d = spline1d.Extrude((1, new[] { 0.0, 1.0 }, 10));
```

Each piece is extruded independently. The new dimension has no knots (empty knot list).

### Slicing

Fix one or more dimensions at specific values, reducing dimensionality:

```csharp
// Fix dimension 1 (T) at 0.5: h(S) = spline(S, 0.5)
var spline1d = spline2d.Slice((1, 0.5));
```

Slicing selects the piece containing the slice value along the sliced dimension, then performs barycentric interpolation within that piece. Only pieces at the correct interval index survive; the rest are discarded.

## Serialization

Save and load splines in JSON format:

```csharp
// Save
spline.Save("payoff_spline.json");

// Load (no rebuild needed)
var loaded = ChebyshevSpline.Load("payoff_spline.json");
double val = loaded.Eval(new[] { 110.0, 0.5 }, new[] { 0, 0 });
```

All piece data is preserved: nodes, weights, differentiation matrices, and tensor values. The original function is **not** saved -- only the numerical data needed for evaluation. A loaded spline has `Function = null` and cannot call `Build()` again without assigning a new function.

## Nodes and FromValues

The `Nodes` and `FromValues` static methods support a "nodes first, values later" workflow, useful when function evaluations are expensive or need to be distributed across machines.

### Nodes

Generate all Chebyshev node positions for every piece without evaluating any function:

```csharp
var nodeInfo = ChebyshevSpline.Nodes(
    numDimensions: 2,
    domain: new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
    nNodes: new[] { 15, 15 },
    knots: new[] { new[] { 100.0 }, Array.Empty<double>() }
);

// nodeInfo.NumPieces   -- total number of pieces (2)
// nodeInfo.PieceShape  -- per-dimension piece counts ([2, 1])
// nodeInfo.Pieces      -- per-piece node info (sub-domain, nodes, grid)
```

### FromValues

Construct a spline from pre-computed values on each piece:

```csharp
var nodeInfo = ChebyshevSpline.Nodes(2, domain, nNodes, knots);

// Evaluate your function at each piece's nodes (can be parallelized)
double[][] pieceValues = new double[nodeInfo.NumPieces][];
for (int p = 0; p < nodeInfo.NumPieces; p++)
{
    var grid = nodeInfo.Pieces[p].FullGrid;
    pieceValues[p] = new double[grid.Length];
    for (int i = 0; i < grid.Length; i++)
        pieceValues[p][i] = MyFunction(grid[i], null);
}

var spline = ChebyshevSpline.FromValues(
    pieceValues: pieceValues,
    numDimensions: 2,
    domain: domain,
    nNodes: nNodes,
    knots: knots
);
```

`FromValues` produces a result identical to `Build()` -- all pre-computed data (weights, differentiation matrices) depends only on the node positions.

## When to Use ChebyshevSpline

| Scenario | Recommended class | Why |
|----------|-------------------|-----|
| Smooth function, low-D | `ChebyshevApproximation` | Full tensor is feasible; spectral convergence without knots |
| Function with kinks at known locations | **`ChebyshevSpline`** | Restores spectral convergence by splitting at singularities |
| High-dimensional (5+D), general | `ChebyshevTT` | TT-Cross builds from $O(d \cdot n \cdot r^2)$ evaluations |
| High-dimensional, additively separable | `ChebyshevSlider` | Additive decomposition; cheapest build |

Use `ChebyshevSpline` when:

- The function has **known non-smooth points** (kinks, discontinuities).
- The dimension count is low enough for full tensor grids (typically 5D or fewer).
- You need **analytical derivatives** (not finite differences).
- You want **spectral accuracy** on a function that would otherwise converge slowly.

Do **not** use `ChebyshevSpline` when:

- The function is smooth everywhere -- a plain `ChebyshevApproximation` is simpler and equally accurate.
- You do not know where the singularities are -- the spline cannot detect them for you.
- The dimension count is high -- each piece requires a full tensor grid, so the build cost grows as $\text{num\_pieces} \times \prod_d n_d$. Many knots in many dimensions creates many pieces: 3 knots in each of 4 dimensions means $4^4 = 256$ pieces.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM. Chapters 8--9.
2. Ruiz, I. & Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View.* Wiley Finance. Section 3.8.
3. Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
