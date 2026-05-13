---
title: Tensor Train Interpolation
---

# Tensor Train Interpolation

The **Tensor Train (TT)** format represents a high-dimensional Chebyshev tensor as a chain of small 3-index cores. When the sampled tensor has low numerical rank, storage drops from $O(n^d)$ to about $O(d \cdot n \cdot r^2)$, where $r$ is a typical TT rank. The default `Build(method: "cross")` path samples adaptively so it can avoid the full $n^d$ grid; `svd`, `als`, `FromValues`, and `ToDense()` are dense-grid paths for intentionally small tensors.

## Motivation

Consider a 5D Black-Scholes pricer $V(S, K, T, \sigma, r)$ with 11 nodes per dimension. A full tensor grid requires $11^5 = 161{,}051$ function evaluations and stores the same number of coefficients. For 7 or more dimensions, the grid size exceeds billions of elements.

The TT decomposition represents the sampled tensor as a chain of 3-index cores:

$$
A[i_1, \ldots, i_d] =
\sum_{\alpha_0, \ldots, \alpha_d}
G_1[\alpha_0, i_1, \alpha_1] \cdot
G_2[\alpha_1, i_2, \alpha_2] \cdots
G_d[\alpha_{d-1}, i_d, \alpha_d]
$$

where each core $G_k$ has shape $(r_{k-1}, n_k, r_k)$ with $r_0 = r_d = 1$. The integers $r_1, \ldots, r_{d-1}$ are the **TT ranks**, which control both approximation quality and cost. Smooth finance examples often work with moderate ranks, but rank is a property of the sampled function and node grid, not a guarantee from the API. Validate held-out points before relying on a rank choice.

The total storage is $\sum_k r_{k-1} \cdot n_k \cdot r_k$, which is linear in $d$ when ranks stay bounded. For the 5D Black-Scholes example with rank 15, this is roughly 9,000 elements instead of 161,000 -- an 18x compression.

### Mental model

Think of each TT core as a local lookup table for one dimension plus two small
rank links to its neighbors:

```text
        rank r1        rank r2                 rank r(d-1)
[1 x n1 x r1] -- [r1 x n2 x r2] -- ... -- [r(d-1) x nd x 1]
      G1                 G2                            Gd
```

Evaluation builds the Chebyshev basis vector for each physical coordinate,
contracts it into the matching core, then multiplies the resulting small matrices
from left to right. Low rank means that the function's cross-variable structure
can pass through those rank links without materializing the full tensor grid.

## When to Use ChebyshevTT

| Scenario | Recommended class |
|----------|-------------------|
| Smooth function, 1--5 dimensions | `ChebyshevApproximation` |
| Function with discontinuities or singularities | `ChebyshevSpline` |
| 6+ dimensions, additively separable or nearly so | `ChebyshevSlider` |
| **5+ dimensions, general coupling between variables** | **`ChebyshevTT`** |

`ChebyshevTT` fills the gap between `ChebyshevApproximation` (limited to ~5D by the curse of dimensionality) and `ChebyshevSlider` (requires near-separability). It handles general cross-variable coupling at the cost of using finite-difference derivatives instead of analytical spectral differentiation.

## Build Methods

`ChebyshevTT.Build(method: ...)` supports three build paths:

### TT-Cross (default)

TT-Cross builds the TT decomposition from a subset of function evaluations using alternating left-to-right and right-to-left sweeps with **maxvol pivoting** (Oseledets & Tyrtyshnikov 2010). At each sweep, it:

1. Assembles a cross-section matrix from the function at selected grid index combinations
2. Computes a truncated SVD to determine the adaptive rank
3. Applies the maxvol algorithm to select the most informative rows (pivot indices)
4. Updates the TT core and index sets for the next mode

The implementation stops when the relative error measured on a small random set
of grid-index checks drops below the requested tolerance, or when the
sweep/improvement limits are reached. This is an internal convergence diagnostic,
not a certified bound on the continuous function error. An evaluation cache
avoids redundant function calls across sweeps.

**Complexity:** TT-Cross targets about $O(d \cdot n \cdot r^2)$ function evaluations for bounded rank, where $d$ is the number of dimensions, $n$ is the typical node count, and $r$ is the TT rank. The exact count depends on ranks, sweeps, cache reuse, seed, and stopping behavior.

### TT-SVD

TT-SVD evaluates the function on the full tensor grid, then decomposes it via
sequential truncated SVD (Oseledets 2011). This is deterministic and gives a
Frobenius-controlled low-rank approximation to the sampled tensor, but it
requires $\prod_i n_i$ function evaluations and dense storage during the build.
Use TT-SVD only when the full grid is feasible and you need a deterministic
reference.

### ALS

`Build(method: "als")` is a rank-adaptive alternating least-squares path. It
materializes the full grid, starts at rank 1, and increases rank until the dense
grid residual falls below `tolerance` or `maxRank` is reached. If the rank cap
is reached first, `BuildWarning` is set. Use ALS for small-grid experiments and
diagnostics, not for high-dimensional production builds.

> **Dense-grid guard.**
> TT-Cross is the production path for high-dimensional tensors. TT-SVD, ALS, `FromValues`, and `ToDense()` intentionally validate dense element counts and byte sizes before allocation. A shape such as 35 nodes in 7 dimensions is a reasonable TT-Cross target but an invalid dense materialization target; ChebyshevSharp throws a clear overflow error instead of wrapping fixed-width products or attempting an impossible allocation.

## Quick Start

```csharp
using ChebyshevSharp;

// 5D smooth pricing-style model: V(S, K, T, sigma, r)
double SmoothValue(double[] x)
{
    double S = x[0], K = x[1], T = x[2], sigma = x[3], r = x[4];
    double moneyness = (S - K) / 20.0;
    double softPayoff = Math.Log(1.0 + Math.Exp(moneyness));
    return Math.Exp(-r * T) * softPayoff * (1.0 + 0.1 * sigma * S / K);
}

var tt = new ChebyshevTT(
    function: SmoothValue,
    numDimensions: 5,
    domain: new[] {
        new[] { 80.0, 120.0 },   // Spot
        new[] { 90.0, 110.0 },   // Strike
        new[] { 0.25, 1.0 },     // Maturity
        new[] { 0.15, 0.35 },    // Volatility
        new[] { 0.01, 0.08 }     // Rate
    },
    nNodes: new[] { 11, 11, 11, 11, 11 },
    maxRank: 15,
    tolerance: 1e-6,
    maxSweeps: 10
);

// Build with TT-Cross (default)
tt.Build(verbose: true, seed: 42);

double[] point = [100.0, 100.0, 0.5, 0.25, 0.05];
double exact = SmoothValue(point);
double approx = tt.Eval(point);

Console.WriteLine($"f(point) = {approx:F8}");
Console.WriteLine($"exact    = {exact:F8}");
Console.WriteLine($"abs err  = {Math.Abs(approx - exact):E2}");
Console.WriteLine($"TT ranks = [{string.Join(", ", tt.TtRanks)}]");
Console.WriteLine($"evals    = {tt.TotalBuildEvals:N0}");
Console.WriteLine($"storage  = {tt.CompressionRatio:F1}x compression");
```

Verbose build output shows the sweep progress, rank evolution, and compression
ratio. The exact ranks, timings, and function-evaluation counts depend on the
function, seed, node counts, rank cap, and stopping path:

```
Building 5D ChebyshevTT (max_rank=15, method='cross')...
  Full tensor would need 161,051 evaluations
  Running TT-Cross...
    Sweep 1 L->R: rel error = 2.13E-03, unique evals = 4,235, ranks = [1, 5, 8, 6, 4, 1]
    Sweep 1 R->L: rel error = 5.67E-05, unique evals = 6,812
    Converged after 2 sweeps (L->R)
  Built in 0.312s (7,401 function evaluations)
  TT ranks: [1, 5, 10, 8, 5, 1]
  Compression: 161,051 -> 8,855 elements (18.2x)
```

Read the first run as a diagnostic, not a certificate. Useful signs are ranks
well below `maxRank`, held-out errors that match your tolerance requirements, and
stable results when you retry a few seeds. A rank vector pinned near
`maxRank` or seed-sensitive validation error means the model is not yet tuned.

### Using TT-SVD

For small problems where you want deterministic reference results:

```csharp
// 3D function -- small enough for full tensor
var ttSvd = new ChebyshevTT(
    function: x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
    numDimensions: 3,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11, 11 },
    maxRank: 5
);
ttSvd.Build(verbose: false, method: "svd");
```

TT-SVD is deterministic (no random seed), so it produces identical results for
the same platform and numerical libraries up to floating-point linear-algebra
differences. This makes it useful for small-grid reference builds and regression
tests.

### Validate a TT-Cross build

Always test a TT-Cross model on points that were not part of the adaptive build.
This checks the combined effect of node count, rank cap, seed, and tolerance:

```csharp
double maxRel = 0.0;
var rng = new Random(123);

for (int i = 0; i < 100; i++)
{
    double[] p = new double[5];
    for (int d = 0; d < 5; d++)
    {
        double lo = tt.Domain[d][0], hi = tt.Domain[d][1];
        p[d] = lo + (hi - lo) * rng.NextDouble();
    }

    double exact = SmoothValue(p);
    double approx = tt.Eval(p);
    if (Math.Abs(exact) > 1e-12)
        maxRel = Math.Max(maxRel, Math.Abs(approx - exact) / Math.Abs(exact));
}

Console.WriteLine($"held-out max relative error = {maxRel:E2}");
```

Use this validation loop as the acceptance gate:

1. Fix a seed while tuning so changes are comparable.
2. Check held-out value error on physical points, not only grid-index checks.
3. Inspect `TtRanks`; ranks at or near `maxRank` usually mean the cap is active.
4. Increase `nNodes` only when node resolution is the bottleneck. If error does
   not improve, rank or sampling is likely the limiting factor.
5. Retry several seeds before publishing a TT-Cross model whose error is close to
   the acceptance threshold.

## Evaluation

### Single point

```csharp
double value = tt.Eval(new[] { 100.0, 100.0, 0.5, 0.25, 0.05 });
```

Evaluation contracts the Chebyshev polynomial basis vectors $T_0(x), T_1(x), \ldots, T_{n-1}(x)$ against each coefficient core in sequence. Cost: $O(d \cdot n \cdot r^2)$ per point.

### Batch evaluation

```csharp
// Evaluate at N points simultaneously
double[,] points = new double[1000, 5];
// ... fill points ...

double[] values = tt.EvalBatch(points);
```

`EvalBatch` vectorizes the contraction across all points, avoiding repeated allocation of intermediate vectors. It can be materially faster than calling `Eval` in a loop, especially for many points, but benchmark your point count and rank profile before treating the speedup as fixed.

### Derivatives via finite differences

```csharp
double[] results = tt.EvalMulti(
    new[] { 100.0, 100.0, 0.5, 0.25, 0.05 },
    new[] {
        new[] { 0, 0, 0, 0, 0 },  // Price
        new[] { 1, 0, 0, 0, 0 },  // Delta (dV/dS)
        new[] { 2, 0, 0, 0, 0 },  // Gamma (d²V/dS²)
        new[] { 0, 0, 0, 1, 0 },  // Vega  (dV/dsigma)
        new[] { 0, 0, 0, 0, 1 },  // Rho   (dV/dr)
    }
);
```

`EvalMulti` computes derivatives using **central finite differences**:

- First derivative: $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$
- Second derivative: $f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$
- Mixed partial: $\frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)}{4 h_i h_j}$

The step size is $h = (b - a) \times 10^{-4}$ per dimension, with automatic boundary nudging when the evaluation point is within $1.5h$ of a domain edge. Maximum supported derivative order per dimension is 2.

> **Finite differences vs analytical derivatives.**
> Unlike `ChebyshevApproximation`, which computes derivatives of the
> interpolating polynomial with spectral differentiation matrices,
> `ChebyshevTT` uses finite differences on the TT interpolant. The finite
> difference error depends on the step size, domain scale, smoothness, TT rank
> truncation, and proximity to the boundary. Validate Greeks against analytical
> formulas or held-out finite-difference checks when derivative accuracy matters.

## Error Estimation

```csharp
double error = tt.ErrorEstimate();
```

The error estimate is computed as the sum of $\max |G_k[:, n_k{-}1, :]|$ across all dimensions $k$, where $G_k[:, n_k{-}1, :]$ is the slice of the coefficient core corresponding to the highest-order Chebyshev polynomial. This parallels the DCT-II coefficient decay logic used by `ChebyshevApproximation`, adapted to the TT core structure.

Like the other classes, the result is cached after the first call.

> **Diagnostic scope.**
> `ErrorEstimate()` checks trailing coefficient-core magnitude. It does not
> include TT-Cross sampling error, rank truncation error, or finite-difference
> derivative error. Use it with held-out validation points.

## Serialization

```csharp
// Save to JSON
tt.Save("tt_model.json");

// Load without the original function
var loaded = ChebyshevTT.Load("tt_model.json");
double val = loaded.Eval(new[] { 100.0, 100.0, 0.5, 0.25, 0.05 });
```

The serialized file contains all coefficient cores, TT ranks, domain, node counts, and build metadata. The original function is not saved. Loaded objects can evaluate, compute derivatives, and estimate errors immediately.

If the file was saved with a different library version, a `LoadWarning` property is set with the version mismatch details.

## Properties

| Property | Description |
|----------|-------------|
| `NumDimensions` | Number of input dimensions |
| `Domain` | Bounds $[a_d, b_d]$ for each dimension |
| `NNodes` | Number of Chebyshev nodes per dimension |
| `MaxRank` | Maximum TT rank specified at construction |
| `TtRanks` | Actual TT ranks $[1, r_1, \ldots, r_{d-1}, 1]$ after build |
| `CompressionRatio` | Ratio of full tensor size to TT storage size |
| `TotalBuildEvals` | Number of function evaluations used during build |
| `Method` | Build method that produced the current cores: `"cross"`, `"svd"`, or `"als"` |
| `BuildWarning` | Warning from ALS when `maxRank` is reached before tolerance is satisfied |
| `LoadWarning` | Version mismatch warning (null if none) |

## Build Modes

`Build(method: ...)` accepts three algorithms:

```csharp
tt.Build(method: "cross");  // TT-Cross (default): sparse adaptive sampling
tt.Build(method: "svd");    // TT-SVD: full O(n^d) tensor; for validation
tt.Build(method: "als");    // Alternating LS: rank-adaptive, full-grid evals
```

ALS starts at rank 1 and grows the TT rank by +1 per outer iteration until the
grid residual falls below `tolerance` or the rank reaches `maxRank`. If the
cap is hit before tolerance is satisfied, `BuildWarning` is set.

```csharp
tt.Build(method: "als", seed: 42);
if (tt.BuildWarning != null)
    Console.Error.WriteLine(tt.BuildWarning);
```

## Refining a Built TT — `RunCompletion`

`RunCompletion(tolerance, maxIter)` runs fixed-rank ALS sweeps on an
already-built TT. Rank is preserved; only per-core coefficients are refined.
It requires the original callable function, so it cannot be called on a TT
loaded from disk or created with `FromValues()`.

```csharp
tt.Build(method: "cross", tolerance: 1e-3, maxRank: 5);
tt.RunCompletion(tolerance: 1e-10, maxIter: 20);
```

## Canonicalization — `OrthLeft` / `OrthRight`

Push R factors through the TT chain so cores up to `position` are
left-orthogonal (`Q^T Q = I` after the `(rL*n, rR)` unfolding) or so cores
beyond `position` are right-orthogonal. The represented tensor is unchanged.
Useful as a primitive for downstream algorithms.

```csharp
tt.OrthLeft(position: 2);   // cores 0 and 1 become left-orthogonal
tt.OrthRight(position: 0);  // cores 1, 2, ... become right-orthogonal
```

## Inner Product

Frobenius inner product of two TTs' Chebyshev coefficient tensors:

```csharp
double ip = ttA.InnerProduct(ttB);  // sum_i C_a[i] * C_b[i]
```

Both TTs must share the same `NumDimensions`, `Domain`, and `NNodes`.

## Static Factories — `Nodes` / `FromValues`

```csharp
var (nodesPerDim, shape) = ChebyshevTT.Nodes(2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });

// Build a TT from a precomputed dense tensor (skip TT-Cross):
double[] dense = /* row-major Π nNodes values */;
var tt = ChebyshevTT.FromValues(dense, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 },
    maxRank: 10, tolerance: 1e-6);
```

`FromValues` compresses an already materialized dense tensor with TT-SVD. It is
not a sparse sampling path and should be used only when the full value array is
deliberately feasible.

## Materialization — `ToDense`

```csharp
double[] flat = tt.ToDense();   // row-major Π nNodes
```

Throws `OverflowException` if the dense shape or byte count exceeds supported
array limits. Use for inspection and round-trip testing, not high-dimensional
production workflows.

## Slicing & Extrusion — `Slice` / `Extrude`

```csharp
var sliced = tt.Slice(dim: 0, value: 0.5);                // dim 0 fixed at 0.5
var extruded = tt.Extrude(dim: 1, newDomain: (0, 1), newN: 5);  // add a constant dim
```

`Slice` uses barycentric interpolation along the sliced dim's value-space core
and absorbs the resulting matrix into a neighbor. A fast path triggers when
`value` coincides with a Chebyshev node within `1e-14`.

## Algebra

```csharp
var sum  = ttA + ttB;          // block-diagonal stacking, then TT-SVD round
var diff = ttA - ttB;
var neg  = -ttA;               // unary
var dbl  = 2.0 * ttA;          // scalar mul (commutative)
var half = ttA / 2.0;          // throws DivideByZeroException on zero

// In-place equivalents (mutate the receiver, return void):
ttA.AddInPlace(ttB);
ttA.SubInPlace(ttB);
ttA.ScalarMulInPlace(2.0);
ttA.ScalarDivInPlace(2.0);
ttA.NegateInPlace();
ttA.RoundInPlace(1e-10);       // explicit TT-SVD recompression
```

`+` and `-` round to the larger of the two operands' `MaxRank` with a default
TT-SVD tolerance of `1e-12`. Use `RoundInPlace(tolerance)` when you need
tighter or looser control.

## Choosing Parameters

### maxRank

The maximum TT rank controls the trade-off between accuracy and cost. Higher rank allows more complex cross-variable interactions to be captured.

- **Separable or nearly additive functions** often need low rank.
- **Moderate coupling** may need ranks around 10--15 in smooth option-pricing-style examples.
- **Strong or localized coupling** may need higher rank, more sweeps, more nodes, or a different approximation strategy.

TT-Cross adaptively selects the actual rank up to `maxRank` based on the SVD
singular value decay. Setting `maxRank` higher than needed can increase work
because it raises the allowed rank caps, but the adaptive truncation can still
keep the final ranks below that cap.

### nNodes

Node counts follow the same guidelines as `ChebyshevApproximation`: 7--15 nodes per dimension is often a useful starting range for smooth functions. For analytic functions, polynomial interpolation can converge rapidly as nodes increase, but TT accuracy also depends on rank truncation and TT-Cross sampling. If increasing `nNodes` does not improve held-out error, inspect `maxRank`, `tolerance`, and seed sensitivity.

### tolerance

The convergence tolerance for TT-Cross (default $10^{-6}$). The algorithm stops when its internal relative error diagnostic on random grid-index checks drops below this threshold. Lower tolerance may require more sweeps and higher effective rank, and it does not replace held-out validation on physical points.

### seed

The random seed for TT-Cross initialization (default: system random). Setting a fixed seed makes the adaptive sampling path reproducible. Different seeds may produce different ranks and held-out errors; retry seeds when ranks are near `maxRank` or validation error is unstable.

## Troubleshooting

| Symptom | Likely cause | First action |
|---------|--------------|--------------|
| Held-out error is high and ranks are near `maxRank` | Rank cap is active | Raise `maxRank`, then check whether error falls for the same seed |
| Held-out error is high but ranks are low | Node resolution or sampling path may be limiting | Increase `nNodes`; if error is seed-sensitive, retry seeds or increase `maxSweeps` |
| `ErrorEstimate()` is small but held-out error is not | Trailing coefficient-core diagnostic misses TT-Cross sampling or rank truncation error | Trust held-out validation over `ErrorEstimate()` |
| Different seeds give materially different errors | Adaptive cross pivots are unstable for this function/rank cap | Run a small seed sweep and choose parameters that are robust, not just the best seed |
| Dense methods throw overflow or allocation errors | `svd`, `als`, `FromValues`, or `ToDense()` need the full tensor | Use `Build(method: "cross")` or reduce the dense problem size |
| Greeks are noisy near boundaries | TT derivatives use finite differences on the interpolant | Validate derivative order, compare against analytical checks, and avoid relying on boundary-adjacent finite differences |

## Theory

### TT decomposition

The Tensor Train format (Oseledets 2011) represents a $d$-dimensional tensor $A[i_1, \ldots, i_d]$ as a chain of matrix products:

$$A[i_1, \ldots, i_d] = G_1[i_1] \cdot G_2[i_2] \cdots G_d[i_d]$$

where each $G_k[i_k]$ is an $r_{k-1} \times r_k$ matrix (a slice of the 3-index core along the node axis). The boundary conditions $r_0 = r_d = 1$ ensure the product yields a scalar.

The TT ranks $r_k$ measure the coupling between the first $k$ and the remaining $d - k$ dimensions in the sampled tensor. For a function that factors as $f(x_1, \ldots, x_d) = g(x_1, \ldots, x_k) \cdot h(x_{k+1}, \ldots, x_d)$, the rank across that split is 1. Additive or weakly coupled structure usually gives low rank; general functions can require higher rank.

### Maxvol algorithm

The maxvol algorithm (Goreinov, Tyrtyshnikov & Zamarashkin 1997) finds $r$ rows of an $m \times r$ tall matrix $A$ such that the $r \times r$ submatrix has approximately maximal determinant. This is used within TT-Cross to select the most informative grid index combinations for function evaluation.

The implementation uses:
1. **Column-pivoted QR** on $A^T$ for initialization (selects the $r$ most linearly independent rows)
2. **Iterative row-swapping refinement** until the coefficient matrix $B = A \cdot A[\text{idx}]^{-1}$ satisfies $\|B\|_{\max} \leq 1.05$

### Value-to-coefficient conversion

After the TT decomposition produces value cores (containing function values at Chebyshev nodes), each core is converted to a coefficient core via the DCT-II along the node axis. This reuses the same `ChebyshevCoefficients1D` routine used by `ChebyshevApproximation`, applying it fiber-by-fiber: for each fixed $(i, k)$ pair in the left and right rank indices, the 1D fiber along the node axis is transformed from values to Chebyshev coefficients.

Evaluation then contracts the Chebyshev polynomial vectors $[T_0(x), T_1(x), \ldots, T_{n-1}(x)]$ against the coefficient cores, which is a standard TT inner product.

## References

See [Citations](citations.md) for the references used by this page.

1. Oseledets, I. V. (2011). "Tensor-Train Decomposition." *SIAM Journal on Scientific Computing* 33(5):2295--2317.
2. Oseledets, I. V. & Tyrtyshnikov, E. E. (2010). "TT-cross approximation for multidimensional arrays." *Linear Algebra and its Applications* 432(1):70--88.
3. Goreinov, S. A., Tyrtyshnikov, E. E. & Zamarashkin, N. L. (1997). "A theory of pseudoskeleton approximations." *Linear Algebra and its Applications* 261:1--21.
4. Ruiz, I. & Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View.* Wiley Finance. Chapter 6.
5. Savostyanov, D. V. & Oseledets, I. V. (2011). "Fast adaptive interpolation of multi-dimensional arrays in tensor train format." *7th International Workshop on Multidimensional Systems*, pp. 1--8.
6. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
7. Bigoni, D., Engsig-Karup, A. P. & Marzouk, Y. M. (2016). "Spectral Tensor-Train Decomposition." *SIAM Journal on Scientific Computing* 38(4):A2405--A2439.
8. Glau, K., Kressner, D. & Statti, F. (2019). "Low-Rank Tensor Approximation for Chebyshev Interpolation in Parametric Option Pricing." arXiv:1902.04367.
