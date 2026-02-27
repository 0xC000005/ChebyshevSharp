---
title: Tensor Train Interpolation
---

# Tensor Train Interpolation

The **Tensor Train (TT)** format enables Chebyshev interpolation of functions with 5 or more dimensions by decomposing the full coefficient tensor into a chain of small 3-index cores. This reduces storage from $O(n^d)$ to $O(d \cdot n \cdot r^2)$, where $r$ is the TT rank, and builds the approximation from $O(d \cdot n \cdot r^2)$ function evaluations instead of the full $n^d$ tensor grid.

## Motivation

Consider a 5D Black-Scholes pricer $V(S, K, T, \sigma, r)$ with 11 nodes per dimension. A full tensor grid requires $11^5 = 161{,}051$ function evaluations and stores the same number of coefficients. For 7 or more dimensions, the grid size exceeds billions of elements.

The TT decomposition represents the same function as a chain of 3-index cores:

$$
f(x_1, \ldots, x_d) \approx \sum_{\alpha_0, \ldots, \alpha_d} G_1[\alpha_0, x_1, \alpha_1] \cdot G_2[\alpha_1, x_2, \alpha_2] \cdots G_d[\alpha_{d-1}, x_d, \alpha_d]
$$

where each core $G_k$ has shape $(r_{k-1}, n_k, r_k)$ with $r_0 = r_d = 1$. The integers $r_1, \ldots, r_{d-1}$ are the **TT ranks**, which control approximation quality. For many functions arising in finance, $r \leq 15$ suffices for high accuracy.

The total storage is $\sum_k r_{k-1} \cdot n_k \cdot r_k$, which is linear in $d$ for bounded rank. For the 5D Black-Scholes example with rank 15, this is roughly 9,000 elements instead of 161,000 -- an 18x compression.

## When to Use ChebyshevTT

| Scenario | Recommended class |
|----------|-------------------|
| Smooth function, 1--5 dimensions | `ChebyshevApproximation` |
| Function with discontinuities or singularities | `ChebyshevSpline` |
| 6+ dimensions, additively separable or nearly so | `ChebyshevSlider` |
| **5+ dimensions, general coupling between variables** | **`ChebyshevTT`** |

`ChebyshevTT` fills the gap between `ChebyshevApproximation` (limited to ~5D by the curse of dimensionality) and `ChebyshevSlider` (requires near-separability). It handles general cross-variable coupling at the cost of using finite-difference derivatives instead of analytical spectral differentiation.

## Build Methods

ChebyshevTT supports two decomposition methods:

### TT-Cross (default)

TT-Cross builds the TT decomposition from a subset of function evaluations using alternating left-to-right and right-to-left sweeps with **maxvol pivoting** (Oseledets & Tyrtyshnikov 2010). At each sweep, it:

1. Assembles a cross-section matrix from the function at selected grid index combinations
2. Computes a truncated SVD to determine the adaptive rank
3. Applies the maxvol algorithm to select the most informative rows (pivot indices)
4. Updates the TT core and index sets for the next mode

The algorithm converges when the relative approximation error drops below the specified tolerance. An evaluation cache avoids redundant function calls across sweeps.

**Complexity:** $O(d \cdot n \cdot r^2)$ function evaluations, where $d$ is the number of dimensions, $n$ is the typical node count, and $r$ is the TT rank. For 5D Black-Scholes with rank 15, this is roughly 7,400 evaluations instead of 161,051.

### TT-SVD

TT-SVD evaluates the function on the full tensor grid, then decomposes it via sequential truncated SVD (Oseledets 2011). This is deterministic and produces the optimal rank-$r$ approximation in the Frobenius norm, but requires $n^d$ function evaluations. Use TT-SVD only when the full grid is feasible (typically $d \leq 6$) and you need a deterministic reference or the best possible accuracy at a given rank.

## Quick Start

```csharp
using ChebyshevSharp;

// 5D Black-Scholes pricer: V(S, K, T, sigma, r)
double BsPrice(double[] x)
{
    double S = x[0], K = x[1], T = x[2], sigma = x[3], r = x[4];
    // ... your pricing model here ...
    return price;
}

var tt = new ChebyshevTT(
    function: BsPrice,
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
```

Build output shows the sweep progress, rank evolution, and compression ratio:

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

### Using TT-SVD

For small problems where you want deterministic, optimal results:

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

TT-SVD is deterministic (no random seed), so it produces identical results across runs. This makes it useful for cross-language validation and regression testing.

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

`EvalBatch` vectorizes the contraction across all points, avoiding repeated allocation of intermediate vectors. It is 15--20x faster than calling `Eval` in a loop.

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
> Unlike `ChebyshevApproximation`, which computes exact derivatives of the interpolating polynomial via spectral differentiation matrices, `ChebyshevTT` uses finite differences. This is because the spectral differentiation matrix requires the full tensor, which TT avoids storing. The finite-difference approach loses 4--8 digits of accuracy relative to analytical derivatives, but remains adequate for most financial applications where Greeks at 0.01--1% relative error are acceptable.

## Error Estimation

```csharp
double error = tt.ErrorEstimate();
```

The error estimate is computed as the sum of $\max |G_k[:, n_k{-}1, :]|$ across all dimensions $k$, where $G_k[:, n_k{-}1, :]$ is the slice of the coefficient core corresponding to the highest-order Chebyshev polynomial. This parallels the DCT-II coefficient decay logic used by `ChebyshevApproximation`, adapted to the TT core structure.

Like the other classes, the result is cached after the first call.

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
| `LoadWarning` | Version mismatch warning (null if none) |

## Limitations

`ChebyshevTT` does **not** support:

- **Analytical derivatives** -- uses finite differences (order 1 and 2 only)
- **Arithmetic operators** ($+$, $-$, $\times$, $\div$)
- **Extrusion and slicing**
- **Integration, root-finding, or optimization**
- **`Nodes()` and `FromValues()` factories**

These operations require the full tensor grid or differentiation matrices, which the TT format does not store. Use `ChebyshevApproximation` or `ChebyshevSpline` when these features are needed.

## Choosing Parameters

### maxRank

The maximum TT rank controls the trade-off between accuracy and cost. Higher rank allows more complex cross-variable interactions to be captured.

- **Separable functions** (e.g., $\sin(x) + \sin(y) + \sin(z)$): rank 2 suffices
- **Moderate coupling** (e.g., Black-Scholes): rank 10--15 gives sub-percent accuracy
- **Strong coupling** (e.g., $\sin(x \cdot y \cdot z)$): may need rank 20+

TT-Cross adaptively selects the actual rank up to `maxRank` based on the SVD singular value decay. Setting `maxRank` higher than needed does not increase cost significantly -- the adaptive truncation keeps the rank minimal.

### nNodes

Node counts follow the same guidelines as `ChebyshevApproximation`: 7--15 nodes per dimension for smooth functions. The Chebyshev spectral convergence guarantee applies within each core, so adding nodes improves accuracy exponentially for analytic functions.

### tolerance

The convergence tolerance for TT-Cross (default $10^{-6}$). The algorithm stops when the relative approximation error (measured at random test points) drops below this threshold. Lower tolerance may require more sweeps and higher effective rank.

### seed

The random seed for TT-Cross initialization (default: system random). Setting a fixed seed ensures reproducible builds. Different seeds may produce slightly different TT ranks and accuracy, but all should converge to comparable quality.

## Theory

### TT decomposition

The Tensor Train format (Oseledets 2011) represents a $d$-dimensional tensor $A[i_1, \ldots, i_d]$ as a chain of matrix products:

$$A[i_1, \ldots, i_d] = G_1[i_1] \cdot G_2[i_2] \cdots G_d[i_d]$$

where each $G_k[i_k]$ is an $r_{k-1} \times r_k$ matrix (a slice of the 3-index core along the node axis). The boundary conditions $r_0 = r_d = 1$ ensure the product yields a scalar.

The TT ranks $r_k$ measure the entanglement between the first $k$ and the remaining $d - k$ dimensions. For a function that factors as $f(x_1, \ldots, x_d) = g(x_1, \ldots, x_k) \cdot h(x_{k+1}, \ldots, x_d)$, the rank $r_k = 1$. Additive separability gives $r_k = 2$ (one component for each addend plus a coupling term). General functions have higher rank.

### Maxvol algorithm

The maxvol algorithm (Goreinov, Tyrtyshnikov & Zamarashkin 1997) finds $r$ rows of an $m \times r$ tall matrix $A$ such that the $r \times r$ submatrix has approximately maximal determinant. This is used within TT-Cross to select the most informative grid index combinations for function evaluation.

The implementation uses:
1. **Column-pivoted QR** on $A^T$ for initialization (selects the $r$ most linearly independent rows)
2. **Iterative row-swapping refinement** until the coefficient matrix $B = A \cdot A[\text{idx}]^{-1}$ satisfies $\|B\|_{\max} \leq 1.05$

### Value-to-coefficient conversion

After the TT decomposition produces value cores (containing function values at Chebyshev nodes), each core is converted to a coefficient core via the DCT-II along the node axis. This reuses the same `ChebyshevCoefficients1D` routine used by `ChebyshevApproximation`, applying it fiber-by-fiber: for each fixed $(i, k)$ pair in the left and right rank indices, the 1D fiber along the node axis is transformed from values to Chebyshev coefficients.

Evaluation then contracts the Chebyshev polynomial vectors $[T_0(x), T_1(x), \ldots, T_{n-1}(x)]$ against the coefficient cores, which is a standard TT inner product.

## References

1. Oseledets, I. V. (2011). "Tensor-Train Decomposition." *SIAM Journal on Scientific Computing* 33(5):2295--2317.
2. Oseledets, I. V. & Tyrtyshnikov, E. E. (2010). "TT-cross approximation for multidimensional arrays." *Linear Algebra and its Applications* 432(1):70--88.
3. Goreinov, S. A., Tyrtyshnikov, E. E. & Zamarashkin, N. L. (1997). "A theory of pseudoskeleton approximations." *Linear Algebra and its Applications* 261:1--21.
4. Ruiz, I. & Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View.* Wiley Finance. Chapters 4--5.
5. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
