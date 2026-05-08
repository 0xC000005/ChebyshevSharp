---
title: Adaptive Refinement
---

# Adaptive Refinement

ChebyshevSharp provides adaptive-refinement APIs for automatically selecting
piecewise knots and per-dimension node counts.

## ChebyshevSpline.AutoKnots

Auto-place knots at function kinks via a curvature-spike scan. Useful for piecewise-smooth functions like `|x|`, `max(0, x)`, or piecewise polynomials.

```csharp
double F(double[] p, object? _) => Math.Abs(p[0]);
var sp = ChebyshevSpline.AutoKnots(F, 1,
    new[] { new[] { -1.0, 1.0 } },
    new[] { 16 });
// Discovers a knot near x=0; the resulting Spline has 2 pieces.
```

Tuning parameters: `thresholdFactor` (default 5.0), `maxKnotsPerDim` (default 5), `nScanPoints` (default 200).

## SobolIndices

Variance decomposition from spectral Chebyshev coefficients. No Monte Carlo, no extra evaluations beyond what's already in `TensorValues`.
The reported variance uses the Chebyshev orthogonality weight on each normalized input dimension, not a sampled uniform input distribution.

```csharp
double F(double[] p, object? _) => Math.Sin(p[0]) + p[1] * p[2];
var ap = new ChebyshevApproximation(
    function: F,
    numDimensions: 3,
    domain: new[]
    {
        new[] { -1.0, 1.0 },
        new[] { -1.0, 1.0 },
        new[] { -1.0, 1.0 }
    },
    nNodes: new[] { 16, 16, 16 });
ap.Build();
SobolResult s = ap.SobolIndices();
Console.WriteLine($"FirstOrder: [{string.Join(", ", s.FirstOrder)}]");
Console.WriteLine($"TotalOrder: [{string.Join(", ", s.TotalOrder)}]");
Console.WriteLine($"Variance: {s.Variance}");
```

`s.Variance == 0` or a value at the documented numerical noise floor indicates a constant function; the indices are zero and meaningless. For `ChebyshevSpline`, indices are computed from the full piecewise expansion, combining within-piece Chebyshev variance, between-piece mean variance, and interactions between interval membership and local Chebyshev modes.

For a Chebyshev expansion

$$
f(x) = \sum_{\alpha} c_\alpha \prod_{j=1}^{d} T_{\alpha_j}(x_j),
$$

the variance under the Chebyshev weight is the weighted energy of all non-constant coefficients:

$$
V = \sum_{\alpha \ne 0} c_\alpha^2 \prod_{j=1}^{d} \langle T_{\alpha_j}, T_{\alpha_j}\rangle,
\qquad
\langle T_0,T_0\rangle=\pi,\quad
\langle T_k,T_k\rangle=\pi/2\ (k>0).
$$

First-order energy for dimension $j$ sums coefficients with $\alpha_j>0$ and all other degrees zero. Total-order energy for dimension $j$ sums every coefficient with $\alpha_j>0$. `ChebyshevTT.SobolIndices()` computes the same quantities by contracting TT coefficient cores, avoiding dense coefficient materialization.

## ChebyshevTT.WithAutoOrder + Reorder

TT compression rank depends on dim order; some functions admit much lower-rank TTs under a non-canonical permutation.

```csharp
double F(double[] p) => Math.Sin(p[0] * p[2]) + Math.Cos(p[1]);
var tt = ChebyshevTT.WithAutoOrder(F, 3,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 16, 16, 16 },
    nTrials: 5, method: "greedy_swap");
Console.WriteLine($"DimOrder: [{string.Join(", ", tt.DimOrder)}]");
// Eval/Slice/Extrude/etc. transparently remap user coordinates by tt.DimOrder.
double v = tt.Eval(new[] { 0.3, -0.4, 0.5 });

// Manual realignment to a different permutation:
var realigned = tt.Reorder(new[] { 1, 2, 0 }, maxRank: 16, tolerance: 1e-10);
```

Binary algebra (`+`, `-`) between TTs requires matching `DimOrder`; call `Reorder` on one operand first if they differ.

## References

- Sobol, I. M. (2001). "Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates."
- Saltelli, A. et al. (2010). "Variance Based Sensitivity Analysis of Model Output."
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
