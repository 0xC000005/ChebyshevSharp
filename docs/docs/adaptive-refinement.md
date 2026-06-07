---
title: Adaptive Refinement
---

# Adaptive Refinement

ChebyshevSharp provides adaptive-refinement APIs for selecting piecewise knots
and per-dimension node counts.

For smooth dense-grid interpolation, start with
[Error-Driven Construction](error-driven-construction.md). It describes the
`ChebyshevApproximation` auto-node loop controlled by `errorThreshold`, `maxN`,
and nullable entries in `nNodes`.

## ChebyshevSpline.AutoKnots

`AutoKnots` builds a `ChebyshevSpline` after scanning each dimension for
second-difference spikes with the other dimensions fixed at their midpoints.
It is a heuristic for candidate knot placement, not a proof that every
non-smooth point has been found.

```csharp
double F(double[] p, object? _) => Math.Abs(p[0]);
var sp = ChebyshevSpline.AutoKnots(F, 1,
    new[] { new[] { -1.0, 1.0 } },
    new[] { 16 });
// Discovers a knot near x=0; the resulting Spline has 2 pieces.
```

Tuning parameters: `thresholdFactor` (default 5.0), `maxKnotsPerDim` (default
5), and `nScanPoints` (default 200). Set `maxKnotsPerDim: 0` to disable
auto-knot insertion and build a single-piece spline. Always inspect
`sp.Knots` and validate held-out points after using the scan.

### Chebfun edge detection vs `AutoKnots`

`AutoKnots` is ChebyshevSharp's automatic kink detector -- the analogue of Chebfun's *splitting* mode. The tools map onto each other:

| Chebfun | ChebyshevSharp |
| --- | --- |
| exact rootfinding for `abs`/`min`/`max`/`sign` (a corner placed at a root of $f$) | declare the location with [`WithSpecialPoints`](special-points.md) |
| automatic edge detection (`splitting on`) | `AutoKnots` |
| piecewise representation (smooth pieces joined at breakpoints) | `ChebyshevSpline` |

The detection *philosophy* differs, and `AutoKnots` is deliberately the simpler tool. Chebfun's `detectedge` recursively bisects and judges each candidate by *convergence* -- refining a singularity's location toward machine precision and classifying jump-vs-kink by which finite-difference derivative blows up. `AutoKnots` instead does a single fixed-grid second-difference scan, which has three consequences worth knowing:

- **Resolution is bounded by the scan grid:** a knot is placed no more accurately than `(hi - lo) / (nScanPoints - 1)`, and the scan does not iteratively refine.
- **It scans one mid-point line per dimension** (other dimensions fixed at their midpoint), so a kink whose location depends on another coordinate -- a sloped boundary, a barrier varying with another input -- can be missed or mis-placed.
- **It thresholds raw $|d^2 f|$,** so it does not distinguish a true kink from steep-but-smooth curvature by convergence behaviour.

For a *known* kink (a strike, a barrier, an exercise boundary already located by root-finding), prefer declaring it with `WithSpecialPoints` -- exact and cheap -- over relying on the scan. Reference: Pachón, R., Platte, R. B. & Trefethen, L. N. (2010), "Piecewise smooth chebfuns," *IMA J. Numer. Anal.* 30, 898--916, and the Chebfun guide on `splitting on`.

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

First-order energy for dimension $j$ sums coefficients with $\alpha_j>0$ and all other degrees zero — the main-effect (first-order) index of Sobol (2001). Total-order energy for dimension $j$ sums every coefficient with $\alpha_j>0$ — the total-effect (total-order) index introduced by Homma & Saltelli (1996), with the efficient estimator of Saltelli et al. (2010). The two are distinct: $S_j \le S_{T_j}$, with equality only when dimension $j$ has no interactions. `ChebyshevTT.SobolIndices()` computes the same quantities by contracting TT coefficient cores, avoiding dense coefficient materialization.

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

- Sobol, I. M. (2001). "Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates." (first-order/main-effect index)
- Homma, T. & Saltelli, A. (1996). "Importance Measures in Global Sensitivity Analysis of Nonlinear Models." *Reliability Engineering & System Safety* 52(1):1--17. (total-order/total-effect index)
- Saltelli, A. et al. (2010). "Variance Based Sensitivity Analysis of Model Output." (efficient total-index estimator)
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
