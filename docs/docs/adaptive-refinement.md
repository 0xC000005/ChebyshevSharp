---
title: Adaptive Refinement
---

# Adaptive Refinement

ChebyshevSharp v0.10.0 adds three adaptive-refinement APIs derived from PyChebyshev v0.20.0 + v0.20.1.

## ChebyshevSpline.AutoKnots

Auto-place knots at function kinks via a curvature-spike scan. Useful for piecewise-smooth functions like `|x|`, `max(0, x)`, or piecewise polynomials.

```csharp
double F(double[] p, object? _) => Math.Abs(p[0]);
var sp = ChebyshevSpline.AutoKnots(F, 1,
    new[] { new[] { -1.0, 1.0 } },
    new[] { 16 });
// Discovers a knot near x=0; the resulting Spline has 2 pieces.
```

Tuning kwargs: `thresholdFactor` (default 5.0), `maxKnotsPerDim` (default 5), `nScanPoints` (default 200).

## SobolIndices

Variance decomposition from spectral Chebyshev coefficients. No Monte Carlo, no extra evaluations beyond what's already in `TensorValues`.

```csharp
double F(double[] p, object? _) => Math.Sin(p[0]) + p[1] * p[2];
var ap = new ChebyshevApproximation(F, 3, ..., new[] { 16, 16, 16 });
ap.Build();
SobolResult s = ap.SobolIndices();
Console.WriteLine($"FirstOrder: [{string.Join(", ", s.FirstOrder)}]");
Console.WriteLine($"TotalOrder: [{string.Join(", ", s.TotalOrder)}]");
Console.WriteLine($"Variance: {s.Variance}");
```

`s.Variance == 0` indicates a constant function — the indices are zero and meaningless. For `ChebyshevSpline`, indices are aggregated across pieces (volume-weighted variance).

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
