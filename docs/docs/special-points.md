---
title: Special Points
---

# Special Points (Kink Declaration)

`ChebyshevSpline.WithSpecialPoints` is the C# entry point for declaring
known kinks at construction time. It is equivalent to passing the same values
as `knots` to a regular `ChebyshevSpline` constructor, but the name makes the
intent clearer when the values represent discontinuities, barriers, strikes, or
other non-smooth locations.

## Why Declare Kinks

Without a kink declaration, spectral methods converge slowly on non-smooth
functions. Declaring the kink as a sub-interval boundary restores spectral
convergence on each smooth piece.

```csharp
using ChebyshevSharp;

// Without kink declaration: one global polynomial must approximate the kink.
var bad = new ChebyshevApproximation(
    (x, _) => Math.Abs(x[0]),
    1, new[] { new[] { -1.0, 1.0 } }, new[] { 31 });
bad.Build(verbose: false);

// With kink declaration: this piecewise-linear example reaches machine precision
// with 11 nodes per piece in the regression tests.
var good = ChebyshevSpline.WithSpecialPoints(
    function: (x, _) => Math.Abs(x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    specialPoints: new[] { new[] { 0.0 } },
    nNodesNested: new[] { new[] { 11, 11 } });
good.Build(verbose: false);

double value = good.Eval(new[] { -0.25 }, new[] { 0 });  // approximately 0.25
```

## Factory vs Constructor

`specialPoints` is intentionally not a `ChebyshevApproximation` constructor
parameter. Declaring special points changes the representation from one dense
tensor to a piecewise spline, so the C# API exposes it through
`ChebyshevSpline.WithSpecialPoints(...)`.

Use [Adaptive Refinement](adaptive-refinement.md) if you want a heuristic scan
for candidate knots instead of declaring known locations yourself.

## Validation Rules

- `specialPoints` must have one array per dimension.
- Points must be finite, sorted, unique, and strictly inside the domain; endpoints
  are already piece boundaries.
- Exactly one construction mode must be supplied: `nNodesNested`, `nNodes`, or
  `errorThreshold`.
- For `nNodesNested`, each dimension needs `specialPoints[d].Length + 1` node
  counts, one for each sub-interval.

## Per-Sub-Interval Node Counts

Pass nested arrays to `nNodesNested` for per-piece refinement:

```csharp
using ChebyshevSharp;

var spl = ChebyshevSpline.WithSpecialPoints(
    function: (x, _) => Math.Abs(x[0]) + x[1] * x[1],
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    specialPoints: new[] { new[] { 0.0 }, Array.Empty<double>() },
    nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } });
// Dim 0: 2 pieces (split at 0.0) with 7 and 9 nodes.
// Dim 1: 1 piece (no kink) with 11 nodes.
```

Use nested counts when one side of a kink needs more resolution than the other.
Use flat `nNodes` when every sub-interval can use the same node count. Use
`errorThreshold` when you want each piece to choose its node count by the
standard error-driven build loop.
