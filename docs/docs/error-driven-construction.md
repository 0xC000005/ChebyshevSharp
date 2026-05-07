# Error-Driven Construction

`ChebyshevApproximation` and `ChebyshevSpline` accept an `errorThreshold`
parameter that drives an automatic node-count selection loop. Pass `null` for
each dimension you want auto-sized; the build doubles the worst-contributing
dim each iteration until both the coefficient-tail estimate and an off-grid
validation pass are at or below `errorThreshold` (or `maxN` is reached, in which
case `BuildWarning` is set).

`errorThreshold` must be finite and greater than zero. Use `maxN` to cap
runtime, not `NaN`, infinity, zero, or a negative threshold.

## Quick Start

```csharp
using ChebyshevSharp;

// Auto-N: every dim resolves automatically.
var cheb = new ChebyshevApproximation(
    function: (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: null,
    errorThreshold: 1e-8);
cheb.Build(verbose: true);  // [auto-N] nNodes=[3, 3], error=...
                            // [auto-N] validation error=...
                            // [auto-N] nNodes=[6, 3], error=...
                            // ...

// Mixed: fix dim 1 at 15, auto-size dim 0 against the threshold.
var mixed = new ChebyshevApproximation(
    function: (x, _) => Math.Sin(8 * x[0]) + Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new int?[] { null, 15 },
    errorThreshold: 1e-6);
```

## Capacity Pre-Check (`GetOptimalN1`)

Before committing to a multi-dimensional build, you can size a single
dimension:

```csharp
int n = ChebyshevApproximation.GetOptimalN1(
    function: (x, _) => Math.Sin(x[0]),
    domain: (-1.0, 1.0),
    errorThreshold: 1e-8);
// n is the smallest N at which a 1-D Chebyshev build hits 1e-8.
```

## Acceptance Check

The coefficient-tail estimate is a fast convergence heuristic, not a proof.
Before auto-N accepts a candidate grid, ChebyshevSharp evaluates the original
function on denser Type-I probe nodes along each auto-sized dimension and
compares those values with the interpolant. This catches aliasing cases where
the last coefficient is accidentally small on an under-resolved grid.

## Spline Per-Piece Threshold

`ChebyshevSpline` applies the threshold per piece — kink-adjacent pieces
refine more than smooth ones automatically:

```csharp
var spl = new ChebyshevSpline(
    function: (x, _) => Math.Abs(x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: null,
    knots: new[] { new[] { 0.0 } },
    errorThreshold: 1e-6);
spl.Build();
// Each piece in spl.Pieces hits the threshold.
```

## When the Cap Bites

If `maxN` is reached before the threshold is satisfied on any auto dim,
`BuildWarning` is set and the build returns the best result it could
achieve. The interpolant is still usable; you should either raise `maxN`
or relax the threshold.

```csharp
var cheb = new ChebyshevApproximation(
    function: (x, _) => Math.Sin(50 * x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: null,
    errorThreshold: 1e-12,
    maxN: 16);
cheb.Build();
if (cheb.BuildWarning != null) Console.Error.WriteLine(cheb.BuildWarning);
```
