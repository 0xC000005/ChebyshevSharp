# Special Points (Kink Declaration)

`ChebyshevSpline.WithSpecialPoints` is the C# entry point for declaring
known kinks at construction time. Equivalent to passing the same values
as `knots` to a regular `ChebyshevSpline` constructor, but the name
matches PyChebyshev's `special_points` kwarg for cross-language
discoverability.

## Why Declare Kinks

Without a kink declaration, spectral methods plateau at low precision on
non-smooth functions (Gibbs phenomenon). Declaring the kink as a
sub-interval boundary restores spectral convergence on each piece.

```csharp
// Without kink declaration: plateaus around 1e-3 even at N=31.
var bad = new ChebyshevApproximation(
    (x, _) => Math.Abs(x[0]),
    1, new[] { new[] { -1.0, 1.0 } }, new[] { 31 });

// With kink declaration: machine precision at N=11 per piece.
var good = ChebyshevSpline.WithSpecialPoints(
    function: (x, _) => Math.Abs(x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    specialPoints: new[] { new[] { 0.0 } },
    nNodesNested: new[] { new[] { 11, 11 } });
```

## API Note (Python vs C# Difference)

In Python, `ChebyshevApproximation(special_points=[[...]])` returns a
`ChebyshevSpline` at construction time, leveraging Python's `__new__`
polymorphism. C# constructors cannot return a different type, so the
`specialPoints` kwarg is intentionally absent from
`ChebyshevApproximation`'s constructor. Use `ChebyshevSpline.WithSpecialPoints(...)`
instead.

## Per-Sub-Interval Node Counts

Pass nested arrays to `nNodesNested` for per-piece refinement:

```csharp
var spl = ChebyshevSpline.WithSpecialPoints(
    function: ...,
    numDimensions: 2,
    domain: ...,
    specialPoints: new[] { new[] { 0.0 }, Array.Empty<double>() },
    nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } });
// Dim 0: 2 pieces (split at 0.0) with 7 and 9 nodes.
// Dim 1: 1 piece (no kink) with 11 nodes.
```
