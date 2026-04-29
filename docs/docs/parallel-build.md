---
title: Parallel Build & Progress Reporting
---

# Parallel Build & Progress Reporting

ChebyshevSharp v0.10.0 adds two ctor-time kwargs to all four interpolant classes:

- `nWorkers` (`int?`): null (sequential, default), `-1` (`Environment.ProcessorCount`), or positive int (`Parallel.For` pool size).
- `progress` (`IProgress<int>?`): per-evaluation cumulative count for Approx/Spline/Slider; per-sweep for TT.

## Thread-safety contract

When `nWorkers` is non-null, the user-supplied function may be invoked concurrently from multiple threads via `Parallel.For`. Functions that capture mutable state must use locks or external synchronization. Functions that are pure (no captured mutable state) need no special handling.

```csharp
// Pure function: thread-safe by construction.
double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
var ap = new ChebyshevApproximation(F, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 32, 32 },
    nWorkers: -1);  // use all cores
ap.Build();
```

```csharp
// Captures shared state — wrap in lock if used with nWorkers != null.
int callCount = 0;
object lockObj = new();
double F2(double[] p, object? _)
{
    lock (lockObj) callCount++;
    return p[0] * p[0];
}
```

## Progress reporting

Progress reports are cumulative integer counts. The total is derivable upfront via the `GetNumEvaluationPoints()` getter (Phase 4) on Approx/Spline/Slider, or `maxSweeps` on TT.

```csharp
var counter = new Progress<int>(n => Console.Write($"\r{n} evaluations done"));
var ap = new ChebyshevApproximation(F, 2, ..., progress: counter);
ap.Build();
```

`progress: null` is a no-op; the parallel path skips the increment entirely.

## TT and `nWorkers`

`ChebyshevTT` accepts `nWorkers` for API symmetry but ignores it: TT-Cross is an adaptive sampling algorithm, not a pre-grid evaluation, so per-grid parallelism does not apply. `progress` on TT fires once per TT-Cross sweep.
