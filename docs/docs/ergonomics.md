---
uid: ergonomics
title: Ergonomics
---

# Ergonomics

ChebyshevSharp v0.8.0 adds a suite of ergonomic accessors and constructor
sugar features across all four interpolant classes. This page summarizes the
new surface.

## Descriptors

A free-form text label you can attach to any interpolant. Survives save/load.

```csharp
var approx = new ChebyshevApproximation(
    (p, _) => p[0] + p[1],
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 5, 5 });
approx.Build();
approx.SetDescriptor("Black-Scholes 2D, vol=20%");

string? label = approx.GetDescriptor();
```

## Additional Data

Pass calibration parameters through every `f(point, data)` call during build:

```csharp
var calibration = LoadCalibration("EOD-2026-04-28.json");
var approx = new ChebyshevApproximation(
    (p, data) => BlackScholesPrice(p, (Calibration)data!),
    numDimensions: 2,
    domain: ...,
    nNodes: ...,
    additionalData: calibration);
approx.Build();

object? stored = approx.GetAdditionalData();
```

`additionalData` is **not** serialized in JSON or `.pcb` files.

`ChebyshevTT` stores `additionalData` for introspection but doesn't thread
it (its function signature is `Func<double[], double>`).

## Derivative ID Registry

Stable session-local int per derivative-orders tuple:

```csharp
int delta = approx.GetDerivativeId(new[] { 1, 0 });
int gamma = approx.GetDerivativeId(new[] { 2, 0 });

double d = approx.Eval(point, delta);
double g = approx.Eval(point, gamma);
```

The registry is per-instance and **does** persist across Save/Load on all
four classes (added in Phase 4).

## Introspection

```csharp
bool ready = approx.IsConstructionFinished();
string how = approx.GetConstructorType();
int[] ns = approx.GetUsedNs();
int maxOrder = approx.GetMaxDerivativeOrder();
double? thr = approx.GetErrorThreshold();    // Approx/Spline only
double[][]? sp = approx.GetSpecialPoints();   // Approx/Spline only
int n = approx.GetNumEvaluationPoints();
double[] pts = approx.GetEvaluationPoints();  // length = n × ndim, row-major
```

## Cloning

```csharp
ChebyshevApproximation clone = approx.Clone();
// clone.Function == null; clone is independent of source
```

## Deferred Build (Approximation + Spline)

```csharp
var deferred = new ChebyshevApproximation(
    (_, _) => throw new InvalidOperationException("not used"),
    numDimensions: 2,
    domain: ...,
    nNodes: new[] { 5, 5 },
    deferBuild: true);
// IsConstructionFinished() == false; Eval/Save throw.

double[] precomputed = await FetchValuesAsync();
deferred.SetOriginalFunctionValues(precomputed);
// IsConstructionFinished() == true; constructor type is "from_values".
```

## Typed Domain/Ns/SpecialPoints Records

```csharp
var d = new Domain(new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } });
double[][] back = d;          // implicit Domain -> double[][]
Domain again = back;          // implicit double[][] -> Domain
```
