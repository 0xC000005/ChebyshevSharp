# Getting Started

## Installation

Install ChebyshevSharp from NuGet:

```bash
dotnet add package ChebyshevSharp
```

Or add to your `.csproj`:

```xml
<PackageReference Include="ChebyshevSharp" Version="0.1.0" />
```

## Basic Usage

### 1. Define a function

ChebyshevSharp interpolates functions of the form `f(x, data) -> double`, where `x` is a point in multi-dimensional space:

```csharp
using ChebyshevSharp;

double MyFunction(double[] x, object? data)
{
    return Math.Sin(x[0]) * Math.Cos(x[1]);
}
```

### 2. Build the interpolant

Specify the number of dimensions, domain bounds, and polynomial orders:

```csharp
var cheb = new ChebyshevApproximation(
    function: MyFunction,
    ndim: 2,
    domains: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    orders: new[] { 11, 11 }
);
cheb.Build();
```

### 3. Evaluate

```csharp
// Function value
double value = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });

// Partial derivative df/dx0
double dfdx0 = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });

// Second derivative d²f/dx1²
double d2fdx1 = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 2 });
```

### 4. Check accuracy

```csharp
double error = cheb.ErrorEstimate();
Console.WriteLine($"Estimated max error: {error:E2}");
```

## Next Steps

- [API Reference](../api/ChebyshevSharp.html) — full class and method documentation
