using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class OptimizeVectorizedTests
{
    [Fact]
    public void Test_minimize_finds_known_minimum()
    {
        static double f(double[] p, object? _) => (p[0] - 0.3) * (p[0] - 0.3);
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 12 });
        approx.Build();

        var (value, location) = approx.Minimize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_maximize_finds_known_maximum()
    {
        static double f(double[] p, object? _) => -((p[0] - 0.7) * (p[0] - 0.7));
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 12 });
        approx.Build();

        var (value, location) = approx.Maximize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.7, location, precision: 8);
    }

    [Fact]
    public void Test_min_at_endpoint()
    {
        static double f(double[] p, object? _) => p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        approx.Build();

        var (value, location) = approx.Minimize();
        Assert.Equal(-1.0, value, precision: 10);
        Assert.Equal(-1.0, location, precision: 10);
    }

    [Fact]
    public void Test_max_at_endpoint()
    {
        static double f(double[] p, object? _) => p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        approx.Build();

        var (value, location) = approx.Maximize();
        Assert.Equal(1.0, value, precision: 10);
        Assert.Equal(1.0, location, precision: 10);
    }

    [Fact]
    public void Test_polynomial_with_multiple_critical_points()
    {
        // f(x) = x^4 - 2x^2 + 1 = (x^2 - 1)^2. Min at x = ±1 (value 0).
        // Has interior critical point at x = 0 (local max, value 1).
        static double f(double[] p, object? _) => Math.Pow(p[0] * p[0] - 1, 2);
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        approx.Build();

        var (minValue, minLocation) = approx.Minimize();
        Assert.Equal(0.0, minValue, precision: 6);
        Assert.True(Math.Abs(Math.Abs(minLocation) - 1.0) < 1e-6);

        var (maxValue, maxLocation) = approx.Maximize();
        Assert.Equal(1.0, maxValue, precision: 6);
        Assert.Equal(0.0, Math.Abs(maxLocation), precision: 6);
    }
}
