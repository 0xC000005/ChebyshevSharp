using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_error_threshold.py (PyChebyshev v0.11)
// Tests added incrementally across Phase 1 tasks.
public class ErrorThresholdTests
{
}

public class ConstructorValidation
{
    private static readonly Func<double[], object?, double> Sin2D = (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]);
    private static readonly double[][] UnitSq = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };

    [Fact]
    public void Test_explicit_n_unchanged()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, new[] { 11, 11 });
        Assert.Equal(new[] { 11, 11 }, cheb.NNodes);
        Assert.Null(cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_error_threshold_only()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(1e-6, cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_neither_n_nor_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: null));
    }

    [Fact]
    public void Test_none_without_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: new int?[] { null, 11 }, errorThreshold: null));
    }

    [Fact]
    public void Test_max_n_default()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(64, cheb.MaxN);
    }

    [Fact]
    public void Test_max_n_custom()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 128);
        Assert.Equal(128, cheb.MaxN);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(-1)]
    public void Test_max_n_below_minimum_raises(int badMaxN)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: badMaxN));
        Assert.Contains("maxN must be at least 3", ex.Message);
    }

    [Fact]
    public void Test_max_n_equal_to_minimum_accepted()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 3);
        Assert.Equal(3, cheb.MaxN);
    }
}
