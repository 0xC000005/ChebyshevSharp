// tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class SplineErgonomicsTests
{
    private static ChebyshevSpline BuildSimple()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { System.Array.Empty<double>(), System.Array.Empty<double>() });
        spline.Build(verbose: false);
        return spline;
    }

    [Fact]
    public void Descriptor_get_returns_null_when_unset()
    {
        var spline = BuildSimple();
        Assert.Null(spline.GetDescriptor());
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var spline = BuildSimple();
        spline.SetDescriptor("my spline");
        Assert.Equal("my spline", spline.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var spline = BuildSimple();
        Assert.True(spline.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var spline = BuildSimple();
        Assert.Equal("function", spline.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var spline = BuildSimple();
        Assert.Equal(new[] { 5, 5 }, spline.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            maxDerivativeOrder: 4);
        spline.Build(verbose: false);
        Assert.Equal(4, spline.GetMaxDerivativeOrder());
    }
}
