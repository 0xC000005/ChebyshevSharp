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

    [Fact]
    public void AdditionalData_threaded_to_each_piece()
    {
        int callCount = 0;
        string? receivedTag = null;
        var spline = new ChebyshevSpline(
            (p, data) =>
            {
                callCount++;
                receivedTag = (string?)data;
                return p[0];
            },
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { new[] { 0.0 } },
            additionalData: "spline-context");
        spline.Build(verbose: false);

        Assert.Equal(10, callCount);  // 2 pieces × 5 nodes
        Assert.Equal("spline-context", receivedTag);
        Assert.Equal("spline-context", spline.GetAdditionalData());
    }

    [Fact]
    public void AdditionalData_default_is_null()
    {
        var spline = BuildSimple();
        Assert.Null(spline.GetAdditionalData());
    }

    [Fact]
    public void GetEvaluationPoints_layout_is_row_major()
    {
        var spline = BuildSimple();
        double[] pts = spline.GetEvaluationPoints();
        int num = spline.GetNumEvaluationPoints();

        Assert.Equal(25, num);  // single-piece, nNodes=[5,5]
        Assert.Equal(50, pts.Length);
    }

    [Fact]
    public void GetEvaluationPoints_returns_snapshot_on_each_call()
    {
        var spline = BuildSimple();
        double[] first = spline.GetEvaluationPoints();
        first[0] = 123.0;

        double[] second = spline.GetEvaluationPoints();
        Assert.NotSame(first, second);
        Assert.NotEqual(123.0, second[0]);
    }

    [Fact]
    public void GetErrorThreshold_returns_ctor_value_when_set()
    {
        var spline = new ChebyshevSpline(
            (p, _) => Math.Abs(p[0]),
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null },
            knots: new[] { new[] { 0.0 } },
            errorThreshold: 1e-6);
        spline.Build(verbose: false);
        Assert.Equal(1e-6, spline.GetErrorThreshold());
    }

    [Fact]
    public void GetSpecialPoints_returns_knots_used_for_construction()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { new[] { -0.5, 0.5 } });
        spline.Build(verbose: false);

        double[][]? sp = spline.GetSpecialPoints();
        Assert.NotNull(sp);
        Assert.Equal(new[] { -0.5, 0.5 }, sp![0]);
    }

    [Fact]
    public void GetDerivativeId_returns_stable_int_per_orders_tuple()
    {
        var spline = BuildSimple();
        int id1 = spline.GetDerivativeId(new[] { 1, 0 });
        int id2 = spline.GetDerivativeId(new[] { 0, 1 });
        Assert.Equal(0, id1);
        Assert.Equal(1, id2);
        Assert.Equal(0, spline.GetDerivativeId(new[] { 1, 0 }));
    }

    [Fact]
    public void EvalByDerivativeId_matches_EvalByOrders()
    {
        var spline = BuildSimple();
        int id = spline.GetDerivativeId(new[] { 1, 0 });
        double byOrders = spline.Eval(new[] { 0.3, 0.5 }, new[] { 1, 0 });
        double byId = spline.Eval(new[] { 0.3, 0.5 }, id);
        Assert.Equal(byOrders, byId, precision: 12);
    }
}
