// tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtErgonomicsTests
{
    private static ChebyshevTT BuildSimple()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var tt = BuildSimple();
        tt.SetDescriptor("my tt");
        Assert.Equal("my tt", tt.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var tt = BuildSimple();
        Assert.True(tt.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_cross_for_default_method()
    {
        var tt = BuildSimple();
        Assert.Equal("cross", tt.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var tt = BuildSimple();
        Assert.Equal(new[] { 5, 5, 5 }, tt.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_default_is_2()
    {
        var tt = BuildSimple();
        Assert.Equal(2, tt.GetMaxDerivativeOrder());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_kwarg_value()
    {
        var tt = new ChebyshevTT(
            p => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            maxDerivativeOrder: 3);
        tt.Build(verbose: false, seed: 42);
        Assert.Equal(3, tt.GetMaxDerivativeOrder());
    }

    [Fact]
    public void AdditionalData_stored_for_introspection()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            additionalData: "tt-context");
        tt.Build(verbose: false, seed: 42);
        Assert.Equal("tt-context", tt.GetAdditionalData());
    }

    [Fact]
    public void GetEvaluationPoints_full_chebyshev_grid()
    {
        var tt = BuildSimple();
        double[] pts = tt.GetEvaluationPoints();
        int num = tt.GetNumEvaluationPoints();
        Assert.Equal(125, num);  // 5*5*5
        Assert.Equal(375, pts.Length);
    }

    [Fact]
    public void GetEvaluationPoints_returns_cached_array_on_second_call()
    {
        var tt = BuildSimple();
        Assert.Same(tt.GetEvaluationPoints(), tt.GetEvaluationPoints());
    }

    [Fact]
    public void GetDerivativeId_returns_stable_int_per_orders_tuple()
    {
        var tt = BuildSimple();
        int id1 = tt.GetDerivativeId(new[] { 1, 0, 0 });
        int id2 = tt.GetDerivativeId(new[] { 0, 1, 0 });
        Assert.Equal(0, id1);
        Assert.Equal(1, id2);
    }

    [Fact]
    public void EvalByDerivativeId_matches_EvalMulti()
    {
        var tt = BuildSimple();
        int id = tt.GetDerivativeId(new[] { 1, 0, 0 });
        double byMulti = tt.EvalMulti(new[] { 0.3, 0.5, 0.2 }, new[] { new[] { 1, 0, 0 } })[0];
        double byId = tt.Eval(new[] { 0.3, 0.5, 0.2 }, id);
        Assert.Equal(byMulti, byId, precision: 8);  // FD derivatives, looser tolerance
    }
}
