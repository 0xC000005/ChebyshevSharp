// tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs
using ChebyshevSharp.Tests.Helpers;
using Xunit;

namespace ChebyshevSharp.Tests;

public class ApproxErgonomicsTests
{
    private static ChebyshevApproximation BuildSimple()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        return approx;
    }

    [Fact]
    public void Descriptor_get_returns_null_when_unset()
    {
        var approx = BuildSimple();
        Assert.Null(approx.GetDescriptor());
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var approx = BuildSimple();
        approx.SetDescriptor("my approximation");
        Assert.Equal("my approximation", approx.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var approx = BuildSimple();
        Assert.True(approx.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var approx = BuildSimple();
        Assert.Equal("function", approx.GetConstructorType());
    }

    [Fact]
    public void GetConstructorType_returns_from_values_for_FromValues_factory()
    {
        var values = new double[5 * 5];
        for (int i = 0; i < values.Length; i++) values[i] = i * 0.1;
        var approx = ChebyshevApproximation.FromValues(
            values,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        Assert.Equal("from_values", approx.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_resolved_node_counts()
    {
        var approx = BuildSimple();
        Assert.Equal(new[] { 5, 5 }, approx.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            maxDerivativeOrder: 3);
        approx.Build(verbose: false);
        Assert.Equal(3, approx.GetMaxDerivativeOrder());
    }

    [Fact]
    public void AdditionalData_threaded_through_Build()
    {
        int callCount = 0;
        string? receivedTag = null;
        var approx = new ChebyshevApproximation(
            (p, data) =>
            {
                callCount++;
                receivedTag = (string?)data;
                return p[0];
            },
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            additionalData: "context-tag");
        approx.Build(verbose: false);

        Assert.Equal(5, callCount);
        Assert.Equal("context-tag", receivedTag);
        Assert.Equal("context-tag", approx.GetAdditionalData());
    }

    [Fact]
    public void AdditionalData_default_is_null()
    {
        var approx = BuildSimple();
        Assert.Null(approx.GetAdditionalData());
    }
}
