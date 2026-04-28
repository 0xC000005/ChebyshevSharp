// tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderErgonomicsTests
{
    private static ChebyshevSlider BuildSimple()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var slider = BuildSimple();
        slider.SetDescriptor("my slider");
        Assert.Equal("my slider", slider.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var slider = BuildSimple();
        Assert.True(slider.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var slider = BuildSimple();
        Assert.Equal("function", slider.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var slider = BuildSimple();
        Assert.Equal(new[] { 5, 5, 5 }, slider.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            partition: new[] { new[] { 0 } },
            pivotPoint: new[] { 0.0 },
            maxDerivativeOrder: 3);
        slider.Build(verbose: false);
        Assert.Equal(3, slider.GetMaxDerivativeOrder());
    }
}
