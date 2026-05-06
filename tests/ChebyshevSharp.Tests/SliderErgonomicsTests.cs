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
    public void IsConstructionFinished_true_after_arithmetic_result()
    {
        var slider = BuildSimple();
        var result = slider + slider;
        Assert.True(result.IsConstructionFinished());
        Assert.All(result.Slides, slide => Assert.True(slide.IsConstructionFinished()));
    }

    [Fact]
    public void IsConstructionFinished_true_after_Extrude_result()
    {
        var slider = BuildSimple();
        var result = slider.Extrude((3, new[] { -1.0, 1.0 }, 5));
        Assert.True(result.IsConstructionFinished());
        Assert.All(result.Slides, slide => Assert.True(slide.IsConstructionFinished()));
    }

    [Fact]
    public void IsConstructionFinished_true_after_Slice_result()
    {
        var slider = BuildSimple();
        var result = slider.Slice((0, 0.25));
        Assert.True(result.IsConstructionFinished());
        Assert.All(result.Slides, slide => Assert.True(slide.IsConstructionFinished()));
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

    [Fact]
    public void AdditionalData_threaded_through_Build()
    {
        string? receivedTag = null;
        var slider = new ChebyshevSlider(
            (p, data) =>
            {
                receivedTag = (string?)data;
                return p[0] + p[1];
            },
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 },
            additionalData: "slider-context");
        slider.Build(verbose: false);

        Assert.Equal("slider-context", receivedTag);
        Assert.Equal("slider-context", slider.GetAdditionalData());
    }

    [Fact]
    public void AdditionalData_default_is_null()
    {
        var slider = BuildSimple();
        Assert.Null(slider.GetAdditionalData());
    }

    [Fact]
    public void GetEvaluationPoints_layout_is_row_major()
    {
        var slider = BuildSimple();  // 3D, partition=[[0],[1,2]], nNodes=[5,5,5]
        double[] pts = slider.GetEvaluationPoints();
        int num = slider.GetNumEvaluationPoints();
        // Slide 0: 5 points × 3 dims = 15 doubles
        // Slide 1: 25 points × 3 dims = 75 doubles
        Assert.Equal(30, num);  // 5 + 25
        Assert.Equal(90, pts.Length);
    }

    [Fact]
    public void GetEvaluationPoints_returns_cached_array_on_second_call()
    {
        var slider = BuildSimple();
        Assert.Same(slider.GetEvaluationPoints(), slider.GetEvaluationPoints());
    }

    [Fact]
    public void GetDerivativeId_returns_stable_int_per_orders_tuple()
    {
        var slider = BuildSimple();
        int id1 = slider.GetDerivativeId(new[] { 1, 0, 0 });
        int id2 = slider.GetDerivativeId(new[] { 0, 1, 0 });
        Assert.Equal(0, id1);
        Assert.Equal(1, id2);
    }

    [Fact]
    public void EvalByDerivativeId_matches_EvalByOrders()
    {
        var slider = BuildSimple();
        int id = slider.GetDerivativeId(new[] { 1, 0, 0 });
        double byOrders = slider.Eval(new[] { 0.3, 0.5, 0.2 }, new[] { 1, 0, 0 });
        double byId = slider.Eval(new[] { 0.3, 0.5, 0.2 }, id);
        Assert.Equal(byOrders, byId, precision: 12);
    }
}
