// tests/ChebyshevSharp.Tests/JsonMigrationTests.cs
using System.IO;
using Xunit;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Verifies that JSON files saved by pre-v0.8.0 code (before Descriptor,
/// ConstructorType, SpecialPoints, and RegisteredDerivativeOrders were added
/// to SerializationState) still load cleanly with sensible defaults.
/// </summary>
public class PreV080FixtureMigrationTests
{
    private static string FixturePath(string name) =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "json-pre-v080", name);

    [Fact]
    public void Approx_pre_v080_loads_with_default_descriptor()
    {
        var approx = ChebyshevApproximation.Load(FixturePath("approx.json"));
        Assert.Null(approx.GetDescriptor());
        Assert.True(approx.IsConstructionFinished());
        Assert.Equal("load", approx.GetConstructorType());
    }

    [Fact]
    public void Spline_pre_v080_loads_with_default_descriptor()
    {
        var spline = ChebyshevSpline.Load(FixturePath("spline.json"));
        Assert.Null(spline.GetDescriptor());
        Assert.True(spline.IsConstructionFinished());
    }

    [Fact]
    public void Slider_pre_v080_loads_with_default_descriptor()
    {
        var slider = ChebyshevSlider.Load(FixturePath("slider.json"));
        Assert.Null(slider.GetDescriptor());
        Assert.True(slider.IsConstructionFinished());
    }

    [Fact]
    public void Tt_pre_v080_loads_with_default_max_derivative_order()
    {
        var tt = ChebyshevTT.Load(FixturePath("tt.json"));
        Assert.Null(tt.GetDescriptor());
        Assert.Equal(2, tt.GetMaxDerivativeOrder());
        Assert.True(tt.IsConstructionFinished());
    }
}
