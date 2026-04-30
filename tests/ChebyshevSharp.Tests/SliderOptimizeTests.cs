using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderOptimizeTests
{
    private static ChebyshevSlider Build1DSlider(Func<double, double> f, int n = 16, double a = -1.0, double b = 1.0)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0]);
        var slider = new ChebyshevSlider(wrapper, 1,
            new[] { new[] { a, b } },
            new[] { n },
            partition: new[] { new[] { 0 } },
            pivotPoint: new[] { (a + b) / 2.0 });
        slider.Build();
        return slider;
    }

    private static ChebyshevSlider Build2DSlider(Func<double, double, double> f, int n = 16)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0], p[1]);
        var slider = new ChebyshevSlider(wrapper, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();
        return slider;
    }

    [Fact]
    public void Test_1D_minimize_quadratic()
    {
        // f(x) = (x - 0.3)^2 has min at x = 0.3, value = 0
        var slider = Build1DSlider(x => (x - 0.3) * (x - 0.3));
        var (value, location) = slider.Minimize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_1D_maximize_quadratic()
    {
        // f(x) = -(x - 0.3)^2 has max at x = 0.3, value = 0
        var slider = Build1DSlider(x => -(x - 0.3) * (x - 0.3));
        var (value, location) = slider.Maximize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_1D_min_at_endpoint()
    {
        // f(x) = x has min at x = -1
        var slider = Build1DSlider(x => x);
        var (value, location) = slider.Minimize();
        Assert.Equal(-1.0, value, precision: 8);
        Assert.Equal(-1.0, location, precision: 8);
    }

    [Fact]
    public void Test_1D_max_at_endpoint()
    {
        // f(x) = x has max at x = 1
        var slider = Build1DSlider(x => x);
        var (value, location) = slider.Maximize();
        Assert.Equal(1.0, value, precision: 8);
        Assert.Equal(1.0, location, precision: 8);
    }

    [Fact]
    public void Test_2D_minimize_with_fixed()
    {
        // f(x, y) = (x - 0.5)^2 + y, fixing y=-0.5 makes f = (x - 0.5)^2 - 0.5
        // Min at x=0.5, value = -0.5
        var slider = Build2DSlider((x, y) => (x - 0.5) * (x - 0.5) + y);
        var (value, location) = slider.Minimize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, -0.5 } });
        Assert.Equal(-0.5, value, precision: 6);
        Assert.Equal(0.5, location, precision: 6);
    }

    [Fact]
    public void Test_2D_maximize_with_fixed()
    {
        // f(x, y) = -((x - 0.5)^2) + y, fixing y=0.5 makes f = -(x - 0.5)^2 + 0.5
        // Max at x=0.5, value = 0.5
        var slider = Build2DSlider((x, y) => -((x - 0.5) * (x - 0.5)) + y);
        var (value, location) = slider.Maximize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.5 } });
        Assert.Equal(0.5, value, precision: 6);
        Assert.Equal(0.5, location, precision: 6);
    }

    [Fact]
    public void Test_min_max_unbuilt_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var slider = new ChebyshevSlider(f, 1, new[] { new[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        // No Build() call

        Assert.Throws<InvalidOperationException>(() => slider.Minimize());
        Assert.Throws<InvalidOperationException>(() => slider.Maximize());
    }

    [Fact]
    public void Test_multi_d_min_requires_dim()
    {
        var slider = Build2DSlider((x, y) => x + y);
        Assert.Throws<ArgumentException>(() => slider.Minimize());
    }

    [Fact]
    public void Test_multi_d_max_requires_fixed()
    {
        var slider = Build2DSlider((x, y) => x + y);
        Assert.Throws<ArgumentException>(() => slider.Maximize(dim: 0));
    }

    [Fact]
    public void Test_min_max_returns_tuple_value_first()
    {
        // Order: (value, location). Testing this convention explicitly.
        var slider = Build1DSlider(x => x * x);  // min at 0
        var (value, location) = slider.Minimize();
        Assert.Equal(0.0, value, precision: 8);  // value first
        Assert.Equal(0.0, location, precision: 8);
    }
}
