using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderRootsTests
{
    private static readonly double Tolerance = 1e-8;

    [Fact]
    public void Test_1D_slider_finds_known_root()
    {
        // f(x) = x - 0.5 has a root at x = 0.5
        Func<double[], object?, double> f = (p, _) => p[0] - 0.5;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        slider.Build();

        double[] roots = slider.Roots();

        Assert.Single(roots);
        Assert.Equal(0.5, roots[0], precision: 8);
    }

    [Fact]
    public void Test_1D_slider_no_roots_returns_empty()
    {
        // f(x) = x^2 + 1 has no real roots
        Func<double[], object?, double> f = (p, _) => p[0] * p[0] + 1.0;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        slider.Build();

        double[] roots = slider.Roots();

        Assert.Empty(roots);
    }

    [Fact]
    public void Test_2D_slider_with_fixed_finds_root()
    {
        // f(x, y) = (x - 0.3) + (y - 0.7), fixing y=0.7 leaves f(x, 0.7) = x - 0.3
        Func<double[], object?, double> f = (p, _) => (p[0] - 0.3) + (p[1] - 0.7);
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        double[] roots = slider.Roots(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.7 } });

        Assert.Single(roots);
        Assert.Equal(0.3, roots[0], precision: 8);
    }

    [Fact]
    public void Test_multi_d_requires_dim_param()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        Assert.Throws<ArgumentException>(() => slider.Roots());
    }

    [Fact]
    public void Test_multi_d_requires_fixed_for_other_dims()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        // Specifying dim=0 but no fixedDims for dim=1 should fail.
        Assert.Throws<ArgumentException>(() => slider.Roots(dim: 0, fixedDims: null));
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] - 0.5;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        // No Build() call

        Assert.Throws<InvalidOperationException>(() => slider.Roots());
    }
}
