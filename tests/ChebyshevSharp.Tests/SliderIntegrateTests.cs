using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestSliderFullIntegrate (Phase 5)
// ======================================================================

public class TestSliderFullIntegrate
{
    [Fact]
    public void Test_pivot_only_function()
    {
        // f(x, y) = constant 5; integral = 5 * 2 * 3 = 30 over [0,2]x[0,3].
        static double F(double[] x, object? _) => 5.0;
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 3.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 1.0, 1.5 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(30.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_additive_function_sum_of_x()
    {
        // f(x, y) = x + y over [-1, 1]^2; integral = 0 (odd in both).
        static double F(double[] x, object? _) => x[0] + x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_separable_function_against_analytical()
    {
        // f(x, y) = sin(x) + cos(y) over [-1, 1]^2.
        // ∫∫ sin(x) dx dy = 0; ∫∫ cos(y) dx dy = 4 sin(1).
        static double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]);
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        double expected = 4.0 * Math.Sin(1.0);
        TestFixtures.AssertClose(expected, result, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_5d_additive_against_analytical()
    {
        // f(x) = sum_i sin(x_i) over [-1, 1]^5.
        // ∫ sin(x_i) dx_i = 0 contributes 0; sum is 0 across all 5 dims with
        // outside-volume = 2^4 each. Closed form: 5 * 0 = 0.
        static double F(double[] x, object? _)
        {
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += Math.Sin(x[i]);
            return s;
        }
        var domain = new[] {
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }
        };
        var slider = new ChebyshevSlider(
            F, 5, domain, new[] { 8, 8, 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 }, new[] { 3 }, new[] { 4 } },
            new[] { 0.0, 0.0, 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-6, atol: 1e-6);
    }
}

// ======================================================================
// TestSliderPartialIntegrate (Phase 5)
// ======================================================================

public class TestSliderPartialIntegrate
{
    [Fact]
    public void Test_returns_slider_over_surviving_dims()
    {
        // f(x, y, z) = sin(x) + cos(y) + z; integrate dim 1 -> Slider over (0, 2)
        static double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]) + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = slider.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevSlider>(result);
        var resultSlider = (ChebyshevSlider)result;
        Assert.Equal(2, resultSlider.NumDimensions);
    }

    [Fact]
    public void Test_partial_disjoint_slide_passes_through()
    {
        // f(x, y) = sin(x) + y^2 over [-1,1]^2 with partition [[0], [1]].
        // Integrate dim 1 -> slide 0 (group [0]) is "none" (passes through),
        // slide 1 (group [1]) is "full". Expected eval at x=0: 2/3.
        static double F(double[] x, object? _) => Math.Sin(x[0]) + x[1] * x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        // ∫_{-1}^{1} (sin(x) + y^2) dy = 2 sin(x) + 2/3.
        double evalAtZero = resultSlider.Eval(new[] { 0.0 }, new[] { 0 });
        TestFixtures.AssertClose(2.0 / 3.0, evalAtZero, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_full_partial_consistency()
    {
        // Joint integrate(dims=[0, 1, 2]) should equal
        // step1=integrate(dims=[0, 1]) then step2=integrate(dims=[0]).
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) + Math.Cos(x[1]) + x[2] * x[2];
        var sliderA = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sliderA.Build(verbose: false);
        double joint = (double)sliderA.Integrate(dims: new[] { 0, 1, 2 });

        // Independent slider for the chained path.
        var sliderB = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sliderB.Build(verbose: false);
        var step1 = (ChebyshevSlider)sliderB.Integrate(dims: new[] { 0, 1 });
        // After integrating original dims 0 and 1, only original dim 2 remains
        // → 1D slider; integrating its dim 0 yields the joint integral.
        double step2 = (double)step1.Integrate(dims: new[] { 0 });

        TestFixtures.AssertClose(joint, step2, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_with_multi_dim_group()
    {
        // f(x, y) = sin(x) * cos(y); partition=[[0, 1]] (one 2D slide).
        // Integrate dim 0 -> "partial" classification on the only slide.
        // ∫_{-1}^{1} sin(x) cos(y) dx = cos(y) * 0 = 0 for all y.
        static double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 },
            new[] { new[] { 0, 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal(1, resultSlider.NumDimensions);
        double evalAtHalf = resultSlider.Eval(new[] { 0.5 }, new[] { 0 });
        TestFixtures.AssertClose(0.0, evalAtHalf, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_with_3d_group_partial_integration()
    {
        // f(x, y, z) = sin(x) + cos(y) + z^2; partition=[[0, 1, 2]] (one 3D slide).
        // Integrate dim 1 -> "partial" classification: 3D slide reduces to 2D over (0, 2).
        // ∫_{-1}^{1} (sin(x) + cos(y) + z^2) dy = 2 sin(x) + 2 sin(1) + 2 z^2.
        // At x=0, z=0: 2 sin(1).
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) + Math.Cos(x[1]) + x[2] * x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1, 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        Assert.Equal(2, resultSlider.NumDimensions);
        double evalAtOrigin = resultSlider.Eval(new[] { 0.0, 0.0 }, new[] { 0, 0 });
        TestFixtures.AssertClose(2.0 * Math.Sin(1.0), evalAtOrigin, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_mixed_classifications()
    {
        // partition=[[0, 1], [2]]; integrate=[0]:
        //   slide 0 (group [0,1]) -> "partial", reduces to 1D over dim 1
        //   slide 1 (group [2])    -> "none", passes through
        // f(x, y, z) = sin(x) cos(y) + z.
        // ∫_{-1}^{1} (sin(x) cos(y) + z) dx = 0 + 2z. So result(y, z) ≈ 2z.
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) * Math.Cos(x[1]) + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal(2, resultSlider.NumDimensions);
        // At (y=0, z=0.5): expected 2 * 0.5 = 1.0
        double evalA = resultSlider.Eval(new[] { 0.0, 0.5 }, new[] { 0, 0 });
        TestFixtures.AssertClose(1.0, evalA, rtol: 1e-6, atol: 1e-6);
        // At (y=0.5, z=0.5): also expected 1.0 (independent of y)
        double evalB = resultSlider.Eval(new[] { 0.5, 0.5 }, new[] { 0, 0 });
        TestFixtures.AssertClose(1.0, evalB, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_result_eval_works()
    {
        // Sanity: result of partial integrate is fully functional.
        static double F(double[] x, object? _) => x[0] + x[1] + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        // Eval should not throw; partition validity preserved.
        Assert.True(resultSlider.Built);
        Assert.Equal(2, resultSlider.NumDimensions);
        // f(x, y, z) = x + y + z, integrate dim 1 over [-1, 1]:
        // g(x, z) = ∫_{-1}^{1} (x + y + z) dy = 2x + 0 + 2z = 2(x + z).
        // At (x=0.3, z=0.7): expected = 2 * (0.3 + 0.7) = 2.0.
        double v = resultSlider.Eval(new[] { 0.3, 0.7 }, new[] { 0, 0 });
        TestFixtures.AssertClose(2.0, v, rtol: 1e-6, atol: 1e-6);
    }
}

// ======================================================================
// TestSliderIntegrateValidation (Phase 5)
// ======================================================================

public class TestSliderIntegrateValidation
{
    private static ChebyshevSlider Make1D()
    {
        static double F(double[] x, object? _) => x[0];
        var slider = new ChebyshevSlider(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 },
            new[] { new[] { 0 } }, new[] { 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    [Fact]
    public void Test_unbuilt_slider_raises()
    {
        static double F(double[] x, object? _) => x[0];
        var slider = new ChebyshevSlider(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 },
            new[] { new[] { 0 } }, new[] { 0.0 });
        // Don't build.
        Assert.Throws<InvalidOperationException>(() => slider.Integrate());
    }

    [Fact]
    public void Test_dims_out_of_range_raises()
    {
        var slider = Make1D();
        var ex = Assert.Throws<ArgumentException>(() => slider.Integrate(dims: new[] { 5 }));
        Assert.Contains("out-of-range", ex.Message);
    }

    [Fact]
    public void Test_negative_dim_raises()
    {
        var slider = Make1D();
        Assert.Throws<ArgumentException>(() => slider.Integrate(dims: new[] { -1 }));
    }

    [Fact]
    public void Test_bounds_outside_domain_raises()
    {
        var slider = Make1D();
        Assert.Throws<ArgumentException>(() =>
            slider.Integrate(
                dims: new[] { 0 },
                bounds: new[] { (-2.0, 2.0) }));
    }
}

// ======================================================================
// TestSliderIntegrateErgonomics (Phase 5)
// ======================================================================

public class TestSliderIntegrateErgonomics
{
    [Fact]
    public void Test_descriptor_preserved_on_partial_result()
    {
        static double F(double[] x, object? _) => x[0] + x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        slider.SetDescriptor("source");
        var result = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal("source", result.GetDescriptor());
    }

    [Fact]
    public void Test_additional_data_preserved_on_partial_result()
    {
        var sentinel = new Dictionary<string, int> { ["k"] = 42 };
        double F(double[] x, object? data)
        {
            Assert.NotNull(data);
            return x[0] + x[1];
        }
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 },
            additionalData: sentinel);
        slider.Build(verbose: false);
        var result = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Same(sentinel, result.GetAdditionalData());
    }
}
