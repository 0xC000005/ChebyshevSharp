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
