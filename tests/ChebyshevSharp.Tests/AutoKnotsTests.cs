using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestAutoKnots (Phase 6 Task 8)
// ======================================================================

public class TestAutoKnots
{
    [Fact]
    public void Test_abs_x_finds_knot_near_zero()
    {
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.True(sp.NumPieces >= 2,
            $"Expected at least 2 pieces (knot near 0); got {sp.NumPieces}");
        var knots0 = sp.Knots[0];
        Assert.Contains(knots0, k => Math.Abs(k) < 0.05);
    }

    [Fact]
    public void Test_relu_finds_knot_near_half()
    {
        static double F(double[] p, object? _) => Math.Max(0.0, p[0] - 0.5);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 8 });
        Assert.True(sp.NumPieces >= 2);
        var knots0 = sp.Knots[0];
        Assert.Contains(knots0, k => Math.Abs(k - 0.5) < 0.05);
    }

    [Fact]
    public void Test_2d_additive_abs_finds_knots_per_dim()
    {
        static double F(double[] p, object? _) => Math.Abs(p[0]) + Math.Abs(p[1]);
        var sp = ChebyshevSpline.AutoKnots(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        Assert.True(sp.Knots[0].Length >= 1);
        Assert.True(sp.Knots[1].Length >= 1);
    }

    [Fact]
    public void Test_smooth_function_finds_no_knots()
    {
        static double F(double[] p, object? _) => p[0] * p[0];
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        // Smooth function: |d²f| is constant, no spikes above threshold.
        Assert.Empty(sp.Knots[0]);
        Assert.Equal(1, sp.NumPieces);
    }

    [Fact]
    public void Test_high_threshold_finds_no_knots_for_abs()
    {
        // Threshold so high that even |x|'s spike is filtered out.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            thresholdFactor: 1000.0);
        Assert.Empty(sp.Knots[0]);
    }

    [Fact]
    public void Test_max_knots_per_dim_caps_count()
    {
        // f(x) with many bumps; cap at 1.
        static double F(double[] p, object? _) =>
            Math.Abs(p[0] - 0.2) + Math.Abs(p[0] - 0.5) + Math.Abs(p[0] - 0.8);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 8 },
            maxKnotsPerDim: 1);
        Assert.True(sp.Knots[0].Length <= 1);
    }

    [Fact]
    public void Test_n_scan_points_too_small_throws()
    {
        static double F(double[] p, object? _) => p[0];
        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 },
            nScanPoints: 2));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Test_threshold_factor_must_be_finite_and_positive(double thresholdFactor)
    {
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            thresholdFactor: thresholdFactor));
    }

    [Fact]
    public void Test_function_returning_nan_throws()
    {
        static double F(double[] p, object? _) => double.NaN;
        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 }));
    }

    [Fact]
    public void Test_null_function_throws_argument_null()
    {
        Assert.Throws<ArgumentNullException>(() => ChebyshevSpline.AutoKnots(null!, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 }));
    }

    [Fact]
    public void Test_invalid_domain_is_rejected_before_scan()
    {
        int calls = 0;
        double F(double[] p, object? _)
        {
            calls++;
            return p[0];
        }

        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            Array.Empty<double[]>(), new[] { 5 }));
        Assert.Equal(0, calls);
    }

    [Fact]
    public void Test_invalid_num_nodes_is_rejected_before_scan()
    {
        int calls = 0;
        double F(double[] p, object? _)
        {
            calls++;
            return p[0];
        }

        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 0 }));
        Assert.Equal(0, calls);
    }

    [Fact]
    public void Test_max_knots_zero_returns_no_knot_spline()
    {
        // maxKnotsPerDim=0 means "no auto-knots, just build a single-piece spline".
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            maxKnotsPerDim: 0);
        Assert.Empty(sp.Knots[0]);
        Assert.Equal(1, sp.NumPieces);
    }

    [Fact]
    public void Test_result_is_fully_functional()
    {
        // The returned ChebyshevSpline must Eval correctly.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        TestFixtures.AssertClose(0.5, sp.Eval(new[] { 0.5 }, new[] { 0 }), rtol: 1e-3, atol: 1e-3);
        TestFixtures.AssertClose(0.5, sp.Eval(new[] { -0.5 }, new[] { 0 }), rtol: 1e-3, atol: 1e-3);
    }
}
