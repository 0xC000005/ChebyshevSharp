using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestApproxSobolIndices (Phase 6 Task 6)
// ======================================================================

public class TestApproxSobolIndices
{
    [Fact]
    public void Test_additive_function_first_order_sums_to_one()
    {
        // f(x, y) = sin(x) + cos(y) — additive, no interaction term.
        // FirstOrder[0] + FirstOrder[1] ≈ 1; both TotalOrder ≈ FirstOrder (no mixing).
        static double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 16, 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0] + s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(s.FirstOrder[0], s.TotalOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(s.FirstOrder[1], s.TotalOrder[1], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_pure_first_dim_function()
    {
        // f(x, y) = sin(x) — constant in y.
        // FirstOrder[0] ≈ 1; FirstOrder[1] ≈ 0; TotalOrder[1] ≈ 0.
        static double F(double[] p, object? _) => Math.Sin(p[0]);
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 16, 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(0.0, s.FirstOrder[1], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(0.0, s.TotalOrder[1], rtol: 0, atol: 1e-10);
    }

    [Fact]
    public void Test_multiplicative_function_total_order_is_one()
    {
        // f(x, y) = x * y on [-1,1]^2 — pure interaction term, no additive part.
        // FirstOrder[*] ≈ 0; TotalOrder[0] ≈ TotalOrder[1] ≈ 1.
        static double F(double[] p, object? _) => p[0] * p[1];
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(0.0, s.FirstOrder[0], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(0.0, s.FirstOrder[1], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[1], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_total_order_at_least_first_order()
    {
        // Invariant: FirstOrder[d] <= TotalOrder[d] for every d.
        static double F(double[] p, object? _) => Math.Sin(p[0] * p[1]) + Math.Cos(p[2]);
        var ap = new ChebyshevApproximation(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        for (int d = 0; d < 3; d++)
            Assert.True(s.FirstOrder[d] <= s.TotalOrder[d] + 1e-12,
                $"FirstOrder[{d}]={s.FirstOrder[d]} > TotalOrder[{d}]={s.TotalOrder[d]}");
    }

    [Fact]
    public void Test_dim_importance_ranking()
    {
        // f(x,y,z) = 100*sin(x) + 1*y + 0.01*z*z — clearly x > y > z by sensitivity.
        static double F(double[] p, object? _) => 100 * Math.Sin(p[0]) + p[1] + 0.01 * p[2] * p[2];
        var ap = new ChebyshevApproximation(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.True(s.TotalOrder[0] > s.TotalOrder[1]);
        Assert.True(s.TotalOrder[1] > s.TotalOrder[2]);
    }

    [Fact]
    public void Test_1d_function_first_order_equals_total_order_one()
    {
        // 1D function: FirstOrder[0] = TotalOrder[0] = 1 (no interaction possible).
        static double F(double[] p, object? _) => Math.Sin(p[0]);
        var ap = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_constant_function_zero_variance()
    {
        // f(x, y) = 5 — constant. DCT-II may leave ~1e-29 noise, which is
        // below the relative floor scale * 1e-28 and therefore reports zero indices.
        static double F(double[] p, object? _) => 5.0;
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.True(s.Variance < 1e-20,
            $"Constant function should have near-zero variance, got {s.Variance}");
        Assert.Equal(0.0, s.FirstOrder[0]);
        Assert.Equal(0.0, s.TotalOrder[1]);
    }

    [Fact]
    public void Test_zero_function_zero_variance()
    {
        static double F(double[] p, object? _) => 0.0;
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.Equal(0.0, s.Variance);
        Assert.Equal(0.0, s.FirstOrder[0]);
        Assert.Equal(0.0, s.TotalOrder[1]);
    }

    [Fact]
    public void Test_large_constant_plus_tiny_signal_recovers_dim()
    {
        // A fixed absolute variance cutoff can incorrectly suppress valid
        // low-amplitude sensitivity when a large constant offset is present.
        static double F(double[] p, object? _) => 1.0 + 1e-12 * p[0];
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.True(s.Variance > 1e-25,
            $"Variance={s.Variance} should preserve the tiny nonconstant signal");
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-6, atol: 1e-6);
        Assert.True(s.TotalOrder[1] < 1e-5,
            $"TotalOrder[1]={s.TotalOrder[1]} should remain at numerical noise level");
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        static double F(double[] p, object? _) => p[0];
        var ap = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 }, deferBuild: true);
        Assert.Throws<InvalidOperationException>(() => ap.SobolIndices());
    }
}

// ======================================================================
// TestSplineSobolIndices (Phase 6 Task 7)
// ======================================================================

public class TestSplineSobolIndices
{
    [Fact]
    public void Test_single_piece_matches_approx()
    {
        // Single-piece spline (no interior knots) matches Approx exactly.
        static double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 16, 16 };

        var ap = new ChebyshevApproximation(F, 2, domain, nNodes);
        ap.Build(verbose: false);
        var apSob = ap.SobolIndices();

        var sp = new ChebyshevSpline(F, 2, domain, nNodes,
            new[] { Array.Empty<double>(), Array.Empty<double>() });
        sp.Build(verbose: false);
        var spSob = sp.SobolIndices();

        TestFixtures.AssertClose(apSob.FirstOrder[0], spSob.FirstOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(apSob.FirstOrder[1], spSob.FirstOrder[1], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(apSob.TotalOrder[0], spSob.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_piecewise_abs_x_1d()
    {
        // f(x) = |x| on [-1,1] with knot at 0; per-piece smooth, both pieces equal volume.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();
        // 1D: FirstOrder[0] = TotalOrder[0] = 1 (or 0 if Variance is 0; |x| is non-constant so Variance > 0).
        Assert.True(s.Variance > 0);
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_piecewise_step_has_between_piece_variance()
    {
        static double F(double[] p, object? _) => p[0] < 0.0 ? 0.0 : 1.0;
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();

        Assert.True(s.Variance > 0.0,
            "A step spline is nonconstant even though each constant piece has zero internal variance.");
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-12, atol: 1e-12);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void Test_piecewise_x_active_on_one_y_piece_counts_interaction()
    {
        static double F(double[] p, object? _) => p[1] < 0.0 ? p[0] : 0.0;
        var sp = new ChebyshevSpline(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 3 },
            new[] { Array.Empty<double>(), new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();

        Assert.True(s.Variance > 0.0);
        TestFixtures.AssertClose(0.5, s.FirstOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(0.0, s.FirstOrder[1], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(0.5, s.TotalOrder[1], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_piecewise_abs_x_plus_abs_y_2d()
    {
        // f(x,y) = |x| + |y| with knots at 0 in both dims. Additive → both first-orders sum to 1.
        static double F(double[] p, object? _) => Math.Abs(p[0]) + Math.Abs(p[1]);
        var sp = new ChebyshevSpline(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0.0 }, new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0] + s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
        // Both dims contribute roughly equally (symmetric function on symmetric domain).
        TestFixtures.AssertClose(s.FirstOrder[0], s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        static double F(double[] p, object? _) => p[0];
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 5 },
            new[] { Array.Empty<double>() });
        // sp not built — should throw.
        Assert.Throws<InvalidOperationException>(() => sp.SobolIndices());
    }
}
