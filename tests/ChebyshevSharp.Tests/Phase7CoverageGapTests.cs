using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Coverage gap fillers for Phase 7. Targets defensive paths and edge cases
/// that the per-feature test files don't naturally exercise.
/// </summary>
public class Phase7CoverageGapTests
{
    [Fact]
    public void Test_slider_roots_dim_out_of_range_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();
        Assert.Throws<ArgumentException>(() => slider.Roots(dim: 5,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 } }));
    }

    [Fact]
    public void Test_tt_roots_dim_out_of_range_throws()
    {
        Func<double[], double> f = (p) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 5,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 } }));
    }

    [Fact]
    public void Test_tt_sobol_indices_1d_function()
    {
        // 1-D edge case: SobolIndices on a 1-D TT.
        Func<double[], double> f = (p) => Math.Sin(p[0]);
        var tt = new ChebyshevTT(f, 1, new[] { new[] { -1.0, 1.0 } },
            new[] { 8 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        var result = tt.SobolIndices();
        Assert.Single(result.FirstOrder);
        Assert.True(result.FirstOrder[0] > 0.99);  // single dim explains all variance
    }

    [Fact]
    public void Test_slider_min_with_partial_partition()
    {
        // A slider with multi-dim group: ensure Minimize path works
        // when reducing through a multi-dim partition.
        Func<double[], object?, double> f = (p, _) => p[0] * p[1] + p[2];
        var slider = new ChebyshevSlider(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            partition: new[] { new[] { 0, 1 }, new[] { 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build();
        var (value, _) = slider.Minimize(dim: 2,
            fixedDims: new Dictionary<int, double> { { 0, 0.5 }, { 1, 0.5 } });
        Assert.False(double.IsNaN(value));
    }

    [Fact]
    public void Test_tt_inner_product_self_returns_positive()
    {
        Func<double[], double> f = (p) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        double ip = tt.InnerProduct(tt);
        Assert.True(ip > 0);
    }

    [Fact]
    public void Test_doubles_all_close_handles_empty_arrays()
    {
        Assert.True(Internal.Algebra.DoublesAllClose(Array.Empty<double>(), Array.Empty<double>()));
    }

    [Fact]
    public void Test_doubles_all_close_handles_length_mismatch()
    {
        Assert.False(Internal.Algebra.DoublesAllClose(new[] { 1.0 }, new[] { 1.0, 2.0 }));
    }

    [Fact]
    public void Test_tt_user_frame_domain_validation_on_roots()
    {
        // Custom domains: Roots should correctly validate against user-frame bounds.
        Func<double[], double> f = (p) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -2.0, 3.0 }, new[] { 5.0, 7.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        // Verify Roots correctly validates against [5, 7] for dim 1:
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 100.0 } }));  // 100 outside [5, 7]
    }

    [Fact]
    public void Test_optimize_1d_with_small_order()
    {
        // Edge case: small n, minimize should still work.
        Func<double[], object?, double> f = (p, _) => p[0] * p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        approx.Build();
        var (value, location) = approx.Minimize();
        Assert.Equal(0.0, value, precision: 4);
        Assert.Equal(0.0, location, precision: 4);
    }

    [Fact]
    public void Test_optimize_1d_constant_function()
    {
        // Edge case: constant function. min == max == any node.
        Func<double[], object?, double> f = (p, _) => 7.0;
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        approx.Build();
        var (minVal, _) = approx.Minimize();
        var (maxVal, _) = approx.Maximize();
        Assert.Equal(7.0, minVal, precision: 8);
        Assert.Equal(7.0, maxVal, precision: 8);
    }

    [Fact]
    public void Test_integrate_reversed_bounds_throws_user_frame_dim()
    {
        // bd.lo > bd.hi must throw with user-frame dim index. Exercises line 753.
        Func<double[], double> f = (p) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);

        var ex = Assert.Throws<ArgumentException>(() =>
            tt.Integrate(dims: new[] { 0 }, bounds: new[] { (0.5, -0.5) }));
        // The lo > hi check triggers first; message must mention "dim 0" or "lo=" or "hi="
        Assert.Contains("dim 0", ex.Message);
    }

    [Fact]
    public void Test_integrate_bounds_length_mismatch_throws()
    {
        // bounds.Length != dims.Length must throw before reaching domain validation.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);

        // 2 dims, 1 bound — should throw.
        var ex = Assert.Throws<ArgumentException>(() =>
            tt.Integrate(dims: new[] { 0, 1 }, bounds: new[] { (-0.5, 0.5) }));
        Assert.Contains("bounds length", ex.Message);
    }

    [Fact]
    public void Test_get_evaluation_points_after_explicit_reorder()
    {
        // Use Reorder to force non-identity _dimOrder; exercises line 2014.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { 0.0, 1.0 }, new[] { 5.0, 7.0 }, new[] { 100.0, 200.0 } },
            new[] { 6, 6, 6 }, maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);

        var reordered = tt.Reorder(new[] { 2, 0, 1 });
        // Confirm the dim order is non-identity (forces the permutation branch).
        Assert.False(reordered.DimOrder.SequenceEqual(new[] { 0, 1, 2 }),
            "Reorder must produce non-identity _dimOrder for this test to exercise the fix");

        // Verify columns are in user-frame order: column 0 in [0,1], column 1 in [5,7], column 2 in [100,200].
        double[] flat = reordered.GetEvaluationPoints();
        int n = reordered.GetNumEvaluationPoints();
        for (int i = 0; i < n; i++)
        {
            Assert.InRange(flat[i * 3 + 0], 0.0, 1.0);
            Assert.InRange(flat[i * 3 + 1], 5.0, 7.0);
            Assert.InRange(flat[i * 3 + 2], 100.0, 200.0);
        }

        // Round-trip: Eval at the first returned point must produce a finite value.
        var pt = new[] { flat[0], flat[1], flat[2] };
        double v = reordered.Eval(pt);
        Assert.False(double.IsNaN(v) || double.IsInfinity(v));
    }
}
