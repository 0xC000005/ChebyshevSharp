using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtCalculusTests
{
    private static ChebyshevTT Build1DTt(Func<double, double> f, int n = 12)
    {
        var tt = new ChebyshevTT(
            (double[] p) => f(p[0]),
            1,
            new[] { new[] { -1.0, 1.0 } },
            new[] { n },
            maxRank: 4,
            tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    private static ChebyshevTT Build3DTt(Func<double, double, double, double> f, int n = 8, int rank = 6)
    {
        var tt = new ChebyshevTT(
            (double[] p) => f(p[0], p[1], p[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n, n },
            maxRank: rank,
            tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Test_1D_roots_finds_known_root()
    {
        var tt = Build1DTt(x => x - 0.4);
        double[] roots = tt.Roots();
        Assert.Single(roots);
        Assert.Equal(0.4, roots[0], precision: 6);
    }

    [Fact]
    public void Test_1D_minimize()
    {
        var tt = Build1DTt(x => (x - 0.2) * (x - 0.2));
        var (value, location) = tt.Minimize();
        Assert.Equal(0.0, value, precision: 6);
        Assert.Equal(0.2, location, precision: 6);
    }

    [Fact]
    public void Test_1D_maximize()
    {
        var tt = Build1DTt(x => -(x - 0.2) * (x - 0.2));
        var (value, location) = tt.Maximize();
        Assert.Equal(0.0, value, precision: 6);
        Assert.Equal(0.2, location, precision: 6);
    }

    [Fact]
    public void Test_3D_roots_with_fixed()
    {
        // f(x, y, z) = (x - 0.5) + (y - 0.5)^2 + z. Fix y=0.5, z=-0.5.
        // Then f(x, 0.5, -0.5) = x - 1, root at x = 1.
        // But x ∈ [-1, 1], so root is at x = 1 (endpoint).
        var tt = Build3DTt((x, y, z) => (x - 0.5) + (y - 0.5) * (y - 0.5) + z);
        double[] roots = tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 0.5 }, { 2, -0.5 } });
        Assert.Single(roots);
        Assert.Equal(1.0, roots[0], precision: 6);
    }

    [Fact]
    public void Test_3D_minimize_with_fixed()
    {
        var tt = Build3DTt((x, y, z) => (x - 0.3) * (x - 0.3) + y + z);
        var (value, location) = tt.Minimize(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, -1.0 }, { 2, -1.0 } });
        Assert.Equal(-2.0, value, precision: 5);
        Assert.Equal(0.3, location, precision: 5);
    }

    [Fact]
    public void Test_3D_maximize_with_fixed()
    {
        var tt = Build3DTt((x, y, z) => -((x - 0.3) * (x - 0.3)) + y + z);
        var (value, location) = tt.Maximize(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 1.0 }, { 2, 1.0 } });
        Assert.Equal(2.0, value, precision: 5);
        Assert.Equal(0.3, location, precision: 5);
    }

    [Fact]
    public void Test_unbuilt_roots_throws()
    {
        var tt = new ChebyshevTT((double[] p) => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.Throws<InvalidOperationException>(() => tt.Roots());
    }

    [Fact]
    public void Test_multi_d_roots_requires_dim()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Roots());
    }

    [Fact]
    public void Test_multi_d_minimize_requires_fixed()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Minimize(dim: 0));
    }

    [Fact]
    public void Test_fixed_includes_target_throws()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_fixed_value_out_of_domain_throws()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        // Domain is [-1, 1]^3; passing 5.0 should throw.
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 5.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_under_with_auto_order_user_frame_dim()
    {
        // Build a TT with strong dim-0 dependence; WithAutoOrder may permute.
        // After permutation, user passes dim=0 (user-frame) and expects
        // the Roots to work transparently.
        var tt = ChebyshevTT.WithAutoOrder(
            (double[] p) => p[0] + 0.1 * p[1] + 0.01 * p[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 6,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // Validate that even though _dimOrder is potentially non-identity,
        // user-frame dim=0 finds the root of f(x, fixed=0, fixed=0) = x.
        double[] roots = tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 }, { 2, 0.0 } });

        Assert.Single(roots);
        Assert.Equal(0.0, roots[0], precision: 4);
    }

    [Fact]
    public void Test_under_with_auto_order_user_frame_fixed_validation()
    {
        // Same setup as above; ensure user-frame fixedDims validation works
        // regardless of internal _dimOrder permutation.
        var tt = ChebyshevTT.WithAutoOrder(
            (double[] p) => p[0] + p[1] + p[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // user-frame dim=1 with out-of-domain value should throw with user-frame error.
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 5.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_no_roots_returns_empty()
    {
        var tt = Build1DTt(x => x * x + 0.5);  // No real roots
        double[] roots = tt.Roots();
        Assert.Empty(roots);
    }

    [Fact]
    public void Test_min_at_endpoint()
    {
        var tt = Build1DTt(x => x);
        var (value, location) = tt.Minimize();
        Assert.Equal(-1.0, value, precision: 6);
        Assert.Equal(-1.0, location, precision: 6);
    }
}
