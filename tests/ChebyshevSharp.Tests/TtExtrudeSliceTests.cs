using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTToDense + TestTTExtrude + TestTTSlice (PyChebyshev v0.18.0).
// Tests added in Phase 2 Task 8.
public class TtExtrudeSliceTests
{
}

public class ToDenseTests
{
    [Fact]
    public void Test_to_dense_returns_array_with_product_size()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        Assert.Equal(16, dense.Length);
    }

    [Fact]
    public void Test_to_dense_shape_matches_n_nodes()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1] + p[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 5, 6 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        Assert.Equal(120, dense.Length);
    }

    [Fact]
    public void Test_to_dense_values_match_eval_at_nodes()
    {
        int n = 5;
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Cos(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double expected = tt.Eval(new[] { nodes[0][i], nodes[1][j] });
                TestFixtures.AssertClose(expected, dense[i * n + j], atol: 1e-10);
            }
    }

    [Fact]
    public void Test_to_dense_round_trip_via_from_values()
    {
        var ttA = new ChebyshevTT(p => p[0] * Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });
        ttA.Build(verbose: false);
        double[] dense = ttA.ToDense();
        var ttB = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });
        double[] xTest = { 0.3, -0.4 };
        TestFixtures.AssertClose(ttA.Eval(xTest), ttB.Eval(xTest), atol: 1e-8);
    }

    [Fact]
    public void Test_to_dense_constant_function()
    {
        var tt = new ChebyshevTT(p => 3.0, 2,
            new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 4, 4 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        foreach (double v in dense)
            TestFixtures.AssertClose(3.0, v, atol: 1e-10);
    }

    [Fact]
    public void Test_to_dense_overflow_guard()
    {
        // 9 dims, 80 nodes each = 80^9 elements * 8 bytes >> int.MaxValue.
        var tt = new ChebyshevTT(p => p[0], 9,
            Enumerable.Repeat(new[] { -1.0, 1.0 }, 9).ToArray(),
            Enumerable.Repeat(80, 9).ToArray());
        tt.Build(verbose: false);
        Assert.Throws<OverflowException>(() => tt.ToDense());
    }
}

public class ExtrudeTests
{
    [Fact]
    public void Test_extrude_returns_tt_with_new_dim()
    {
        var tt = new ChebyshevTT(p => p[0] * p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 4);
        Assert.Equal(2, result.NumDimensions);
    }

    [Fact]
    public void Test_extrude_preserves_eval_at_existing_dims()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 5);
        // Result(x, y) should equal sin(x) for any y.
        double[] xs = { -0.5, 0.0, 0.3 };
        double[] ys = { 0.1, 0.5, 0.9 };
        foreach (double x in xs)
            foreach (double y in ys)
                TestFixtures.AssertClose(Math.Sin(x), result.Eval(new[] { x, y }), atol: 1e-6);
    }

    [Fact]
    public void Test_extrude_constant_value()
    {
        var tt = new ChebyshevTT(p => 7.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 5.0), newN: 4);
        TestFixtures.AssertClose(7.0, result.Eval(new[] { 0.5, 2.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_extrude_validates_dim_idx()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.Extrude(dim: 5, newDomain: (0.0, 1.0), newN: 4));
    }

    [Fact]
    public void Test_extrude_validates_domain_order()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (1.0, 0.0), newN: 4));
    }

    [Fact]
    public void Test_extrude_validates_finite_domain()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var loEx = Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (double.NaN, 1.0), newN: 4));
        Assert.Contains("finite", loEx.Message);

        var hiEx = Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (0.0, double.PositiveInfinity), newN: 4));
        Assert.Contains("finite", hiEx.Message);
    }

    [Fact]
    public void Test_extrude_validates_nn_minimum()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 1));
    }
}

public class SliceTests
{
    [Fact]
    public void Test_slice_returns_lower_dim_tt()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 6, 6 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 0, value: 0.5);
        Assert.Equal(1, result.NumDimensions);
    }

    [Fact]
    public void Test_slice_at_node_uses_fast_path()
    {
        int n = 6;
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) * Math.Cos(p[1]), 2, domain, new[] { n, n });
        tt.Build(verbose: false);
        var (nodes, _) = ChebyshevTT.Nodes(2, domain, new[] { n, n });
        var result = tt.Slice(dim: 0, value: nodes[0][2]);
        foreach (double y in new[] { -0.5, 0.0, 0.5 })
            TestFixtures.AssertClose(
                tt.Eval(new[] { nodes[0][2], y }),
                result.Eval(new[] { y }),
                atol: 1e-10);
    }

    [Fact]
    public void Test_slice_at_interior_value_matches_eval()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Cos(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 10, 10 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 1, value: 0.3);
        foreach (double x in new[] { -0.5, 0.0, 0.4 })
            TestFixtures.AssertClose(
                Math.Sin(x) + Math.Cos(0.3),
                result.Eval(new[] { x }),
                atol: 1e-6);
    }

    [Fact]
    public void Test_slice_endpoint_dim_left()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 6 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 0, value: 0.5);
        TestFixtures.AssertClose(0.3, result.Eval(new[] { 0.6 }), atol: 1e-8);
    }

    [Fact]
    public void Test_slice_endpoint_dim_right()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 6, 4 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 1, value: 0.5);
        TestFixtures.AssertClose(0.15, result.Eval(new[] { 0.3 }), atol: 1e-8);
    }

    [Fact]
    public void Test_slice_validates_value_within_domain()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.Slice(dim: 0, value: 5.0));
    }

    [Fact]
    public void Test_slice_validates_finite_value()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        tt.Build(verbose: false);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => tt.Slice(dim: 0, value: double.NaN));
        Assert.Contains("finite", ex.Message);
    }

    [Fact]
    public void Test_slice_validates_dim_idx()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.Slice(dim: 5, value: 0.0));
    }

    [Fact]
    public void Test_slice_then_to_dense()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        tt.Build(verbose: false);
        var sliced = tt.Slice(dim: 0, value: 0.5);
        double[] dense = sliced.ToDense();
        Assert.Equal(5, dense.Length);
    }

    [Fact]
    public void Test_slice_1d_tt_throws()
    {
        // Slicing a 1D TT would produce a 0D result — explicitly disallowed.
        var tt = new ChebyshevTT(p => p[0] * p[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 5 },
            tolerance: 1e-4, maxRank: 3);
        tt.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => tt.Slice(0, 0.5));
        Assert.Contains("1D", ex.Message);
    }
}
