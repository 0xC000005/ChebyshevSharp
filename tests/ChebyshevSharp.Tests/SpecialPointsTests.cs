using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_special_points.py (PyChebyshev v0.12)
// Tests added incrementally across Phase 1 tasks.
public class SpecialPointsTests
{
}

public class NestedNNodesTests
{
    [Fact]
    public void Test_nested_n_nodes_per_piece()
    {
        // 1D abs(x) with knot at 0; left piece uses 11 nodes, right piece 13.
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 11, 13 } },
            knots: new[] { new[] { 0.0 } });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 13 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_2d_per_sub_interval()
    {
        // 2D: dim 0 has knot at 0.2 (2 pieces with 7,9 nodes), dim 1 has no knot (1 piece, 11 nodes)
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + x[1] * x[1] * x[1] * x[1],
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } },
            knots: new[] { new[] { 0.2 }, Array.Empty<double>() });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 7, 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 9, 11 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_outer_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]),
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11, 11 } },  // only 1 entry, should be 2
                knots: new[] { new[] { 0.0 }, Array.Empty<double>() }));
        Assert.Contains("must have 2 entries", ex.Message);
    }

    [Fact]
    public void Test_nested_inner_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]),
                1, new[] { new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11 } },  // 1 entry, should be 2 (since knots has 1 knot)
                knots: new[] { new[] { 0.0 } }));
        Assert.Contains("must have 2 entries", ex.Message);
    }
}
