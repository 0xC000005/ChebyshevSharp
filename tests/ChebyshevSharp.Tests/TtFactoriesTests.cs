using System;
using System.Linq;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTNodes + TestTTFromValues (PyChebyshev v0.18.0).
// Tests added in Phase 2 Task 7.
public class TtFactoriesTests
{
}

public class NodesTests
{
    [Fact]
    public void Test_nodes_returns_per_dim_arrays()
    {
        var (nodes, shape) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 5 });
        Assert.Equal(2, nodes.Length);
        Assert.Equal(4, nodes[0].Length);
        Assert.Equal(5, nodes[1].Length);
        Assert.Equal(new[] { 4, 5 }, shape);
    }

    [Fact]
    public void Test_nodes_within_domain()
    {
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { 0.0, 2.0 }, new[] { -3.0, 3.0 } }, new[] { 6, 6 });
        Assert.True(nodes[0].Min() >= 0.0 - 1e-12);
        Assert.True(nodes[0].Max() <= 2.0 + 1e-12);
        Assert.True(nodes[1].Min() >= -3.0 - 1e-12);
        Assert.True(nodes[1].Max() <= 3.0 + 1e-12);
    }

    [Fact]
    public void Test_nodes_consistency_with_approximation_nodes()
    {
        var (ttNodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        var chebInfo = ChebyshevApproximation.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        double[][] chebNodes = chebInfo.NodesPerDim;
        for (int d = 0; d < 2; d++)
            for (int j = 0; j < 5; j++)
                TestFixtures.AssertClose(chebNodes[d][j], ttNodes[d][j], atol: 1e-14);
    }
}

public class FromValuesTests
{
    [Fact]
    public void Test_from_values_round_trip_at_node()
    {
        int n = 8;
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        // dense[i, j] = sin(nodes_x[i]) * cos(nodes_y[j])
        var dense = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dense[i * n + j] = Math.Sin(nodes[0][i]) * Math.Cos(nodes[1][j]);

        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        Assert.Equal(2, tt.NumDimensions);
        TestFixtures.AssertClose(dense[2 * n + 3],
            tt.Eval(new[] { nodes[0][2], nodes[1][3] }), atol: 1e-10);
    }

    [Fact]
    public void Test_from_values_constant_function_recovers()
    {
        var dense = Enumerable.Repeat(7.0, 25).ToArray();
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 5, 5 });
        TestFixtures.AssertClose(7.0, tt.Eval(new[] { 0.3, 0.4 }), atol: 1e-10);
    }

    [Fact]
    public void Test_from_values_validates_tensor_shape()
    {
        var bad = new double[20]; // 4*5 != 5*5
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_validates_nan()
    {
        var bad = new double[25];
        bad[0] = double.NaN;
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_validates_infinity()
    {
        var bad = new double[25];
        bad[0] = double.PositiveInfinity;
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_max_rank_caps_rank()
    {
        var rng = new Random(42);
        var dense = Enumerable.Range(0, 6 * 6 * 6).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        var tt = ChebyshevTT.FromValues(dense, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, maxRank: 3);
        foreach (int r in tt.TtRanks)
            Assert.True(r <= 3, $"max_rank=3 violated, got rank {r}");
    }

    [Fact]
    public void Test_from_values_function_is_null()
    {
        var dense = new double[16];
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        // Loaded/factory-built TTs cannot do RunCompletion (Function == null).
        Assert.Throws<InvalidOperationException>(() => tt.RunCompletion());
    }

    [Fact]
    public void Test_from_values_method_is_svd()
    {
        var dense = new double[16];
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        Assert.Equal("svd", tt.Method);
    }
}

public class TtFactoryValidationTests
{
    [Fact]
    public void Test_constructor_validates_null_function()
    {
        var ex = Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevTT(
                null!,
                1,
                new[] { new[] { -1.0, 1.0 } },
                new[] { 5 }));
        Assert.Contains("function", ex.Message);
    }

    [Fact]
    public void Test_constructor_validates_malformed_grid_arguments()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevTT(
                x => x[0],
                1,
                null!,
                new[] { 5 }));

        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevTT(
                x => x[0],
                1,
                new[] { new[] { -1.0, 1.0 } },
                null!));

        var nullEntry = Assert.Throws<ArgumentException>(() =>
            new ChebyshevTT(
                x => x[0],
                1,
                new double[][] { null! },
                new[] { 5 }));
        Assert.Contains("domain[0]", nullEntry.Message);

        var badBounds = Assert.Throws<ArgumentException>(() =>
            new ChebyshevTT(
                x => x[0],
                1,
                new[] { new[] { 1.0, 1.0 } },
                new[] { 5 }));
        Assert.Contains("domain[0]", badBounds.Message);

        var badCount = Assert.Throws<ArgumentException>(() =>
            new ChebyshevTT(
                x => x[0],
                1,
                new[] { new[] { -1.0, 1.0 } },
                new[] { 0 }));
        Assert.Contains("nNodes[0]", badCount.Message);
    }

    [Fact]
    public void Test_constructor_clones_grid_arguments()
    {
        double[][] domain = { new[] { -1.0, 1.0 } };
        int[] nNodes = { 5 };

        var tt = new ChebyshevTT(x => x[0], 1, domain, nNodes);

        domain[0][0] = 123.0;
        nNodes[0] = 99;

        Assert.Equal(-1.0, tt.Domain[0][0]);
        Assert.Equal(5, tt.NNodes[0]);
    }

    [Fact]
    public void Test_nodes_validates_domain_length()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(2, new[] { new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
        Assert.Contains("domain", ex.Message);
    }

    [Fact]
    public void Test_nodes_validates_nNodes_length()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5 }));
        Assert.Contains("nNodes", ex.Message);
    }

    [Fact]
    public void Test_nodes_validates_malformed_grid_arguments()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevTT.Nodes(1, null!, new[] { 5 }));

        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevTT.Nodes(1, new[] { new[] { -1.0, 1.0 } }, null!));

        var nullEntry = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(1, new double[][] { null! }, new[] { 5 }));
        Assert.Contains("domain[0]", nullEntry.Message);

        var malformedEntry = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(1, new[] { new[] { -1.0 } }, new[] { 5 }));
        Assert.Contains("domain[0]", malformedEntry.Message);

        var nonFiniteBounds = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(1, new[] { new[] { double.NaN, 1.0 } }, new[] { 5 }));
        Assert.Contains("domain[0]", nonFiniteBounds.Message);

        var reversedBounds = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(1, new[] { new[] { 2.0, -2.0 } }, new[] { 5 }));
        Assert.Contains("domain[0]", reversedBounds.Message);

        var badCount = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.Nodes(1, new[] { new[] { -1.0, 1.0 } }, new[] { -1 }));
        Assert.Contains("nNodes[0]", badCount.Message);
    }

    [Fact]
    public void Test_from_values_validates_domain_length()
    {
        var values = new double[25];
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(values, 2, new[] { new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
        Assert.Contains("domain", ex.Message);
    }

    [Fact]
    public void Test_from_values_validates_nNodes_length()
    {
        var values = new double[5];
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(values, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5 }));
        Assert.Contains("nNodes", ex.Message);
    }

    [Fact]
    public void Test_from_values_validates_malformed_grid_arguments()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevTT.FromValues(null!, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 }));

        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevTT.FromValues(new double[5], 1, null!, new[] { 5 }));

        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevTT.FromValues(new double[5], 1, new[] { new[] { -1.0, 1.0 } }, null!));

        var nullEntry = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(new double[5], 1, new double[][] { null! }, new[] { 5 }));
        Assert.Contains("domain[0]", nullEntry.Message);

        var badBounds = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(new double[5], 1, new[] { new[] { double.NegativeInfinity, 1.0 } }, new[] { 5 }));
        Assert.Contains("domain[0]", badBounds.Message);

        var badCount = Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(Array.Empty<double>(), 1, new[] { new[] { -1.0, 1.0 } }, new[] { 0 }));
        Assert.Contains("nNodes[0]", badCount.Message);
    }
}
