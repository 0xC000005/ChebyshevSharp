using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// Test data
// ======================================================================

file static class ExtrudeSliceTestData
{
    public static readonly double[] TestPoints1D = { -0.7, -0.3, 0.0, 0.4, 0.8 };

    public static readonly double[][] TestPoints2D =
    {
        new[] { 0.5, 0.3 },
        new[] { -0.7, 0.8 },
        new[] { 0.0, 0.0 },
        new[] { 0.9, -0.9 },
        new[] { -0.2, 0.6 },
    };
}

// ======================================================================
// TestApproxExtrude
// ======================================================================

public class TestApproxExtrude
{
    private static ChebyshevApproximation Cheb1D => TestFixtures.ExtrudeCheb1D;
    private static ChebyshevApproximation Cheb2D => TestFixtures.ExtrudeCheb2D;

    [Fact]
    public void Extrude_1d_to_2d_append()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { -2.0, 2.0 }, 7));
        Assert.Equal(2, ct2.NumDimensions);
        Assert.Equal(new[] { 11, 7 }, ct2.NNodes);
    }

    [Fact]
    public void Extrude_1d_to_2d_prepend()
    {
        var ct2 = Cheb1D.Extrude((0, new[] { -2.0, 2.0 }, 7));
        Assert.Equal(2, ct2.NumDimensions);
        Assert.Equal(new[] { 7, 11 }, ct2.NNodes);
    }

    [Fact]
    public void Extrude_2d_to_3d_middle()
    {
        var ct3 = Cheb2D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(3, ct3.NumDimensions);
        Assert.Equal(new[] { 11, 9, 11 }, ct3.NNodes);
    }

    [Fact]
    public void Extrude_2d_to_3d_end()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(3, ct3.NumDimensions);
        Assert.Equal(new[] { 11, 11, 9 }, ct3.NNodes);
    }

    [Fact]
    public void Extrude_multi_dim()
    {
        var ct3 = Cheb1D.Extrude(
            (0, new[] { -2.0, 2.0 }, 7),
            (2, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(3, ct3.NumDimensions);
        Assert.Equal(new[] { 7, 11, 9 }, ct3.NNodes);
    }

    [Fact]
    public void Extrude_value_preserved()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        foreach (var p in ExtrudeSliceTestData.TestPoints2D)
        {
            double orig = Cheb2D.VectorizedEval(p, new[] { 0, 0 });
            foreach (double newCoord in new[] { 0.0, 1.5, 3.7, 5.0 })
            {
                double extruded = ct3.VectorizedEval(
                    new[] { p[0], p[1], newCoord }, new[] { 0, 0, 0 });
                Assert.True(Math.Abs(extruded - orig) < 1e-12,
                    $"Value mismatch at [{p[0]},{p[1]}]+[{newCoord}]: {extruded} vs {orig}");
            }
        }
    }

    [Fact]
    public void Extrude_any_new_coord()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { -10.0, 10.0 }, 11));
        double x = 0.5;
        double refVal = Cheb1D.VectorizedEval(new[] { x }, new[] { 0 });
        foreach (double y in new[] { -10.0, -3.3, 0.0, 5.5, 10.0 })
        {
            double val = ct2.VectorizedEval(new[] { x, y }, new[] { 0, 0 });
            Assert.True(Math.Abs(val - refVal) < 1e-12);
        }
    }

    [Fact]
    public void Extrude_derivative_original_preserved()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        foreach (var p in ExtrudeSliceTestData.TestPoints2D)
        {
            double orig = Cheb2D.VectorizedEval(p, new[] { 1, 0 });
            double ext = ct3.VectorizedEval(
                new[] { p[0], p[1], 2.5 }, new[] { 1, 0, 0 });
            Assert.True(Math.Abs(ext - orig) < 1e-10,
                $"Derivative mismatch at [{p[0]},{p[1]}]: {ext} vs {orig}");
        }
    }

    [Fact]
    public void Extrude_derivative_new_dim_zero()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        foreach (var p in ExtrudeSliceTestData.TestPoints2D)
        {
            double deriv = ct3.VectorizedEval(
                new[] { p[0], p[1], 2.5 }, new[] { 0, 0, 1 });
            Assert.True(Math.Abs(deriv) < 1e-10,
                $"Derivative along new dim not zero at [{p[0]},{p[1]}]: {deriv}");
        }
    }

    [Fact]
    public void Extrude_metadata()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { 0.0, 1.0 }, 5));
        Assert.Null(ct2.Function);
        Assert.Equal(0.0, ct2.BuildTime);
        Assert.Equal(0, ct2.NEvaluations);
    }

    [Fact]
    public void Extrude_domain_updated()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(2, ct2.Domain.Length);
        Assert.Equal(new[] { -1.0, 1.0 }, ct2.Domain[0]);
        Assert.Equal(new[] { 0.0, 5.0 }, ct2.Domain[1]);
        Assert.Equal(new[] { 11, 9 }, ct2.NNodes);
    }

    [Fact]
    public void Extrude_error_not_built()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var ex = Assert.Throws<InvalidOperationException>(
            () => cheb.Extrude((1, new[] { 0.0, 1.0 }, 5)));
        Assert.Contains("build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Extrude_error_bad_dim_index()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Cheb1D.Extrude((5, new[] { 0.0, 1.0 }, 5)));
        Assert.Contains("out of range", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Extrude_error_duplicate_dim()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Cheb1D.Extrude(
                (0, new[] { 0.0, 1.0 }, 5),
                (0, new[] { 0.0, 1.0 }, 5)));
        Assert.Contains("uplicate", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Extrude_serialization()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        string path = Path.GetTempFileName();
        try
        {
            ct2.Save(path);
            var loaded = ChebyshevApproximation.Load(path);
            double[] p = { 0.5, 2.5 };
            double orig = ct2.VectorizedEval(p, new[] { 0, 0 });
            double back = loaded.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(orig - back) < 1e-15);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// TestApproxSlice
// ======================================================================

public class TestApproxSlice
{
    private static ChebyshevApproximation Cheb2D => TestFixtures.ExtrudeCheb2D;

    [Fact]
    public void Slice_2d_to_1d()
    {
        double yFixed = 0.3;
        var ct1 = Cheb2D.Slice((1, yFixed));
        Assert.Equal(1, ct1.NumDimensions);
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, yFixed }, new[] { 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10,
                $"Slice 2D->1D failed at x={x}: {sliced} vs {exact}");
        }
    }

    [Fact]
    public void Slice_3d_to_2d()
    {
        var ct3 = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 });
        ct3.Build(verbose: false);

        double yFixed = 0.5;
        var ct2 = ct3.Slice((1, yFixed));
        Assert.Equal(2, ct2.NumDimensions);

        double[][] testPts = { new[] { 0.3, -0.7 }, new[] { -0.5, 0.8 }, new[] { 0.0, 0.0 } };
        foreach (var p in testPts)
        {
            double exact = ct3.VectorizedEval(new[] { p[0], yFixed, p[1] }, new[] { 0, 0, 0 });
            double sliced = ct2.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10);
        }
    }

    [Fact]
    public void Slice_exact_node()
    {
        double nodeVal = Cheb2D.NodeArrays[1][3];
        var ct1 = Cheb2D.Slice((1, nodeVal));
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, nodeVal }, new[] { 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-13);
        }
    }

    [Fact]
    public void Slice_between_nodes()
    {
        double yFixed = 0.1234;
        var ct1 = Cheb2D.Slice((1, yFixed));
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, yFixed }, new[] { 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10);
        }
    }

    [Fact]
    public void Slice_boundary()
    {
        var ct1 = Cheb2D.Slice((1, -1.0));
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, -1.0 }, new[] { 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10);
        }
    }

    [Fact]
    public void Slice_multi_dim()
    {
        var ct3 = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 });
        ct3.Build(verbose: false);

        double yFixed = 0.3;
        double zFixed = -0.5;
        var ct1 = ct3.Slice((1, yFixed), (2, zFixed));
        Assert.Equal(1, ct1.NumDimensions);

        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = ct3.VectorizedEval(
                new[] { x, yFixed, zFixed }, new[] { 0, 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10);
        }
    }

    [Fact]
    public void Slice_derivative_preserved()
    {
        double yFixed = 0.4;
        var ct1 = Cheb2D.Slice((1, yFixed));
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, yFixed }, new[] { 1, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 1 });
            Assert.True(Math.Abs(sliced - exact) < 1e-8);
        }
    }

    [Fact]
    public void Slice_metadata()
    {
        var ct1 = Cheb2D.Slice((1, 0.5));
        Assert.Null(ct1.Function);
        Assert.Equal(0.0, ct1.BuildTime);
    }

    [Fact]
    public void Slice_domain_updated()
    {
        var ct1 = Cheb2D.Slice((1, 0.5));
        Assert.Single(ct1.Domain);
        Assert.Equal(new[] { -1.0, 1.0 }, ct1.Domain[0]);
        Assert.Equal(new[] { 11 }, ct1.NNodes);
    }

    [Fact]
    public void Slice_error_not_built()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 });
        var ex = Assert.Throws<InvalidOperationException>(
            () => cheb.Slice((1, 0.5)));
        Assert.Contains("build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Slice_error_bad_dim_index()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Cheb2D.Slice((5, 0.5)));
        Assert.Contains("out of range", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Slice_error_out_of_domain()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Cheb2D.Slice((0, 5.0)));
        Assert.Contains("outside", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Slice_error_all_dims()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Cheb2D.Slice((0, 0.0), (1, 0.0)));
        Assert.Contains("annot slice all", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Slice_serialization()
    {
        var ct1 = Cheb2D.Slice((1, 0.5));
        string path = Path.GetTempFileName();
        try
        {
            ct1.Save(path);
            var loaded = ChebyshevApproximation.Load(path);
            double x = 0.3;
            double orig = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            double back = loaded.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(orig - back) < 1e-15);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// TestSliceExtrudeInverse
// ======================================================================

public class TestSliceExtrudeInverse
{
    [Fact]
    public void Extrude_then_slice_identity()
    {
        var cheb2D = TestFixtures.ExtrudeCheb2D;
        var ct3 = cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        var ct2Back = ct3.Slice((2, 2.5));

        foreach (var p in ExtrudeSliceTestData.TestPoints2D)
        {
            double orig = cheb2D.VectorizedEval(p, new[] { 0, 0 });
            double back = ct2Back.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(back - orig) < 1e-12,
                $"Round-trip failed at [{p[0]},{p[1]}]: {back} vs {orig}");
        }
    }

    [Fact]
    public void Round_trip_multiple_points()
    {
        var cheb1D = TestFixtures.ExtrudeCheb1D;
        var ct2 = cheb1D.Extrude((0, new[] { -3.0, 3.0 }, 7));
        var ct1Back = ct2.Slice((0, 1.0));

        // Deterministic pseudo-random points with seed 42, uniform(-1, 1)
        var rng = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            double x = rng.NextDouble() * 2.0 - 1.0; // uniform(-1, 1)
            double orig = cheb1D.VectorizedEval(new[] { x }, new[] { 0 });
            double back = ct1Back.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(back - orig) < 1e-12);
        }
    }
}

// ======================================================================
// TestExtrudeAlgebraUseCase
// ======================================================================

public class TestExtrudeAlgebraUseCase
{
    // Build Trade A(spot,rate) and Trade B(spot,vol), extrude to common 3D, add.
    private static readonly Lazy<(ChebyshevApproximation ctA, ChebyshevApproximation ctB, ChebyshevApproximation portfolio)> _portfolio = new(() =>
    {
        var ctA = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + 0.5 * Math.Cos(x[1]),
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.01, 0.08 } },
            new[] { 11, 11 });
        ctA.Build(verbose: false);

        var ctB = new ChebyshevApproximation(
            (x, _) => Math.Cos(x[0]) + 0.3 * Math.Sin(x[1]),
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.15, 0.35 } },
            new[] { 11, 11 });
        ctB.Build(verbose: false);

        // Extrude to common 3D: (spot, rate, vol)
        var ctA3D = ctA.Extrude((2, new[] { 0.15, 0.35 }, 11)); // add vol dim
        var ctB3D = ctB.Extrude((1, new[] { 0.01, 0.08 }, 11)); // add rate dim

        var portfolio = ctA3D + ctB3D;
        return (ctA, ctB, portfolio);
    });

    [Fact]
    public void Extrude_add_different_variables()
    {
        var (_, _, portfolio) = _portfolio.Value;
        Assert.Equal(3, portfolio.NumDimensions);
        Assert.Equal(new[] { 11, 11, 11 }, portfolio.NNodes);
    }

    [Fact]
    public void Portfolio_value_correct()
    {
        var (ctA, ctB, portfolio) = _portfolio.Value;
        double[][] testPts =
        {
            new[] { 0.5, 0.03, 0.25 },
            new[] { -0.3, 0.06, 0.2 },
            new[] { 0.0, 0.04, 0.3 },
            new[] { 0.8, 0.02, 0.18 },
        };
        foreach (var pt in testPts)
        {
            double spot = pt[0], rate = pt[1], vol = pt[2];
            double valA = ctA.VectorizedEval(new[] { spot, rate }, new[] { 0, 0 });
            double valB = ctB.VectorizedEval(new[] { spot, vol }, new[] { 0, 0 });
            double portVal = portfolio.VectorizedEval(pt, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(portVal - (valA + valB)) < 1e-10,
                $"Portfolio mismatch at [{spot},{rate},{vol}]: {portVal} vs {valA + valB}");
        }
    }

    [Fact]
    public void Portfolio_greeks()
    {
        var (ctA, ctB, portfolio) = _portfolio.Value;
        double[] pt = { 0.5, 0.04, 0.25 };
        double spot = pt[0], rate = pt[1], vol = pt[2];

        // d/dspot of portfolio = d/dspot(trade_a) + d/dspot(trade_b)
        double deltaA = ctA.VectorizedEval(new[] { spot, rate }, new[] { 1, 0 });
        double deltaB = ctB.VectorizedEval(new[] { spot, vol }, new[] { 1, 0 });
        double deltaPort = portfolio.VectorizedEval(pt, new[] { 1, 0, 0 });
        Assert.True(Math.Abs(deltaPort - (deltaA + deltaB)) < 1e-8);

        // d/drate of portfolio = d/drate(trade_a) + 0
        double rhoA = ctA.VectorizedEval(new[] { spot, rate }, new[] { 0, 1 });
        double rhoPort = portfolio.VectorizedEval(pt, new[] { 0, 1, 0 });
        Assert.True(Math.Abs(rhoPort - rhoA) < 1e-8);

        // d/dvol of portfolio = 0 + d/dvol(trade_b)
        double vegaB = ctB.VectorizedEval(new[] { spot, vol }, new[] { 0, 1 });
        double vegaPort = portfolio.VectorizedEval(pt, new[] { 0, 0, 1 });
        Assert.True(Math.Abs(vegaPort - vegaB) < 1e-8);
    }
}

// ======================================================================
// TestEdgeCases (CA-only tests)
// ======================================================================

public class TestExtrudeSliceEdgeCases
{
    private static ChebyshevApproximation Cheb1D => TestFixtures.ExtrudeCheb1D;
    private static ChebyshevApproximation Cheb2D => TestFixtures.ExtrudeCheb2D;

    [Fact]
    public void Extrude_min_nodes()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { 0.0, 10.0 }, 2));
        Assert.Equal(2, ct2.NumDimensions);
        Assert.Equal(new[] { 11, 2 }, ct2.NNodes);

        double refVal = Cheb1D.VectorizedEval(new[] { 0.5 }, new[] { 0 });
        double val = ct2.VectorizedEval(new[] { 0.5, 5.0 }, new[] { 0, 0 });
        Assert.True(Math.Abs(val - refVal) < 1e-12);
    }

    [Fact]
    public void Slice_boundary_right()
    {
        var ct1 = Cheb2D.Slice((1, 1.0));
        foreach (double x in ExtrudeSliceTestData.TestPoints1D)
        {
            double exact = Cheb2D.VectorizedEval(new[] { x, 1.0 }, new[] { 0, 0 });
            double sliced = ct1.VectorizedEval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10);
        }
    }

    [Fact]
    public void Error_estimate_extruded()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        double err = ct3.ErrorEstimate();
        Assert.True(err >= 0.0);
        double origErr = Cheb2D.ErrorEstimate();
        Assert.True(err < origErr * 100); // generous bound
    }

    [Fact]
    public void Error_estimate_sliced()
    {
        var ct1 = Cheb2D.Slice((1, 0.5));
        double err = ct1.ErrorEstimate();
        Assert.True(err >= 0.0);
    }

    [Fact]
    public void Vectorized_eval_batch_extruded()
    {
        var ct2 = Cheb1D.Extrude((1, new[] { -2.0, 2.0 }, 7));
        double[][] pts =
        {
            new[] { 0.5, 0.0 },
            new[] { -0.3, 1.0 },
            new[] { 0.8, -1.5 },
        };
        double[] results = ct2.VectorizedEvalBatch(pts, new[] { 0, 0 });
        double refVal = Cheb1D.VectorizedEval(new[] { 0.5 }, new[] { 0 });
        Assert.True(Math.Abs(results[0] - refVal) < 1e-12);
    }

    [Fact]
    public void Vectorized_eval_batch_sliced()
    {
        var ct1 = Cheb2D.Slice((1, 0.3));
        double[][] pts =
        {
            new[] { 0.5 },
            new[] { -0.3 },
            new[] { 0.8 },
        };
        double[] results = ct1.VectorizedEvalBatch(pts, new[] { 0 });
        double[] xValues = { 0.5, -0.3, 0.8 };
        for (int i = 0; i < xValues.Length; i++)
        {
            double refVal = Cheb2D.VectorizedEval(new[] { xValues[i], 0.3 }, new[] { 0, 0 });
            Assert.True(Math.Abs(results[i] - refVal) < 1e-10);
        }
    }

    [Fact]
    public void Vectorized_eval_multi_extruded()
    {
        var ct3 = Cheb2D.Extrude((2, new[] { 0.0, 5.0 }, 9));
        double[] pt = { 0.5, -0.3, 2.5 };
        int[][] derivOrders =
        {
            new[] { 0, 0, 0 },
            new[] { 1, 0, 0 },
            new[] { 0, 0, 1 },
        };
        double[] results = ct3.VectorizedEvalMulti(pt, derivOrders);
        Assert.Equal(3, results.Length);
        // d/d(new_dim) should be ~0
        Assert.True(Math.Abs(results[2]) < 1e-10);
    }

    [Fact]
    public void Vectorized_eval_multi_sliced()
    {
        var ct1 = Cheb2D.Slice((1, 0.4));
        int[][] derivOrders =
        {
            new[] { 0 },
            new[] { 1 },
        };
        double[] results = ct1.VectorizedEvalMulti(new[] { 0.5 }, derivOrders);
        Assert.Equal(2, results.Length);
        double refVal = Cheb2D.VectorizedEval(new[] { 0.5, 0.4 }, new[] { 0, 0 });
        Assert.True(Math.Abs(results[0] - refVal) < 1e-10);
    }

    [Fact]
    public void Spline_slice_at_knot()
    {
        // Slice a 2D spline (extruded from 1D |x|) at exact knot value.
        var spAbs1D = TestFixtures.SplineAbs1D;
        var sp2 = spAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        // Slice dim 0 at the knot x=0
        var sp1 = sp2.Slice((0, 0.0));
        Assert.Equal(1, sp1.NumDimensions);
        // |0| = 0, so value should be ~0 everywhere in new dim
        foreach (double y in new[] { 0.0, 2.5, 5.0 })
        {
            double val = sp1.Eval(new[] { y }, new[] { 0 });
            Assert.True(Math.Abs(val) < 1e-10);
        }
    }
}

// ======================================================================
// TestSplineExtrude (ported from Python TestSplineExtrude)
// ======================================================================

public class TestSplineExtrude
{
    private static ChebyshevSpline SpAbs1D => TestFixtures.SplineAbs1D;

    [Fact]
    public void Spline_extrude_basic()
    {
        // Extrude 1D spline -> 2D.
        var sp2 = SpAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(2, sp2.NumDimensions);
        Assert.Equal(new[] { 15, 9 }, sp2.NNodes);
    }

    [Fact]
    public void Spline_extrude_knots_preserved()
    {
        // Original knots unchanged, new dim has empty knots.
        var sp2 = SpAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(new[] { 0.0 }, sp2.Knots[0]); // original
        Assert.Empty(sp2.Knots[1]); // new dim
    }

    [Fact]
    public void Spline_extrude_pieces_count()
    {
        // Number of pieces unchanged (multiplied by 1 for new knotless dim).
        var sp2 = SpAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(SpAbs1D.NumPieces, sp2.NumPieces);
    }

    [Fact]
    public void Spline_extrude_value_preserved()
    {
        // Evaluations match original regardless of new coordinate.
        var sp2 = SpAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        foreach (double x in new[] { -0.8, -0.3, 0.0, 0.3, 0.8 })
        {
            double orig = SpAbs1D.Eval(new[] { x }, new[] { 0 });
            foreach (double y in new[] { 0.0, 2.5, 5.0 })
            {
                double ext = sp2.Eval(new[] { x, y }, new[] { 0, 0 });
                Assert.True(Math.Abs(ext - orig) < 1e-10,
                    $"Value mismatch at ({x}, {y}): {ext} vs {orig}");
            }
        }
    }

    [Fact]
    public void Spline_extrude_derivative_new_dim_zero()
    {
        // Derivative along new dim is zero.
        var sp2 = SpAbs1D.Extrude((1, new[] { 0.0, 5.0 }, 9));
        foreach (double x in new[] { 0.3, 0.8 }) // avoid knot at 0
        {
            double deriv = sp2.Eval(new[] { x, 2.5 }, new[] { 0, 1 });
            Assert.True(Math.Abs(deriv) < 1e-10);
        }
    }
}

// ======================================================================
// TestSplineSlice (ported from Python TestSplineSlice)
// ======================================================================

public class TestSplineSlice
{
    private static ChebyshevSpline SpBs2D => TestFixtures.SplineBs2D;

    [Fact]
    public void Spline_slice_basic()
    {
        // Slice 2D spline -> 1D.
        double sFixed = 95.0;
        var sp1 = SpBs2D.Slice((0, sFixed));
        Assert.Equal(1, sp1.NumDimensions);
    }

    [Fact]
    public void Spline_slice_piece_reduction()
    {
        // After slicing, we keep only pieces from the interval containing the value.
        // spline_bs_2d has knots [[100.0], []], domain [[80,120],[0.25,1.0]]
        // Slicing dim 0 at S=95 -> picks the left piece (80-100)
        var sp1 = SpBs2D.Slice((0, 95.0));
        // Original had 2 pieces along dim 0 (knot at 100), 1 along dim 1
        // After slicing dim 0, should have 1 piece
        Assert.Equal(1, sp1.NumPieces);
    }

    [Fact]
    public void Spline_slice_value_matches()
    {
        // Evaluation matches direct 2D eval at sliced point.
        double sFixed = 95.0;
        var sp1 = SpBs2D.Slice((0, sFixed));
        foreach (double t in new[] { 0.3, 0.5, 0.8 })
        {
            double exact = SpBs2D.Eval(new[] { sFixed, t }, new[] { 0, 0 });
            double sliced = sp1.Eval(new[] { t }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-10,
                $"Spline slice mismatch at T={t}: {sliced} vs {exact}");
        }
    }

    [Fact]
    public void Spline_slice_knots_preserved()
    {
        // Remaining knots unchanged after slicing.
        var sp1 = SpBs2D.Slice((0, 95.0));
        Assert.Single(sp1.Knots);
        Assert.Empty(sp1.Knots[0]); // dim 1's knots were []
    }

    [Fact]
    public void Spline_slice_serialization()
    {
        // Save/load round-trip works.
        var sp1 = SpBs2D.Slice((0, 95.0));
        string path = Path.Combine(Path.GetTempPath(), $"spline_slice_test_{Guid.NewGuid()}.json");
        try
        {
            sp1.Save(path);
            var loaded = ChebyshevSpline.Load(path);
            double valOrig = sp1.Eval(new[] { 0.5 }, new[] { 0 });
            double valLoaded = loaded.Eval(new[] { 0.5 }, new[] { 0 });
            Assert.True(Math.Abs(valOrig - valLoaded) < 1e-15);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

// ======================================================================
// C#-Specific: Spline Extrude/Slice Edge Cases
// ======================================================================

/// <summary>
/// C#-specific extrude/slice edge case tests for ChebyshevSpline.
/// </summary>
public class TestSplineExtrudeSliceCSharpEdgeCases
{
    [Fact]
    public void Test_extrude_spline_preserves_knots()
    {
        // Extruding a 1D spline with knot at 0 should:
        // - Preserve the existing knot [0.0] for dim 0
        // - Add an empty knot list for the new dim
        var sp1D = TestFixtures.SplineAbs1D;
        var sp2D = sp1D.Extrude((1, new[] { 0.0, 5.0 }, 9));

        Assert.Equal(2, sp2D.Knots.Length);
        Assert.Equal(new[] { 0.0 }, sp2D.Knots[0]); // original knots preserved
        Assert.Empty(sp2D.Knots[1]); // new dim has no knots
    }

    [Fact]
    public void Test_slice_spline_at_knot_value()
    {
        // Slicing a 2D spline (with knots along dim 0) at a knot value
        // in that dimension should succeed and remove the knot dimension.
        // spline_bs_2d has knots [[100.0], []], domain [[80,120],[0.25,1.0]]
        var sp2D = TestFixtures.SplineBs2D;

        // Slice dim 0 at the knot value S=100
        var sp1D = sp2D.Slice((0, 100.0));

        Assert.Equal(1, sp1D.NumDimensions);
        // The remaining dim's knots (originally dim 1) should be empty
        Assert.Single(sp1D.Knots);
        Assert.Empty(sp1D.Knots[0]);

        // At S=100, max(S-100, 0)*exp(-rT) = 0 for all T
        foreach (double t in new[] { 0.3, 0.5, 0.8 })
        {
            double val = sp1D.Eval(new[] { t }, new[] { 0 });
            Assert.True(Math.Abs(val) < 1e-8,
                $"Expected ~0 at T={t} (S=100 is at the kink), got {val}");
        }
    }
}

// ======================================================================
// TestSliderExtrude (ported from Python TestSliderExtrude)
// ======================================================================

public class TestSliderExtrude
{
    private static ChebyshevSlider SlF => TestFixtures.AlgebraSliderF;

    [Fact]
    public void Slider_extrude_basic()
    {
        // Extrude 3D slider -> 4D, verify partition updated.
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(4, sl4.NumDimensions);
        Assert.Equal(4, sl4.Partition.Length); // original 3 groups + 1 new
    }

    [Fact]
    public void Slider_extrude_pivot_extended()
    {
        // Pivot point has new entry (midpoint of new domain).
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        Assert.Equal(4, sl4.PivotPoint.Length);
        Assert.True(Math.Abs(sl4.PivotPoint[3] - 2.5) < 1e-14); // midpoint of [0, 5]
    }

    [Fact]
    public void Slider_extrude_value_preserved()
    {
        // Evaluations match original regardless of new coordinate.
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        var pts = new[] { new[] { 0.5, 0.3, 0.7 }, new[] { -0.5, 0.8, -0.2 }, new[] { 0.0, 0.0, 0.0 } };
        foreach (var p in pts)
        {
            double orig = SlF.Eval(p, new[] { 0, 0, 0 });
            foreach (double newCoord in new[] { 0.0, 2.5, 5.0 })
            {
                double ext = sl4.Eval(new[] { p[0], p[1], p[2], newCoord }, new[] { 0, 0, 0, 0 });
                Assert.True(Math.Abs(ext - orig) < 1e-10,
                    $"Slider extrude mismatch at [{string.Join(", ", p)}]+[{newCoord}]");
            }
        }
    }

    [Fact]
    public void Slider_extrude_derivative_new_dim_zero()
    {
        // Derivative along new dim is zero.
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        double[] p = { 0.5, 0.3, 0.7, 2.5 };
        double deriv = sl4.Eval(p, new[] { 0, 0, 0, 1 });
        Assert.True(Math.Abs(deriv) < 1e-10);
    }

    [Fact]
    public void Slider_extrude_multigroup()
    {
        // Extrude slider with multi-dim group partition [[0,1],[2]].
        double F(double[] x, object? _) => Math.Sin(x[0] + x[1]) + Math.Cos(x[2]);
        var sl = new ChebyshevSlider(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sl.Build(verbose: false);
        var sl4 = sl.Extrude((3, new[] { 0.0, 5.0 }, 7));
        Assert.Equal(4, sl4.NumDimensions);
        Assert.Equal(3, sl4.Partition.Length); // [[0,1],[2],[3]]
        // Value preserved
        double refVal = sl.Eval(new[] { 0.3, 0.5, -0.2 }, new[] { 0, 0, 0 });
        double val = sl4.Eval(new[] { 0.3, 0.5, -0.2, 2.5 }, new[] { 0, 0, 0, 0 });
        Assert.True(Math.Abs(val - refVal) < 1e-8);
    }
}

// ======================================================================
// TestSliderSlice (ported from Python TestSliderSlice)
// ======================================================================

public class TestSliderSlice
{
    private static ChebyshevSlider SlF => TestFixtures.AlgebraSliderF;

    [Fact]
    public void Slider_slice_single_dim_group()
    {
        // Slice removes a single-dim group, pivot_value updated.
        // algebra_slider_f: partition [[0],[1],[2]], 3D sin(x)+sin(y)+sin(z)
        double xFixed = 0.5;
        var sl2 = SlF.Slice((0, xFixed));
        Assert.Equal(2, sl2.NumDimensions);
        Assert.Equal(2, sl2.Partition.Length);
    }

    [Fact]
    public void Slider_slice_multi_dim_group()
    {
        // Slice within a multi-dim group.
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]) + Math.Sin(x[2]);
        var sl = new ChebyshevSlider(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sl.Build(verbose: false);
        // Slice dim 0 from the 2-dim group [0,1]
        var sl2 = sl.Slice((0, 0.3));
        Assert.Equal(2, sl2.NumDimensions);
        // Group that was [0,1] should now be just [0] (remapped from original dim 1)
        Assert.Equal(2, sl2.Partition.Length);
    }

    [Fact]
    public void Slider_slice_value_matches()
    {
        // Evaluation matches original at sliced point.
        double xFixed = 0.5;
        var sl2 = SlF.Slice((0, xFixed));
        var yzPts = new[] { new[] { 0.3, 0.7 }, new[] { -0.5, 0.2 }, new[] { 0.8, -0.8 } };
        foreach (var yz in yzPts)
        {
            double orig = SlF.Eval(new[] { xFixed, yz[0], yz[1] }, new[] { 0, 0, 0 });
            double sliced = sl2.Eval(yz, new[] { 0, 0 });
            Assert.True(Math.Abs(sliced - orig) < 1e-8,
                $"Slider slice mismatch at yz=[{yz[0]}, {yz[1]}]: {sliced} vs {orig}");
        }
    }

    [Fact]
    public void Slider_slice_partition_remap()
    {
        // After slicing dim 0, remaining partition indices remapped correctly.
        var sl2 = SlF.Slice((0, 0.5));
        // Original partition was [[0],[1],[2]], after removing dim 0:
        // dim 1->0, dim 2->1, so partition should be [[0],[1]]
        Assert.Equal(2, sl2.Partition.Length);
        Assert.Equal(new[] { 0 }, sl2.Partition[0]);
        Assert.Equal(new[] { 1 }, sl2.Partition[1]);
    }

    [Fact]
    public void Slider_slice_to_single_group()
    {
        // Slice slider from 3 groups to 1 group (2 consecutive slices).
        // algebra_slider_f: partition [[0],[1],[2]], sin(x)+sin(y)+sin(z)
        var sl2 = SlF.Slice((0, 0.5));
        var sl1 = sl2.Slice((0, 0.3)); // now dim 0 was original dim 1
        Assert.Equal(1, sl1.NumDimensions);
        Assert.Single(sl1.Partition);
        // Verify: sin(0.5) + sin(0.3) + sin(z)
        foreach (double z in new[] { -0.5, 0.0, 0.7 })
        {
            double exact = SlF.Eval(new[] { 0.5, 0.3, z }, new[] { 0, 0, 0 });
            double sliced = sl1.Eval(new[] { z }, new[] { 0 });
            Assert.True(Math.Abs(sliced - exact) < 1e-8,
                $"Slider double-slice at z={z}: {sliced} vs {exact}");
        }
    }
}

// ======================================================================
// C#-Specific: Slider Extrude/Slice Edge Cases
// ======================================================================

/// <summary>
/// C#-specific extrude/slice edge case tests for ChebyshevSlider.
/// Covers boundary slicing, extrude-then-slice roundtrips, extrude at dim 0,
/// and serialization of extruded/sliced sliders.
/// </summary>
public class TestSliderExtrudeSliceCSharpEdgeCases
{
    private static ChebyshevSlider SlF => TestFixtures.AlgebraSliderF;

    [Fact]
    public void Test_slider_slice_at_domain_lower_bound()
    {
        // Slice at the lower bound of the domain should work.
        // SlF domain is [[-1,1],[-1,1],[-1,1]]
        var sl2 = SlF.Slice((0, -1.0));
        Assert.Equal(2, sl2.NumDimensions);
        double[] pt = { 0.3, 0.7 };
        double expected = SlF.Eval(new[] { -1.0, 0.3, 0.7 }, new[] { 0, 0, 0 });
        double actual = sl2.Eval(pt, new[] { 0, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-8);
    }

    [Fact]
    public void Test_slider_slice_at_domain_upper_bound()
    {
        // Slice at the upper bound of the domain should work.
        var sl2 = SlF.Slice((2, 1.0));
        Assert.Equal(2, sl2.NumDimensions);
        double[] pt = { 0.3, 0.7 };
        double expected = SlF.Eval(new[] { 0.3, 0.7, 1.0 }, new[] { 0, 0, 0 });
        double actual = sl2.Eval(pt, new[] { 0, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-8);
    }

    [Fact]
    public void Test_slider_extrude_then_slice_roundtrip()
    {
        // Extrude at dim 3 then slice it back should yield same values.
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        var sl3 = sl4.Slice((3, 2.5));
        Assert.Equal(3, sl3.NumDimensions);

        var pts = new[] { new[] { 0.5, 0.3, 0.7 }, new[] { -0.5, 0.8, -0.2 } };
        foreach (var p in pts)
        {
            double orig = SlF.Eval(p, new[] { 0, 0, 0 });
            double rt = sl3.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(rt - orig) < 1e-8,
                $"Extrude+slice roundtrip failed at [{string.Join(", ", p)}]");
        }
    }

    [Fact]
    public void Test_slider_extrude_at_dim_zero()
    {
        // Extrude at dim 0 (prepend). Should shift all existing partition indices.
        var sl4 = SlF.Extrude((0, new[] { -2.0, 2.0 }, 7));
        Assert.Equal(4, sl4.NumDimensions);
        Assert.Equal(7, sl4.NNodes[0]);
        // Original nodes shifted to indices 1, 2, 3
        Assert.Equal(SlF.NNodes[0], sl4.NNodes[1]);
        Assert.Equal(SlF.NNodes[1], sl4.NNodes[2]);
        Assert.Equal(SlF.NNodes[2], sl4.NNodes[3]);

        // Value preserved: new dim 0 doesn't contribute
        double orig = SlF.Eval(new[] { 0.5, 0.3, 0.7 }, new[] { 0, 0, 0 });
        double ext = sl4.Eval(new[] { 1.5, 0.5, 0.3, 0.7 }, new[] { 0, 0, 0, 0 });
        Assert.True(Math.Abs(ext - orig) < 1e-10);
    }

    [Fact]
    public void Test_slider_extrude_middle_dim()
    {
        // Extrude at dim 1 (middle). Original dims 1,2 shift to 2,3.
        var sl4 = SlF.Extrude((1, new[] { 0.0, 10.0 }, 5));
        Assert.Equal(4, sl4.NumDimensions);
        Assert.Equal(5, sl4.NNodes[1]);

        double orig = SlF.Eval(new[] { 0.5, 0.3, 0.7 }, new[] { 0, 0, 0 });
        double ext = sl4.Eval(new[] { 0.5, 5.0, 0.3, 0.7 }, new[] { 0, 0, 0, 0 });
        Assert.True(Math.Abs(ext - orig) < 1e-10);
    }

    [Fact]
    public void Test_slider_sliced_result_serializable()
    {
        // Save/load a sliced slider should preserve values.
        var sl2 = SlF.Slice((0, 0.5));
        string path = Path.GetTempFileName();
        try
        {
            sl2.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            double[] pt = { 0.3, 0.7 };
            double orig = sl2.Eval(pt, new[] { 0, 0 });
            double rest = loaded.Eval(pt, new[] { 0, 0 });
            Assert.Equal(orig, rest, 14);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_slider_extruded_result_serializable()
    {
        // Save/load an extruded slider should preserve values.
        var sl4 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 9));
        string path = Path.GetTempFileName();
        try
        {
            sl4.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            double[] pt = { 0.5, 0.3, 0.7, 2.5 };
            double orig = sl4.Eval(pt, new[] { 0, 0, 0, 0 });
            double rest = loaded.Eval(pt, new[] { 0, 0, 0, 0 });
            Assert.Equal(orig, rest, 14);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_slider_slice_derivative_preserved()
    {
        // Derivative in remaining dims should match original after slicing.
        var sl2 = SlF.Slice((0, 0.5));
        double[] pt = { 0.3, 0.7 };
        // d/d(dim1) of sin(y) -> cos(y)
        double expected = SlF.Eval(new[] { 0.5, 0.3, 0.7 }, new[] { 0, 1, 0 });
        double actual = sl2.Eval(pt, new[] { 1, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-8,
            $"Derivative after slice: {actual} vs {expected}");
    }

    [Fact]
    public void Test_slider_multiple_extrude()
    {
        // Extrude twice to go from 3D -> 5D.
        var sl5 = SlF.Extrude((3, new[] { 0.0, 5.0 }, 7), (4, new[] { -1.0, 1.0 }, 5));
        Assert.Equal(5, sl5.NumDimensions);
        Assert.Equal(5, sl5.Partition.Length); // 3 original + 2 new singleton groups

        double orig = SlF.Eval(new[] { 0.5, 0.3, 0.7 }, new[] { 0, 0, 0 });
        double ext = sl5.Eval(new[] { 0.5, 0.3, 0.7, 2.5, 0.0 }, new[] { 0, 0, 0, 0, 0 });
        Assert.True(Math.Abs(ext - orig) < 1e-10);
    }

    [Fact]
    public void Test_slider_slice_coupled_group()
    {
        // Slice from a coupled slider (multi-dim group)
        // and verify derivative in remaining coupled dim still works.
        double F(double[] x, object? _) => x[0] * x[0] * x[1] * x[1] + x[2];
        var sl = new ChebyshevSlider(F, 3,
            new[] { new[] { -2.0, 2.0 }, new[] { -2.0, 2.0 }, new[] { -2.0, 2.0 } },
            new[] { 12, 12, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sl.Build(verbose: false);

        // Slice dim 0 at x0=1.0: remaining function is x1^2 + x2
        var sl2 = sl.Slice((0, 1.0));
        Assert.Equal(2, sl2.NumDimensions);

        // Check value
        double expected = sl.Eval(new[] { 1.0, 0.5, -1.0 }, new[] { 0, 0, 0 });
        double actual = sl2.Eval(new[] { 0.5, -1.0 }, new[] { 0, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-6,
            $"Coupled slice value: {actual} vs {expected}");
    }
}
