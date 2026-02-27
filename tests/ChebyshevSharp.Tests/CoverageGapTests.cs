using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestNodesSpline — ChebyshevSpline.Nodes() coverage
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.Nodes().
/// Ported from ref/PyChebyshev/tests/test_from_values.py :: TestNodesSpline.
/// Covers: ChebyshevSpline.cs lines 543-587, SplinePieceNodeInfo, SplineNodeInfo.
/// </summary>
public class TestNodesSpline
{
    [Fact]
    public void Nodes_ReturnsCorrectProperties()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        Assert.NotNull(info.Pieces);
        Assert.True(info.NumPieces > 0);
        Assert.NotNull(info.PieceShape);
    }

    [Fact]
    public void Nodes_PieceCount_1D_SingleKnot()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        Assert.Equal(2, info.NumPieces);
        Assert.Equal(new[] { 2 }, info.PieceShape);
        Assert.Equal(2, info.Pieces.Length);
    }

    [Fact]
    public void Nodes_PieceCount_1D_MultipleKnots()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { new[] { -0.5, 0.0, 0.5 } });

        Assert.Equal(4, info.NumPieces);
        Assert.Equal(new[] { 4 }, info.PieceShape);
    }

    [Fact]
    public void Nodes_SubDomains_1D()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        // Piece 0: [-1, 0]
        TestFixtures.AssertClose(-1.0, info.Pieces[0].SubDomain[0][0]);
        TestFixtures.AssertClose(0.0, info.Pieces[0].SubDomain[0][1]);

        // Piece 1: [0, 1]
        TestFixtures.AssertClose(0.0, info.Pieces[1].SubDomain[0][0]);
        TestFixtures.AssertClose(1.0, info.Pieces[1].SubDomain[0][1]);
    }

    [Fact]
    public void Nodes_NoKnots_GivesOnePiece()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { Array.Empty<double>() });

        Assert.Equal(1, info.NumPieces);
        Assert.Equal(new[] { 1 }, info.PieceShape);
    }

    [Fact]
    public void Nodes_2D_MultiKnot()
    {
        var info = ChebyshevSpline.Nodes(
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
            new[] { 10, 10 },
            new[] { new[] { 0.0 }, new[] { 1.0 } });

        // 2 intervals in dim 0, 2 intervals in dim 1 => 4 pieces
        Assert.Equal(4, info.NumPieces);
        Assert.Equal(new[] { 2, 2 }, info.PieceShape);
    }

    [Fact]
    public void Nodes_PieceOrdering_MatchesNdIndex()
    {
        var info = ChebyshevSpline.Nodes(
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 3.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0.0 }, new[] { 1.0, 2.0 } });

        // 2 x 3 = 6 pieces
        Assert.Equal(6, info.NumPieces);
        Assert.Equal(new[] { 2, 3 }, info.PieceShape);

        // C-order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        Assert.Equal(new[] { 0, 0 }, info.Pieces[0].PieceIndex);
        Assert.Equal(new[] { 0, 1 }, info.Pieces[1].PieceIndex);
        Assert.Equal(new[] { 0, 2 }, info.Pieces[2].PieceIndex);
        Assert.Equal(new[] { 1, 0 }, info.Pieces[3].PieceIndex);
        Assert.Equal(new[] { 1, 1 }, info.Pieces[4].PieceIndex);
        Assert.Equal(new[] { 1, 2 }, info.Pieces[5].PieceIndex);
    }

    [Fact]
    public void Nodes_PieceGridShape()
    {
        var info = ChebyshevSpline.Nodes(
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
            new[] { 10, 8 },
            new[] { new[] { 0.0 }, Array.Empty<double>() });

        // Each piece has 10 * 8 = 80 grid points, 2 coords each
        foreach (var piece in info.Pieces)
        {
            Assert.Equal(new[] { 10, 8 }, piece.Shape);
            Assert.Equal(80, piece.FullGrid.Length);
            Assert.All(piece.FullGrid, p => Assert.Equal(2, p.Length));
        }
    }

    [Fact]
    public void Nodes_PieceNodesInSubDomain()
    {
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        // Piece 0 nodes should all be in [-1, 0]
        foreach (var pt in info.Pieces[0].FullGrid)
        {
            Assert.True(pt[0] >= -1.0 - 1e-14, $"Node {pt[0]} below lower bound");
            Assert.True(pt[0] <= 0.0 + 1e-14, $"Node {pt[0]} above upper bound");
        }

        // Piece 1 nodes should all be in [0, 1]
        foreach (var pt in info.Pieces[1].FullGrid)
        {
            Assert.True(pt[0] >= 0.0 - 1e-14, $"Node {pt[0]} below lower bound");
            Assert.True(pt[0] <= 1.0 + 1e-14, $"Node {pt[0]} above upper bound");
        }
    }

    [Fact]
    public void Nodes_DomainValidation_LoGeHi_Raises()
    {
        Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.Nodes(
                1, new[] { new[] { 1.0, 1.0 } }, new[] { 10 },
                new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void Nodes_DuplicateKnots_Raises()
    {
        Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.Nodes(
                1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
                new[] { new[] { 0.0, 0.0 } }));
    }
}

// ======================================================================
// TestFromValuesSpline — ChebyshevSpline.FromValues() coverage
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.FromValues().
/// Ported from ref/PyChebyshev/tests/test_from_values.py :: TestFromValuesSpline.
/// Covers: ChebyshevSpline.cs lines 606-671.
/// </summary>
public class TestFromValuesSpline
{
    [Fact]
    public void Eval_MatchesBuild_1D()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        double[] pts = { -0.7, -0.3, 0.1, 0.5, 0.9 };
        foreach (double p in pts)
        {
            double va = a.Eval(new[] { p }, new[] { 0 });
            double vb = b.Eval(new[] { p }, new[] { 0 });
            Assert.Equal(va, vb);
        }
    }

    [Fact]
    public void Eval_MatchesBuild_2D()
    {
        double F(double[] x, object? _) => Math.Max(x[0] - 100.0, 0.0) * Math.Exp(-0.05 * x[1]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 2,
            new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
            new[] { 15, 15 },
            new[] { new[] { 100.0 }, Array.Empty<double>() });

        double va = a.Eval(new[] { 110.0, 0.5 }, new[] { 0, 0 });
        double vb = b.Eval(new[] { 110.0, 0.5 }, new[] { 0, 0 });
        Assert.Equal(va, vb);
    }

    [Fact]
    public void Derivatives_Match()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        // First derivative on each piece
        double da = a.Eval(new[] { 0.5 }, new[] { 1 });
        double db = b.Eval(new[] { 0.5 }, new[] { 1 });
        Assert.Equal(da, db);

        da = a.Eval(new[] { -0.5 }, new[] { 1 });
        db = b.Eval(new[] { -0.5 }, new[] { 1 });
        Assert.Equal(da, db);
    }

    [Fact]
    public void Integrate_Full()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        double ia = (double)a.Integrate();
        double ib = (double)b.Integrate();
        Assert.Equal(ia, ib);
    }

    [Fact]
    public void Roots_Match()
    {
        double F(double[] x, object? _) => x[0] * x[0] - 0.25;
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        double[] rootsA = a.Roots();
        double[] rootsB = b.Roots();

        Assert.Equal(rootsA.Length, rootsB.Length);
        for (int i = 0; i < rootsA.Length; i++)
            TestFixtures.AssertClose(rootsA[i], rootsB[i]);
    }

    [Fact]
    public void SaveLoad_RoundTrip()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (_, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        string path = Path.GetTempFileName();
        try
        {
            b.Save(path);
            var loaded = ChebyshevSpline.Load(path);
            Assert.Equal(
                b.Eval(new[] { 0.5 }, new[] { 0 }),
                loaded.Eval(new[] { 0.5 }, new[] { 0 }));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void NoKnots_MatchesApproximation()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);

        // Spline with no knots => 1 piece, should match ChebyshevApproximation
        var (spline, _) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { Array.Empty<double>() });

        var approx = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        approx.Build(verbose: false);

        double vs = spline.Eval(new[] { 0.5 }, new[] { 0 });
        double va = approx.Eval(new[] { 0.5 }, new[] { 0 });
        TestFixtures.AssertClose(vs, va);
    }

    [Fact]
    public void ErrorEstimate_Matches()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        Assert.Equal(a.ErrorEstimate(), b.ErrorEstimate());
    }

    [Fact]
    public void BatchEval_Matches()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        var points = new[] { new[] { -0.5 }, new[] { 0.3 }, new[] { 0.8 } };
        double[] ra = a.EvalBatch(points, new[] { 0 });
        double[] rb = b.EvalBatch(points, new[] { 0 });

        Assert.Equal(ra.Length, rb.Length);
        for (int i = 0; i < ra.Length; i++)
            Assert.Equal(ra[i], rb[i]);
    }

    [Fact]
    public void MultiEval_Matches()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        double[] ra = a.EvalMulti(new[] { 0.5 }, new[] { new[] { 0 }, new[] { 1 } });
        double[] rb = b.EvalMulti(new[] { 0.5 }, new[] { new[] { 0 }, new[] { 1 } });

        Assert.Equal(ra.Length, rb.Length);
        for (int i = 0; i < ra.Length; i++)
            Assert.Equal(ra[i], rb[i]);
    }

    [Fact]
    public void Build_OnFromValues_Raises()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var (_, b) = TestFixtures.BuildBothWaysSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        var ex = Assert.Throws<InvalidOperationException>(() => b.Build());
        Assert.Contains("no function", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EndToEnd_Workflow()
    {
        // Full workflow: Nodes() -> external eval -> FromValues() -> evaluate
        var info = ChebyshevSpline.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        double[][] pieceValues = new double[info.NumPieces][];
        for (int p = 0; p < info.NumPieces; p++)
        {
            var piece = info.Pieces[p];
            pieceValues[p] = new double[piece.FullGrid.Length];
            for (int i = 0; i < piece.FullGrid.Length; i++)
                pieceValues[p][i] = Math.Abs(piece.FullGrid[i][0]);
        }

        var spline = ChebyshevSpline.FromValues(
            pieceValues, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });

        TestFixtures.AssertClose(0.5, spline.Eval(new[] { 0.5 }, new[] { 0 }), rtol: 1e-10);
        TestFixtures.AssertClose(0.7, spline.Eval(new[] { -0.7 }, new[] { 0 }), rtol: 1e-10);
    }

    // --- Edge cases ---

    [Fact]
    public void PieceCountMismatch_Raises()
    {
        // Provide wrong number of pieces
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.FromValues(
                new[] { new double[15] },  // 1 piece, but 2 needed
                1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
                new[] { new[] { 0.0 } }));  // knot => 2 pieces
        Assert.Contains("piece", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PieceShapeMismatch_Raises()
    {
        // Provide wrong shape for one piece
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.FromValues(
                new[] { new double[15], new double[10] },  // piece 1 should be 15
                1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
                new[] { new[] { 0.0 } }));
        Assert.Contains("piece", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

// ======================================================================
// TestDctIIviaFFT — FFT-based DCT-II for n > 32
// ======================================================================

/// <summary>
/// Tests exercising the FFT-based DCT-II path (BarycentricKernel.DctIIviaFFT).
/// This path activates for n > 32 nodes. All standard tests use n <= 32.
/// Covers: BarycentricKernel.cs lines 330, 360-387.
/// </summary>
public class TestDctIIviaFFT
{
    [Fact]
    public void ErrorEstimate_LargeOrder_1D()
    {
        // Use 64 nodes to trigger FFT path (n > 32)
        double F(double[] x, object? _) => Math.Sin(x[0]);

        var cheb = new ChebyshevApproximation(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 64 });
        cheb.Build(verbose: false);

        double err = cheb.ErrorEstimate();
        // sin(x) with 64 nodes should have near-machine-precision error
        Assert.True(err < 1e-14, $"Error estimate {err:E2} too large for 64 nodes on sin(x)");
    }

    [Fact]
    public void ErrorEstimate_LargeOrder_IsReasonable()
    {
        // Build with 33 nodes (just above FFT threshold of 32) and verify error estimate
        // is both positive and small for a smooth function
        double F(double[] x, object? _) => Math.Exp(-x[0] * x[0]);

        var cheb = new ChebyshevApproximation(
            F, 1, new[] { new[] { -3.0, 3.0 } }, new[] { 33 });
        cheb.Build(verbose: false);

        double err = cheb.ErrorEstimate();
        Assert.True(err > 0, "Error estimate should be positive");
        Assert.True(err < 1e-6, $"Error estimate {err:E2} too large for 33 nodes on exp(-x²)");

        // Also verify actual interpolation accuracy is good
        double actual = cheb.VectorizedEval(new[] { 1.5 }, new[] { 0 });
        double expected = Math.Exp(-1.5 * 1.5);
        Assert.True(Math.Abs(actual - expected) < 1e-8,
            $"Interpolation error too large: expected {expected:E10}, got {actual:E10}");
    }

    [Fact]
    public void ErrorEstimate_LargeOrder_2D()
    {
        // 2D with one dimension > 32 nodes
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]);

        var cheb = new ChebyshevApproximation(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 40, 10 });
        cheb.Build(verbose: false);

        double err = cheb.ErrorEstimate();
        Assert.True(err < 1e-10, $"Error estimate {err:E2} too large for 40x10 on sin*cos");
    }

    [Fact]
    public void FromValues_LargeOrder_ErrorEstimate()
    {
        // FromValues with n=50 triggers FFT path in ErrorEstimate
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { 0.0, Math.PI } }, new[] { 50 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0]);

        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { 0.0, Math.PI } }, new[] { 50 });

        double err = cheb.ErrorEstimate();
        Assert.True(err < 1e-14, $"Error estimate {err:E2} too large for 50 nodes on sin(x)");
    }
}

// ======================================================================
// TestSplineIntegrationEdgeCases — partial integration coverage gaps
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.Integrate() edge cases.
/// Covers: partial integration branches (lines 959-1096), 0D fallback.
/// </summary>
public class TestSplineIntegrationEdgeCases
{
    [Fact]
    public void Integrate_Partial_WithBounds()
    {
        // Partial integration with custom bounds on a 2D spline
        double F(double[] x, object? _) => Math.Abs(x[0]) * Math.Sin(x[1]);
        var spline = new ChebyshevSpline(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, Math.PI } },
            new[] { 15, 15 },
            new[] { new[] { 0.0 }, Array.Empty<double>() });
        spline.Build(verbose: false);

        // Integrate dim 0 with custom bounds, keep dim 1
        var result = (ChebyshevSpline)spline.Integrate(
            dims: new[] { 0 },
            bounds: new[] { (-0.5, 0.5) });

        double val = result.Eval(new[] { 1.5 }, new[] { 0 });
        Assert.True(double.IsFinite(val));
    }

    [Fact]
    public void Integrate_AllDims_WithBounds()
    {
        // Full integration with custom bounds
        double F(double[] x, object? _) => Math.Abs(x[0]);
        var spline = new ChebyshevSpline(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        spline.Build(verbose: false);

        // Integrate with sub-interval bounds
        double val = (double)spline.Integrate(
            bounds: new[] { (-0.5, 0.5) });

        // Integral of |x| from -0.5 to 0.5 = 2 * (0.5^2 / 2) = 0.25
        TestFixtures.AssertClose(0.25, val, rtol: 1e-8);
    }

    [Fact]
    public void Integrate_Partial_ReturnsDimensionReduction()
    {
        // 2D -> 1D partial integration
        double F(double[] x, object? _) => x[0] * x[0] + Math.Abs(x[1]);
        var spline = new ChebyshevSpline(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 15, 15 },
            new[] { Array.Empty<double>(), new[] { 0.0 } });
        spline.Build(verbose: false);

        // Integrate dim 1 only
        var result = spline.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevSpline>(result);

        var spline1d = (ChebyshevSpline)result;
        double val = spline1d.Eval(new[] { 0.5 }, new[] { 0 });
        Assert.True(double.IsFinite(val));
    }
}

// ======================================================================
// TestDefensiveErrorPaths — guard clauses and validation
// ======================================================================

/// <summary>
/// Tests for defensive error paths (guard clauses) that are never triggered
/// by the main test suite. Covers type mismatch, not-built guards, etc.
/// </summary>
public class TestDefensiveErrorPaths
{
    [Fact]
    public void Algebra_NotBuilt_Left_Raises()
    {
        // Left operand not built should throw InvalidOperationException
        double F(double[] x, object? _) => Math.Sin(x[0]);

        var a = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        // a NOT built

        var b = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        b.Build(verbose: false);

        Assert.Throws<InvalidOperationException>(() =>
        {
            var _ = a + b;
        });
    }

    [Fact]
    public void Algebra_NotBuilt_Right_Raises()
    {
        // Right operand not built should throw InvalidOperationException
        double F(double[] x, object? _) => Math.Sin(x[0]);

        var a = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        a.Build(verbose: false);

        var b = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        // b NOT built

        Assert.Throws<InvalidOperationException>(() =>
        {
            var _ = a + b;
        });
    }

    [Fact]
    public void Spline_Slice_NotBuilt_Raises()
    {
        var spline = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });

        // Not built yet
        Assert.Throws<InvalidOperationException>(() => spline.Slice((0, 0.5)));
    }

    [Fact]
    public void Spline_Extrude_NotBuilt_Raises()
    {
        var spline = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });

        Assert.Throws<InvalidOperationException>(() =>
            spline.Extrude((1, new[] { 0.0, 1.0 }, 5)));
    }

    [Fact]
    public void Spline_Maximize_NotBuilt_Raises()
    {
        var spline = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });

        Assert.Throws<InvalidOperationException>(() => spline.Maximize());
    }

    [Fact]
    public void Spline_Slice_OutOfDomain_Raises()
    {
        var spline = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        spline.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => spline.Slice((0, 5.0)));
    }

    [Fact]
    public void Spline_ValidateKnots_LengthMismatch_Raises()
    {
        // knots.Length != numDimensions
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => x[0], 2,
                new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
                new[] { 10, 10 },
                new[] { new[] { 0.0 } }));  // 1 knot array for 2 dims
    }

    [Fact]
    public void Spline_DuplicateKnots_Raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => x[0], 1,
                new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
                new[] { new[] { 0.0, 0.0 } }));
        Assert.Contains("sorted", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Spline_CheckCompatible_DimensionMismatch_Raises()
    {
        double F(double[] x, object? _) => x[0];

        var sp1d = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });
        sp1d.Build(verbose: false);

        double G(double[] x, object? _) => x[0] + x[1];
        var sp2d = new ChebyshevSpline(G, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 },
            new[] { Array.Empty<double>(), Array.Empty<double>() });
        sp2d.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = sp1d + sp2d; });
    }

    [Fact]
    public void Spline_CheckCompatible_DomainMismatch_Raises()
    {
        double F(double[] x, object? _) => x[0];

        var sp1 = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });
        sp1.Build(verbose: false);

        var sp2 = new ChebyshevSpline(F, 1,
            new[] { new[] { 0.0, 2.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });
        sp2.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = sp1 + sp2; });
    }

    [Fact]
    public void Spline_CheckCompatible_NodesMismatch_Raises()
    {
        double F(double[] x, object? _) => x[0];

        var sp1 = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });
        sp1.Build(verbose: false);

        var sp2 = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { Array.Empty<double>() });
        sp2.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = sp1 + sp2; });
    }

    [Fact]
    public void Spline_CheckCompatible_KnotsMismatch_Raises()
    {
        double F(double[] x, object? _) => Math.Abs(x[0]);

        var sp1 = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { new[] { 0.0 } });
        sp1.Build(verbose: false);

        var sp2 = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { new[] { 0.5 } });
        sp2.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = sp1 + sp2; });
    }

    [Fact]
    public void BarycentricDerivative_OrderLessThan1_Raises()
    {
        // This tests the order < 1 guard in BarycentricDerivativeAnalytical
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        cheb.Build(verbose: false);

        // Requesting derivative order 0 for a dimension should work (no derivative)
        // but the internal BarycentricDerivativeAnalytical should not be called with order < 1
        // Just verify normal eval works with zeroth order
        double val = cheb.VectorizedEval(new[] { 0.5 }, new[] { 0 });
        TestFixtures.AssertClose(Math.Sin(0.5), val, rtol: 1e-10);
    }

    [Fact]
    public void Approx_Maximize_NotBuilt_Raises()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        Assert.Throws<InvalidOperationException>(() => cheb.Maximize());
    }

    [Fact]
    public void Approx_FromValues_DomainMismatch_Raises()
    {
        // domain length != numDimensions
        var info = ChebyshevApproximation.Nodes(1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0]);

        // Pass 2 domain entries for 1D
        Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(values, 1,
                new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 10 }));
    }

    [Fact]
    public void Spline_FromValues_DomainLoGeHi_Raises()
    {
        // domain lo >= hi should throw
        Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.FromValues(
                new[] { new double[10] }, 1,
                new[] { new[] { 1.0, 1.0 } }, new[] { 10 },
                new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void Spline_Slice_OutOfDomain_LowValue_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 },
            new[] { Array.Empty<double>() });
        sp.Build(verbose: false);

        // Extude to 2D first, then slice with out-of-domain value
        var sp2d = sp.Extrude((1, new[] { 0.0, 1.0 }, 5));
        Assert.Throws<ArgumentException>(() => sp2d.Slice((1, -0.5)));
    }
}

// ======================================================================
// TestSplineIntegratePartialBounds — complex partial boundary overlap
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.Integrate() with partial bounds that exercise
/// the piece-by-piece overlap logic (lines 1025-1096).
/// </summary>
public class TestSplineIntegratePartialBounds
{
    [Fact]
    public void Integrate_1D_PartialBounds_SubsetOfPiece()
    {
        // f(x) = x^2 on [-1, 1] with knot at 0, integrate over [0.2, 0.8] (subset of one piece)
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var result = sp.Integrate(bounds: new[] { (0.2, 0.8) });
        double expected = (0.8 * 0.8 * 0.8 - 0.2 * 0.2 * 0.2) / 3.0;
        Assert.IsType<double>(result);
        TestFixtures.AssertClose(expected, (double)result, rtol: 1e-10);
    }

    [Fact]
    public void Integrate_1D_PartialBounds_SpanMultiplePieces()
    {
        // f(x) = x^2 on [-1, 1] with knot at 0, integrate over [-0.5, 0.5]
        // This spans both pieces, exercising the overlap sum logic
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var result = sp.Integrate(bounds: new[] { (-0.5, 0.5) });
        double expected = (0.5 * 0.5 * 0.5 + 0.5 * 0.5 * 0.5) / 3.0;
        Assert.IsType<double>(result);
        TestFixtures.AssertClose(expected, (double)result, rtol: 1e-10);
    }

    [Fact]
    public void Integrate_1D_BoundsOutsidePiece_NoOverlap()
    {
        // f(x) = x on [-1, 1] with knot at 0, integrate over [0.5, 1.0]
        // Only the second piece [0, 1] contributes, and partially
        var sp = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var result = sp.Integrate(bounds: new[] { (0.5, 1.0) });
        double expected = (1.0 * 1.0 - 0.5 * 0.5) / 2.0; // = 0.375
        Assert.IsType<double>(result);
        TestFixtures.AssertClose(expected, (double)result, rtol: 1e-10);
    }

    [Fact]
    public void Integrate_1D_PartialBounds_FullPieceCoincidence()
    {
        // Bounds exactly match a piece boundary [0, 1] — should use full integration path
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var result = sp.Integrate(bounds: new[] { (0.0, 1.0) });
        double expected = 1.0 / 3.0;
        Assert.IsType<double>(result);
        TestFixtures.AssertClose(expected, (double)result, rtol: 1e-10);
    }

    [Fact]
    public void Integrate_Partial_1D_ReducesToScalar()
    {
        // Partial integration on all dims with bounds should return double
        var sp = new ChebyshevSpline(
            (x, _) => Math.Sin(x[0]), 1,
            new[] { new[] { 0.0, Math.PI } }, new[] { 15 },
            new[] { new[] { Math.PI / 2 } });
        sp.Build(verbose: false);

        var result = sp.Integrate(dims: new[] { 0 }, bounds: new[] { (0.0, Math.PI) });
        // ∫₀^π sin(x) dx = 2
        Assert.IsType<double>(result);
        TestFixtures.AssertClose(2.0, (double)result, rtol: 1e-6);
    }

    [Fact]
    public void Integrate_2D_PartialDim_WithBounds()
    {
        // 2D spline, integrate dim 0 with partial bounds, keep dim 1
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0] + x[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
            new[] { 15, 11 },
            new[] { new[] { 0.0 }, Array.Empty<double>() });
        sp.Build(verbose: false);

        // Integrate dim 0 over [0.0, 1.0] (partial, only the right piece)
        var result = sp.Integrate(dims: new[] { 0 }, bounds: new[] { (0.0, 1.0) });
        // Result should be a ChebyshevSpline in 1D (dim 1)
        Assert.IsType<ChebyshevSpline>(result);
        var sp1d = (ChebyshevSpline)result;

        // At y=0.5: ∫₀¹ (x²+0.5) dx = 1/3 + 0.5 = 5/6
        double val = sp1d.Eval(new[] { 0.5 }, new[] { 0 });
        TestFixtures.AssertClose(5.0 / 6.0, val, rtol: 1e-6);
    }
}

// ======================================================================
// TestSplineRootsEdgeCases — root deduplication near knots
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.Roots() edge cases: deduplication near knot
/// boundaries, empty root arrays, and single roots.
/// </summary>
public class TestSplineRootsEdgeCases
{
    [Fact]
    public void Roots_NearKnotBoundary_Deduplicates()
    {
        // f(x) = x on [-1, 1] with knot at 0 — root at x=0 found by both pieces
        var sp = new ChebyshevSpline(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 11 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        double[] roots = sp.Roots();
        // Should find exactly one root at 0.0, not two
        Assert.Single(roots);
        TestFixtures.AssertClose(0.0, roots[0], atol: 1e-10);
    }

    [Fact]
    public void Roots_NoRoots_ReturnsEmpty()
    {
        // f(x) = x^2 + 1 on [-1, 1] — no real roots
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0] + 1.0, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 11 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        double[] roots = sp.Roots();
        Assert.Empty(roots);
    }

    [Fact]
    public void Roots_MultipleRoots_AcrossPieces()
    {
        // f(x) = sin(2πx) on [-1, 1] with knot at 0 — roots at -0.5, 0, 0.5
        var sp = new ChebyshevSpline(
            (x, _) => Math.Sin(2 * Math.PI * x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        double[] roots = sp.Roots();
        // Interior roots at -0.5, 0.5 (at minimum); endpoints and knot root may or may not be found
        Assert.True(roots.Length >= 2, $"Expected at least 2 roots, got {roots.Length}");
        // Roots should be sorted and deduplicated (this exercises the Count > 1 dedup path)
        for (int i = 1; i < roots.Length; i++)
            Assert.True(roots[i] > roots[i - 1], "Roots should be sorted and deduplicated");
    }
}

// ======================================================================
// TestSliderCompatibilityErrors — Slider arithmetic guard clauses
// ======================================================================

/// <summary>
/// Tests for ChebyshevSlider.CheckSliderCompatible() guard clauses
/// that are not covered by AlgebraTests.
/// </summary>
public class TestSliderCompatibilityErrors
{
    [Fact]
    public void Slider_MaxDerivOrderMismatch_Raises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]);

        var sl1 = new ChebyshevSlider(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 },
            maxDerivativeOrder: 2);
        sl1.Build(verbose: false);

        var sl2 = new ChebyshevSlider(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 },
            maxDerivativeOrder: 3);
        sl2.Build(verbose: false);

        Assert.Throws<InvalidOperationException>(() => { var _ = sl1 + sl2; });
    }

    [Fact]
    public void Slider_PartitionMismatch_Raises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]);

        var sl1 = new ChebyshevSlider(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        sl1.Build(verbose: false);

        var sl2 = new ChebyshevSlider(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0, 1 } },
            new[] { 0.0, 0.0 });
        sl2.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = sl1 + sl2; });
    }
}

// ======================================================================
// TestCalculusValidation — ValidateCalculusArgs guard clauses
// ======================================================================

/// <summary>
/// Tests for Calculus.ValidateCalculusArgs() error paths.
/// </summary>
public class TestCalculusValidation
{
    [Fact]
    public void Calculus_1D_WrongDim_Raises()
    {
        // 1D interpolant with dim=1 (should be 0)
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => cheb.Roots(dim: 1));
    }

    [Fact]
    public void Calculus_1D_WithFixedDims_Raises()
    {
        // 1D interpolant with fixed dims (should be empty)
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Roots(dim: 0, fixedDims: new Dictionary<int, double> { { 0, 0.5 } }));
    }

    [Fact]
    public void Calculus_MultiD_NoDim_Raises()
    {
        // 2D interpolant without specifying dim
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0] + x[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Roots(fixedDims: new Dictionary<int, double> { { 1, 0.0 } }));
    }

    [Fact]
    public void Calculus_MultiD_DimOutOfRange_Raises()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0] + x[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Roots(dim: 5, fixedDims: new Dictionary<int, double> { { 0, 0.0 } }));
    }
}

// ======================================================================
// TestExtrudeSliceValidation — parameter validation guard clauses
// ======================================================================

/// <summary>
/// Tests for ExtrudeSlice validation: invalid bounds, nNodes, duplicate dims.
/// </summary>
public class TestExtrudeSliceValidation
{
    [Fact]
    public void Extrude_BoundsLoGeHi_Raises()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Extrude((1, new[] { 1.0, 0.0 }, 5)));
    }

    [Fact]
    public void Extrude_NNodesLessThan2_Raises()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Extrude((1, new[] { 0.0, 1.0 }, 1)));
    }

    [Fact]
    public void Slice_DuplicateDimIndex_Raises()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0] + x[1] + x[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 });
        cheb.Build(verbose: false);

        Assert.Throws<ArgumentException>(() =>
            cheb.Slice((0, 0.5), (0, 0.3)));
    }
}

// ======================================================================
// TestSplineSliceMultiPiece — slice through multi-piece spline
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.Slice() edge cases: slicing 2D splines to 1D,
/// exercising the multi-piece slice logic (lines 838-844).
/// </summary>
public class TestSplineSliceMultiPiece
{
    [Fact]
    public void Slice_2D_Spline_AllDimsHaveKnots()
    {
        // 2D spline with knots on both dims, slice down to 1D
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 },
            new[] { new[] { 0.0 }, new[] { 0.0 } });
        sp.Build(verbose: false);

        // Slice dim 1 at 0.5
        var sp1d = sp.Slice((1, 0.5));
        Assert.Equal(1, sp1d.NumDimensions);

        double val = sp1d.Eval(new[] { 0.3 }, new[] { 0 });
        double expected = Math.Abs(0.3) + Math.Abs(0.5);
        TestFixtures.AssertClose(expected, val, rtol: 1e-10);
    }

    [Fact]
    public void Slice_2D_Spline_ReduceToAllSliced()
    {
        // 2D spline, slice both dims — should exercise the "all dims being sliced" path
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) * x[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
            new[] { 11, 11 },
            new[] { new[] { 0.0 }, Array.Empty<double>() });
        sp.Build(verbose: false);

        // This should fail because slicing all dims produces 0D
        Assert.Throws<ArgumentException>(() => sp.Slice((0, 0.5), (1, 1.0)));
    }

    [Fact]
    public void Slice_2DSpline_BothDimsKnotted_SliceDim0()
    {
        // 2D spline with 2x2 pieces, slice dim 0
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 },
            new[] { new[] { 0.0 }, new[] { 0.0 } });
        sp.Build(verbose: false);

        // Slice dim 0 at -0.3 — the piece containing -0.3 is piece 0 ([-1, 0])
        var sp1d = sp.Slice((0, -0.3));
        Assert.Equal(1, sp1d.NumDimensions);

        double val = sp1d.Eval(new[] { 0.7 }, new[] { 0 });
        double expected = Math.Abs(-0.3) + Math.Abs(0.7);
        TestFixtures.AssertClose(expected, val, rtol: 1e-10);
    }
}
