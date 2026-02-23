using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestNodesApprox
// ======================================================================

/// <summary>
/// Tests for ChebyshevApproximation.Nodes().
/// Ported from ref/PyChebyshev/tests/test_from_values.py :: TestNodesApprox.
/// </summary>
public class TestNodesApprox
{
    [Fact]
    public void Nodes_ReturnsCorrectProperties()
    {
        var info = ChebyshevApproximation.Nodes(2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 5, 7 });

        Assert.NotNull(info.NodesPerDim);
        Assert.NotNull(info.FullGrid);
        Assert.NotNull(info.Shape);
    }

    [Fact]
    public void Nodes_Shape_1D()
    {
        var info = ChebyshevApproximation.Nodes(1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });

        Assert.Equal(new[] { 10 }, info.Shape);
        Assert.Single(info.NodesPerDim);
        Assert.Equal(10, info.NodesPerDim[0].Length);
    }

    [Fact]
    public void Nodes_Shape_2D()
    {
        var info = ChebyshevApproximation.Nodes(2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 3.0 } }, new[] { 8, 12 });

        Assert.Equal(new[] { 8, 12 }, info.Shape);
        Assert.Equal(96, info.FullGrid.Length);
    }

    [Fact]
    public void Nodes_Shape_3D()
    {
        var info = ChebyshevApproximation.Nodes(
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 }, new[] { 2.0, 4.0 } },
            new[] { 5, 6, 7 });

        Assert.Equal(new[] { 5, 6, 7 }, info.Shape);
        Assert.Equal(210, info.FullGrid.Length);
    }

    [Fact]
    public void Nodes_FullGrid_Shape()
    {
        var info = ChebyshevApproximation.Nodes(
            2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 4, 5 });

        Assert.Equal(20, info.FullGrid.Length);
        // Each grid point has 2 coordinates
        Assert.All(info.FullGrid, p => Assert.Equal(2, p.Length));

        // All x-coords should be in [-1, 1]
        double xMin = info.FullGrid.Min(p => p[0]);
        double xMax = info.FullGrid.Max(p => p[0]);
        Assert.True(xMin >= -1.0, $"x min {xMin} < -1.0");
        Assert.True(xMax <= 1.0, $"x max {xMax} > 1.0");

        // All y-coords should be in [0, 1]
        double yMin = info.FullGrid.Min(p => p[1]);
        double yMax = info.FullGrid.Max(p => p[1]);
        Assert.True(yMin >= 0.0, $"y min {yMin} < 0.0");
        Assert.True(yMax <= 1.0, $"y max {yMax} > 1.0");
    }

    [Fact]
    public void Nodes_MatchBuildNodes()
    {
        // nodes() returns same nodes as a built interpolant
        double Func(double[] x, object? _) => Math.Sin(x[0]) + x[1];

        var cheb = new ChebyshevApproximation(
            Func, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
            new[] { 8, 6 });
        cheb.Build(verbose: false);

        var info = ChebyshevApproximation.Nodes(
            2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 8, 6 });

        for (int d = 0; d < 2; d++)
        {
            Assert.Equal(cheb.NodeArrays[d].Length, info.NodesPerDim[d].Length);
            for (int i = 0; i < cheb.NodeArrays[d].Length; i++)
            {
                TestFixtures.AssertClose(cheb.NodeArrays[d][i], info.NodesPerDim[d][i]);
            }
        }
    }

    [Fact]
    public void Nodes_DomainMismatchRaises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.Nodes(2, new[] { new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
        Assert.Contains("numDimensions", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Nodes_NdimMismatchRaises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.Nodes(
                2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 5 }));
        Assert.Contains("numDimensions", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

// ======================================================================
// TestFromValuesApprox
// ======================================================================

/// <summary>
/// Tests for ChebyshevApproximation.FromValues().
/// Ported from ref/PyChebyshev/tests/test_from_values.py :: TestFromValuesApprox.
/// </summary>
public class TestFromValuesApprox
{
    // --- Core equivalence: bit-identical to Build() ---

    [Fact]
    public void Eval_MatchesBuild_1D()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

        double[] pts = { 0.5, 1.0, 2.0, 3.0 };
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
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Exp(-x[1]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 12, 10 });

        double va = a.VectorizedEval(new[] { 0.3, 0.7 }, new[] { 0, 0 });
        double vb = b.VectorizedEval(new[] { 0.3, 0.7 }, new[] { 0, 0 });
        Assert.Equal(va, vb);
    }

    [Fact]
    public void Eval_MatchesBuild_3D()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { 1.0, 3.0 } },
            new[] { 8, 8, 6 });

        double va = a.VectorizedEval(new[] { 0.5, -0.3, 2.0 }, new[] { 0, 0, 0 });
        double vb = b.VectorizedEval(new[] { 0.5, -0.3, 2.0 }, new[] { 0, 0, 0 });
        Assert.Equal(va, vb);
    }

    [Fact]
    public void Derivatives_1st()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 15, 15 });

        int[][] derivs = { new[] { 1, 0 }, new[] { 0, 1 } };
        foreach (int[] deriv in derivs)
        {
            double da = a.VectorizedEval(new[] { 0.3, 0.5 }, deriv);
            double db = b.VectorizedEval(new[] { 0.3, 0.5 }, deriv);
            Assert.Equal(da, db);
        }
    }

    [Fact]
    public void Derivatives_2nd()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 15, 15 });

        int[][] derivs = { new[] { 2, 0 }, new[] { 0, 2 }, new[] { 1, 1 } };
        foreach (int[] deriv in derivs)
        {
            double da = a.VectorizedEval(new[] { 0.3, 0.5 }, deriv);
            double db = b.VectorizedEval(new[] { 0.3, 0.5 }, deriv);
            Assert.Equal(da, db);
        }
    }

    [Fact]
    public void Integrate_Full()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

        double ia = (double)a.Integrate();
        double ib = (double)b.Integrate();
        Assert.Equal(ia, ib);
    }

    [Fact]
    public void Integrate_Partial()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Exp(-x[1]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 15, 12 });

        // Partial integration returns a ChebyshevApproximation
        var ra = (ChebyshevApproximation)a.Integrate(dims: new[] { 1 });
        var rb = (ChebyshevApproximation)b.Integrate(dims: new[] { 1 });

        double[] pt = { 0.3 };
        Assert.Equal(
            ra.VectorizedEval(pt, new[] { 0 }),
            rb.VectorizedEval(pt, new[] { 0 }));
    }

    [Fact]
    public void Integrate_SubInterval()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

        double ia = (double)a.Integrate(bounds: new[] { (0.0, 1.0) });
        double ib = (double)b.Integrate(bounds: new[] { (0.0, 1.0) });
        Assert.Equal(ia, ib);
    }

    [Fact]
    public void Roots()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { 0.5, 5.0 } }, new[] { 25 });

        double[] rootsA = a.Roots();
        double[] rootsB = b.Roots();

        Assert.Equal(rootsA.Length, rootsB.Length);
        for (int i = 0; i < rootsA.Length; i++)
        {
            TestFixtures.AssertClose(rootsA[i], rootsB[i]);
        }
    }

    [Fact]
    public void Minimize()
    {
        double F(double[] x, object? _) => (x[0] - 0.3) * (x[0] - 0.3);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        var (va, xa) = a.Minimize();
        var (vb, xb) = b.Minimize();
        Assert.Equal(xa, xb);
        Assert.Equal(va, vb);
    }

    [Fact]
    public void Maximize()
    {
        double F(double[] x, object? _) => -(x[0] - 0.3) * (x[0] - 0.3);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        var (va, xa) = a.Maximize();
        var (vb, xb) = b.Maximize();
        Assert.Equal(xa, xb);
        Assert.Equal(va, vb);
    }

    [Fact]
    public void Algebra_Add()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        double G(double[] x, object? _) => Math.Cos(x[0]);

        var (_, fb) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        var (_, gb) = TestFixtures.BuildBothWaysApprox(
            G, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        var result = fb + gb;
        double expected = Math.Sin(0.5) + Math.Cos(0.5);
        Assert.True(Math.Abs(result.VectorizedEval(new[] { 0.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void Algebra_Mul()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (_, fb) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        var result = fb * 3.0;
        double expected = 3.0 * Math.Sin(0.5);
        Assert.True(Math.Abs(result.VectorizedEval(new[] { 0.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void Extrude()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (_, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        var extruded = b.Extrude((1, new[] { 0.0, 2.0 }, 5));
        double val = extruded.VectorizedEval(new[] { 0.5, 1.0 }, new[] { 0, 0 });
        Assert.True(Math.Abs(val - Math.Sin(0.5)) < 1e-10);
    }

    [Fact]
    public void Slice()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Exp(-x[1]);
        var (_, b) = TestFixtures.BuildBothWaysApprox(
            F, 2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 15, 12 });

        var sliced = b.Slice((1, 1.0));
        double expected = Math.Sin(0.5) * Math.Exp(-1.0);
        Assert.True(Math.Abs(sliced.VectorizedEval(new[] { 0.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void SaveLoad()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (_, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        string path = Path.GetTempFileName();
        try
        {
            b.Save(path);
            var loaded = ChebyshevApproximation.Load(path);
            Assert.Equal(
                b.VectorizedEval(new[] { 0.5 }, new[] { 0 }),
                loaded.VectorizedEval(new[] { 0.5 }, new[] { 0 }));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ErrorEstimate()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        Assert.Equal(a.ErrorEstimate(), b.ErrorEstimate());
    }

    [Fact]
    public void BatchEval()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        // Python test uses vectorized_eval_multi with multiple points and single deriv order
        // In our API that is VectorizedEvalMulti with multiple points
        double[] ra = a.VectorizedEvalMulti(new[] { 0.1 }, new[] { new[] { 0 } });
        double[] rb = b.VectorizedEvalMulti(new[] { 0.1 }, new[] { new[] { 0 } });
        Assert.Equal(ra[0], rb[0]);

        ra = a.VectorizedEvalMulti(new[] { 0.5 }, new[] { new[] { 0 } });
        rb = b.VectorizedEvalMulti(new[] { 0.5 }, new[] { new[] { 0 } });
        Assert.Equal(ra[0], rb[0]);

        ra = a.VectorizedEvalMulti(new[] { 0.9 }, new[] { new[] { 0 } });
        rb = b.VectorizedEvalMulti(new[] { 0.9 }, new[] { new[] { 0 } });
        Assert.Equal(ra[0], rb[0]);
    }

    [Fact]
    public void MultiEval()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var (a, b) = TestFixtures.BuildBothWaysApprox(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        // Value and first derivative at same point
        double[] ra = a.VectorizedEvalMulti(new[] { 0.5 }, new[] { new[] { 0 }, new[] { 1 } });
        double[] rb = b.VectorizedEvalMulti(new[] { 0.5 }, new[] { new[] { 0 }, new[] { 1 } });
        Assert.Equal(ra.Length, rb.Length);
        for (int i = 0; i < ra.Length; i++)
            Assert.Equal(ra[i], rb[i]);
    }

    [Fact]
    public void EndToEnd_Workflow()
    {
        // Full workflow: Nodes() -> external eval -> FromValues() -> evaluate
        var info = ChebyshevApproximation.Nodes(
            2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 12, 10 });

        // Simulate external evaluation
        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
        {
            values[i] = Math.Sin(info.FullGrid[i][0]) * Math.Exp(-info.FullGrid[i][1]);
        }

        var cheb = ChebyshevApproximation.FromValues(
            values, 2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } }, new[] { 12, 10 });

        double expected = Math.Sin(0.5) * Math.Exp(-1.0);
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { 0.5, 1.0 }, new[] { 0, 0 }) - expected) < 1e-10);
    }

    // --- Edge cases ---

    [Fact]
    public void ShapeMismatch_Raises()
    {
        // 5 * 5 = 25 values, but we expect 5 * 7 = 35
        double[] values = new double[25];
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                values, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
                new[] { 5, 7 }));
        Assert.Contains("shape", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Nan_Raises()
    {
        double[] values = { 1.0, 1.0, double.NaN, 1.0, 1.0 };
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                values, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 }));
        Assert.Contains("NaN or Inf", ex.Message);
    }

    [Fact]
    public void Inf_Raises()
    {
        double[] values = { 1.0, 1.0, double.PositiveInfinity, 1.0, 1.0 };
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                values, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 }));
        Assert.Contains("NaN or Inf", ex.Message);
    }

    [Fact]
    public void OneNodeDimension()
    {
        // n_nodes=[1, 15]: constant in first dimension
        double F(double[] x, object? _) => Math.Sin(x[1]);

        var info = ChebyshevApproximation.Nodes(
            2, new[] { new[] { -1.0, 1.0 }, new[] { 0.0, Math.PI } }, new[] { 1, 15 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = F(info.FullGrid[i], null);

        var cheb = ChebyshevApproximation.FromValues(
            values, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, Math.PI } },
            new[] { 1, 15 });

        // Should be able to evaluate (dim 0 is constant)
        double val = cheb.VectorizedEval(new[] { 0.0, 1.0 }, new[] { 0, 0 });
        Assert.True(Math.Abs(val - Math.Sin(1.0)) < 1e-10);
    }

    [Fact]
    public void MaxDerivativeOrder()
    {
        // Custom max_derivative_order=3, verify 3rd derivative works
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { 0.0, Math.PI } }, new[] { 25 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0]);

        var cheb = ChebyshevApproximation.FromValues(
            values, 1,
            new[] { new[] { 0.0, Math.PI } },
            new[] { 25 },
            maxDerivativeOrder: 3);

        // 3rd derivative of sin(x) = -cos(x)
        double d3 = cheb.VectorizedEval(new[] { 1.0 }, new[] { 3 });
        double expected = -Math.Cos(1.0);
        Assert.True(Math.Abs(d3 - expected) < 1e-4,
            $"3rd deriv: {d3} vs {expected}");
    }

    [Fact]
    public void Build_OnFromValues_Raises()
    {
        // Build() on a from_values object raises InvalidOperationException
        double[] values = { 1.0, 1.0, 1.0, 1.0, 1.0 };
        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });

        var ex = Assert.Throws<InvalidOperationException>(() => cheb.Build());
        Assert.Contains("no function assigned", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CrossCompat_Algebra()
    {
        // Algebra between from_values and build objects works

        // f via from_values
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        double[] valsF = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            valsF[i] = Math.Sin(info.FullGrid[i][0]);

        var chebF = ChebyshevApproximation.FromValues(
            valsF, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        // g via build
        var chebG = new ChebyshevApproximation(
            (x, _) => Math.Cos(x[0]), 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        chebG.Build(verbose: false);

        var result = chebF + chebG;
        double expected = Math.Sin(0.5) + Math.Cos(0.5);
        Assert.True(Math.Abs(result.VectorizedEval(new[] { 0.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void Str_Output()
    {
        // ToString() shows 'built' with build_time=0.000 and 0 evaluations
        double[] values = { 1.0, 1.0, 1.0, 1.0, 1.0 };
        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });

        string s = cheb.ToString();
        Assert.Contains("built", s, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("0.000s", s);
        Assert.Contains("0 evaluations", s);
    }

    [Fact]
    public void Repr()
    {
        // ToReprString() works without error
        double[] values = { 1.0, 1.0, 1.0, 1.0, 1.0 };
        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });

        string r = cheb.ToReprString();
        Assert.Contains("ChebyshevApproximation", r);
        Assert.Contains("built=True", r);
    }
}

// ======================================================================
// TestEdgeCases (ChebyshevApproximation-only tests)
// ======================================================================

/// <summary>
/// Edge cases identified during code review for ChebyshevApproximation.
/// Ported from ref/PyChebyshev/tests/test_from_values.py :: TestEdgeCases.
/// </summary>
public class TestFromValuesEdgeCases
{
    [Fact]
    public void NegativeDomain()
    {
        // Domain with negative bounds works correctly
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { -10.0, -5.0 } }, new[] { 15 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0]);

        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { -10.0, -5.0 } }, new[] { 15 });

        double expected = Math.Sin(-7.5);
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { -7.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void WideDomain()
    {
        // Very wide domain still works
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { -1000.0, 1000.0 } }, new[] { 25 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0] / 100.0);

        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { -1000.0, 1000.0 } }, new[] { 25 });

        double expected = Math.Sin(500.0 / 100.0);
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { 500.0 }, new[] { 0 }) - expected) < 1e-6);
    }

    [Fact]
    public void TightDomain()
    {
        // Very narrow domain works
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { 0.999, 1.001 } }, new[] { 10 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = info.FullGrid[i][0] * info.FullGrid[i][0];

        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { 0.999, 1.001 } }, new[] { 10 });

        double expected = 1.0 * 1.0;
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { 1.0 }, new[] { 0 }) - expected) < 1e-12);
    }

    [Fact]
    public void BoundaryEvaluation()
    {
        // Evaluation at exact domain boundaries
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
            values[i] = Math.Sin(info.FullGrid[i][0]);

        var cheb = ChebyshevApproximation.FromValues(
            values, 1, new[] { new[] { 0.0, Math.PI } }, new[] { 20 });

        // Left boundary: sin(0) = 0
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { 0.0 }, new[] { 0 })) < 1e-10);
        // Right boundary: sin(pi) ~ 0
        Assert.True(Math.Abs(cheb.VectorizedEval(new[] { Math.PI }, new[] { 0 })) < 1e-10);
    }

    [Fact]
    public void AlgebraChain()
    {
        // Chained algebra operations on from_values objects
        var info = ChebyshevApproximation.Nodes(
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        double[] valsF = new double[info.FullGrid.Length];
        double[] valsG = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
        {
            valsF[i] = Math.Sin(info.FullGrid[i][0]);
            valsG[i] = Math.Cos(info.FullGrid[i][0]);
        }

        var chebF = ChebyshevApproximation.FromValues(
            valsF, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        var chebG = ChebyshevApproximation.FromValues(
            valsG, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });

        // (f + g) * 2 - g = 2*sin + 2*cos - cos = 2*sin + cos
        var result = (chebF + chebG) * 2.0 - chebG;
        double expected = 2.0 * Math.Sin(0.5) + Math.Cos(0.5);
        Assert.True(Math.Abs(result.VectorizedEval(new[] { 0.5 }, new[] { 0 }) - expected) < 1e-10);
    }

    [Fact]
    public void DomainLoGeHi_Raises()
    {
        // from_values rejects domain with lo >= hi
        double[] values = { 1.0, 1.0, 1.0, 1.0, 1.0 };

        // lo == hi
        var ex1 = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                values, 1, new[] { new[] { 1.0, 1.0 } }, new[] { 5 }));
        Assert.Contains("lo=", ex1.Message);
        Assert.Contains("must be strictly less than hi", ex1.Message);

        // lo > hi
        var ex2 = Assert.Throws<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                values, 1, new[] { new[] { 2.0, 1.0 } }, new[] { 5 }));
        Assert.Contains("lo=", ex2.Message);
        Assert.Contains("must be strictly less than hi", ex2.Message);
    }

    [Fact]
    public void FourD_FromValues()
    {
        // 4D from_values builds and evaluates correctly
        int ndim = 4;
        double[][] domain = { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        int[] nNodes = { 5, 5, 5, 5 };

        var info = ChebyshevApproximation.Nodes(ndim, domain, nNodes);

        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
        {
            double sum = 0;
            for (int d = 0; d < ndim; d++)
                sum += info.FullGrid[i][d] * info.FullGrid[i][d];
            values[i] = sum;
        }

        var cheb = ChebyshevApproximation.FromValues(values, ndim, domain, nNodes);

        // f(0.5, 0.5, 0.5, 0.5) = 4 * 0.25 = 1.0
        double expected = 1.0;
        double actual = cheb.VectorizedEval(new[] { 0.5, 0.5, 0.5, 0.5 }, new[] { 0, 0, 0, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-10);
    }
}
