using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ---------- helper functions (exact) ----------

file static class AlgebraHelpers
{
    public static double F2D(double[] x) => Math.Sin(x[0]) + Math.Sin(x[1]);
    public static double G2D(double[] x) => Math.Cos(x[0]) * Math.Cos(x[1]);
    public static double DF_DX0(double[] x) => Math.Cos(x[0]);
    public static double DG_DX0(double[] x) => -Math.Sin(x[0]) * Math.Cos(x[1]);
    public static double D2F_DX0(double[] x) => -Math.Sin(x[0]);
    public static double D2G_DX0(double[] x) => -Math.Cos(x[0]) * Math.Cos(x[1]);

    public static readonly double[][] TestPoints2D =
    {
        new[] { 0.5, 0.3 },
        new[] { -0.7, 0.8 },
        new[] { 0.0, 0.0 },
        new[] { 0.9, -0.9 },
        new[] { -0.2, 0.6 },
    };
}

// ================================================================
// TestApproxArithmetic (16 tests)
// ================================================================

public class TestApproxArithmetic
{
    private static ChebyshevApproximation F => TestFixtures.AlgebraChebF;
    private static ChebyshevApproximation G => TestFixtures.AlgebraChebG;

    [Fact]
    public void AddValues()
    {
        var c = F + G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) + AlgebraHelpers.G2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"Add failed at [{p[0]}, {p[1]}]: {approx} vs {exact}");
        }
    }

    [Fact]
    public void AddDerivative()
    {
        var c = F + G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.DF_DX0(p) + AlgebraHelpers.DG_DX0(p);
            double approx = c.VectorizedEval(p, new[] { 1, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"Add deriv failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void AddSecondDerivative()
    {
        var c = F + G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.D2F_DX0(p) + AlgebraHelpers.D2G_DX0(p);
            double approx = c.VectorizedEval(p, new[] { 2, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6,
                $"Add 2nd deriv failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void SubValues()
    {
        var c = F - G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) - AlgebraHelpers.G2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"Sub failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void SubSelfIsZero()
    {
        var c = F - F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double val = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(val) < 1e-14,
                $"f-f not zero at [{p[0]}, {p[1]}]: {val}");
        }
    }

    [Fact]
    public void MulScalar()
    {
        var c = 3.0 * F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = 3.0 * AlgebraHelpers.F2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-9,
                $"Scalar mul failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void RmulScalar()
    {
        var c1 = 3.0 * F;
        var c2 = F * 3.0;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double v1 = c1.VectorizedEval(p, new[] { 0, 0 });
            double v2 = c2.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(v1 - v2) < 1e-15,
                $"rmul != mul at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void TruedivScalar()
    {
        var c = F / 2.0;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) / 2.0;
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"Div failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void Neg()
    {
        var c = -F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = -AlgebraHelpers.F2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"Neg failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void Iadd()
    {
        // Create a copy via scalar mul by 1.0 to avoid mutating the fixture
        var c = F * 1.0;
        c = c + G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) + AlgebraHelpers.G2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"iadd failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void Isub()
    {
        var c = F * 1.0;
        c = c - G;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) - AlgebraHelpers.G2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"isub failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void Imul()
    {
        var c = F * 1.0;
        c = c * 2.0;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = 2.0 * AlgebraHelpers.F2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"imul failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void Itruediv()
    {
        var c = F * 1.0;
        c = c / 2.0;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = AlgebraHelpers.F2D(p) / 2.0;
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"itruediv failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void ResultFunctionIsNull()
    {
        var c = F + G;
        Assert.Null(c.Function);
    }

    [Fact]
    public void ResultBuildTimeZero()
    {
        var c = F + G;
        Assert.Equal(0.0, c.BuildTime);
    }

    [Fact]
    public void ResultSerializable()
    {
        var c = F + G;
        string path = Path.Combine(Path.GetTempPath(), $"algebra_test_{Guid.NewGuid()}.json");
        try
        {
            c.Save(path);
            var loaded = ChebyshevApproximation.Load(path);
            foreach (var p in AlgebraHelpers.TestPoints2D.Take(2))
            {
                double vOrig = c.VectorizedEval(p, new[] { 0, 0 });
                double vLoaded = loaded.VectorizedEval(p, new[] { 0, 0 });
                Assert.True(Math.Abs(vOrig - vLoaded) < 1e-15);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

// ================================================================
// TestApproxBatchAndMulti (4 tests)
// ================================================================

public class TestApproxBatchAndMulti
{
    private static ChebyshevApproximation F => TestFixtures.AlgebraChebF;
    private static ChebyshevApproximation G => TestFixtures.AlgebraChebG;

    [Fact]
    public void BatchEval()
    {
        var c = F + G;
        var pts = AlgebraHelpers.TestPoints2D;
        double[] batchVals = c.VectorizedEvalBatch(pts, new[] { 0, 0 });
        for (int i = 0; i < pts.Length; i++)
        {
            double single = c.VectorizedEval(pts[i], new[] { 0, 0 });
            Assert.True(Math.Abs(batchVals[i] - single) < 1e-14);
        }
    }

    [Fact]
    public void EvalMulti()
    {
        var c = F + G;
        double[] p = { 0.5, 0.3 };
        int[][] derivs = { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 0, 1 } };
        double[] results = c.VectorizedEvalMulti(p, derivs);
        for (int i = 0; i < derivs.Length; i++)
        {
            double single = c.VectorizedEval(p, derivs[i]);
            Assert.True(Math.Abs(results[i] - single) < 1e-14);
        }
    }

    [Fact]
    public void MulScalarBatch()
    {
        var c = 3.0 * F;
        var pts = AlgebraHelpers.TestPoints2D;
        double[] batchC = c.VectorizedEvalBatch(pts, new[] { 0, 0 });
        double[] batchF = F.VectorizedEvalBatch(pts, new[] { 0, 0 });
        for (int i = 0; i < pts.Length; i++)
        {
            Assert.True(Math.Abs(batchC[i] - 3.0 * batchF[i]) < 1e-14);
        }
    }

    [Fact]
    public void ChainedOps()
    {
        // 0.5*f + 0.3*g - 0.2*f = 0.3*f + 0.3*g
        var c = 0.5 * F + 0.3 * G - 0.2 * F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double exact = 0.3 * AlgebraHelpers.F2D(p) + 0.3 * AlgebraHelpers.G2D(p);
            double approx = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }
}

// ================================================================
// TestEdgeCases (8 CA tests)
// ================================================================

public class TestEdgeCases
{
    private static ChebyshevApproximation F => TestFixtures.AlgebraChebF;
    private static ChebyshevApproximation G => TestFixtures.AlgebraChebG;

    [Fact]
    public void IntScalar()
    {
        // C# operator*(double, CA) won't accept int directly - use 2.0
        var c = 2.0 * F;
        double[] p = { 0.5, 0.3 };
        double exact = 2.0 * AlgebraHelpers.F2D(p);
        double approx = c.VectorizedEval(p, new[] { 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-10);
    }

    [Fact]
    public void MulZero()
    {
        var c = 0.0 * F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            Assert.True(Math.Abs(c.VectorizedEval(p, new[] { 0, 0 })) < 1e-15);
        }
    }

    [Fact]
    public void MulOneIdentity()
    {
        var c = 1.0 * F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double v1 = F.VectorizedEval(p, new[] { 0, 0 });
            double v2 = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(v1 - v2) < 1e-15);
        }
    }

    [Fact]
    public void DoubleNeg()
    {
        var c = -(-F);
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double v1 = F.VectorizedEval(p, new[] { 0, 0 });
            double v2 = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(v1 - v2) < 1e-15);
        }
    }

    [Fact]
    public void DivOneIdentity()
    {
        var c = F / 1.0;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double v1 = F.VectorizedEval(p, new[] { 0, 0 });
            double v2 = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(v1 - v2) < 1e-15);
        }
    }

    [Fact]
    public void ReprAlgebraicResult()
    {
        var c = F + G;
        string r = c.ToReprString();
        Assert.Contains("built=True", r);
        Assert.Contains("ChebyshevApproximation", r);
    }

    [Fact]
    public void StrAlgebraicResult()
    {
        var c = F + G;
        string s = c.ToString()!;
        Assert.Contains("built", s);
        Assert.Contains("0.000s", s);
        Assert.Contains("0 evaluations", s);
    }

    [Fact]
    public void ErrorEstimateOnCombined()
    {
        var c = F + G;
        double est = c.ErrorEstimate();
        Assert.True(est >= 0);
        Assert.True(est < 1e-5, $"Error estimate {est} is not < 1e-5");
    }
}

// ================================================================
// TestCompatibility (6 CA tests)
// ================================================================

public class TestCompatibility
{
    [Fact]
    public void DifferentNNodesRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var a = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var b = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        a.Build(verbose: false);
        b.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Matches("[Nn]ode", ex.Message);
    }

    [Fact]
    public void DifferentDomainRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var a = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var b = new ChebyshevApproximation(F, 1, new[] { new[] { -2.0, 2.0 } }, new[] { 11 });
        a.Build(verbose: false);
        b.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Matches("[Dd]omain", ex.Message);
    }

    [Fact]
    public void DifferentDimensionsRaises()
    {
        double F1(double[] x, object? _) => Math.Sin(x[0]);
        double F2(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]);
        var a = new ChebyshevApproximation(F1, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var b = new ChebyshevApproximation(F2, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        a.Build(verbose: false);
        b.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Matches("[Dd]imension", ex.Message);
    }

    [Fact]
    public void DifferentMaxDerivativeOrderRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var a = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 }, maxDerivativeOrder: 2);
        var b = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 }, maxDerivativeOrder: 3);
        a.Build(verbose: false);
        b.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Contains("max_derivative_order", ex.Message);
    }

    [Fact]
    public void UnbuiltLeftRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var a = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var b = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        b.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = a + b; });
        Assert.Contains("not built", ex.Message);
    }

    [Fact]
    public void UnbuiltRightRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]);
        var a = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        var b = new ChebyshevApproximation(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        a.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = a + b; });
        Assert.Contains("not built", ex.Message);
    }
}

// ================================================================
// TestPortfolioUseCase (3 tests)
// ================================================================

public class TestPortfolioUseCase
{
    private static readonly Lazy<(ChebyshevApproximation call, ChebyshevApproximation put, ChebyshevApproximation straddle)>
        _instruments = new(() =>
        {
            double CallLike(double[] x, object? _) => Math.Max(x[0] - 0.5, 0.0) * Math.Exp(-0.05 * x[1]);
            double PutLike(double[] x, object? _) => Math.Max(0.5 - x[0], 0.0) * Math.Exp(-0.05 * x[1]);
            double Straddle(double[] x, object? _) => Math.Abs(x[0] - 0.5) * Math.Exp(-0.05 * x[1]);

            var domain = new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } };
            int[] ns = { 20, 12 };

            var c = new ChebyshevApproximation(CallLike, 2, domain, ns);
            var p = new ChebyshevApproximation(PutLike, 2, domain, ns);
            var s = new ChebyshevApproximation(Straddle, 2, domain, ns);
            c.Build(verbose: false);
            p.Build(verbose: false);
            s.Build(verbose: false);
            return (c, p, s);
        });

    private static readonly double[][] PortfolioTestPoints =
    {
        new[] { 0.7, 0.5 },
        new[] { 0.3, 0.5 },
        new[] { 0.5, 0.5 },
        new[] { 0.8, 0.2 },
        new[] { 0.2, 0.8 },
        new[] { 0.6, 0.3 },
    };

    [Fact]
    public void WeightedSum3Instruments()
    {
        var (call, put, straddle) = _instruments.Value;
        var portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle;

        foreach (var p in PortfolioTestPoints)
        {
            // Compare portfolio eval to weighted sum of individual evals
            // This tests the algebra, not interpolation accuracy
            double weighted =
                0.4 * call.VectorizedEval(p, new[] { 0, 0 })
                + 0.3 * put.VectorizedEval(p, new[] { 0, 0 })
                + 0.3 * straddle.VectorizedEval(p, new[] { 0, 0 });
            double approx = portfolio.VectorizedEval(p, new[] { 0, 0 });
            Assert.True(Math.Abs(approx - weighted) < 1e-14,
                $"Portfolio algebra failed at [{p[0]}, {p[1]}]");
        }
    }

    [Fact]
    public void PortfolioBatchEval()
    {
        var (call, put, straddle) = _instruments.Value;
        var portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle;

        double[] batch = portfolio.VectorizedEvalBatch(PortfolioTestPoints, new[] { 0, 0 });
        for (int i = 0; i < PortfolioTestPoints.Length; i++)
        {
            double single = portfolio.VectorizedEval(PortfolioTestPoints[i], new[] { 0, 0 });
            Assert.True(Math.Abs(batch[i] - single) < 1e-14);
        }
    }

    [Fact]
    public void PortfolioGreeks()
    {
        var (call, put, straddle) = _instruments.Value;
        var portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle;

        // Test at ITM points (avoid kink at x=0.5 where derivative is discontinuous)
        double[][] itmPoints = { new[] { 0.7, 0.5 }, new[] { 0.3, 0.5 }, new[] { 0.8, 0.2 } };
        foreach (var p in itmPoints)
        {
            double deltaPort = portfolio.VectorizedEval(p, new[] { 1, 0 });
            double deltaCall = call.VectorizedEval(p, new[] { 1, 0 });
            double deltaPut = put.VectorizedEval(p, new[] { 1, 0 });
            double deltaStrad = straddle.VectorizedEval(p, new[] { 1, 0 });
            double weighted = 0.4 * deltaCall + 0.3 * deltaPut + 0.3 * deltaStrad;
            Assert.True(Math.Abs(deltaPort - weighted) < 1e-10,
                $"Delta mismatch at [{p[0]}, {p[1]}]");
        }
    }
}

// ================================================================
// TestSplineArithmetic (ported from Python TestSplineArithmetic)
// ================================================================

public class TestSplineArithmetic
{
    private static ChebyshevSpline SF => TestFixtures.AlgebraSplineF; // |x|
    private static ChebyshevSpline SG => TestFixtures.AlgebraSplineG; // x^2

    // Test points in both pieces (x < 0 and x > 0) within [-1, 1] domain
    private static readonly double[] SplinePts = { -0.8, -0.5, -0.2, 0.2, 0.5, 0.8 };

    [Fact]
    public void Spline_AddValues()
    {
        var c = SF + SG;
        foreach (double x in SplinePts)
        {
            double exact = Math.Abs(x) + x * x;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10,
                $"Spline add failed at {x}");
        }
    }

    [Fact]
    public void Spline_AddDerivative()
    {
        // d/dx(|x| + x^2) on right side = 1 + 2x
        var c = SF + SG;
        foreach (double x in new[] { 0.2, 0.5, 0.8 })
        {
            double exact = 1.0 + 2.0 * x;
            double approx = c.Eval(new[] { x }, new[] { 1 });
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"Spline deriv failed at {x}");
        }
    }

    [Fact]
    public void Spline_SubValues()
    {
        var c = SF - SG;
        foreach (double x in SplinePts)
        {
            double exact = Math.Abs(x) - x * x;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_MulScalar()
    {
        var c = 2.5 * SF;
        foreach (double x in SplinePts)
        {
            double exact = 2.5 * Math.Abs(x);
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_Neg()
    {
        var c = -SF;
        foreach (double x in SplinePts)
        {
            double exact = -Math.Abs(x);
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_TruedivScalar()
    {
        var c = SF / 2.0;
        foreach (double x in SplinePts)
        {
            double exact = Math.Abs(x) / 2.0;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_RmulScalar()
    {
        var c1 = 2.5 * SF;
        var c2 = SF * 2.5;
        foreach (double x in SplinePts)
        {
            double v1 = c1.Eval(new[] { x }, new[] { 0 });
            double v2 = c2.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(v1 - v2) < 1e-15);
        }
    }

    [Fact]
    public void Spline_Isub()
    {
        var c = 1.0 * SF;
        c = c - SG;
        foreach (double x in new[] { -0.5, 0.5 })
        {
            double exact = Math.Abs(x) - x * x;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_Itruediv()
    {
        var c = 1.0 * SF;
        c = c / 2.0;
        foreach (double x in new[] { -0.5, 0.5 })
        {
            double exact = Math.Abs(x) / 2.0;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_EvalBatch()
    {
        var c = SF + SG;
        double[][] pts = SplinePts.Select(x => new[] { x }).ToArray();
        double[] batchVals = c.EvalBatch(pts, new[] { 0 });
        for (int i = 0; i < SplinePts.Length; i++)
        {
            double single = c.Eval(new[] { SplinePts[i] }, new[] { 0 });
            Assert.True(Math.Abs(batchVals[i] - single) < 1e-14);
        }
    }

    [Fact]
    public void Spline_DifferentKnotsRaises()
    {
        // Splines with different knots cannot be combined.
        var s2 = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.5 } });
        s2.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = SF + s2; });
        Assert.Matches("[Kk]not", ex.Message);
    }

    [Fact]
    public void Spline_ResultErrorEstimate()
    {
        var c = SF + SG;
        Assert.True(c.ErrorEstimate() >= 0);
    }

    [Fact]
    public void Spline_ResultNumPieces()
    {
        var c = SF + SG;
        Assert.Equal(SF.NumPieces, c.NumPieces);
    }
}

// ================================================================
// TestSplineExtended (ported from Python TestSplineExtended)
// ================================================================

public class TestSplineExtended
{
    private static ChebyshevSpline SF => TestFixtures.AlgebraSplineF;
    private static ChebyshevSpline SG => TestFixtures.AlgebraSplineG;

    [Fact]
    public void Spline_Iadd()
    {
        var c = 1.0 * SF;
        c = c + SG;
        foreach (double x in new[] { -0.5, 0.5 })
        {
            double exact = Math.Abs(x) + x * x;
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_Imul()
    {
        var c = 1.0 * SF;
        c = c * 3.0;
        foreach (double x in new[] { -0.5, 0.5 })
        {
            double exact = 3.0 * Math.Abs(x);
            double approx = c.Eval(new[] { x }, new[] { 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-10);
        }
    }

    [Fact]
    public void Spline_Serializable()
    {
        var c = SF + SG;
        string path = Path.Combine(Path.GetTempPath(), $"spline_algebra_test_{Guid.NewGuid()}.json");
        try
        {
            c.Save(path);
            var loaded = ChebyshevSpline.Load(path);
            double vOrig = c.Eval(new[] { 0.5 }, new[] { 0 });
            double vLoaded = loaded.Eval(new[] { 0.5 }, new[] { 0 });
            Assert.True(Math.Abs(vOrig - vLoaded) < 1e-15);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Spline_Repr()
    {
        var c = SF + SG;
        string r = c.ToReprString();
        Assert.Contains("built=True", r);
        Assert.Contains("ChebyshevSpline", r);
    }
}

// ================================================================
// TestSplineCompatibility (ported from Python TestCompatibility spline tests)
// ================================================================

public class TestSplineCompatibility
{
    [Fact]
    public void UnbuiltSplineRaises()
    {
        var a = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        var b = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        b.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = a + b; });
        Assert.Contains("not built", ex.Message);
    }
}

// ================================================================
// C#-Specific: Spline Arithmetic Edge Cases
// ================================================================

/// <summary>
/// C#-specific spline arithmetic edge case tests not in the Python baseline.
/// </summary>
public class TestSplineArithmeticCSharpEdgeCases
{
    [Fact]
    public void Test_spline_add_different_knots_throws()
    {
        // Two splines on the same domain but with different knot positions.
        var s1 = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        s1.Build(verbose: false);

        var s2 = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.5 } });
        s2.Build(verbose: false);

        var ex = Assert.Throws<ArgumentException>(() => { var _ = s1 + s2; });
        Assert.Matches("[Kk]not", ex.Message);
    }

    [Fact]
    public void Test_spline_divide_by_zero_scalar()
    {
        // Division by 0.0 should produce Infinity values, not throw.
        var sp = TestFixtures.AlgebraSplineF;
        var result = sp / 0.0;

        // Evaluate at a point where |x| > 0 (non-zero value)
        double val = result.Eval(new[] { 0.5 }, new[] { 0 });
        Assert.True(double.IsInfinity(val),
            $"Expected Infinity from division by zero, got {val}");
    }

    [Fact]
    public void Test_spline_subtract_self_exactly_zero()
    {
        var sp = TestFixtures.AlgebraSplineF;
        var diff = sp - sp;

        // Evaluate at non-knot points
        foreach (double x in new[] { -0.8, -0.3, 0.2, 0.5, 0.9 })
        {
            double val = diff.Eval(new[] { x }, new[] { 0 });
            Assert.Equal(0.0, val);
        }
    }
}

// ================================================================
// C#-specific tests: Arithmetic Edge Cases
// ================================================================

/// <summary>
/// Tests for arithmetic edge cases with special floating-point values.
/// These are C#-specific concerns around IEEE 754 behavior that don't
/// exist in the Python baseline.
/// </summary>
public class TestArithmeticEdgeCases
{
    private static ChebyshevApproximation F => TestFixtures.AlgebraChebF;

    /// <summary>
    /// Dividing by zero scalar should produce Infinity values in the tensor
    /// (not throw an exception), consistent with IEEE 754 behavior.
    /// </summary>
    [Fact]
    public void Test_divide_by_zero_scalar()
    {
        var c = F / 0.0;
        double[] p = { 0.5, 0.3 };
        double val = c.VectorizedEval(p, new[] { 0, 0 });
        Assert.True(double.IsInfinity(val) || double.IsNaN(val),
            $"Expected Infinity or NaN, got {val}");
    }

    /// <summary>
    /// Multiplying by NaN should produce NaN values in the result.
    /// </summary>
    [Fact]
    public void Test_multiply_by_nan()
    {
        var c = F * double.NaN;
        double[] p = { 0.5, 0.3 };
        double val = c.VectorizedEval(p, new[] { 0, 0 });
        Assert.True(double.IsNaN(val), $"Expected NaN, got {val}");
    }

    /// <summary>
    /// (f - f) evaluated at any point should give exactly 0.0.
    /// This verifies pointwise cancellation produces exact zeros in TensorValues.
    /// </summary>
    [Fact]
    public void Test_subtract_self_exactly_zero()
    {
        var c = F - F;
        foreach (var p in AlgebraHelpers.TestPoints2D)
        {
            double val = c.VectorizedEval(p, new[] { 0, 0 });
            Assert.Equal(0.0, val);
        }
    }
}

// ================================================================
// TestSliderArithmetic (ported from Python TestSliderArithmetic)
// ================================================================

public class TestSliderArithmetic
{
    private static ChebyshevSlider SlF => TestFixtures.AlgebraSliderF; // sin(x)+sin(y)+sin(z)
    private static ChebyshevSlider SlG => TestFixtures.AlgebraSliderG; // cos(x)+cos(y)+cos(z)

    private static readonly double[][] SliderTestPoints =
    {
        new[] { 0.5, 0.3, 0.7 },
        new[] { -0.5, 0.8, -0.2 },
    };

    [Fact]
    public void Slider_AddValues()
    {
        var c = SlF + SlG;
        var pts = new[] { new[] { 0.5, 0.3, 0.7 }, new[] { -0.5, 0.8, -0.2 }, new[] { 0.1, -0.3, 0.9 } };
        foreach (var p in pts)
        {
            double exact = 0;
            for (int i = 0; i < p.Length; i++)
                exact += Math.Sin(p[i]) + Math.Cos(p[i]);
            double approx = c.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6,
                $"Slider add failed at [{string.Join(", ", p)}]");
        }
    }

    [Fact]
    public void Slider_SubValues()
    {
        var c = SlF - SlG;
        foreach (var p in SliderTestPoints)
        {
            double exact = 0;
            for (int i = 0; i < p.Length; i++)
                exact += Math.Sin(p[i]) - Math.Cos(p[i]);
            double approx = c.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6);
        }
    }

    [Fact]
    public void Slider_MulScalar()
    {
        var c = 2.0 * SlF;
        foreach (var p in SliderTestPoints)
        {
            double exact = 2.0 * (Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2]));
            double approx = c.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6);
        }
    }

    [Fact]
    public void Slider_MulScalarDerivative()
    {
        // d/dx0(2*sin(x)+...) = 2*cos(x)
        var c = 2.0 * SlF;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = 2.0 * Math.Cos(p[0]);
        double approx = c.Eval(p, new[] { 1, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void Slider_Neg()
    {
        var c = -SlF;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = -(Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2]));
        double approx = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void Slider_TruedivScalar()
    {
        var c = SlF / 2.0;
        foreach (var p in SliderTestPoints)
        {
            double exact = (Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2])) / 2.0;
            double approx = c.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6);
        }
    }

    [Fact]
    public void Slider_RmulScalar()
    {
        var c1 = 2.0 * SlF;
        var c2 = SlF * 2.0;
        double[] p = { 0.5, 0.3, 0.7 };
        double v1 = c1.Eval(p, new[] { 0, 0, 0 });
        double v2 = c2.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(v1 - v2) < 1e-15);
    }

    [Fact]
    public void Slider_AddDerivative()
    {
        // d/dx0(sin(x)+cos(x)+...) = cos(x0) - sin(x0)
        var c = SlF + SlG;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = Math.Cos(p[0]) - Math.Sin(p[0]);
        double approx = c.Eval(p, new[] { 1, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-5);
    }

    [Fact]
    public void Slider_SubDerivative()
    {
        // d/dx0(sin(x)-cos(x)+...) = cos(x0) + sin(x0)
        var c = SlF - SlG;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = Math.Cos(p[0]) + Math.Sin(p[0]);
        double approx = c.Eval(p, new[] { 1, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-5);
    }

    [Fact]
    public void Slider_Iadd()
    {
        var c = 1.0 * SlF;
        c = c + SlG;
        foreach (var p in SliderTestPoints)
        {
            double exact = 0;
            for (int i = 0; i < p.Length; i++)
                exact += Math.Sin(p[i]) + Math.Cos(p[i]);
            double approx = c.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(approx - exact) < 1e-6,
                $"Slider iadd failed at [{string.Join(", ", p)}]");
        }
    }

    [Fact]
    public void Slider_Isub()
    {
        var c = 1.0 * SlF;
        c = c - SlG;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = (Math.Sin(p[0]) - Math.Cos(p[0]))
                     + (Math.Sin(p[1]) - Math.Cos(p[1]))
                     + (Math.Sin(p[2]) - Math.Cos(p[2]));
        double approx = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void Slider_Imul()
    {
        var c = 1.0 * SlF;
        c = c * 2.0;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = 2.0 * (Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2]));
        double approx = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void Slider_Itruediv()
    {
        var c = 1.0 * SlF;
        c = c / 2.0;
        double[] p = { 0.5, 0.3, 0.7 };
        double exact = (Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2])) / 2.0;
        double approx = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void Slider_DifferentPartitionRaises()
    {
        double H(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var s2 = new ChebyshevSlider(H, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        s2.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = SlF + s2; });
        Assert.Matches("[Pp]artition", ex.Message);
    }

    [Fact]
    public void Slider_DifferentPivotRaises()
    {
        double H(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var s2 = new ChebyshevSlider(H, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.5, 0.0, 0.0 });
        s2.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => { var _ = SlF + s2; });
        Assert.Matches("[Pp]ivot", ex.Message);
    }

    [Fact]
    public void Slider_ResultPivotValue()
    {
        var c = SlF + SlG;
        double expected = SlF.PivotValue + SlG.PivotValue;
        Assert.True(Math.Abs(c.PivotValue - expected) < 1e-14);
    }

    [Fact]
    public void Slider_Serializable()
    {
        var c = SlF + SlG;
        string path = Path.Combine(Path.GetTempPath(), $"slider_algebra_test_{Guid.NewGuid()}.json");
        try
        {
            c.Save(path);
            var loaded = ChebyshevSlider.Load(path);
            double[] p = { 0.5, 0.3, 0.7 };
            double vOrig = c.Eval(p, new[] { 0, 0, 0 });
            double vLoaded = loaded.Eval(p, new[] { 0, 0, 0 });
            Assert.True(Math.Abs(vOrig - vLoaded) < 1e-15);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Slider_Repr()
    {
        var c = SlF + SlG;
        string r = c.ToReprString();
        Assert.Contains("built=True", r);
        Assert.Contains("ChebyshevSlider", r);
    }

    [Fact]
    public void Slider_UnbuiltRaises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var a = new ChebyshevSlider(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        var b = new ChebyshevSlider(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        b.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = a + b; });
        Assert.Contains("built", ex.Message);
    }
}

// ================================================================
// C#-Specific: Slider Arithmetic Edge Cases
// ================================================================

/// <summary>
/// C#-specific slider arithmetic edge case tests not in the Python baseline.
/// Covers IEEE 754 behavior, compatibility errors, and cross-type checks.
/// </summary>
public class TestSliderArithmeticCSharpEdgeCases
{
    private static ChebyshevSlider SlF => TestFixtures.AlgebraSliderF;
    private static ChebyshevSlider SlG => TestFixtures.AlgebraSliderG;

    [Fact]
    public void Test_slider_divide_by_zero_scalar()
    {
        var c = SlF / 0.0;
        double[] p = { 0.5, 0.3, 0.7 };
        double val = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(double.IsInfinity(val) || double.IsNaN(val),
            $"Expected Infinity or NaN from division by zero, got {val}");
    }

    [Fact]
    public void Test_slider_multiply_by_nan()
    {
        var c = SlF * double.NaN;
        double[] p = { 0.5, 0.3, 0.7 };
        double val = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(double.IsNaN(val), $"Expected NaN, got {val}");
    }

    [Fact]
    public void Test_slider_subtract_self_exactly_zero()
    {
        var c = SlF - SlF;
        foreach (var p in new[] { new[] { 0.5, 0.3, 0.7 }, new[] { -0.5, 0.8, -0.2 }, new[] { 0.0, 0.0, 0.0 } })
        {
            double val = c.Eval(p, new[] { 0, 0, 0 });
            Assert.Equal(0.0, val);
        }
    }

    [Fact]
    public void Test_slider_multiply_by_zero_gives_zero()
    {
        var c = SlF * 0.0;
        double[] p = { 0.5, 0.3, 0.7 };
        double val = c.Eval(p, new[] { 0, 0, 0 });
        Assert.Equal(0.0, val);
    }

    [Fact]
    public void Test_slider_add_subtract_roundtrip()
    {
        // (F + G) - G should approximately equal F.
        var c = (SlF + SlG) - SlG;
        double[] p = { 0.5, 0.3, 0.7 };
        double expected = SlF.Eval(p, new[] { 0, 0, 0 });
        double actual = c.Eval(p, new[] { 0, 0, 0 });
        Assert.True(Math.Abs(actual - expected) < 1e-10,
            $"(F+G)-G != F: {actual} vs {expected}");
    }

    [Fact]
    public void Test_slider_different_dim_count_raises()
    {
        // Sliders with different dimension counts cannot be combined.
        double F2(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]);
        var sl2d = new ChebyshevSlider(F2, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        sl2d.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = SlF + sl2d; });
        Assert.Contains("mismatch", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_slider_different_nNodes_raises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var s2 = new ChebyshevSlider(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 }, // different from [8,8,8]
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        s2.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = SlF + s2; });
        Assert.Contains("mismatch", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_slider_different_domain_raises()
    {
        double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
        var s2 = new ChebyshevSlider(F, 3,
            new[] { new[] { -2.0, 2.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, // different domain
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        s2.Build(verbose: false);
        var ex = Assert.Throws<InvalidOperationException>(() => { var _ = SlF + s2; });
        Assert.Contains("mismatch", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_slider_scalar_mul_preserves_pivot_value()
    {
        var c = SlF * 3.0;
        Assert.True(Math.Abs(c.PivotValue - SlF.PivotValue * 3.0) < 1e-14);
    }

    [Fact]
    public void Test_slider_neg_pivot_value()
    {
        var c = -SlF;
        Assert.True(Math.Abs(c.PivotValue - (-SlF.PivotValue)) < 1e-14);
    }
}
