namespace ChebyshevSharp.Tests.Helpers;

/// <summary>
/// Shared test fixtures and assertion helpers.
/// Ported from ref/PyChebyshev/tests/conftest.py.
/// </summary>
public static class TestFixtures
{
    // ---------------------------------------------------------------
    // Test functions
    // ---------------------------------------------------------------

    public static double SinSum3D(double[] x, object? _)
    {
        return Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
    }

    // ---------------------------------------------------------------
    // Fixture builders (lazy singletons via Lazy<T>)
    // ---------------------------------------------------------------

    private static readonly Lazy<ChebyshevApproximation> _chebSin3D = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            SinSum3D, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { 1.0, 3.0 } },
            new[] { 10, 8, 4 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _chebBs3D = new(() =>
    {
        double K = 100.0, r = 0.05, q = 0.02;
        var cheb = new ChebyshevApproximation(
            (x, _) => BlackScholes.BsCallPrice(S: x[0], K: K, T: x[1], r: r, sigma: x[2], q: q),
            3,
            new[] { new[] { 50.0, 150.0 }, new[] { 0.1, 2.0 }, new[] { 0.1, 0.5 } },
            new[] { 15, 12, 10 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _chebBs5D = new(() =>
    {
        double q = 0.02;
        var cheb = new ChebyshevApproximation(
            (x, _) => BlackScholes.BsCallPrice(S: x[0], K: x[1], T: x[2], r: x[4], sigma: x[3], q: q),
            5,
            new[] {
                new[] { 80.0, 120.0 }, new[] { 90.0, 110.0 }, new[] { 0.25, 1.0 },
                new[] { 0.15, 0.35 }, new[] { 0.01, 0.08 }
            },
            new[] { 11, 11, 11, 11, 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _algebraChebF = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _algebraChebG = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Cos(x[0]) * Math.Cos(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _extrudeCheb1D = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _extrudeCheb2D = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _calculusChebSin1D = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { 0.0, Math.PI } }, new[] { 15 });
        cheb.Build(verbose: false);
        return cheb;
    });

    private static readonly Lazy<ChebyshevApproximation> _calculusCheb2D = new(() =>
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        return cheb;
    });

    // ---------------------------------------------------------------
    // Spline fixture builders
    // ---------------------------------------------------------------

    private static readonly Lazy<ChebyshevSpline> _splineAbs1D = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);
        return sp;
    });

    private static readonly Lazy<ChebyshevSpline> _splineBs2D = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Max(x[0] - 100.0, 0.0) * Math.Exp(-0.05 * x[1]),
            2,
            new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 1.0 } },
            new[] { 15, 15 },
            new[] { new[] { 100.0 }, Array.Empty<double>() });
        sp.Build(verbose: false);
        return sp;
    });

    private static readonly Lazy<ChebyshevSpline> _algebraSplineF = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);
        return sp;
    });

    private static readonly Lazy<ChebyshevSpline> _algebraSplineG = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0] * x[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);
        return sp;
    });

    private static readonly Lazy<ChebyshevSpline> _calculusSplineAbs = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 11 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);
        return sp;
    });

    // Public accessors
    public static ChebyshevApproximation ChebSin3D => _chebSin3D.Value;
    public static ChebyshevApproximation ChebBs3D => _chebBs3D.Value;
    public static ChebyshevApproximation ChebBs5D => _chebBs5D.Value;
    public static ChebyshevApproximation AlgebraChebF => _algebraChebF.Value;
    public static ChebyshevApproximation AlgebraChebG => _algebraChebG.Value;
    public static ChebyshevApproximation ExtrudeCheb1D => _extrudeCheb1D.Value;
    public static ChebyshevApproximation ExtrudeCheb2D => _extrudeCheb2D.Value;
    public static ChebyshevApproximation CalculusChebSin1D => _calculusChebSin1D.Value;
    public static ChebyshevApproximation CalculusCheb2D => _calculusCheb2D.Value;

    public static ChebyshevSpline SplineAbs1D => _splineAbs1D.Value;
    public static ChebyshevSpline SplineBs2D => _splineBs2D.Value;
    public static ChebyshevSpline AlgebraSplineF => _algebraSplineF.Value;
    public static ChebyshevSpline AlgebraSplineG => _algebraSplineG.Value;
    public static ChebyshevSpline CalculusSplineAbs => _calculusSplineAbs.Value;

    // ---------------------------------------------------------------
    // Slider fixture builders
    // ---------------------------------------------------------------

    private static readonly Lazy<ChebyshevSlider> _algebraSliderF = new(() =>
    {
        var sl = new ChebyshevSlider(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sl.Build(verbose: false);
        return sl;
    });

    private static readonly Lazy<ChebyshevSlider> _algebraSliderG = new(() =>
    {
        var sl = new ChebyshevSlider(
            (x, _) => Math.Cos(x[0]) + Math.Cos(x[1]) + Math.Cos(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sl.Build(verbose: false);
        return sl;
    });

    public static ChebyshevSlider AlgebraSliderF => _algebraSliderF.Value;
    public static ChebyshevSlider AlgebraSliderG => _algebraSliderG.Value;

    // ---------------------------------------------------------------
    // TensorTrain fixture builders
    // ---------------------------------------------------------------

    public static readonly double[][] TT_5D_BS_DOMAIN = new[] {
        new[] { 80.0, 120.0 },
        new[] { 90.0, 110.0 },
        new[] { 0.25, 1.0 },
        new[] { 0.15, 0.35 },
        new[] { 0.01, 0.08 },
    };

    private static readonly int[] TT_5D_BS_NODES = { 11, 11, 11, 11, 11 };

    /// <summary>5D BS call price: V(S, K, T, sigma, r), q=0.02.</summary>
    public static double Bs5DFunc(double[] x) =>
        BlackScholes.BsCallPrice(S: x[0], K: x[1], T: x[2], r: x[4], sigma: x[3], q: 0.02);

    private static readonly Lazy<ChebyshevTT> _ttSin3D = new(() =>
    {
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 },
            maxRank: 5);
        tt.Build(verbose: false, seed: 42);
        return tt;
    });

    private static readonly Lazy<ChebyshevTT> _ttSin3DSvd = new(() =>
    {
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");
        return tt;
    });

    private static readonly Lazy<ChebyshevTT> _ttBs5D = new(() =>
    {
        var tt = new ChebyshevTT(
            Bs5DFunc, 5,
            TT_5D_BS_DOMAIN,
            TT_5D_BS_NODES,
            maxRank: 15, maxSweeps: 5);
        tt.Build(verbose: false, seed: 42);
        return tt;
    });

    public static ChebyshevTT TtSin3D => _ttSin3D.Value;
    public static ChebyshevTT TtSin3DSvd => _ttSin3DSvd.Value;
    public static ChebyshevTT TtBs5D => _ttBs5D.Value;

    // ---------------------------------------------------------------
    // Assertion helpers
    // ---------------------------------------------------------------

    public static void AssertClose(double expected, double actual, double rtol = 1e-10, double atol = 1e-14)
    {
        double tolerance = atol + rtol * Math.Abs(expected);
        Assert.True(Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:E15}, got {actual:E15} (diff={Math.Abs(expected - actual):E2}, rtol={rtol}, atol={atol})");
    }

    public static void AssertRelativeError(double expected, double actual, double maxRelPercent)
    {
        double relErr = Math.Abs(actual - expected) / Math.Abs(expected) * 100.0;
        Assert.True(relErr < maxRelPercent,
            $"Relative error {relErr:F4}% exceeds {maxRelPercent}% (expected={expected}, actual={actual})");
    }

    /// <summary>
    /// Build same function via Build() and via Nodes()+FromValues(), return both.
    /// </summary>
    public static (ChebyshevApproximation a, ChebyshevApproximation b) BuildBothWaysApprox(
        Func<double[], object?, double> func, int ndim, double[][] domain, int[] nNodes,
        int maxDerivativeOrder = 2)
    {
        var chebA = new ChebyshevApproximation(func, ndim, domain, nNodes, maxDerivativeOrder);
        chebA.Build(verbose: false);

        var info = ChebyshevApproximation.Nodes(ndim, domain, nNodes);
        double[] values = new double[info.FullGrid.Length];
        for (int i = 0; i < info.FullGrid.Length; i++)
        {
            double[] point = info.FullGrid[i];
            values[i] = func(point, null);
        }
        var chebB = ChebyshevApproximation.FromValues(values, ndim, domain, nNodes, maxDerivativeOrder);

        return (chebA, chebB);
    }

    /// <summary>
    /// Build same function via Build() and via Nodes()+FromValues() for ChebyshevSpline, return both.
    /// </summary>
    public static (ChebyshevSpline a, ChebyshevSpline b) BuildBothWaysSpline(
        Func<double[], object?, double> func, int ndim, double[][] domain, int[] nNodes,
        double[][] knots, int maxDerivativeOrder = 2)
    {
        var splineA = new ChebyshevSpline(func, ndim, domain, nNodes, knots, maxDerivativeOrder);
        splineA.Build(verbose: false);

        var info = ChebyshevSpline.Nodes(ndim, domain, nNodes, knots);
        double[][] pieceValues = new double[info.NumPieces][];
        for (int p = 0; p < info.NumPieces; p++)
        {
            var piece = info.Pieces[p];
            pieceValues[p] = new double[piece.FullGrid.Length];
            for (int i = 0; i < piece.FullGrid.Length; i++)
                pieceValues[p][i] = func(piece.FullGrid[i], null);
        }
        var splineB = ChebyshevSpline.FromValues(pieceValues, ndim, domain, nNodes, knots, maxDerivativeOrder);

        return (splineA, splineB);
    }
}
