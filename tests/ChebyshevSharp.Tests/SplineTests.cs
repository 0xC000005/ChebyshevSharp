using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// Shared spline fixtures (lazy singletons, matching conftest.py)
// ======================================================================

internal static class SplineFixtures
{
    // spline_abs_1d: 1D |x| with knot at x=0, domain [-1,1], 15 nodes
    private static readonly Lazy<ChebyshevSpline> _splineAbs1D = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [15], [[0.0]]);
        sp.Build(verbose: false);
        return sp;
    });

    // spline_bs_2d: 2D discounted call payoff max(S-K,0)*exp(-rT) with knot at K=100
    private static readonly Lazy<ChebyshevSpline> _splineBs2D = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Max(x[0] - 100.0, 0.0) * Math.Exp(-0.05 * x[1]),
            2, [[80.0, 120.0], [0.25, 1.0]], [15, 15], [[100.0], []]);
        sp.Build(verbose: false);
        return sp;
    });

    public static ChebyshevSpline SplineAbs1D => _splineAbs1D.Value;
    public static ChebyshevSpline SplineBs2D => _splineBs2D.Value;
}

// ======================================================================
// TestConstruction
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline construction.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestConstruction.
/// </summary>
public class TestSplineConstruction
{
    [Fact]
    public void ValidConstruction_CorrectNumPiecesAndShape()
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + x[1],
            2, [[-1.0, 1.0], [0.0, 2.0]], [10, 10], [[0.0], [1.0]]);

        Assert.Equal(4, sp.NumPieces);  // 2 x 2
        Assert.Equal(new[] { 2, 2 }, sp.Shape);
    }

    [Fact]
    public void KnotOutsideDomain_Raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => x[0],
                1, [[-1.0, 1.0]], [10], [[2.0]]));

        Assert.Contains("not strictly inside", ex.Message);
    }

    [Fact]
    public void UnsortedKnots_Raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => x[0],
                1, [[-1.0, 1.0]], [10], [[0.5, -0.5]]));

        Assert.Contains("sorted", ex.Message);
    }

    [Fact]
    public void EmptyKnots_GivesOnePiece()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0] + x[1],
            2, [[-1.0, 1.0], [-1.0, 1.0]], [10, 10], [[], []]);

        Assert.Equal(1, sp.NumPieces);
        Assert.Equal(new[] { 1, 1 }, sp.Shape);
    }
}

// ======================================================================
// TestBuildRequired
// ======================================================================

/// <summary>
/// Tests that operations before Build() raise InvalidOperationException.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestBuildRequired.
/// </summary>
public class TestSplineBuildRequired
{
    [Fact]
    public void EvalBeforeBuild_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            sp.Eval([0.5], [0]));

        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ErrorEstimateBeforeBuild_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            sp.ErrorEstimate());

        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

// ======================================================================
// Test1DAccuracy
// ======================================================================

/// <summary>
/// Tests for 1D spline accuracy on |x| with knot at 0.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: Test1DAccuracy.
/// </summary>
public class TestSpline1DAccuracy
{
    [Fact]
    public void AbsValue_AtHalf()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double val = sp.Eval([0.5], [0]);
        Assert.True(Math.Abs(val - 0.5) < 1e-10,
            $"Expected 0.5, got {val}");
    }

    [Fact]
    public void AbsValue_AtNeg()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double val = sp.Eval([-0.3], [0]);
        Assert.True(Math.Abs(val - 0.3) < 1e-10,
            $"Expected 0.3, got {val}");
    }

    [Fact]
    public void LeftPiece_Derivative()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double deriv = sp.Eval([-0.5], [1]);
        Assert.True(Math.Abs(deriv - (-1.0)) < 1e-8,
            $"Expected -1.0, got {deriv}");
    }

    [Fact]
    public void RightPiece_Derivative()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double deriv = sp.Eval([0.5], [1]);
        Assert.True(Math.Abs(deriv - 1.0) < 1e-8,
            $"Expected 1.0, got {deriv}");
    }

    [Fact]
    public void SplineVsGlobal_Accuracy()
    {
        var sp = SplineFixtures.SplineAbs1D;

        // Build global interpolant with same total nodes
        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [30]);
        globalCheb.Build(verbose: false);

        double[] testPts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.6];
        double splineMaxErr = 0.0;
        double globalMaxErr = 0.0;
        foreach (double x in testPts)
        {
            double exact = Math.Abs(x);
            double splineErr = Math.Abs(sp.Eval([x], [0]) - exact);
            double globalErr = Math.Abs(globalCheb.VectorizedEval([x], [0]) - exact);
            splineMaxErr = Math.Max(splineMaxErr, splineErr);
            globalMaxErr = Math.Max(globalMaxErr, globalErr);
        }

        // Global should have algebraic error > 1e-3 for |x|
        Assert.True(globalMaxErr > 1e-3,
            $"Global error {globalMaxErr:E2} should be > 1e-3");
        // Spline should have spectral (near-zero) error
        Assert.True(splineMaxErr < 1e-10,
            $"Spline error {splineMaxErr:E2} should be < 1e-10");
    }
}

// ======================================================================
// Test2DAccuracy
// ======================================================================

/// <summary>
/// Tests for 2D spline accuracy on discounted call payoff.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: Test2DAccuracy.
/// </summary>
public class TestSpline2DAccuracy
{
    [Fact]
    public void ItmValue()
    {
        var sp = SplineFixtures.SplineBs2D;
        double S = 110.0, T = 0.5;
        double exact = Math.Max(S - 100.0, 0.0) * Math.Exp(-0.05 * T);
        double val = sp.Eval([S, T], [0, 0]);
        Assert.True(Math.Abs(val - exact) < 1e-8,
            $"Expected {exact}, got {val}");
    }

    [Fact]
    public void OtmValue()
    {
        var sp = SplineFixtures.SplineBs2D;
        double S = 90.0, T = 0.5;
        double exact = Math.Max(S - 100.0, 0.0) * Math.Exp(-0.05 * T);
        double val = sp.Eval([S, T], [0, 0]);
        Assert.True(Math.Abs(val - exact) < 1e-8,
            $"Expected {exact}, got {val}");
        Assert.True(Math.Abs(val) < 1e-8,
            $"OTM value should be ~0, got {val}");
    }

    [Fact]
    public void ItmDerivativeWrtS()
    {
        var sp = SplineFixtures.SplineBs2D;
        double S = 110.0, T = 0.5;
        double deriv = sp.Eval([S, T], [1, 0]);
        double expected = Math.Exp(-0.05 * T);  // d/dS of (S-100)*exp(-rT) = exp(-rT)
        Assert.True(Math.Abs(deriv - expected) < 1e-8,
            $"Expected {expected}, got {deriv}");
    }

    [Fact]
    public void OtmDerivativeWrtS()
    {
        var sp = SplineFixtures.SplineBs2D;
        double S = 90.0, T = 0.5;
        double deriv = sp.Eval([S, T], [1, 0]);
        Assert.True(Math.Abs(deriv) < 1e-8,
            $"OTM derivative should be ~0, got {deriv}");
    }
}

// ======================================================================
// TestBatchEval
// ======================================================================

/// <summary>
/// Tests for batch evaluation of ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestBatchEval.
/// </summary>
public class TestSplineBatchEval
{
    [Fact]
    public void BatchMatchesLoop()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double[][] pts = [[-0.7], [-0.3], [0.1], [0.5], [0.9]];
        double[] batchResults = sp.EvalBatch(pts, [0]);
        for (int i = 0; i < pts.Length; i++)
        {
            double loopResult = sp.Eval(pts[i], [0]);
            Assert.True(Math.Abs(batchResults[i] - loopResult) < 1e-12,
                $"Point {pts[i][0]}: batch={batchResults[i]}, loop={loopResult}");
        }
    }

    [Fact]
    public void BatchSpanningMultiplePieces()
    {
        var sp = SplineFixtures.SplineBs2D;
        double[][] pts =
        [
            [110.0, 0.5],  // ITM
            [90.0, 0.5],   // OTM
            [105.0, 1.0],  // ITM
            [85.0, 0.25],  // OTM
        ];
        double[] batchResults = sp.EvalBatch(pts, [0, 0]);
        for (int i = 0; i < pts.Length; i++)
        {
            double exact = Math.Max(pts[i][0] - 100.0, 0.0) * Math.Exp(-0.05 * pts[i][1]);
            Assert.True(Math.Abs(batchResults[i] - exact) < 1e-8,
                $"Point [{pts[i][0]}, {pts[i][1]}]: batch={batchResults[i]}, exact={exact}");
        }
    }

    [Fact]
    public void SinglePieceBatch()
    {
        var sp = SplineFixtures.SplineAbs1D;
        // All points in right piece (x > 0)
        double[][] pts = [[0.1], [0.3], [0.5], [0.7], [0.9]];
        double[] batchResults = sp.EvalBatch(pts, [0]);
        for (int i = 0; i < pts.Length; i++)
        {
            Assert.True(Math.Abs(batchResults[i] - pts[i][0]) < 1e-10,
                $"Expected {pts[i][0]}, got {batchResults[i]}");
        }
    }
}

// ======================================================================
// TestEvalMulti
// ======================================================================

/// <summary>
/// Tests for multi-output evaluation of ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestEvalMulti.
/// </summary>
public class TestSplineEvalMulti
{
    [Fact]
    public void EvalMultiMatchesIndividual()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double[] pt = [0.5];
        int[][] derivs = [[0], [1]];
        double[] multi = sp.EvalMulti(pt, derivs);
        double[] singles = derivs.Select(d => sp.Eval(pt, d)).ToArray();
        for (int i = 0; i < multi.Length; i++)
        {
            Assert.True(Math.Abs(multi[i] - singles[i]) < 1e-12,
                $"deriv {i}: multi={multi[i]}, single={singles[i]}");
        }
    }

    [Fact]
    public void ValueAndDerivative()
    {
        var sp = SplineFixtures.SplineBs2D;
        double[] pt = [110.0, 0.5];
        int[][] derivs = [[0, 0], [1, 0], [0, 1]];
        double[] multi = sp.EvalMulti(pt, derivs);
        double[] singles = derivs.Select(d => sp.Eval(pt, d)).ToArray();
        for (int i = 0; i < multi.Length; i++)
        {
            Assert.True(Math.Abs(multi[i] - singles[i]) < 1e-12,
                $"deriv {i}: multi={multi[i]}, single={singles[i]}");
        }
    }
}

// ======================================================================
// TestDerivatives
// ======================================================================

/// <summary>
/// Tests for derivative evaluation of ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestDerivatives.
/// </summary>
public class TestSplineDerivatives
{
    [Fact]
    public void AnalyticalVsFD_WithinPiece()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double[] pt = [0.5];
        double h = 1e-6;
        double valUp = sp.Eval([pt[0] + h], [0]);
        double valDn = sp.Eval([pt[0] - h], [0]);
        double fdDeriv = (valUp - valDn) / (2 * h);
        double analytical = sp.Eval(pt, [1]);
        Assert.True(Math.Abs(analytical - fdDeriv) < 1e-5,
            $"Analytical={analytical}, FD={fdDeriv}");
    }

    [Fact]
    public void DerivativeAtKnot_Raises()
    {
        var sp = SplineFixtures.SplineAbs1D;
        var ex = Assert.Throws<ArgumentException>(() =>
            sp.Eval([0.0], [1]));
        Assert.Contains("not defined", ex.Message);
    }

    [Fact]
    public void PureValueAtKnot_Ok()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double val = sp.Eval([0.0], [0]);
        Assert.True(Math.Abs(val - 0.0) < 1e-10,
            $"Expected 0.0, got {val}");
    }

    [Fact]
    public void SecondDerivativeWithinPiece()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double d2 = sp.Eval([0.5], [2]);
        Assert.True(Math.Abs(d2) < 1e-8,
            $"Expected ~0 for second derivative of |x| on x>0 piece, got {d2}");
    }
}

// ======================================================================
// TestMultipleKnots
// ======================================================================

/// <summary>
/// Tests for splines with multiple knots.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestMultipleKnots.
/// </summary>
public class TestSplineMultipleKnots
{
    private static readonly Lazy<ChebyshevSpline> _splineMultiKnot = new(() =>
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0] - 1.0) + Math.Abs(x[0] + 1.0) + Math.Abs(x[1]),
            2,
            [[-3.0, 3.0], [-2.0, 2.0]],
            [10, 10],
            [[-1.0, 1.0], [0.0]]);
        sp.Build(verbose: false);
        return sp;
    });

    private static ChebyshevSpline SplineMultiKnot => _splineMultiKnot.Value;

    [Fact]
    public void CorrectPieceCount()
    {
        var sp = SplineMultiKnot;
        Assert.Equal(6, sp.NumPieces);
        Assert.Equal(new[] { 3, 2 }, sp.Shape);
    }

    [Fact]
    public void RoutingCorrectness()
    {
        var sp = SplineMultiKnot;
        double[][] testPts =
        [
            [-2.0, -1.0],  // dim0: left of -1, dim1: below 0
            [-2.0, 1.0],   // dim0: left of -1, dim1: above 0
            [0.0, -1.0],   // dim0: between -1 and 1, dim1: below 0
            [0.0, 1.0],    // dim0: between -1 and 1, dim1: above 0
            [2.0, -1.0],   // dim0: right of 1, dim1: below 0
            [2.0, 1.0],    // dim0: right of 1, dim1: above 0
        ];
        foreach (var pt in testPts)
        {
            double exact = Math.Abs(pt[0] - 1.0) + Math.Abs(pt[0] + 1.0) + Math.Abs(pt[1]);
            double approx = sp.Eval(pt, [0, 0]);
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"Error at [{pt[0]}, {pt[1]}]: approx={approx}, exact={exact}");
        }
    }

    [Fact]
    public void AccuracyAcrossAllPieces()
    {
        var sp = SplineMultiKnot;
        var rng = new Random(42);
        double maxErr = 0.0;
        for (int i = 0; i < 50; i++)
        {
            double x0 = -3.0 + 6.0 * rng.NextDouble();
            double x1 = -2.0 + 4.0 * rng.NextDouble();
            double exact = Math.Abs(x0 - 1.0) + Math.Abs(x0 + 1.0) + Math.Abs(x1);
            double approx = sp.Eval([x0, x1], [0, 0]);
            maxErr = Math.Max(maxErr, Math.Abs(approx - exact));
        }
        Assert.True(maxErr < 1e-8,
            $"Max error across 50 random points: {maxErr:E2}");
    }
}

// ======================================================================
// TestErrorEstimate
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline.ErrorEstimate().
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestErrorEstimate.
/// </summary>
public class TestSplineErrorEstimate
{
    [Fact]
    public void ErrorEstimatePositive()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double est = sp.ErrorEstimate();
        Assert.True(est > 0, $"Error estimate should be > 0, got {est}");
    }

    [Fact]
    public void SplineErrorLessThanGlobal()
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [15], [[0.0]]);
        sp.Build(verbose: false);

        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [15]);
        globalCheb.Build(verbose: false);

        Assert.True(sp.ErrorEstimate() < globalCheb.ErrorEstimate(),
            $"Spline error est {sp.ErrorEstimate():E2} should be < global {globalCheb.ErrorEstimate():E2}");
    }
}

// ======================================================================
// TestSerialization
// ======================================================================

/// <summary>
/// Tests for save/load round-trip of ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestSerialization.
/// </summary>
public class TestSplineSerialization
{
    [Fact]
    public void SaveLoadRoundtrip()
    {
        var sp = SplineFixtures.SplineAbs1D;
        string path = Path.GetTempFileName();
        try
        {
            sp.Save(path);
            var loaded = ChebyshevSpline.Load(path);

            double[][] testPts = [[-0.7], [-0.3], [0.1], [0.5], [0.9]];
            foreach (var pt in testPts)
            {
                double orig = sp.Eval(pt, [0]);
                double rest = loaded.Eval(pt, [0]);
                Assert.Equal(orig, rest, 15);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void FunctionIsNullAfterLoad()
    {
        var sp = SplineFixtures.SplineAbs1D;
        string path = Path.GetTempFileName();
        try
        {
            sp.Save(path);
            var loaded = ChebyshevSpline.Load(path);
            Assert.Null(loaded.Function);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void WrongType_Raises()
    {
        string path = Path.GetTempFileName();
        try
        {
            // Write a JSON object with wrong Type field
            File.WriteAllText(path, """{"Type":"NotASpline"}""");
            var ex = Assert.Throws<InvalidOperationException>(() =>
                ChebyshevSpline.Load(path));
            Assert.Contains("ChebyshevSpline", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SavedFileContainsVersion()
    {
        // Port of test_version_mismatch_warning: C# uses JSON (not pickle),
        // so we verify the version field is included in the serialized state.
        var sp = SplineFixtures.SplineAbs1D;
        string path = Path.GetTempFileName();
        try
        {
            sp.Save(path);
            string json = File.ReadAllText(path);
            Assert.Contains("\"Version\"", json);
            Assert.Contains("0.1.0", json);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// TestReprStr
// ======================================================================

/// <summary>
/// Tests for ToReprString() and ToString() of ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestReprStr.
/// </summary>
public class TestSplineReprStr
{
    [Fact]
    public void ReprContainsKeyInfo()
    {
        var sp = SplineFixtures.SplineAbs1D;
        string r = sp.ToReprString();
        Assert.Contains("dims=1", r);
        Assert.Contains("pieces=2", r);
        Assert.Contains("built=True", r);
    }

    [Fact]
    public void StrContainsKnotsPiecesDomain()
    {
        var sp = SplineFixtures.SplineAbs1D;
        string s = sp.ToString()!;
        Assert.Contains("Knots:", s);
        Assert.Contains("Pieces:", s);
        Assert.Contains("Domain:", s);
        Assert.Contains("Error est:", s);
    }
}

// ======================================================================
// TestSplineVsGlobal
// ======================================================================

/// <summary>
/// Tests comparing ChebyshevSpline accuracy vs global ChebyshevApproximation.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestSplineVsGlobal.
/// </summary>
public class TestSplineVsGlobal
{
    [Fact]
    public void AbsX_GlobalPoor_SplineExcellent()
    {
        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [15]);
        globalCheb.Build(verbose: false);

        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [15], [[0.0]]);
        sp.Build(verbose: false);

        double[] testPts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8];
        double globalMaxErr = testPts.Max(x =>
            Math.Abs(globalCheb.VectorizedEval([x], [0]) - Math.Abs(x)));
        double splineMaxErr = testPts.Max(x =>
            Math.Abs(sp.Eval([x], [0]) - Math.Abs(x)));

        Assert.True(globalMaxErr > 1e-2,
            $"Global error {globalMaxErr:E2} should be > 1e-2");
        Assert.True(splineMaxErr < 1e-10,
            $"Spline error {splineMaxErr:E2} should be < 1e-10");
    }

    [Fact]
    public void CallPayoffAccuracy()
    {
        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Max(x[0] - 100.0, 0.0),
            1, [[80.0, 120.0]], [15]);
        globalCheb.Build(verbose: false);

        var sp = new ChebyshevSpline(
            (x, _) => Math.Max(x[0] - 100.0, 0.0),
            1, [[80.0, 120.0]], [15], [[100.0]]);
        sp.Build(verbose: false);

        double[] testPts = [85, 90, 95, 105, 110, 115];
        double globalMaxErr = testPts.Max(x =>
            Math.Abs(globalCheb.VectorizedEval([x], [0]) - Math.Max(x - 100.0, 0.0)));
        double splineMaxErr = testPts.Max(x =>
            Math.Abs(sp.Eval([x], [0]) - Math.Max(x - 100.0, 0.0)));

        Assert.True(splineMaxErr < globalMaxErr,
            $"Spline error {splineMaxErr:E2} should be < global error {globalMaxErr:E2}");
    }

    [Fact]
    public void SmoothFunction_SimilarAccuracy()
    {
        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, [[-1.0, 1.0]], [15]);
        globalCheb.Build(verbose: false);

        var sp = new ChebyshevSpline(
            (x, _) => Math.Sin(x[0]),
            1, [[-1.0, 1.0]], [15], [[0.0]]);
        sp.Build(verbose: false);

        double[] testPts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8];
        double globalMaxErr = testPts.Max(x =>
            Math.Abs(globalCheb.VectorizedEval([x], [0]) - Math.Sin(x)));
        double splineMaxErr = testPts.Max(x =>
            Math.Abs(sp.Eval([x], [0]) - Math.Sin(x)));

        // Both should be < 1e-10 for smooth function with 15 nodes
        Assert.True(globalMaxErr < 1e-10,
            $"Global error {globalMaxErr:E2} should be < 1e-10");
        Assert.True(splineMaxErr < 1e-10,
            $"Spline error {splineMaxErr:E2} should be < 1e-10");
    }

    [Fact]
    public void HigherOrderKink_MaxXSquared()
    {
        var globalCheb = new ChebyshevApproximation(
            (x, _) => Math.Max(x[0], 0.0) * Math.Max(x[0], 0.0),
            1, [[-1.0, 1.0]], [15]);
        globalCheb.Build(verbose: false);

        var sp = new ChebyshevSpline(
            (x, _) => Math.Max(x[0], 0.0) * Math.Max(x[0], 0.0),
            1, [[-1.0, 1.0]], [15], [[0.0]]);
        sp.Build(verbose: false);

        double[] testPts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8];
        Func<double, double> f = x => Math.Max(x, 0.0) * Math.Max(x, 0.0);
        double globalMaxErr = testPts.Max(x =>
            Math.Abs(globalCheb.VectorizedEval([x], [0]) - f(x)));
        double splineMaxErr = testPts.Max(x =>
            Math.Abs(sp.Eval([x], [0]) - f(x)));

        Assert.True(splineMaxErr < globalMaxErr,
            $"Spline error {splineMaxErr:E2} should be < global error {globalMaxErr:E2}");
    }
}

// ======================================================================
// TestBatchEvalDerivatives
// ======================================================================

/// <summary>
/// Tests for batch evaluation with derivatives.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestBatchEvalDerivatives.
/// </summary>
public class TestSplineBatchEvalDerivatives
{
    [Fact]
    public void BatchWithFirstDerivative()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double[][] pts = [[-0.7], [-0.3], [0.3], [0.7]];
        double[] batchResults = sp.EvalBatch(pts, [1]);
        for (int i = 0; i < pts.Length; i++)
        {
            double loopResult = sp.Eval(pts[i], [1]);
            Assert.True(Math.Abs(batchResults[i] - loopResult) < 1e-12,
                $"Point {pts[i][0]}: batch={batchResults[i]}, loop={loopResult}");
        }
    }

    [Fact]
    public void BatchDerivativesSpanningPieces()
    {
        var sp = SplineFixtures.SplineBs2D;
        double[][] pts =
        [
            [110.0, 0.5],  // ITM: dV/dS ~ exp(-rT)
            [90.0, 0.5],   // OTM: dV/dS ~ 0
        ];
        double[] batchResults = sp.EvalBatch(pts, [1, 0]);
        Assert.True(Math.Abs(batchResults[0] - Math.Exp(-0.05 * 0.5)) < 1e-8,
            $"ITM derivative: expected {Math.Exp(-0.05 * 0.5)}, got {batchResults[0]}");
        Assert.True(Math.Abs(batchResults[1]) < 1e-8,
            $"OTM derivative: expected ~0, got {batchResults[1]}");
    }

    [Fact]
    public void BatchAtKnotBoundaryValues()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double[][] pts = [[0.0], [0.5], [-0.5]];
        double[] results = sp.EvalBatch(pts, [0]);
        double[] expected = [0.0, 0.5, 0.5];
        for (int i = 0; i < pts.Length; i++)
        {
            Assert.True(Math.Abs(results[i] - expected[i]) < 1e-10,
                $"Point {pts[i][0]}: expected {expected[i]}, got {results[i]}");
        }
    }
}

// ======================================================================
// TestDomainBoundary
// ======================================================================

/// <summary>
/// Tests for evaluation at domain boundaries.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestDomainBoundary.
/// </summary>
public class TestSplineDomainBoundary
{
    [Fact]
    public void EvalAtLeftBoundary()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double val = sp.Eval([-1.0], [0]);
        Assert.True(Math.Abs(val - 1.0) < 1e-10,
            $"Expected 1.0, got {val}");
    }

    [Fact]
    public void EvalAtRightBoundary()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double val = sp.Eval([1.0], [0]);
        Assert.True(Math.Abs(val - 1.0) < 1e-10,
            $"Expected 1.0, got {val}");
    }

    [Fact]
    public void DerivativeAtDomainBoundary()
    {
        var sp = SplineFixtures.SplineAbs1D;
        double deriv = sp.Eval([0.99], [1]);
        Assert.True(Math.Abs(deriv - 1.0) < 1e-6,
            $"Expected ~1.0, got {deriv}");
    }
}

// ======================================================================
// TestProperties
// ======================================================================

/// <summary>
/// Tests for ChebyshevSpline properties.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestProperties.
/// </summary>
public class TestSplineProperties
{
    [Fact]
    public void TotalBuildEvals_1D()
    {
        var sp = SplineFixtures.SplineAbs1D;
        Assert.Equal(2 * 15, sp.TotalBuildEvals);  // 2 pieces x 15 nodes
    }

    [Fact]
    public void TotalBuildEvals_2D()
    {
        var sp = SplineFixtures.SplineBs2D;
        Assert.Equal(2 * 15 * 15, sp.TotalBuildEvals);  // 2 pieces x 15x15
    }

    [Fact]
    public void BuildTimePositive()
    {
        var sp = SplineFixtures.SplineAbs1D;
        Assert.True(sp.BuildTime > 0,
            $"BuildTime should be > 0, got {sp.BuildTime}");
    }
}

// ======================================================================
// TestBuildRequiredExtended
// ======================================================================

/// <summary>
/// Extended tests for build-required guards.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestBuildRequiredExtended.
/// </summary>
public class TestSplineBuildRequiredExtended
{
    [Fact]
    public void EvalMultiBeforeBuild_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            sp.EvalMulti([0.5], [[0]]));

        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EvalBatchBeforeBuild_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            sp.EvalBatch([[0.5]], [0]));

        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SaveBeforeBuild_Raises()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        string path = Path.GetTempFileName();
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                sp.Save(path));
            Assert.Contains("build", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// TestReprStrUnbuilt
// ======================================================================

/// <summary>
/// Tests for repr/str of unbuilt ChebyshevSpline.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestReprStrUnbuilt.
/// </summary>
public class TestSplineReprStrUnbuilt
{
    [Fact]
    public void ReprUnbuilt()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        string r = sp.ToReprString();
        Assert.Contains("built=False", r);
    }

    [Fact]
    public void StrUnbuilt()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x[0],
            1, [[-1.0, 1.0]], [10], [[0.0]]);

        string s = sp.ToString()!;
        Assert.Contains("not built", s);
        Assert.DoesNotContain("Error est:", s);
        Assert.DoesNotContain("Build:", s);
    }
}

// ======================================================================
// TestVerboseAndDisplay
// ======================================================================

/// <summary>
/// Tests for verbose build and display.
/// Ported from ref/PyChebyshev/tests/test_spline.py :: TestVerboseAndDisplay.
/// </summary>
public class TestSplineVerboseAndDisplay
{
    [Fact]
    public void VerboseBuild()
    {
        var sp = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, [[-1.0, 1.0]], [5], [[0.0]]);

        // Capture console output
        var sw = new StringWriter();
        var original = Console.Out;
        Console.SetOut(sw);
        try
        {
            sp.Build(verbose: true);
        }
        finally
        {
            Console.SetOut(original);
        }
        string output = sw.ToString();
        Assert.Contains("Building", output);
        Assert.Contains("Piece", output);
        Assert.Contains("Build complete", output);
    }

    [Fact]
    public void HighDimStrTruncation()
    {
        var sp = new ChebyshevSpline(
            (x, _) => x.Sum(v => Math.Abs(v)),
            7,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [3, 3, 3, 3, 3, 3, 3],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]);

        string s = sp.ToString()!;
        Assert.Contains("...]", s);
        Assert.Contains("...", s);
    }
}

// ======================================================================
// C#-Specific Tests: Knot Validation
// ======================================================================

/// <summary>
/// C#-specific tests for knot validation edge cases not covered in the
/// Python baseline.
/// </summary>
public class TestSplineKnotValidation
{
    [Fact]
    public void Test_duplicate_knots_throws()
    {
        // Duplicate knots are caught by either the "sorted" check (since
        // equal values violate strict ordering) or the explicit "duplicates"
        // check. Either way, ArgumentException is thrown.
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]),
                1, [[-1.0, 1.0]], [15], [[0.5, 0.5]]));

        // The validation catches this via the sorted check (0.5 <= 0.5)
        Assert.True(
            ex.Message.Contains("sorted", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("duplicate", StringComparison.OrdinalIgnoreCase),
            $"Expected message about sorted or duplicate, got: {ex.Message}");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void Test_knots_at_domain_boundary_throws(double knotValue)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => x[0],
                1, [[0.0, 1.0]], [10], [[knotValue]]));

        Assert.Contains("not strictly inside", ex.Message);
    }

    [Fact]
    public void Test_many_knots_correct_piece_count()
    {
        // 10 knots on [-1, 1] creates 11 pieces along dim 0
        double[] knots = new double[10];
        for (int i = 0; i < 10; i++)
            knots[i] = -0.9 + 0.18 * i; // evenly spaced in [-0.9, 0.9]

        var sp = new ChebyshevSpline(
            (x, _) => Math.Sin(x[0]),
            1, [[-1.0, 1.0]], [5], [knots]);

        Assert.Equal(new[] { 11 }, sp.Shape);
        Assert.Equal(11, sp.NumPieces);
    }
}

// ======================================================================
// C#-Specific Tests: Empty Knots / Single-Piece
// ======================================================================

/// <summary>
/// C#-specific test verifying that a spline with no knots is equivalent
/// to a plain ChebyshevApproximation.
/// </summary>
public class TestSplineEmptyKnotsEquivalence
{
    [Fact]
    public void Test_empty_knots_equivalent_to_approx()
    {
        static double f(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[0]);

        var sp = new ChebyshevSpline(f, 1, [[-1.0, 1.0]], [15], [[]]);
        sp.Build(verbose: false);

        var ca = new ChebyshevApproximation(f, 1, [[-1.0, 1.0]], [15]);
        ca.Build(verbose: false);

        var rng = new Random(42);
        for (int i = 0; i < 10; i++)
        {
            double x = -1.0 + 2.0 * rng.NextDouble();
            double spVal = sp.Eval([x], [0]);
            double caVal = ca.VectorizedEval([x], [0]);
            Assert.True(Math.Abs(spVal - caVal) < 1e-13,
                $"Spline vs Approx mismatch at x={x}: {spVal} vs {caVal}");
        }
    }
}

// ======================================================================
// C#-Specific Tests: Piece Routing
// ======================================================================

/// <summary>
/// C#-specific test for piece routing near knot boundaries.
/// </summary>
public class TestSplinePieceRouting
{
    [Fact]
    public void Test_point_slightly_below_knot_routes_left()
    {
        // For a smooth function, values just below and just above the knot
        // should be very close (continuity).
        var sp = SplineFixtures.SplineAbs1D;
        double knot = 0.0;
        double eps = 1e-12;

        double valLeft = sp.Eval([knot - eps], [0]);
        double valRight = sp.Eval([knot + eps], [0]);

        // Both should be close to |knot| = 0
        Assert.True(Math.Abs(valLeft) < 1e-8,
            $"Value at knot - eps should be near 0, got {valLeft}");
        Assert.True(Math.Abs(valRight) < 1e-8,
            $"Value at knot + eps should be near 0, got {valRight}");

        // And close to each other (continuity)
        Assert.True(Math.Abs(valLeft - valRight) < 1e-8,
            $"Values across knot should be close: {valLeft} vs {valRight}");
    }
}

// ======================================================================
// C#-Specific Tests: Thread Safety
// ======================================================================

/// <summary>
/// C#-specific test for thread safety of concurrent Eval calls.
/// </summary>
public class TestSplineThreadSafety
{
    [Fact]
    public void Test_concurrent_spline_eval_thread_safe()
    {
        var sp = SplineFixtures.SplineAbs1D;
        int numTasks = 100;
        var results = new double[numTasks];
        var expected = new double[numTasks];
        var rng = new Random(42);

        // Pre-compute random points and expected values
        double[] points = new double[numTasks];
        for (int i = 0; i < numTasks; i++)
        {
            points[i] = -1.0 + 2.0 * rng.NextDouble();
            expected[i] = Math.Abs(points[i]);
        }

        Parallel.For(0, numTasks, i =>
        {
            results[i] = sp.Eval([points[i]], [0]);
        });

        for (int i = 0; i < numTasks; i++)
        {
            Assert.True(Math.Abs(results[i] - expected[i]) < 1e-10,
                $"Thread safety failure at point {points[i]}: got {results[i]}, expected {expected[i]}");
        }
    }
}

// ======================================================================
// C#-Specific Tests: Serialization Edge Cases
// ======================================================================

/// <summary>
/// C#-specific serialization edge case tests.
/// </summary>
public class TestSplineSerializationEdgeCases
{
    [Fact]
    public void Test_spline_save_load_roundtrip_bit_identical()
    {
        var sp = SplineFixtures.SplineBs2D;
        string path = Path.GetTempFileName();
        try
        {
            sp.Save(path);
            var loaded = ChebyshevSpline.Load(path);

            var rng = new Random(123);
            for (int i = 0; i < 10; i++)
            {
                double s = 80.0 + 40.0 * rng.NextDouble();
                double t = 0.25 + 0.75 * rng.NextDouble();
                double origVal = sp.Eval([s, t], [0, 0]);
                double loadedVal = loaded.Eval([s, t], [0, 0]);
                // Bit-identical: exactly equal, not just close
                Assert.Equal(origVal, loadedVal);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_spline_load_nonexistent_file_throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ChebyshevSpline.Load("nonexistent_file_that_does_not_exist.json"));
    }

    [Fact]
    public void Test_load_approx_as_spline_throws()
    {
        // Save a ChebyshevApproximation, then try ChebyshevSpline.Load() on it.
        var ca = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, [[-1.0, 1.0]], [10]);
        ca.Build(verbose: false);

        string path = Path.GetTempFileName();
        try
        {
            ca.Save(path);
            // ChebyshevApproximation doesn't write a "Type" field, so the
            // deserialized SplineSerializationState.Type will be "ChebyshevSpline"
            // (its default), but PieceStates will be empty, causing a failure.
            // Either way, it should not silently succeed.
            Assert.ThrowsAny<Exception>(() => ChebyshevSpline.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// C#-Specific Tests: Null Safety
// ======================================================================

/// <summary>
/// C#-specific null safety test.
/// </summary>
public class TestSplineNullSafety
{
    [Fact]
    public void Test_null_knots_throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            new ChebyshevSpline(
                (x, _) => x[0],
                1, [[-1.0, 1.0]], [10], null!));
    }
}
