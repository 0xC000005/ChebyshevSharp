using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ---------------------------------------------------------------------------
// 3D sin tests
// ---------------------------------------------------------------------------

public class TestSimple3D
{
    [Fact]
    public void Test_price_accuracy()
    {
        var cheb = TestFixtures.ChebSin3D;
        double[] p = [0.1, 0.3, 1.7];
        double exact = TestFixtures.SinSum3D(p, null);
        double approx = cheb.Eval(p, [0, 0, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 1.0);
    }

    [Fact]
    public void Test_derivative_dy()
    {
        var cheb = TestFixtures.ChebSin3D;
        double[] p = [0.1, 0.3, 1.7];
        double exact = Math.Cos(p[1]);
        double approx = cheb.Eval(p, [0, 1, 0]);
        Assert.True(Math.Abs(approx - exact) < 1e-4,
            $"Expected {exact}, got {approx}, diff={Math.Abs(approx - exact)}");
    }

    [Fact]
    public void Test_vectorized_matches_eval()
    {
        var cheb = TestFixtures.ChebSin3D;
        double[] p = [0.1, 0.3, 1.7];
        double v1 = cheb.Eval(p, [0, 0, 0]);
        double v2 = cheb.VectorizedEval(p, [0, 0, 0]);
        Assert.True(Math.Abs(v1 - v2) < 1e-12,
            $"Eval={v1}, VectorizedEval={v2}, diff={Math.Abs(v1 - v2)}");
    }
}

// ---------------------------------------------------------------------------
// 3D Black-Scholes tests
// ---------------------------------------------------------------------------

public class TestBlackScholes3D
{
    [Theory]
    [InlineData(100.0, "ATM")]
    [InlineData(120.0, "ITM")]
    [InlineData(80.0, "OTM")]
    public void Test_price(double S, string name)
    {
        _ = name; // used for test display name
        var cheb = TestFixtures.ChebBs3D;
        double K = 100.0, r = 0.05, q = 0.02;
        double[] p = [S, 1.0, 0.25];
        double exact = BlackScholes.BsCallPrice(S: S, K: K, T: 1.0, r: r, sigma: 0.25, q: q);
        double approx = cheb.Eval(p, [0, 0, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 0.5);
    }

    [Fact]
    public void Test_delta()
    {
        var cheb = TestFixtures.ChebBs3D;
        double K = 100.0, r = 0.05, q = 0.02;
        double[] p = [100, 1.0, 0.25];
        double exact = BlackScholes.BsCallDelta(S: 100, K: K, T: 1.0, r: r, sigma: 0.25, q: q);
        double approx = cheb.Eval(p, [1, 0, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 1.0);
    }
}

// ---------------------------------------------------------------------------
// 5D Black-Scholes tests
// ---------------------------------------------------------------------------

public class TestBlackScholes5D
{
    private const double Q = 0.02;

    [Theory]
    [InlineData(100, 100, 1.0, 0.25, 0.05, "ATM")]
    [InlineData(110, 100, 1.0, 0.25, 0.05, "ITM")]
    [InlineData(90, 100, 1.0, 0.25, 0.05, "OTM")]
    [InlineData(100, 100, 0.5, 0.25, 0.05, "ShortT")]
    [InlineData(100, 100, 1.0, 0.35, 0.05, "HighVol")]
    public void Test_price(double S, double K, double T, double sigma, double r, string name)
    {
        _ = name; // used for test display name
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [S, K, T, sigma, r];
        double exact = BlackScholes.BsCallPrice(S: S, K: K, T: T, r: r, sigma: sigma, q: Q);
        double approx = cheb.Eval(p, [0, 0, 0, 0, 0]);
        double err = Math.Abs(approx - exact) / exact * 100;
        Assert.True(err < 0.01, $"{name}: {err:F4}% error");
    }

    [Fact]
    public void Test_delta()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        double exact = BlackScholes.BsCallDelta(S: 100, K: 100, T: 1.0, r: 0.05, sigma: 0.25, q: Q);
        double approx = cheb.Eval(p, [1, 0, 0, 0, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 1.0);
    }

    [Fact]
    public void Test_gamma()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        double exact = BlackScholes.BsCallGamma(S: 100, K: 100, T: 1.0, r: 0.05, sigma: 0.25, q: Q);
        double approx = cheb.Eval(p, [2, 0, 0, 0, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 1.0);
    }

    [Fact]
    public void Test_vega()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        double exact = BlackScholes.BsCallVega(S: 100, K: 100, T: 1.0, r: 0.05, sigma: 0.25, q: Q);
        double approx = cheb.Eval(p, [0, 0, 0, 1, 0]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 3.0);
    }

    [Fact]
    public void Test_rho()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        double exact = BlackScholes.BsCallRho(S: 100, K: 100, T: 1.0, r: 0.05, sigma: 0.25, q: Q);
        double approx = cheb.Eval(p, [0, 0, 0, 0, 1]);
        TestFixtures.AssertRelativeError(exact, approx, maxRelPercent: 1.0);
    }
}

// ---------------------------------------------------------------------------
// Evaluation method consistency
// ---------------------------------------------------------------------------

public class TestEvalMethods
{
    [Fact]
    public void Test_vectorized_matches_eval()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        double v1 = cheb.Eval(p, [0, 0, 0, 0, 0]);
        double v2 = cheb.VectorizedEval(p, [0, 0, 0, 0, 0]);
        Assert.True(Math.Abs(v1 - v2) < 1e-12,
            $"Eval={v1}, VectorizedEval={v2}, diff={Math.Abs(v1 - v2)}");
    }

    [Fact]
    public void Test_multi_matches_single()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] p = [100, 100, 1.0, 0.25, 0.05];
        int[][] derivs =
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ];
        double[] multi = cheb.VectorizedEvalMulti(p, derivs);
        for (int i = 0; i < derivs.Length; i++)
        {
            double single = cheb.VectorizedEval(p, derivs[i]);
            Assert.True(Math.Abs(multi[i] - single) < 1e-12,
                $"Multi[{i}]={multi[i]}, Single={single}, diff={Math.Abs(multi[i] - single)}");
        }
    }

    [Fact]
    public void Test_vectorized_eval_batch()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[][] points =
        [
            [100, 100, 1.0, 0.25, 0.05],
            [110, 100, 1.0, 0.25, 0.05]
        ];
        double[] results = cheb.VectorizedEvalBatch(points, [0, 0, 0, 0, 0]);
        Assert.Equal(2, results.Length);
        for (int i = 0; i < 2; i++)
        {
            double single = cheb.VectorizedEval(points[i], [0, 0, 0, 0, 0]);
            Assert.True(Math.Abs(results[i] - single) < 1e-12,
                $"Batch[{i}]={results[i]}, Single={single}, diff={Math.Abs(results[i] - single)}");
        }
    }

    [Fact]
    public void Test_node_coincidence()
    {
        // Evaluation at exact Chebyshev nodes should not crash.
        var cheb = TestFixtures.ChebBs5D;
        double[] p = new double[5];
        for (int d = 0; d < 5; d++)
            p[d] = cheb.NodeArrays[d][5];

        double v1 = cheb.Eval(p, [0, 0, 0, 0, 0]);
        double v2 = cheb.VectorizedEval(p, [0, 0, 0, 0, 0]);
        Assert.True(Math.Abs(v1 - v2) < 1e-12,
            $"Eval={v1}, VectorizedEval={v2}, diff={Math.Abs(v1 - v2)}");
    }

    [Fact]
    public void Test_build_required()
    {
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, 1.0 }], [5]);
        Assert.Throws<InvalidOperationException>(() =>
            cheb.VectorizedEval([0.5], [0]));
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

public class TestSerialization
{
    private static readonly double[][] TestPoints =
    [
        [0.1, 0.3, 1.7],
        [-0.5, 0.0, 2.5],
        [0.9, -0.9, 1.1],
        [0.0, 0.0, 2.0],
        [-0.3, 0.7, 2.9]
    ];

    [Fact]
    public void Test_save_load_roundtrip()
    {
        var cheb = TestFixtures.ChebSin3D;
        string path = Path.GetTempFileName();
        try
        {
            cheb.Save(path);
            var loaded = ChebyshevApproximation.Load(path);

            foreach (double[] pt in TestPoints)
            {
                double orig = cheb.VectorizedEval(pt, [0, 0, 0]);
                double rest = loaded.VectorizedEval(pt, [0, 0, 0]);
                Assert.Equal(orig, rest, precision: 15);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_function_is_none_after_load()
    {
        var cheb = TestFixtures.ChebSin3D;
        string path = Path.GetTempFileName();
        try
        {
            cheb.Save(path);
            var loaded = ChebyshevApproximation.Load(path);
            Assert.Null(loaded.Function);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_loaded_state_attributes()
    {
        var cheb = TestFixtures.ChebSin3D;
        string path = Path.GetTempFileName();
        try
        {
            cheb.Save(path);
            var loaded = ChebyshevApproximation.Load(path);

            Assert.NotNull(loaded.TensorValues);
            int expectedTotal = 1;
            for (int d = 0; d < cheb.NumDimensions; d++)
                expectedTotal *= cheb.NNodes[d];
            Assert.Equal(expectedTotal, loaded.TensorValues!.Length);

            Assert.NotNull(loaded.Weights);
            Assert.Equal(cheb.NumDimensions, loaded.Weights!.Length);

            Assert.NotNull(loaded.DiffMatrices);
            Assert.Equal(cheb.NumDimensions, loaded.DiffMatrices!.Length);

            Assert.Equal(cheb.NumDimensions, loaded.NodeArrays.Length);
            for (int d = 0; d < cheb.NumDimensions; d++)
            {
                Assert.Equal(cheb.NodeArrays[d].Length, loaded.NodeArrays[d].Length);
                for (int i = 0; i < cheb.NodeArrays[d].Length; i++)
                    Assert.Equal(cheb.NodeArrays[d][i], loaded.NodeArrays[d][i]);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_save_before_build_raises()
    {
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, 1.0 }], [5]);
        string path = Path.GetTempFileName();
        try
        {
            Assert.Throws<InvalidOperationException>(() => cheb.Save(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

// ---------------------------------------------------------------------------
// Repr / Str
// ---------------------------------------------------------------------------

public class TestRepr
{
    [Fact]
    public void Test_repr_unbuilt()
    {
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 2, [new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 }], [11, 11]);
        string r = cheb.ToReprString();
        Assert.Contains("built=False", r);
        Assert.Contains("dims=2", r);
        Assert.Contains("[11, 11]", r);
    }

    [Fact]
    public void Test_repr_built()
    {
        var cheb = TestFixtures.ChebSin3D;
        string r = cheb.ToReprString();
        Assert.Contains("built=True", r);
        Assert.Contains("dims=3", r);
    }

    [Fact]
    public void Test_str_unbuilt()
    {
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 2, [new[] { 0.0, 1.0 }, new[] { 2.0, 3.0 }], [11, 11]);
        string s = cheb.ToString()!;
        Assert.Contains("not built", s);
        Assert.Contains("2D", s);
        Assert.Contains("[11, 11]", s);
        Assert.Contains("[0, 1]", s);
        Assert.DoesNotContain("Build:", s);
    }

    [Fact]
    public void Test_str_built()
    {
        var cheb = TestFixtures.ChebSin3D;
        string s = cheb.ToString()!;
        Assert.Contains("built", s);
        Assert.Contains("3D", s);
        Assert.Contains("Build:", s);
        Assert.Contains("evaluations", s);
        Assert.Contains("Derivatives:", s);
    }
}

// ---------------------------------------------------------------------------
// Error estimation
// ---------------------------------------------------------------------------

public class TestErrorEstimation
{
    [Fact]
    public void Test_error_estimate_decreases_with_n()
    {
        // Error estimate should decrease monotonically as n increases.
        // Uses even node counts to avoid aliasing: sin(x) has only odd
        // Chebyshev coefficients, so for odd n the last coefficient can
        // land on an even index and be spuriously near zero.
        static double sin1d(double[] x, object? _) => Math.Sin(x[0]);

        int[] nValues = [6, 8, 10, 12, 14];
        double[] estimates = new double[nValues.Length];
        for (int i = 0; i < nValues.Length; i++)
        {
            var cheb = new ChebyshevApproximation(sin1d, 1, [new[] { -1.0, 1.0 }], [nValues[i]]);
            cheb.Build(verbose: false);
            estimates[i] = cheb.ErrorEstimate();
        }

        for (int i = 1; i < estimates.Length; i++)
        {
            Assert.True(estimates[i] < estimates[i - 1],
                $"error_estimate did not decrease: n={nValues[i]} gave {estimates[i]:E2} >= {estimates[i - 1]:E2}");
        }
    }

    [Fact]
    public void Test_error_estimate_tracks_empirical_1d()
    {
        // Error estimate should be within 2 orders of magnitude of empirical error.
        static double sin1d(double[] x, object? _) => Math.Sin(x[0]);

        var cheb = new ChebyshevApproximation(sin1d, 1, [new[] { -1.0, 1.0 }], [10]);
        cheb.Build(verbose: false);

        double estimate = cheb.ErrorEstimate();

        // Compute empirical max error on a dense grid
        double maxErr = 0.0;
        for (int i = 0; i < 1000; i++)
        {
            double x = -1.0 + 2.0 * i / 999.0;
            double exact = Math.Sin(x);
            double approx = cheb.VectorizedEval([x], [0]);
            maxErr = Math.Max(maxErr, Math.Abs(exact - approx));
        }

        Assert.True(estimate > 0.01 * maxErr,
            $"estimate {estimate:E2} < 0.01 * empirical {maxErr:E2}");
        Assert.True(estimate < 1000 * maxErr,
            $"estimate {estimate:E2} > 1000 * empirical {maxErr:E2}");
    }

    [Fact]
    public void Test_error_estimate_sin_3d()
    {
        // 3D sin interpolant should have small but positive error estimate.
        var cheb = TestFixtures.ChebSin3D;
        double est = cheb.ErrorEstimate();
        Assert.True(est > 0, $"Expected positive error estimate, got {est}");
        Assert.True(est < 0.1, $"Expected error estimate < 0.1, got {est}");
    }

    [Fact]
    public void Test_error_estimate_bs_3d()
    {
        // 3D Black-Scholes interpolant error estimate should be bounded.
        var cheb = TestFixtures.ChebBs3D;
        double est = cheb.ErrorEstimate();
        Assert.True(est > 0, $"Expected positive error estimate, got {est}");
        Assert.True(est < 1.0, $"Expected error estimate < 1.0, got {est}");
    }

    [Fact]
    public void Test_error_estimate_bs_5d()
    {
        // 5D Black-Scholes interpolant should have small error estimate.
        var cheb = TestFixtures.ChebBs5D;
        double est = cheb.ErrorEstimate();
        Assert.True(est > 0, $"Expected positive error estimate, got {est}");
        Assert.True(est < 1.0, $"Expected error estimate < 1.0, got {est}");
    }

    [Fact]
    public void Test_error_estimate_not_built()
    {
        // error_estimate() should raise InvalidOperationException if not built.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, 1.0 }], [5]);
        Assert.Throws<InvalidOperationException>(() => cheb.ErrorEstimate());
    }

    [Fact]
    public void Test_chebyshev_coefficients_1d_simple()
    {
        // Chebyshev coefficients of x^2 should be c_0=0.5, c_2=0.5, rest ~0.
        static double xSquared(double[] x, object? _) => x[0] * x[0];

        var cheb = new ChebyshevApproximation(xSquared, 1, [new[] { -1.0, 1.0 }], [10]);
        cheb.Build(verbose: false);

        // Extract 1D values (the only slice)
        double[] values1d = cheb.TensorValues!;
        double[] coeffs = ChebyshevApproximation.ChebyshevCoefficients1D(values1d);

        // x^2 = (T_0 + T_2) / 2, so c_0 = 0.5, c_2 = 0.5
        TestFixtures.AssertClose(0.5, coeffs[0], rtol: 0, atol: 1e-12);
        TestFixtures.AssertClose(0.5, coeffs[2], rtol: 0, atol: 1e-12);

        // All other coefficients should be near zero
        for (int i = 0; i < coeffs.Length; i++)
        {
            if (i != 0 && i != 2)
            {
                Assert.True(Math.Abs(coeffs[i]) < 1e-12,
                    $"c_{i} = {coeffs[i]:E2}, expected ~0");
            }
        }
    }

    [Fact]
    public void Test_error_estimate_high_n_near_zero()
    {
        // With 30 nodes, sin(x) error estimate should be near machine epsilon.
        static double sin1d(double[] x, object? _) => Math.Sin(x[0]);

        var cheb = new ChebyshevApproximation(sin1d, 1, [new[] { -1.0, 1.0 }], [30]);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() < 1e-14,
            $"Expected error estimate < 1e-14, got {cheb.ErrorEstimate()}");
    }
}

// ---------------------------------------------------------------------------
// Additional coverage tests
// ---------------------------------------------------------------------------

public class TestCoverageGaps
{
    [Fact]
    public void Test_eval_before_build_raises()
    {
        // eval() before build() should raise InvalidOperationException.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { -1.0, 1.0 }], [5]);
        Assert.Throws<InvalidOperationException>(() =>
            cheb.Eval([0.5], [0]));
    }

    [Fact]
    public void Test_vectorized_eval_multi_before_build_raises()
    {
        // vectorized_eval_multi() before build() should raise InvalidOperationException.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { -1.0, 1.0 }], [5]);
        Assert.Throws<InvalidOperationException>(() =>
            cheb.VectorizedEvalMulti([0.5], [[0]]));
    }

    [Fact]
    public void Test_get_derivative_id()
    {
        // get_derivative_id() should return the input as-is.
        var cheb = TestFixtures.ChebSin3D;
        int[] derivOrder = [1, 0, 0];
        int[] result = cheb.GetDerivativeId(derivOrder);
        Assert.Equal(derivOrder, result);
    }

    [Fact]
    public void Test_high_dim_str_truncation()
    {
        // ToString() for 7D+ should truncate nodes and domain display.
        static double f(double[] x, object? _)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++) sum += x[i];
            return sum;
        }

        var cheb = new ChebyshevApproximation(
            f, 7,
            [
                new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
                new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
                new[] { -1.0, 1.0 }
            ],
            [3, 3, 3, 3, 3, 3, 3]);

        string s = cheb.ToString()!;
        Assert.Contains("...", s);
    }

    [Fact]
    public void Test_verbose_build()
    {
        // Build with verbose=true should print progress messages.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, [new[] { -1.0, 1.0 }], [5]);

        var sw = new StringWriter();
        var original = Console.Out;
        Console.SetOut(sw);
        try
        {
            cheb.Build(verbose: true);
        }
        finally
        {
            Console.SetOut(original);
        }

        string output = sw.ToString();
        Assert.Contains("Building", output);
        Assert.Contains("Built in", output);
    }

    [Fact]
    public void Test_load_wrong_type_raises()
    {
        // Loading a file with invalid content should throw.
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "this is not json at all");
            Assert.ThrowsAny<Exception>(() => ChebyshevApproximation.Load(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

// ---------------------------------------------------------------------------
// C#-specific tests: Null Safety
// ---------------------------------------------------------------------------

/// <summary>
/// Tests for null argument handling in evaluation methods.
/// These are C#-specific concerns that do not exist in the Python baseline.
/// </summary>
public class TestNullSafety
{
    /// <summary>
    /// VectorizedEval with null derivativeOrder should throw.
    /// </summary>
    [Fact]
    public void Test_null_derivative_order_in_eval()
    {
        var cheb = TestFixtures.ChebSin3D;
        Assert.ThrowsAny<NullReferenceException>(() =>
            cheb.VectorizedEval([0.1, 0.3, 1.7], null!));
    }

    /// <summary>
    /// VectorizedEval with null point should throw.
    /// </summary>
    [Fact]
    public void Test_null_point_in_eval()
    {
        var cheb = TestFixtures.ChebSin3D;
        Assert.ThrowsAny<NullReferenceException>(() =>
            cheb.VectorizedEval(null!, [0, 0, 0]));
    }
}

// ---------------------------------------------------------------------------
// C#-specific tests: Boundary Validation
// ---------------------------------------------------------------------------

/// <summary>
/// Tests for argument validation on point dimensions, derivative orders,
/// and domain configuration. C#-specific because these rely on .NET
/// exception types and array bounds checking.
/// </summary>
public class TestBoundaryValidation
{
    /// <summary>
    /// VectorizedEval with a point of wrong dimension should throw.
    /// </summary>
    [Fact]
    public void Test_eval_point_wrong_dimension()
    {
        var cheb = TestFixtures.ChebSin3D; // 3D interpolant
        // Point has 2 dimensions instead of 3
        Assert.ThrowsAny<Exception>(() =>
            cheb.VectorizedEval([0.1, 0.3], [0, 0, 0]));
    }

    /// <summary>
    /// VectorizedEval with derivativeOrder of wrong length should throw.
    /// </summary>
    [Fact]
    public void Test_derivative_order_wrong_length()
    {
        var cheb = TestFixtures.ChebSin3D; // 3D interpolant
        // derivativeOrder has 2 elements instead of 3
        Assert.ThrowsAny<Exception>(() =>
            cheb.VectorizedEval([0.1, 0.3, 1.7], [0, 0]));
    }

    /// <summary>
    /// VectorizedEval with negative derivative order silently behaves like
    /// derivative order 0, because the implementation guards with
    /// <c>if (deriv &gt; 0)</c>. This test documents that behavior.
    /// </summary>
    [Fact]
    public void Test_negative_derivative_order_behaves_like_zero()
    {
        var cheb = TestFixtures.ChebSin3D;
        double valueWithNeg = cheb.VectorizedEval([0.1, 0.3, 1.7], [-1, 0, 0]);
        double valueWithZero = cheb.VectorizedEval([0.1, 0.3, 1.7], [0, 0, 0]);
        Assert.Equal(valueWithZero, valueWithNeg);
    }

    /// <summary>
    /// Constructor with domain where lo == hi should cause an error
    /// when passed to FromValues (which validates domain bounds).
    /// </summary>
    [Fact]
    public void Test_domain_lo_equals_hi_throws()
    {
        // FromValues validates domain lo &lt; hi
        Assert.ThrowsAny<ArgumentException>(() =>
            ChebyshevApproximation.FromValues(
                new double[] { 1.0, 1.0, 1.0, 1.0, 1.0 },
                1, [new[] { 1.0, 1.0 }], [5]));
    }

    /// <summary>
    /// nNodes = [1] is degenerate but should still work for constant interpolation.
    /// Evaluation at the domain midpoint should return the single node value.
    /// </summary>
    [Fact]
    public void Test_single_node_per_dimension()
    {
        static double f(double[] x, object? _) => 42.0;
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, 1.0 }], [1]);
        cheb.Build(verbose: false);
        double val = cheb.VectorizedEval([0.5], [0]);
        TestFixtures.AssertClose(42.0, val, rtol: 1e-10);
    }
}

// ---------------------------------------------------------------------------
// C#-specific tests: Build State
// ---------------------------------------------------------------------------

/// <summary>
/// Tests for build state consistency and boundary evaluation correctness.
/// </summary>
public class TestBuildState
{
    /// <summary>
    /// Calling Build() twice should produce the same evaluation results (idempotent).
    /// </summary>
    [Fact]
    public void Test_double_build_is_idempotent()
    {
        static double f(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]);
        var cheb = new ChebyshevApproximation(f, 2,
            [new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }], [10, 10]);

        cheb.Build(verbose: false);
        double v1 = cheb.VectorizedEval([0.3, 0.7], [0, 0]);

        cheb.Build(verbose: false);
        double v2 = cheb.VectorizedEval([0.3, 0.7], [0, 0]);

        Assert.Equal(v1, v2);
    }

    /// <summary>
    /// Eval at the exact domain minimum should produce the correct value.
    /// </summary>
    [Fact]
    public void Test_eval_at_domain_min_exact()
    {
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, Math.PI }], [20]);
        cheb.Build(verbose: false);

        double expected = Math.Sin(0.0);
        double actual = cheb.VectorizedEval([0.0], [0]);
        TestFixtures.AssertClose(expected, actual, rtol: 0, atol: 1e-10);
    }

    /// <summary>
    /// Eval at the exact domain maximum should produce the correct value.
    /// </summary>
    [Fact]
    public void Test_eval_at_domain_max_exact()
    {
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, [new[] { 0.0, Math.PI }], [20]);
        cheb.Build(verbose: false);

        double expected = Math.Sin(Math.PI);
        double actual = cheb.VectorizedEval([Math.PI], [0]);
        TestFixtures.AssertClose(expected, actual, rtol: 0, atol: 1e-10);
    }
}

// ---------------------------------------------------------------------------
// C#-specific tests: Thread Safety
// ---------------------------------------------------------------------------

/// <summary>
/// Tests for concurrent evaluation safety. A built interpolant should be
/// safe to evaluate from multiple threads simultaneously since evaluation
/// is a read-only operation on pre-computed data.
/// </summary>
public class TestThreadSafety
{
    /// <summary>
    /// 100 parallel VectorizedEval calls on the same built interpolant should
    /// all produce the correct result (no data races on shared state).
    /// </summary>
    [Fact]
    public void Test_concurrent_eval_thread_safe()
    {
        var cheb = TestFixtures.ChebBs5D;
        double[] point = [100, 100, 1.0, 0.25, 0.05];
        int[] deriv = [0, 0, 0, 0, 0];
        double expected = cheb.VectorizedEval(point, deriv);

        const int numParallel = 100;
        double[] results = new double[numParallel];
        Exception? caughtException = null;

        Parallel.For(0, numParallel, i =>
        {
            try
            {
                results[i] = cheb.VectorizedEval(
                    [100, 100, 1.0, 0.25, 0.05], [0, 0, 0, 0, 0]);
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref caughtException, ex, null);
            }
        });

        Assert.Null(caughtException);
        for (int i = 0; i < numParallel; i++)
        {
            Assert.Equal(expected, results[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// C#-specific tests: Serialization Edge Cases
// ---------------------------------------------------------------------------

/// <summary>
/// Tests for serialization edge cases specific to the C# JSON implementation.
/// </summary>
public class TestSerializationEdgeCases
{
    /// <summary>
    /// Load from a nonexistent file path should throw FileNotFoundException.
    /// </summary>
    [Fact]
    public void Test_load_nonexistent_file_throws()
    {
        string path = Path.Combine(Path.GetTempPath(), $"nonexistent_{Guid.NewGuid()}.json");
        Assert.Throws<FileNotFoundException>(() =>
            ChebyshevApproximation.Load(path));
    }

    /// <summary>
    /// Load from an empty file should throw a JSON deserialization exception.
    /// </summary>
    [Fact]
    public void Test_load_empty_file_throws()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "");
            Assert.ThrowsAny<System.Text.Json.JsonException>(() =>
                ChebyshevApproximation.Load(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    /// <summary>
    /// Save, load, then eval at 10 test points — results must be bit-identical
    /// (exactly equal, not just close) since all data is round-tripped through JSON.
    /// </summary>
    [Fact]
    public void Test_save_load_roundtrip_bit_identical()
    {
        var cheb = TestFixtures.ChebBs5D;
        string path = Path.GetTempFileName();
        try
        {
            cheb.Save(path);
            var loaded = ChebyshevApproximation.Load(path);

            double[][] testPoints =
            [
                [100, 100, 1.0, 0.25, 0.05],
                [85, 95, 0.3, 0.20, 0.03],
                [115, 105, 0.8, 0.30, 0.07],
                [90, 100, 0.5, 0.18, 0.02],
                [110, 95, 0.7, 0.25, 0.06],
                [95, 110, 0.9, 0.32, 0.04],
                [105, 90, 0.4, 0.16, 0.05],
                [100, 100, 0.25, 0.15, 0.01],
                [88, 100, 0.6, 0.28, 0.08],
                [112, 108, 1.0, 0.35, 0.05],
            ];

            foreach (double[] pt in testPoints)
            {
                double orig = cheb.VectorizedEval(pt, [0, 0, 0, 0, 0]);
                double rest = loaded.VectorizedEval(pt, [0, 0, 0, 0, 0]);
                Assert.Equal(orig, rest); // Exact equality, not approximate
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
