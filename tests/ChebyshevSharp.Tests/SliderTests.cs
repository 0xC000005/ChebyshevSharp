using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// Shared slider test functions
// ======================================================================

internal static class SliderTestFunctions
{
    /// <summary>x0^3 * x1^2 + x2 -- coupling only within (x0, x1).</summary>
    public static double Coupled2dPlus1d(double[] x, object? _)
    {
        return x[0] * x[0] * x[0] * x[1] * x[1] + x[2];
    }

    /// <summary>Sum of squares: x0^2 + x1^2 + x2^2 + x3^2 + x4^2.</summary>
    public static double Polynomial5d(double[] x, object? _)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
            sum += x[i] * x[i];
        return sum;
    }
}

// ======================================================================
// Shared slider fixtures (lazy singletons)
// ======================================================================

internal static class SliderFixtures
{
    // slider_sin_3d: sin(x)+sin(y)+sin(z), partition [[0],[1],[2]], pivot [0,0,2]
    private static readonly Lazy<ChebyshevSlider> _sliderSin3D = new(() =>
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [1.0, 3.0]],
            [12, 10, 10],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 2.0]);
        slider.Build(verbose: false);
        return slider;
    });

    // slider_coupled: x0^3*x1^2+x2, partition [[0,1],[2]], pivot [0,0,0]
    private static readonly Lazy<ChebyshevSlider> _sliderCoupled = new(() =>
    {
        var slider = new ChebyshevSlider(
            SliderTestFunctions.Coupled2dPlus1d, 3,
            [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]],
            [12, 12, 8],
            partition: [[0, 1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);
        return slider;
    });

    // slider_5d: sum(x[i]^2), partition [[0],[1],[2],[3],[4]], pivot [0,0,0,0,0]
    private static readonly Lazy<ChebyshevSlider> _slider5D = new(() =>
    {
        var slider = new ChebyshevSlider(
            SliderTestFunctions.Polynomial5d, 5,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [6, 6, 6, 6, 6],
            partition: [[0], [1], [2], [3], [4]],
            pivotPoint: [0.0, 0.0, 0.0, 0.0, 0.0]);
        slider.Build(verbose: false);
        return slider;
    });

    // slider_bs_3d: Black-Scholes price, partition [[0],[1],[2]], pivot [100,1,0.3]
    private static readonly Lazy<ChebyshevSlider> _sliderBs3D = new(() =>
    {
        double K = 100.0, r = 0.05, q = 0.02;
        var slider = new ChebyshevSlider(
            (x, _) => BlackScholes.BsCallPrice(S: x[0], K: K, T: x[1], r: r, sigma: x[2], q: q),
            3,
            [[50.0, 150.0], [0.1, 2.0], [0.1, 0.5]],
            [15, 12, 10],
            partition: [[0], [1], [2]],
            pivotPoint: [100.0, 1.0, 0.3]);
        slider.Build(verbose: false);
        return slider;
    });

    public static ChebyshevSlider SliderSin3D => _sliderSin3D.Value;
    public static ChebyshevSlider SliderCoupled => _sliderCoupled.Value;
    public static ChebyshevSlider Slider5D => _slider5D.Value;
    public static ChebyshevSlider SliderBs3D => _sliderBs3D.Value;
}

// ======================================================================
// TestAdditivelySeparable
// ======================================================================

/// <summary>
/// sin(x0) + sin(x1) + sin(x2) with partition [[0],[1],[2]].
/// Ported from test_slider.py :: TestAdditivelySeparable (5 tests).
/// </summary>
public class TestAdditivelySeparable
{
    [Fact]
    public void TestFunctionValue()
    {
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        double expected = Math.Sin(0.5) + Math.Sin(-0.3) + Math.Sin(1.7);
        double result = slider.Eval(pt, [0, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-9,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestFirstDerivativeDim0()
    {
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        double expected = Math.Cos(0.5); // d/dx0 of sin(x0)
        double result = slider.Eval(pt, [1, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-8,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestFirstDerivativeDim2()
    {
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        double expected = Math.Cos(1.7); // d/dx2 of sin(x2)
        double result = slider.Eval(pt, [0, 0, 1]);
        Assert.True(Math.Abs(result - expected) < 1e-6,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestSecondDerivative()
    {
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        double expected = -Math.Sin(0.5); // d^2/dx0^2 of sin(x0)
        double result = slider.Eval(pt, [2, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-6,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestBuildEvalsIsSum()
    {
        var slider = SliderFixtures.SliderSin3D;
        // partition [[0],[1],[2]] with n_nodes [12,10,10]: total = 12 + 10 + 10 = 32
        Assert.Equal(12 + 10 + 10, slider.TotalBuildEvals);
    }
}

// ======================================================================
// TestPartiallyCoupled
// ======================================================================

/// <summary>
/// x0^3 * x1^2 + x2 with partition [[0,1],[2]].
/// Ported from test_slider.py :: TestPartiallyCoupled (4 tests).
/// </summary>
public class TestPartiallyCoupled
{
    [Fact]
    public void TestFunctionValue()
    {
        var slider = SliderFixtures.SliderCoupled;
        double[] pt = [1.0, 0.5, -1.0];
        double expected = 1.0 * 1.0 * 1.0 * 0.5 * 0.5 + (-1.0); // 0.25 - 1.0 = -0.75
        double result = slider.Eval(pt, [0, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-8,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestDerivativeX0()
    {
        // d/dx0 of x0^3*x1^2 + x2 = 3*x0^2*x1^2
        var slider = SliderFixtures.SliderCoupled;
        double[] pt = [1.0, 0.5, -1.0];
        double expected = 3.0 * 1.0 * 1.0 * 0.5 * 0.5; // 0.75
        double result = slider.Eval(pt, [1, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-6,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestDerivativeX2()
    {
        // d/dx2 of x0^3*x1^2 + x2 = 1
        var slider = SliderFixtures.SliderCoupled;
        double[] pt = [1.0, 0.5, -1.0];
        double result = slider.Eval(pt, [0, 0, 1]);
        Assert.True(Math.Abs(result - 1.0) < 1e-8,
            $"Expected 1.0, got {result}");
    }

    [Fact]
    public void TestBuildEvals()
    {
        var slider = SliderFixtures.SliderCoupled;
        // partition [[0,1],[2]] with n_nodes [12,12,8]: total = 12*12 + 8 = 152
        Assert.Equal(12 * 12 + 8, slider.TotalBuildEvals);
    }
}

// ======================================================================
// TestHighDimensional
// ======================================================================

/// <summary>
/// 5D sum-of-squares with partition [[0],[1],[2],[3],[4]].
/// Ported from test_slider.py :: TestHighDimensional (4 tests).
/// </summary>
public class TestHighDimensional
{
    [Fact]
    public void TestFunctionValue()
    {
        var slider = SliderFixtures.Slider5D;
        double[] pt = [0.5, -0.3, 0.7, -0.1, 0.9];
        double expected = 0.5 * 0.5 + 0.3 * 0.3 + 0.7 * 0.7 + 0.1 * 0.1 + 0.9 * 0.9;
        double result = slider.Eval(pt, [0, 0, 0, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-10,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestDerivative()
    {
        var slider = SliderFixtures.Slider5D;
        double[] pt = [0.5, -0.3, 0.7, -0.1, 0.9];
        // d/dx3 of (x3^2) = 2*x3 = -0.2
        double result = slider.Eval(pt, [0, 0, 0, 1, 0]);
        Assert.True(Math.Abs(result - (-0.2)) < 1e-8,
            $"Expected -0.2, got {result}");
    }

    [Fact]
    public void TestBuildEvals()
    {
        var slider = SliderFixtures.Slider5D;
        // 5 slides of 6 nodes each: 5 * 6 = 30, NOT 6^5 = 7776
        Assert.Equal(30, slider.TotalBuildEvals);
    }

    [Fact]
    public void TestEvalMulti()
    {
        var slider = SliderFixtures.Slider5D;
        double[] pt = [0.5, -0.3, 0.7, -0.1, 0.9];
        int[][] derivs =
        [
            [0, 0, 0, 0, 0], // value
            [1, 0, 0, 0, 0], // d/dx0
            [0, 0, 0, 0, 1], // d/dx4
        ];
        double[] results = slider.EvalMulti(pt, derivs);
        double expectedVal = 0.5 * 0.5 + 0.3 * 0.3 + 0.7 * 0.7 + 0.1 * 0.1 + 0.9 * 0.9;
        Assert.True(Math.Abs(results[0] - expectedVal) < 1e-10,
            $"Expected {expectedVal}, got {results[0]}");
        Assert.True(Math.Abs(results[1] - 2.0 * 0.5) < 1e-8,
            $"Expected {2.0 * 0.5}, got {results[1]}"); // 2*x0 = 1.0
        Assert.True(Math.Abs(results[2] - 2.0 * 0.9) < 1e-8,
            $"Expected {2.0 * 0.9}, got {results[2]}"); // 2*x4 = 1.8
    }
}

// ======================================================================
// TestCrossGroupDerivatives
// ======================================================================

/// <summary>
/// Cross-group mixed partials should return 0.
/// Ported from test_slider.py :: TestCrossGroupDerivatives (2 tests).
/// </summary>
public class TestCrossGroupDerivatives
{
    // Uses a separate slider with [10,10,10] nodes (not the shared fixture)
    private static readonly Lazy<ChebyshevSlider> _slider3D = new(() =>
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [1.0, 3.0]],
            [10, 10, 10],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 2.0]);
        slider.Build(verbose: false);
        return slider;
    });

    [Fact]
    public void TestCrossGroupMixedPartialIsZero()
    {
        // d^2f/dx0*dx1 = 0 when x0 and x1 are in different slides
        var slider = _slider3D.Value;
        double[] pt = [0.5, -0.3, 1.7];
        double result = slider.Eval(pt, [1, 1, 0]);
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void TestCrossGroupThreeWayIsZero()
    {
        // d^3f/dx0*dx1*dx2 = 0 when all dims in separate slides
        var slider = _slider3D.Value;
        double[] pt = [0.5, -0.3, 1.7];
        double result = slider.Eval(pt, [1, 1, 1]);
        Assert.Equal(0.0, result);
    }
}

// ======================================================================
// TestMultiDimSlideDerivative
// ======================================================================

/// <summary>
/// Derivatives within a multi-dimensional slide group.
/// Ported from test_slider.py :: TestMultiDimSlideDerivative (2 tests).
/// </summary>
public class TestMultiDimSlideDerivative
{
    // Uses the same slider as TestPartiallyCoupled but from a separate lazy
    // to match the Python fixture scoping exactly.
    private static readonly Lazy<ChebyshevSlider> _sliderGrouped = new(() =>
    {
        var slider = new ChebyshevSlider(
            SliderTestFunctions.Coupled2dPlus1d, 3,
            [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]],
            [12, 12, 8],
            partition: [[0, 1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);
        return slider;
    });

    [Fact]
    public void TestMixedPartialWithinGroup()
    {
        // d^2f/dx0*dx1 of x0^3*x1^2 = 6*x0^2*x1, within slide [0,1]
        var slider = _sliderGrouped.Value;
        double[] pt = [1.0, 0.5, -1.0];
        double expected = 6.0 * 1.0 * 1.0 * 0.5; // 3.0
        double result = slider.Eval(pt, [1, 1, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-4,
            $"Expected {expected}, got {result}");
    }

    [Fact]
    public void TestCrossGroupMixedPartialGrouped()
    {
        // d^2f/dx0*dx2 = 0 when x0 in [0,1] and x2 in [2]
        var slider = _sliderGrouped.Value;
        double[] pt = [1.0, 0.5, -1.0];
        double result = slider.Eval(pt, [1, 0, 1]);
        Assert.Equal(0.0, result);
    }
}

// ======================================================================
// TestValidation
// ======================================================================

/// <summary>
/// Constructor and pre-build validation.
/// Ported from test_slider.py :: TestValidation (3 tests).
/// </summary>
public class TestSliderValidation
{
    [Fact]
    public void TestInvalidPartitionMissingDim()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(
                TestFixtures.SinSum3D, 3,
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                [5, 5, 5],
                partition: [[0], [1]], // missing dim 2
                pivotPoint: [0.0, 0.0, 0.0]));
        Assert.Contains("Partition must cover", ex.Message);
    }

    [Fact]
    public void TestInvalidPartitionDuplicateDim()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(
                TestFixtures.SinSum3D, 3,
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                [5, 5, 5],
                partition: [[0, 1], [1, 2]], // dim 1 appears twice
                pivotPoint: [0.0, 0.0, 0.0]));
        Assert.Contains("Partition must cover", ex.Message);
    }

    [Fact]
    public void TestEvalBeforeBuild()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            slider.Eval([0.5, 0.5, 0.5], [0, 0, 0]));
        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

// ======================================================================
// TestBlackScholesSliding
// ======================================================================

/// <summary>
/// BS 3D with partition [[0],[1],[2]] -- shows cross-coupling error.
/// Ported from test_slider.py :: TestBlackScholesSliding (2 tests).
/// </summary>
public class TestBlackScholesSliding
{
    [Fact]
    public void TestAtPivotNearExact()
    {
        // At the pivot point, the slider should be very close to exact.
        var slider = SliderFixtures.SliderBs3D;
        double[] pivot = [100.0, 1.0, 0.3];
        double expected = BlackScholes.BsCallPrice(S: 100.0, K: 100.0, T: 1.0, r: 0.05, sigma: 0.3, q: 0.02);
        double result = slider.Eval(pivot, [0, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-3,
            $"Expected {expected}, got {result}, diff={Math.Abs(result - expected)}");
    }

    [Fact]
    public void TestAwayFromPivotHasError()
    {
        // Away from pivot, sliding has error due to cross-coupling.
        var slider = SliderFixtures.SliderBs3D;
        double[] pt = [110.0, 0.5, 0.2];
        double expected = BlackScholes.BsCallPrice(S: 110.0, K: 100.0, T: 0.5, r: 0.05, sigma: 0.2, q: 0.02);
        double result = slider.Eval(pt, [0, 0, 0]);
        // Error exists but should be within ~100% for this separable approximation
        double relError = Math.Abs(result - expected) / Math.Abs(expected);
        Assert.True(relError < 1.0,
            $"Relative error {relError:F4} exceeds 1.0 (expected={expected}, got={result})");
    }
}

// ======================================================================
// TestSliderSerialization
// ======================================================================

/// <summary>
/// Tests for save/load round-trip on ChebyshevSlider.
/// Ported from test_slider.py :: TestSliderSerialization (6 tests).
/// </summary>
public class TestSliderSerialization
{
    private static readonly double[][] TestPoints =
    [
        [0.5, -0.3, 1.7],
        [-0.8, 0.6, 2.5],
        [0.0, 0.0, 2.0],
        [0.9, -0.9, 1.1],
        [-0.3, 0.7, 2.9],
    ];

    [Fact]
    public void TestSaveLoadRoundtrip()
    {
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            foreach (var pt in TestPoints)
            {
                double orig = slider.Eval(pt, [0, 0, 0]);
                double rest = loaded.Eval(pt, [0, 0, 0]);
                Assert.Equal(orig, rest, 15);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestDerivativeAfterLoad()
    {
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            foreach (var pt in TestPoints)
            {
                double orig = slider.Eval(pt, [1, 0, 0]);
                double rest = loaded.Eval(pt, [1, 0, 0]);
                Assert.Equal(orig, rest, 15);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestFunctionIsNullAfterLoad()
    {
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            Assert.Null(loaded.Function);
            foreach (var slide in loaded.Slides)
                Assert.Null(slide.Function);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestSliderInternalState()
    {
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            // DimToSlide should match
            Assert.Equal(slider.DimToSlide.Count, loaded.DimToSlide.Count);
            foreach (var kvp in slider.DimToSlide)
                Assert.Equal(kvp.Value, loaded.DimToSlide[kvp.Key]);

            // PivotValue should match
            Assert.Equal(slider.PivotValue, loaded.PivotValue);

            // Partition should match
            Assert.Equal(slider.Partition.Length, loaded.Partition.Length);
            for (int i = 0; i < slider.Partition.Length; i++)
                Assert.Equal(slider.Partition[i], loaded.Partition[i]);

            // PivotPoint should match
            Assert.Equal(slider.PivotPoint, loaded.PivotPoint);

            // Built flag
            Assert.True(loaded.Built);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestSaveBeforeBuildRaises()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);

        string path = Path.GetTempFileName();
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                slider.Save(path));
            Assert.Contains("unbuilt", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestPathlibPath()
    {
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.Combine(Path.GetTempPath(), $"slider_test_{Guid.NewGuid()}.json");
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);
            double[] pt = [0.5, -0.3, 1.7];
            double orig = slider.Eval(pt, [0, 0, 0]);
            double rest = loaded.Eval(pt, [0, 0, 0]);
            Assert.Equal(orig, rest, 15);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}

// ======================================================================
// TestSliderRepr
// ======================================================================

/// <summary>
/// Tests for ToReprString() and ToString() of ChebyshevSlider.
/// Ported from test_slider.py :: TestSliderRepr (4 tests).
/// </summary>
public class TestSliderRepr
{
    [Fact]
    public void TestReprUnbuilt()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [11, 11, 11],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);

        string r = slider.ToReprString();
        Assert.Contains("built=False", r);
        Assert.Contains("dims=3", r);
        Assert.Contains("slides=3", r);
        Assert.Contains("partition=", r);
    }

    [Fact]
    public void TestReprBuilt()
    {
        var slider = SliderFixtures.SliderSin3D;
        string r = slider.ToReprString();
        Assert.Contains("built=True", r);
        Assert.Contains("dims=3", r);
    }

    [Fact]
    public void TestStrUnbuilt()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [1.0, 3.0]],
            [11, 11, 11],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 2.0]);

        string s = slider.ToString()!;
        Assert.Contains("not built", s);
        Assert.Contains("Partition:", s);
        Assert.Contains("Pivot:", s);
        Assert.Contains("Nodes:", s);
        Assert.Contains("Domain:", s);
        Assert.DoesNotContain("Slides:", s); // no slide details when unbuilt
    }

    [Fact]
    public void TestStrBuilt()
    {
        var slider = SliderFixtures.SliderSin3D;
        string s = slider.ToString()!;
        Assert.Contains("built", s);
        Assert.Contains("Partition:", s);
        Assert.Contains("Pivot:", s);
        Assert.Contains("Slides:", s);
        Assert.Contains("[0] dims", s);
    }
}

// ======================================================================
// TestSliderErrorEstimation
// ======================================================================

/// <summary>
/// Tests for error estimation on ChebyshevSlider.
/// Ported from test_slider.py :: TestSliderErrorEstimation (4 tests).
/// </summary>
public class TestSliderErrorEstimation
{
    [Fact]
    public void TestSliderErrorEstimateSeparable()
    {
        // Separable sin sum with partition [[0],[1],[2]] should have tiny error.
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [12, 12, 12],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);

        double est = slider.ErrorEstimate();
        Assert.True(est > 0, $"Expected positive error estimate, got {est}");
        Assert.True(est < 1e-7, $"Expected error < 1e-7, got {est}");
    }

    [Fact]
    public void TestSliderErrorEstimateCoupled()
    {
        // Coupled polynomial with appropriate partition should have small error.
        var slider = new ChebyshevSlider(
            SliderTestFunctions.Coupled2dPlus1d, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [12, 12, 6],
            partition: [[0, 1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);

        double est = slider.ErrorEstimate();
        Assert.True(est > 0, $"Expected positive error estimate, got {est}");
        Assert.True(est < 1e-3, $"Expected error < 1e-3, got {est}");
    }

    [Fact]
    public void TestSliderErrorEstimateEqualsSlideSum()
    {
        // Slider error_estimate() should equal sum of individual slide estimates.
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [12, 12, 12],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);

        double manualSum = 0;
        foreach (var slide in slider.Slides)
            manualSum += slide.ErrorEstimate();

        Assert.Equal(manualSum, slider.ErrorEstimate());
    }

    [Fact]
    public void TestSliderErrorEstimateNotBuilt()
    {
        // error_estimate() should raise if not built.
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            slider.ErrorEstimate());
        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

// ======================================================================
// TestCoverageGaps
// ======================================================================

/// <summary>
/// Additional coverage tests.
/// Ported from test_slider.py :: TestCoverageGaps (4 tests).
/// </summary>
public class TestSliderCoverageGaps
{
    [Fact]
    public void TestVerboseBuild()
    {
        // Build with verbose=true should print progress messages.
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);

        var sw = new StringWriter();
        var original = Console.Out;
        Console.SetOut(sw);
        try
        {
            slider.Build(verbose: true);
        }
        finally
        {
            Console.SetOut(original);
        }

        string output = sw.ToString();
        Assert.Contains("Building", output);
        Assert.Contains("Slide", output);
        Assert.Contains("Build complete", output);
    }

    [Fact]
    public void TestLoadWrongTypeRaises()
    {
        // Loading a non-ChebyshevSlider object should raise.
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, """{"Type":"NotASlider"}""");
            var ex = Assert.Throws<InvalidOperationException>(() =>
                ChebyshevSlider.Load(path));
            Assert.Contains("ChebyshevSlider", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestVersionMismatchWarning()
    {
        // C# uses JSON not pickle, so we test that loading a file with
        // wrong Type field raises an error (equivalent behavior).
        // This replaces the Python version mismatch warning test.
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, """{"Type":"WrongVersion"}""");
            var ex = Assert.Throws<InvalidOperationException>(() =>
                ChebyshevSlider.Load(path));
            Assert.Contains("ChebyshevSlider", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestHighDimStrTruncation()
    {
        // ToString() for 8D should truncate nodes, domain, pivot, partition.
        static double f(double[] x, object? _)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++) sum += x[i];
            return sum;
        }

        var slider = new ChebyshevSlider(
            f, 8,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0],
             [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [3, 3, 3, 3, 3, 3, 3, 3],
            partition: [[0], [1], [2], [3], [4], [5], [6], [7]],
            pivotPoint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        string s = slider.ToString()!;
        Assert.Contains("...]", s);
        Assert.Contains("...", s);
    }
}

// ======================================================================
// C#-Specific: Slider Validation Edge Cases
// ======================================================================

/// <summary>
/// C#-specific validation tests for ChebyshevSlider constructor and methods.
/// Tests null safety, argument validation, and boundary conditions not in Python.
/// </summary>
public class TestSliderValidationCSharp
{
    [Fact]
    public void TestPartitionWithOutOfRangeDim()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(
                TestFixtures.SinSum3D, 3,
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                [5, 5, 5],
                partition: [[0], [1], [5]], // dim 5 out of range
                pivotPoint: [0.0, 0.0, 0.0]));
        Assert.Contains("Partition must cover", ex.Message);
    }

    [Fact]
    public void TestPartitionWithNegativeDim()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(
                TestFixtures.SinSum3D, 3,
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                [5, 5, 5],
                partition: [[-1], [0], [1]], // negative dim
                pivotPoint: [0.0, 0.0, 0.0]));
        Assert.Contains("Partition must cover", ex.Message);
    }

    [Fact]
    public void TestEvalMultiBeforeBuild()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            slider.EvalMulti([0.5, 0.5, 0.5], [[0, 0, 0]]));
        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TestBuildWithNullFunctionRaises()
    {
        // Loaded slider has Function=null; calling Build on it should fail.
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);
            Assert.Null(loaded.Function);
            var ex = Assert.Throws<InvalidOperationException>(() =>
                loaded.Build(verbose: false));
            Assert.Contains("null", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestExtrudeBeforeBuildRaises()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            slider.Extrude((3, new[] { 0.0, 1.0 }, 5)));
        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TestSliceBeforeBuildRaises()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [5, 5, 5],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        var ex = Assert.Throws<InvalidOperationException>(() =>
            slider.Slice((0, 0.5)));
        Assert.Contains("Build", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TestSliceOutsideDomainRaises()
    {
        var slider = SliderFixtures.SliderSin3D;
        // Domain is [[-1,1], [-1,1], [1,3]], try slicing dim 0 at 5.0
        var ex = Assert.Throws<ArgumentException>(() =>
            slider.Slice((0, 5.0)));
        Assert.Contains("outside domain", ex.Message);
    }

    [Fact]
    public void TestDoubleBuildIsIdempotent()
    {
        // Building twice should just overwrite previous state.
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [8, 8, 8],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);
        double val1 = slider.Eval([0.5, 0.3, 0.7], [0, 0, 0]);

        slider.Build(verbose: false); // second build
        double val2 = slider.Eval([0.5, 0.3, 0.7], [0, 0, 0]);

        Assert.Equal(val1, val2, 14);
    }
}

// ======================================================================
// C#-Specific: Slider Serialization Edge Cases
// ======================================================================

/// <summary>
/// C#-specific serialization edge case tests for ChebyshevSlider.
/// </summary>
public class TestSliderSerializationCSharp
{
    [Fact]
    public void TestLoadNonexistentFileThrows()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ChebyshevSlider.Load("/tmp/this_file_does_not_exist_12345.json"));
    }

    [Fact]
    public void TestLoadEmptyFileThrows()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "");
            Assert.ThrowsAny<Exception>(() => ChebyshevSlider.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestLoadInvalidJsonThrows()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "not valid json {{{");
            Assert.ThrowsAny<Exception>(() => ChebyshevSlider.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestLoadChebyshevSplineAsSliderThrows()
    {
        // Save a ChebyshevSpline and try to load as ChebyshevSlider.
        // ChebyshevSpline has Type="ChebyshevSpline", so Slider.Load rejects it.
        var spline = TestFixtures.SplineAbs1D;
        string path = Path.GetTempFileName();
        try
        {
            spline.Save(path);
            var ex = Assert.Throws<InvalidOperationException>(() =>
                ChebyshevSlider.Load(path));
            Assert.Contains("ChebyshevSlider", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestRoundtripCoupledSlider()
    {
        // Coupled slider (multi-dim group) save/load round-trip.
        var slider = SliderFixtures.SliderCoupled;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            var pts = new[]
            {
                new[] { 1.0, 0.5, -1.0 },
                new[] { -0.5, 1.0, 0.3 },
                new[] { 0.0, 0.0, 0.0 },
            };
            foreach (var pt in pts)
            {
                double orig = slider.Eval(pt, [0, 0, 0]);
                double rest = loaded.Eval(pt, [0, 0, 0]);
                Assert.Equal(orig, rest, 14);
            }

            // Derivative round-trip
            double dOrig = slider.Eval([1.0, 0.5, -1.0], [1, 0, 0]);
            double dRest = loaded.Eval([1.0, 0.5, -1.0], [1, 0, 0]);
            Assert.Equal(dOrig, dRest, 14);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestRoundtripBitIdentical()
    {
        // Save/load should be bit-identical (not just close).
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);

            // Check all slides have identical TensorValues
            for (int i = 0; i < slider.Slides.Length; i++)
            {
                var origTv = slider.Slides[i].TensorValues!;
                var loadTv = loaded.Slides[i].TensorValues!;
                Assert.Equal(origTv.Length, loadTv.Length);
                for (int j = 0; j < origTv.Length; j++)
                    Assert.Equal(origTv[j], loadTv[j]);
            }

            // PivotValue bit-identical
            Assert.Equal(slider.PivotValue, loaded.PivotValue);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TestRoundtripErrorEstimate()
    {
        // ErrorEstimate should match after round-trip.
        var slider = SliderFixtures.SliderSin3D;
        string path = Path.GetTempFileName();
        try
        {
            slider.Save(path);
            var loaded = ChebyshevSlider.Load(path);
            Assert.Equal(slider.ErrorEstimate(), loaded.ErrorEstimate());
        }
        finally
        {
            File.Delete(path);
        }
    }
}

// ======================================================================
// C#-Specific: Slider Evaluation Edge Cases
// ======================================================================

/// <summary>
/// C#-specific evaluation edge case tests for ChebyshevSlider.
/// </summary>
public class TestSliderEvalCSharp
{
    [Fact]
    public void TestEvalAtPivotExact()
    {
        // At the pivot point of an additively separable function,
        // the slider should reproduce the exact value (within interpolation tolerance).
        var slider = SliderFixtures.SliderSin3D;
        double[] pivot = slider.PivotPoint;
        double expected = Math.Sin(pivot[0]) + Math.Sin(pivot[1]) + Math.Sin(pivot[2]);
        double result = slider.Eval(pivot, [0, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-8,
            $"At pivot: expected {expected}, got {result}");
    }

    [Fact]
    public void TestEvalAtDomainBoundaries()
    {
        // Evaluate at the corners of the domain.
        var slider = SliderFixtures.SliderSin3D;
        double[] lo = [-1.0, -1.0, 1.0]; // lower bounds
        double[] hi = [1.0, 1.0, 3.0];   // upper bounds

        double valLo = slider.Eval(lo, [0, 0, 0]);
        double expectedLo = Math.Sin(-1.0) + Math.Sin(-1.0) + Math.Sin(1.0);
        Assert.True(Math.Abs(valLo - expectedLo) < 1e-8,
            $"At lower bounds: expected {expectedLo}, got {valLo}");

        double valHi = slider.Eval(hi, [0, 0, 0]);
        double expectedHi = Math.Sin(1.0) + Math.Sin(1.0) + Math.Sin(3.0);
        Assert.True(Math.Abs(valHi - expectedHi) < 1e-8,
            $"At upper bounds: expected {expectedHi}, got {valHi}");
    }

    [Fact]
    public void TestEvalMultiConsistentWithEval()
    {
        // EvalMulti results should match individual Eval calls.
        var slider = SliderFixtures.SliderCoupled;
        double[] pt = [1.0, 0.5, -1.0];
        int[][] derivs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
        double[] multi = slider.EvalMulti(pt, derivs);

        for (int i = 0; i < derivs.Length; i++)
        {
            double single = slider.Eval(pt, derivs[i]);
            Assert.Equal(single, multi[i], 15);
        }
    }

    [Fact]
    public void TestConcurrentEval()
    {
        // Thread safety: parallel Eval calls should not corrupt state.
        var slider = SliderFixtures.SliderSin3D;
        int N = 100;
        var points = new double[N][];
        var expected = new double[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            double x = rng.NextDouble() * 2 - 1;
            double y = rng.NextDouble() * 2 - 1;
            double z = rng.NextDouble() * 2 + 1; // domain [1, 3]
            points[i] = [x, y, z];
            expected[i] = Math.Sin(x) + Math.Sin(y) + Math.Sin(z);
        }

        var results = new double[N];
        Parallel.For(0, N, i =>
        {
            results[i] = slider.Eval(points[i], [0, 0, 0]);
        });

        for (int i = 0; i < N; i++)
        {
            Assert.True(Math.Abs(results[i] - expected[i]) < 1e-8,
                $"Thread safety failure at i={i}: {results[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void TestHigherOrderDerivativeWithinGroup()
    {
        // 3rd derivative of sin(x) within a single-dim slide.
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        // d^3/dx^3 of sin(x) = -cos(x)
        double expected = -Math.Cos(0.5);
        double result = slider.Eval(pt, [3, 0, 0]);
        Assert.True(Math.Abs(result - expected) < 1e-4,
            $"3rd derivative: expected {expected}, got {result}");
    }

    [Fact]
    public void TestAllZeroDerivativeOrderIsValue()
    {
        // derivativeOrder = [0,0,0] should give function value.
        var slider = SliderFixtures.SliderSin3D;
        double[] pt = [0.5, -0.3, 1.7];
        double val = slider.Eval(pt, [0, 0, 0]);
        double expected = Math.Sin(0.5) + Math.Sin(-0.3) + Math.Sin(1.7);
        Assert.True(Math.Abs(val - expected) < 1e-9);
    }
}

// ======================================================================
// C#-Specific: Slider Properties and Immutability
// ======================================================================

/// <summary>
/// C#-specific tests verifying that constructor clones arrays (input mutation safety).
/// </summary>
public class TestSliderPropertiesCSharp
{
    [Fact]
    public void TestConstructorClonesDomain()
    {
        // Mutating the original domain array should not affect the slider.
        double[][] domain = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]];
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3, domain, [5, 5, 5],
            partition: [[0], [1], [2]], pivotPoint: [0.0, 0.0, 0.0]);
        domain[0][0] = 999.0;
        Assert.Equal(-1.0, slider.Domain[0][0]);
    }

    [Fact]
    public void TestConstructorClonesNNodes()
    {
        int[] nNodes = [5, 5, 5];
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], nNodes,
            partition: [[0], [1], [2]], pivotPoint: [0.0, 0.0, 0.0]);
        nNodes[0] = 999;
        Assert.Equal(5, slider.NNodes[0]);
    }

    [Fact]
    public void TestConstructorClonesPivotPoint()
    {
        double[] pivot = [0.0, 0.0, 0.0];
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], [5, 5, 5],
            partition: [[0], [1], [2]], pivotPoint: pivot);
        pivot[0] = 999.0;
        Assert.Equal(0.0, slider.PivotPoint[0]);
    }

    [Fact]
    public void TestConstructorClonesPartition()
    {
        int[][] partition = [[0], [1], [2]];
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], [5, 5, 5],
            partition: partition, pivotPoint: [0.0, 0.0, 0.0]);
        partition[0][0] = 999;
        Assert.Equal(0, slider.Partition[0][0]);
    }

    [Fact]
    public void TestDimToSlideMapping()
    {
        var slider = SliderFixtures.SliderCoupled;
        // Partition [[0,1],[2]]: dim 0->slide 0, dim 1->slide 0, dim 2->slide 1
        Assert.Equal(0, slider.DimToSlide[0]);
        Assert.Equal(0, slider.DimToSlide[1]);
        Assert.Equal(1, slider.DimToSlide[2]);
    }

    [Fact]
    public void TestTotalBuildEvalsMatchesExpected()
    {
        // Coupled: [[0,1],[2]], nNodes=[12,12,8] -> 12*12 + 8 = 152
        Assert.Equal(152, SliderFixtures.SliderCoupled.TotalBuildEvals);

        // 5D singletons: 5*6 = 30
        Assert.Equal(30, SliderFixtures.Slider5D.TotalBuildEvals);

        // Sin3D: 12 + 10 + 10 = 32
        Assert.Equal(32, SliderFixtures.SliderSin3D.TotalBuildEvals);
    }

    [Fact]
    public void TestBuildTimePositive()
    {
        var slider = SliderFixtures.SliderSin3D;
        Assert.True(slider.BuildTime > 0, "BuildTime should be positive");
    }

    [Fact]
    public void TestMaxDerivativeOrderDefault()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], [5, 5, 5],
            partition: [[0], [1], [2]], pivotPoint: [0.0, 0.0, 0.0]);
        Assert.Equal(2, slider.MaxDerivativeOrder);
    }

    [Fact]
    public void TestMaxDerivativeOrderCustom()
    {
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], [5, 5, 5],
            partition: [[0], [1], [2]], pivotPoint: [0.0, 0.0, 0.0],
            maxDerivativeOrder: 4);
        Assert.Equal(4, slider.MaxDerivativeOrder);
    }
}

// ======================================================================
// C#-Specific: Slider Error Estimation Edge Cases
// ======================================================================

/// <summary>
/// C#-specific error estimation edge case tests.
/// </summary>
public class TestSliderErrorEstimationCSharp
{
    [Fact]
    public void TestErrorEstimateCachedAfterFirstCall()
    {
        // Calling ErrorEstimate twice should return the exact same value (cached).
        var slider = new ChebyshevSlider(
            TestFixtures.SinSum3D, 3,
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            [10, 10, 10],
            partition: [[0], [1], [2]],
            pivotPoint: [0.0, 0.0, 0.0]);
        slider.Build(verbose: false);

        double est1 = slider.ErrorEstimate();
        double est2 = slider.ErrorEstimate();
        Assert.Equal(est1, est2); // exact same reference, not just close
    }

    [Fact]
    public void TestErrorEstimateAfterArithmetic()
    {
        // Arithmetic results should have a valid error estimate.
        var c = TestFixtures.AlgebraSliderF + TestFixtures.AlgebraSliderG;
        double est = c.ErrorEstimate();
        Assert.True(est >= 0, $"Error estimate should be non-negative, got {est}");
    }

    [Fact]
    public void TestErrorEstimateNonNegativeForAllFixtures()
    {
        Assert.True(SliderFixtures.SliderSin3D.ErrorEstimate() >= 0);
        Assert.True(SliderFixtures.SliderCoupled.ErrorEstimate() >= 0);
        Assert.True(SliderFixtures.Slider5D.ErrorEstimate() >= 0);
        Assert.True(SliderFixtures.SliderBs3D.ErrorEstimate() >= 0);
    }
}
