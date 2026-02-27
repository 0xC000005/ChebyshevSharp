using BenchmarkDotNet.Attributes;
using ChebyshevSharp;

namespace ChebyshevSharp.Benchmarks;

[MemoryDiagnoser]
public class SplineBenchmarks
{
    private ChebyshevSpline _abs1D = null!;
    private ChebyshevSpline _bs2D = null!;
    private ChebyshevApproximation _absGlobal1D = null!;
    private ChebyshevApproximation _bsGlobal2D = null!;

    private double[] _point1D = null!;
    private double[] _point2D = null!;

    private int[] _noDerivs1D = null!;
    private int[] _noDerivs2D = null!;
    private int[] _deriv1_1D = null!;
    private int[] _deriv1_2D = null!;

    private double[][] _batchPoints2D = null!;
    private int[][] _multiDerivs2D = null!;

    [GlobalSetup]
    public void Setup()
    {
        double K = 100.0, r = 0.05, q = 0.02;

        // 1D: |x| with knot at 0
        _abs1D = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        _abs1D.Build(verbose: false);

        // 1D global: |x| without knot (for comparison)
        _absGlobal1D = new ChebyshevApproximation(
            (x, _) => Math.Abs(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 30 });
        _absGlobal1D.Build(verbose: false);

        // 2D: BS with knot at strike
        _bs2D = new ChebyshevSpline(
            (x, _) => BlackScholes.BsCallPrice(x[0], K, x[1], r, 0.25, q), 2,
            new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 2.0 } },
            new[] { 15, 12 },
            new[] { new[] { K }, Array.Empty<double>() });
        _bs2D.Build(verbose: false);

        // 2D global: BS without knot (same total nodes)
        _bsGlobal2D = new ChebyshevApproximation(
            (x, _) => BlackScholes.BsCallPrice(x[0], K, x[1], r, 0.25, q), 2,
            new[] { new[] { 80.0, 120.0 }, new[] { 0.25, 2.0 } },
            new[] { 30, 12 });
        _bsGlobal2D.Build(verbose: false);

        // Points
        _point1D = new[] { 0.3 };
        _point2D = new[] { 105.0, 1.0 };

        // Derivative orders
        _noDerivs1D = new[] { 0 };
        _noDerivs2D = new[] { 0, 0 };
        _deriv1_1D = new[] { 1 };
        _deriv1_2D = new[] { 1, 0 };

        // Batch points (100 random points in 2D domain)
        var rng = new Random(42);
        _batchPoints2D = new double[100][];
        for (int i = 0; i < 100; i++)
        {
            _batchPoints2D[i] = new[] {
                80.0 + 40.0 * rng.NextDouble(),
                0.25 + 1.75 * rng.NextDouble()
            };
        }

        // Multi derivative orders (value + delta + gamma)
        _multiDerivs2D = new[] {
            new[] { 0, 0 },
            new[] { 1, 0 },
            new[] { 2, 0 }
        };
    }

    // --- 1D Spline vs Global ---

    [Benchmark]
    public double Spline_1D_Value() => _abs1D.Eval(_point1D, _noDerivs1D);

    [Benchmark]
    public double Global_1D_Value() => _absGlobal1D.VectorizedEval(_point1D, _noDerivs1D);

    [Benchmark]
    public double Spline_1D_Deriv1() => _abs1D.Eval(_point1D, _deriv1_1D);

    [Benchmark]
    public double Global_1D_Deriv1() => _absGlobal1D.VectorizedEval(_point1D, _deriv1_1D);

    // --- 2D Spline vs Global ---

    [Benchmark]
    public double Spline_2D_Value() => _bs2D.Eval(_point2D, _noDerivs2D);

    [Benchmark]
    public double Global_2D_Value() => _bsGlobal2D.VectorizedEval(_point2D, _noDerivs2D);

    [Benchmark]
    public double Spline_2D_Delta() => _bs2D.Eval(_point2D, _deriv1_2D);

    [Benchmark]
    public double Global_2D_Delta() => _bsGlobal2D.VectorizedEval(_point2D, _deriv1_2D);

    // --- 2D Batch ---

    [Benchmark]
    public double[] Spline_2D_Batch100() => _bs2D.EvalBatch(_batchPoints2D, _noDerivs2D);

    [Benchmark]
    public double[] Global_2D_Batch100() => _bsGlobal2D.VectorizedEvalBatch(_batchPoints2D, _noDerivs2D);

    // --- 2D Multi (value + delta + gamma) ---

    [Benchmark]
    public double[] Spline_2D_Multi() => _bs2D.EvalMulti(_point2D, _multiDerivs2D);

    [Benchmark]
    public double[] Global_2D_Multi() => _bsGlobal2D.VectorizedEvalMulti(_point2D, _multiDerivs2D);

    // --- Error Estimate ---

    [Benchmark]
    public double Spline_1D_ErrorEstimate()
    {
        // Build fresh to avoid caching
        var s = ChebyshevSpline.FromValues(
            _abs1D.Pieces!.Select(p => p!.TensorValues!).ToArray(),
            1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 15 },
            new[] { new[] { 0.0 } });
        return s.ErrorEstimate();
    }
}
