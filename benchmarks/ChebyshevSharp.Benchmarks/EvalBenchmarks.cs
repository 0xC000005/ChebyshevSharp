using BenchmarkDotNet.Attributes;
using ChebyshevSharp;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp.Benchmarks;

[MemoryDiagnoser]
public class EvalBenchmarks
{
    private ChebyshevApproximation _sin1D = null!;
    private ChebyshevApproximation _bs3D = null!;
    private ChebyshevApproximation _bs5D = null!;

    private double[] _point1D = null!;
    private double[] _point3D = null!;
    private double[] _point5D = null!;

    private int[] _noDerivs1D = null!;
    private int[] _noDerivs3D = null!;
    private int[] _noDerivs5D = null!;
    private int[] _deriv1_3D = null!;
    private int[] _deriv1_5D = null!;

    private double[][] _batchPoints3D = null!;
    private int[][] _multiDerivs3D = null!;

    [GlobalSetup]
    public void Setup()
    {
        // 1D: sin, n=20
        _sin1D = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]), 1,
            new[] { new[] { 0.0, Math.PI } }, new[] { 20 });
        _sin1D.Build(verbose: false);

        // 3D: Black-Scholes, n=[15,12,10]
        double K = 100.0, r = 0.05, q = 0.02;
        _bs3D = new ChebyshevApproximation(
            (x, _) => BlackScholes.BsCallPrice(x[0], K, x[1], r, x[2], q), 3,
            new[] { new[] { 50.0, 150.0 }, new[] { 0.1, 2.0 }, new[] { 0.1, 0.5 } },
            new[] { 15, 12, 10 });
        _bs3D.Build(verbose: false);

        // 5D: Black-Scholes, n=[11,11,11,11,11]
        _bs5D = new ChebyshevApproximation(
            (x, _) => BlackScholes.BsCallPrice(x[0], x[1], x[2], x[4], x[3], q), 5,
            new[] {
                new[] { 80.0, 120.0 }, new[] { 90.0, 110.0 }, new[] { 0.25, 1.0 },
                new[] { 0.15, 0.35 }, new[] { 0.01, 0.08 }
            },
            new[] { 11, 11, 11, 11, 11 });
        _bs5D.Build(verbose: false);

        // Points
        _point1D = new[] { 1.5 };
        _point3D = new[] { 100.0, 1.0, 0.3 };
        _point5D = new[] { 100.0, 100.0, 0.5, 0.25, 0.04 };

        // Derivative orders
        _noDerivs1D = new[] { 0 };
        _noDerivs3D = new[] { 0, 0, 0 };
        _noDerivs5D = new[] { 0, 0, 0, 0, 0 };
        _deriv1_3D = new[] { 1, 0, 0 };
        _deriv1_5D = new[] { 1, 0, 0, 0, 0 };

        // Batch points (100 random points in 3D domain)
        var rng = new Random(42);
        _batchPoints3D = new double[100][];
        for (int i = 0; i < 100; i++)
        {
            _batchPoints3D[i] = new[] {
                50.0 + 100.0 * rng.NextDouble(),
                0.1 + 1.9 * rng.NextDouble(),
                0.1 + 0.4 * rng.NextDouble()
            };
        }

        // Multi derivative orders (price + delta + gamma)
        _multiDerivs3D = new[] {
            new[] { 0, 0, 0 },
            new[] { 1, 0, 0 },
            new[] { 2, 0, 0 }
        };
    }

    // --- 1D Eval ---

    [Benchmark]
    public double Eval_1D_Value() => _sin1D.VectorizedEval(_point1D, _noDerivs1D);

    [Benchmark]
    public double Eval_1D_Deriv1() => _sin1D.VectorizedEval(_point1D, new[] { 1 });

    // --- 3D Eval ---

    [Benchmark]
    public double Eval_3D_Value() => _bs3D.VectorizedEval(_point3D, _noDerivs3D);

    [Benchmark]
    public double Eval_3D_Delta() => _bs3D.VectorizedEval(_point3D, _deriv1_3D);

    // --- 5D Eval ---

    [Benchmark]
    public double Eval_5D_Value() => _bs5D.VectorizedEval(_point5D, _noDerivs5D);

    [Benchmark]
    public double Eval_5D_Delta() => _bs5D.VectorizedEval(_point5D, _deriv1_5D);

    // --- Batch ---

    [Benchmark]
    public double[] Batch_3D_100pts() => _bs3D.VectorizedEvalBatch(_batchPoints3D, _noDerivs3D);

    // --- Multi (price + delta + gamma) ---

    [Benchmark]
    public double[] Multi_3D_Greeks() => _bs3D.VectorizedEvalMulti(_point3D, _multiDerivs3D);

    // --- Error Estimate ---

    [Benchmark]
    public double ErrorEstimate_3D()
    {
        // Must create fresh to avoid caching
        var cheb = ChebyshevApproximation.FromValues(
            _bs3D.TensorValues!,
            3,
            new[] { new[] { 50.0, 150.0 }, new[] { 0.1, 2.0 }, new[] { 0.1, 0.5 } },
            new[] { 15, 12, 10 });
        return cheb.ErrorEstimate();
    }
}

/// <summary>
/// Minimal Black-Scholes for benchmarks (avoids test project dependency).
/// </summary>
internal static class BlackScholes
{
    public static double BsCallPrice(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        double d2 = d1 - sigma * Math.Sqrt(T);
        return S * Math.Exp(-q * T) * NormalCdf(d1) - K * Math.Exp(-r * T) * NormalCdf(d2);
    }

    private static double NormalCdf(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    private static double Erf(double x)
    {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }
}
