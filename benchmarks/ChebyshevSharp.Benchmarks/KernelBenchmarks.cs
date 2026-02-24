using BenchmarkDotNet.Attributes;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp.Benchmarks;

[MemoryDiagnoser]
public class KernelBenchmarks
{
    // Small: 10x10 (typical 1D contraction)
    private double[] _dataSmall = null!;
    private double[] _vecSmall = null!;
    private double[] _matFlatSmall = null!;

    // Medium: 180x10 (3D BS after first contraction: 15*12=180 leading, 10 last)
    private double[] _dataMedium = null!;
    private double[] _vecMedium = null!;
    private double[] _matFlatMedium = null!;

    // Large: 14641x11 (5D BS after first contraction: 11^4=14641 leading, 11 last)
    private double[] _dataLarge = null!;
    private double[] _vecLarge = null!;
    private double[] _matFlatLarge = null!;

    // DCT arrays
    private double[] _dctSmall = null!;  // n=10
    private double[] _dctMedium = null!; // n=20
    private double[] _dctLarge = null!;  // n=50

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _dataSmall = new double[100];
        _vecSmall = new double[10];
        _matFlatSmall = new double[100];
        FillRandom(rng, _dataSmall);
        FillRandom(rng, _vecSmall);
        FillRandom(rng, _matFlatSmall);

        _dataMedium = new double[1800];
        _vecMedium = new double[10];
        _matFlatMedium = new double[100];
        FillRandom(rng, _dataMedium);
        FillRandom(rng, _vecMedium);
        FillRandom(rng, _matFlatMedium);

        _dataLarge = new double[14641 * 11];
        _vecLarge = new double[11];
        _matFlatLarge = new double[121];
        FillRandom(rng, _dataLarge);
        FillRandom(rng, _vecLarge);
        FillRandom(rng, _matFlatLarge);

        _dctSmall = new double[10];
        _dctMedium = new double[20];
        _dctLarge = new double[50];
        FillRandom(rng, _dctSmall);
        FillRandom(rng, _dctMedium);
        FillRandom(rng, _dctLarge);
    }

    // --- MatmulLastAxis (GEMV via BLAS) ---

    [Benchmark]
    public double[] GEMV_Small_10x10() => BarycentricKernel.MatmulLastAxis(_dataSmall, 10, 10, _vecSmall);

    [Benchmark]
    public double[] GEMV_Medium_180x10() => BarycentricKernel.MatmulLastAxis(_dataMedium, 180, 10, _vecMedium);

    [Benchmark]
    public double[] GEMV_Large_14641x11() => BarycentricKernel.MatmulLastAxis(_dataLarge, 14641, 11, _vecLarge);

    // --- MatmulLastAxisMatrixFlat (GEMM via BLAS) ---

    [Benchmark]
    public double[] GEMM_Small_10x10x10() => BarycentricKernel.MatmulLastAxisMatrixFlat(_dataSmall, 10, 10, _matFlatSmall, 10);

    [Benchmark]
    public double[] GEMM_Medium_180x10x10() => BarycentricKernel.MatmulLastAxisMatrixFlat(_dataMedium, 180, 10, _matFlatMedium, 10);

    [Benchmark]
    public double[] GEMM_Large_14641x11x11() => BarycentricKernel.MatmulLastAxisMatrixFlat(_dataLarge, 14641, 11, _matFlatLarge, 11);

    // --- DCT-II (ChebyshevCoefficients1D) ---

    [Benchmark]
    public double[] DCT_n10() => BarycentricKernel.ChebyshevCoefficients1D(_dctSmall);

    [Benchmark]
    public double[] DCT_n20() => BarycentricKernel.ChebyshevCoefficients1D(_dctMedium);

    [Benchmark]
    public double[] DCT_n50() => BarycentricKernel.ChebyshevCoefficients1D(_dctLarge);

    private static void FillRandom(Random rng, double[] arr)
    {
        for (int i = 0; i < arr.Length; i++) arr[i] = rng.NextDouble();
    }
}
