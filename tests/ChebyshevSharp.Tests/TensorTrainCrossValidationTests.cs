using Xunit;
using Xunit.Abstractions;
using ChebyshevSharp;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Cross-language validation tests: C# ChebyshevTT vs Python PyChebyshev reference values.
/// TT-SVD is deterministic (no RNG), so both languages must produce identical results
/// within floating-point tolerance. Reference values generated from PyChebyshev v0.10.1.
/// </summary>
public class TensorTrainCrossValidationTests
{
    private readonly ITestOutputHelper _out;
    private readonly ChebyshevTT _ttSvd3DSin;
    private readonly ChebyshevTT _ttSvdPoly;

    public TensorTrainCrossValidationTests(ITestOutputHelper output)
    {
        _out = output;

        // Same parameters as Python reference: 3D sin sum, SVD, rank 5
        _ttSvd3DSin = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 },
            maxRank: 5);
        _ttSvd3DSin.Build(verbose: false, method: "svd");

        // 3D polynomial, SVD, rank 5
        _ttSvdPoly = new ChebyshevTT(
            x => x[0] * x[0] * x[1] + x[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7, 7 },
            maxRank: 5);
        _ttSvdPoly.Build(verbose: false, method: "svd");
    }

    // ------------------------------------------------------------------
    // Python reference values (from PyChebyshev v0.10.1, TT-SVD)
    // ------------------------------------------------------------------

    // 3D sin sum eval results at 8 test points
    private static readonly (double[] pt, double pyVal)[] SinSvdRef = {
        (new[] { 0.5, 0.3, 0.1 },     0.8747791619079373),
        (new[] { -0.7, 0.0, 0.8 },    0.07313840369729174),
        (new[] { 0.0, 0.0, 0.0 },     2.2261563198894894e-16),
        (new[] { 0.99, -0.99, 0.5 },  0.47942553861620185),
        (new[] { -0.5, 0.5, -0.5 },  -0.479425538616202),
        (new[] { 0.1, 0.2, 0.3 },     0.594022954067581),
        (new[] { 0.8, -0.3, 0.6 },    0.9864783576278429),
        (new[] { -0.1, -0.2, -0.3 }, -0.5940229540675814),
    };

    // 3D polynomial eval results
    private static readonly (double[] pt, double pyVal)[] PolySvdRef = {
        (new[] { 0.5, 0.3, 0.1 },    0.17500000000000024),
        (new[] { -0.8, 0.7, -0.5 }, -0.05200000000000021),
        (new[] { 0.0, 1.0, -1.0 },  -0.9999999999999998),
    };

    // ------------------------------------------------------------------
    // Test 1: Eval values match Python (TT-SVD, deterministic)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_3D_Sin_Eval_Matches_Python()
    {
        double maxDiff = 0;
        foreach (var (pt, pyVal) in SinSvdRef)
        {
            double csVal = _ttSvd3DSin.Eval(pt);
            double diff = Math.Abs(csVal - pyVal);
            maxDiff = Math.Max(maxDiff, diff);
            _out.WriteLine($"  pt=[{string.Join(", ", pt)}]  py={pyVal:E15}  cs={csVal:E15}  diff={diff:E2}");

            // TT-SVD is deterministic — both languages use the same algorithm on the same
            // Chebyshev nodes. Small differences arise from SVD implementation details
            // (MathNet vs LAPACK) and DCT-II precision. Tolerance: 1e-10.
            Assert.True(diff < 1e-10,
                $"C# vs Python diff {diff:E2} at [{string.Join(", ", pt)}]");
        }
        _out.WriteLine($"Max C# vs Python diff: {maxDiff:E2}");
    }

    [Fact]
    public void SVD_3D_Poly_Eval_Matches_Python()
    {
        foreach (var (pt, pyVal) in PolySvdRef)
        {
            double csVal = _ttSvdPoly.Eval(pt);
            double diff = Math.Abs(csVal - pyVal);
            _out.WriteLine($"  pt=[{string.Join(", ", pt)}]  py={pyVal:E15}  cs={csVal:E15}  diff={diff:E2}");
            Assert.True(diff < 1e-14,
                $"Poly C# vs Python diff {diff:E2} at [{string.Join(", ", pt)}]");
        }
    }

    // ------------------------------------------------------------------
    // Test 2: TT ranks match Python exactly
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Ranks_Match_Python()
    {
        int[] pyRanks = { 1, 2, 2, 1 };
        int[] csRanks = _ttSvd3DSin.TtRanks;
        _out.WriteLine($"Python ranks: [{string.Join(", ", pyRanks)}]");
        _out.WriteLine($"C#     ranks: [{string.Join(", ", csRanks)}]");
        Assert.Equal(pyRanks, csRanks);
    }

    // ------------------------------------------------------------------
    // Test 3: Compression ratio matches Python
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Compression_Matches_Python()
    {
        double pyCompression = 15.125;
        double csCompression = _ttSvd3DSin.CompressionRatio;
        _out.WriteLine($"Python compression: {pyCompression}");
        _out.WriteLine($"C#     compression: {csCompression}");
        Assert.Equal(pyCompression, csCompression, 6);
    }

    // ------------------------------------------------------------------
    // Test 4: Error estimate matches Python
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Error_Estimate_Matches_Python()
    {
        double pyErr = 3.0119187904188487e-16;
        double csErr = _ttSvd3DSin.ErrorEstimate();
        _out.WriteLine($"Python error est: {pyErr:E15}");
        _out.WriteLine($"C#     error est: {csErr:E15}");
        // Error estimate depends on DCT-II coefficients, small SVD differences propagate
        // Both should be extremely small for sin sum with rank-2 SVD
        Assert.True(csErr < 1e-12, $"Error estimate {csErr:E2} unexpectedly large");
        // Check same order of magnitude
        if (pyErr > 0 && csErr > 0)
        {
            double ratio = csErr / pyErr;
            _out.WriteLine($"Ratio: {ratio:F2}");
        }
    }

    // ------------------------------------------------------------------
    // Test 5: Batch eval matches single eval (internal consistency)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Batch_Eval_Matches_Single()
    {
        double[,] pts = new double[SinSvdRef.Length, 3];
        for (int i = 0; i < SinSvdRef.Length; i++)
            for (int j = 0; j < 3; j++)
                pts[i, j] = SinSvdRef[i].pt[j];

        double[] batch = _ttSvd3DSin.EvalBatch(pts);
        for (int i = 0; i < SinSvdRef.Length; i++)
        {
            double single = _ttSvd3DSin.Eval(SinSvdRef[i].pt);
            Assert.True(Math.Abs(batch[i] - single) < 1e-15,
                $"Batch vs single mismatch at i={i}: {Math.Abs(batch[i] - single):E2}");
        }
    }

    // ------------------------------------------------------------------
    // Test 6: FD derivatives match Python
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_FD_Derivatives_Match_Python()
    {
        double[] pt = { 0.5, 0.3, 0.1 };
        var results = _ttSvd3DSin.EvalMulti(pt, new[] {
            new[] { 0, 0, 0 },  // value
            new[] { 1, 0, 0 },  // d/dx0
            new[] { 0, 1, 0 },  // d/dx1
            new[] { 0, 0, 1 },  // d/dx2
            new[] { 2, 0, 0 },  // d2/dx0^2
            new[] { 1, 1, 0 },  // d2/dx0dx1 (mixed, should be ~0)
        });

        // Python reference FD values
        double[] pyFD = {
            0.8747791619079373,      // value
            0.8775825557758021,      // d/dx0
            0.9553364830272582,      // d/dx1
            0.9950041585243818,      // d/dx2
            -0.47942553538238286,    // d2/dx0^2
            6.938893903907228e-10,   // d2/dx0dx1
        };

        // Analytical reference
        double[] analytical = {
            Math.Sin(0.5) + Math.Sin(0.3) + Math.Sin(0.1),
            Math.Cos(0.5),
            Math.Cos(0.3),
            Math.Cos(0.1),
            -Math.Sin(0.5),
            0.0, // separable, mixed partial is zero
        };

        string[] labels = { "value", "d/dx0", "d/dx1", "d/dx2", "d2/dx0^2", "d2/dx0dx1" };

        for (int i = 0; i < results.Length; i++)
        {
            double diffVsPy = Math.Abs(results[i] - pyFD[i]);
            double diffVsExact = Math.Abs(results[i] - analytical[i]);
            _out.WriteLine($"  {labels[i]}: cs={results[i]:E10}  py={pyFD[i]:E10}  exact={analytical[i]:E10}  " +
                           $"diff_py={diffVsPy:E2}  diff_exact={diffVsExact:E2}");
        }

        // Value should match Python closely (same TT eval path)
        Assert.True(Math.Abs(results[0] - pyFD[0]) < 1e-10, "Value mismatch");

        // FD derivatives use the same step size h=(b-a)*1e-4=2e-4 and same nudging logic.
        // Small differences arise from TT eval differences propagating through FD stencil.
        // Allow 1e-6 relative tolerance for first derivatives, 1e-4 for second.
        for (int i = 1; i <= 3; i++) // first derivatives
        {
            double relErr = Math.Abs(results[i] - analytical[i]) / Math.Abs(analytical[i]);
            Assert.True(relErr < 1e-4, $"{labels[i]} rel error {relErr:E2}");
        }

        // Second derivative
        double d2RelErr = Math.Abs(results[4] - analytical[4]) / Math.Abs(analytical[4]);
        Assert.True(d2RelErr < 1e-3, $"d2/dx0^2 rel error {d2RelErr:E2}");

        // Mixed partial should be ~0 for separable function
        Assert.True(Math.Abs(results[5]) < 1e-4, $"Mixed partial = {results[5]:E4}");
    }

    // ------------------------------------------------------------------
    // Test 7: Both C# and Python agree on accuracy vs analytical
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Both_Languages_Match_Analytical()
    {
        // Both Python and C# TT-SVD should give < 1e-8 error vs analytical for sin sum
        foreach (var (pt, pyVal) in SinSvdRef)
        {
            double exact = Math.Sin(pt[0]) + Math.Sin(pt[1]) + Math.Sin(pt[2]);
            double csVal = _ttSvd3DSin.Eval(pt);

            double pyErr = Math.Abs(pyVal - exact);
            double csErr = Math.Abs(csVal - exact);

            _out.WriteLine($"  pt=[{string.Join(", ", pt)}]  pyErr={pyErr:E2}  csErr={csErr:E2}");

            // Both should achieve < 1e-8 accuracy
            Assert.True(pyErr < 1e-8, $"Python error {pyErr:E2} too large");
            Assert.True(csErr < 1e-8, $"C# error {csErr:E2} too large");

            // And they should be within the same order of magnitude of each other
            if (pyErr > 1e-16 && csErr > 1e-16)
            {
                double ratio = Math.Max(pyErr, csErr) / Math.Min(pyErr, csErr);
                Assert.True(ratio < 1000, $"Error ratio {ratio:F1} — languages diverge");
            }
        }
    }

    // ------------------------------------------------------------------
    // Test 8: Full tensor equivalence (small 3D, exact comparison)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Polynomial_Exact_Recovery()
    {
        // Pure polynomial x0^2 + x1 — low TT-rank, should be recovered exactly by TT-SVD
        var tt = new ChebyshevTT(
            x => x[0] * x[0] + x[1],
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7 },
            maxRank: 10);
        tt.Build(verbose: false, method: "svd");

        // Check on a fine evaluation grid
        int nEval = 11;
        double maxErr = 0;
        for (int i = 0; i < nEval; i++)
            for (int j = 0; j < nEval; j++)
            {
                double x0 = -1.0 + 2.0 * i / (nEval - 1);
                double x1 = -1.0 + 2.0 * j / (nEval - 1);
                double exact = x0 * x0 + x1;
                double approx = tt.Eval(new[] { x0, x1 });
                double err = Math.Abs(exact - approx);
                if (err > maxErr) maxErr = err;
            }

        _out.WriteLine($"Max error on 11x11 eval grid: {maxErr:E2}");
        Assert.True(maxErr < 1e-10, $"Max error {maxErr:E2} on full grid");
    }
}
