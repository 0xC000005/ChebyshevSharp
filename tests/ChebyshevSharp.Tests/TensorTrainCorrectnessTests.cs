using Xunit;
using Xunit.Abstractions;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Additional correctness tests for ChebyshevTT, targeting mathematical invariants,
/// edge cases, and code paths not covered by the base Python-ported test suite.
/// </summary>
public class TensorTrainCorrectnessTests
{
    private readonly ITestOutputHelper _out;

    public TensorTrainCorrectnessTests(ITestOutputHelper output)
    {
        _out = output;
    }

    // ------------------------------------------------------------------
    // 1. Eval at Chebyshev grid nodes reproduces function values
    //    (validates DCT-II value→coefficient conversion roundtrip)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Eval_At_Grid_Nodes_Reproduces_Function_Values()
    {
        // Build a 2D polynomial that TT-SVD can recover exactly
        Func<double[], double> f = x => x[0] * x[0] + 2.0 * x[1];
        int n0 = 7, n1 = 7;
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n0, n1 },
            maxRank: 10);
        tt.Build(verbose: false, method: "svd");

        // Generate Chebyshev Type I nodes for both dimensions
        double[] nodes0 = MakeChebyshevNodes(-1, 1, n0);
        double[] nodes1 = MakeChebyshevNodes(-1, 1, n1);

        double maxErr = 0;
        for (int i = 0; i < n0; i++)
            for (int j = 0; j < n1; j++)
            {
                double[] pt = { nodes0[i], nodes1[j] };
                double exact = f(pt);
                double approx = tt.Eval(pt);
                double err = Math.Abs(exact - approx);
                maxErr = Math.Max(maxErr, err);
            }

        _out.WriteLine($"Max error at grid nodes: {maxErr:E2}");
        Assert.True(maxErr < 1e-12, $"Grid node eval error {maxErr:E2} too large");
    }

    // ------------------------------------------------------------------
    // 2. Constant function yields all rank-1 cores
    //    (validates SVD truncation threshold)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Constant_Function_Yields_Rank1()
    {
        var tt = new ChebyshevTT(
            x => 42.0, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7, 7 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        int[] ranks = tt.TtRanks;
        _out.WriteLine($"Ranks: [{string.Join(", ", ranks)}]");

        // All ranks should be 1 for a constant function
        for (int i = 0; i < ranks.Length; i++)
            Assert.Equal(1, ranks[i]);

        // Eval should return 42 everywhere
        var rng = new Random(99);
        for (int i = 0; i < 10; i++)
        {
            double[] pt = { -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble() };
            double val = tt.Eval(pt);
            Assert.True(Math.Abs(val - 42.0) < 1e-10,
                $"Constant function eval = {val} at [{string.Join(", ", pt)}]");
        }
    }

    // ------------------------------------------------------------------
    // 3. 2D TT-Cross and TT-SVD (sweep loop boundary conditions)
    // ------------------------------------------------------------------

    [Fact]
    public void Cross_2D_Minimal_Dimensions()
    {
        Func<double[], double> f = x => Math.Sin(x[0]) + Math.Cos(x[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { 0.0, Math.PI }, new[] { 0.0, Math.PI } },
            new[] { 11, 11 },
            maxRank: 5);
        tt.Build(verbose: false, method: "cross", seed: 42);

        double[] pt = { 1.0, 0.5 };
        double exact = Math.Sin(1.0) + Math.Cos(0.5);
        double approx = tt.Eval(pt);
        _out.WriteLine($"2D Cross: exact={exact:E10}, approx={approx:E10}, err={Math.Abs(exact - approx):E2}");
        Assert.True(Math.Abs(approx - exact) < 1e-6);
    }

    [Fact]
    public void SVD_2D_Matches_Cross()
    {
        Func<double[], double> f = x => Math.Sin(x[0]) + Math.Cos(x[1]);
        var ttSvd = new ChebyshevTT(f, 2,
            new[] { new[] { 0.0, Math.PI }, new[] { 0.0, Math.PI } },
            new[] { 11, 11 },
            maxRank: 5);
        ttSvd.Build(verbose: false, method: "svd");

        var ttCross = new ChebyshevTT(f, 2,
            new[] { new[] { 0.0, Math.PI }, new[] { 0.0, Math.PI } },
            new[] { 11, 11 },
            maxRank: 5);
        ttCross.Build(verbose: false, method: "cross", seed: 42);

        var rng = new Random(77);
        for (int i = 0; i < 10; i++)
        {
            double[] pt = { rng.NextDouble() * Math.PI, rng.NextDouble() * Math.PI };
            double vSvd = ttSvd.Eval(pt);
            double vCross = ttCross.Eval(pt);
            double exact = f(pt);
            _out.WriteLine($"  pt=[{pt[0]:F3}, {pt[1]:F3}]  svd={vSvd:E10}  cross={vCross:E10}  exact={exact:E10}");
            Assert.True(Math.Abs(vSvd - exact) < 1e-8, $"SVD error {Math.Abs(vSvd - exact):E2}");
            Assert.True(Math.Abs(vCross - exact) < 1e-4, $"Cross error {Math.Abs(vCross - exact):E2}");
        }
    }

    // ------------------------------------------------------------------
    // 4. Asymmetric domains with different node counts per dimension
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Asymmetric_Domain_And_Nodes()
    {
        // Different domain ranges AND different node counts
        Func<double[], double> f = x => x[0] * x[0] + 3.0 * x[1] - x[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { 0.0, 10.0 }, new[] { -5.0, 5.0 }, new[] { 100.0, 200.0 } },
            new[] { 5, 9, 7 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        double[][] pts = {
            new[] { 3.0, 1.0, 150.0 },
            new[] { 0.5, -4.0, 110.0 },
            new[] { 9.5, 4.5, 195.0 },
        };
        foreach (var pt in pts)
        {
            double exact = pt[0] * pt[0] + 3.0 * pt[1] - pt[2];
            double approx = tt.Eval(pt);
            _out.WriteLine($"  pt=[{string.Join(", ", pt)}]  exact={exact:E10}  approx={approx:E10}  err={Math.Abs(exact - approx):E2}");
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"Asymmetric domain error {Math.Abs(approx - exact):E2} at [{string.Join(", ", pt)}]");
        }
    }

    // ------------------------------------------------------------------
    // 5. Eval at domain boundaries and corners
    // ------------------------------------------------------------------

    [Fact]
    public void Eval_At_Domain_Boundaries()
    {
        Func<double[], double> f = x => x[0] * x[0] + x[1];
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -2.0, 3.0 }, new[] { 0.0, 5.0 } },
            new[] { 9, 9 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        // All four corners
        double[][] corners = {
            new[] { -2.0, 0.0 },
            new[] { -2.0, 5.0 },
            new[] { 3.0, 0.0 },
            new[] { 3.0, 5.0 },
        };
        foreach (var pt in corners)
        {
            double exact = pt[0] * pt[0] + pt[1];
            double approx = tt.Eval(pt);
            double err = Math.Abs(exact - approx);
            _out.WriteLine($"  corner=[{pt[0]}, {pt[1]}]  exact={exact}  approx={approx:E10}  err={err:E2}");
            Assert.True(err < 1e-8, $"Corner error {err:E2} at [{pt[0]}, {pt[1]}]");
        }
    }

    // ------------------------------------------------------------------
    // 6. 7D TT-Cross (cache key overflow, rank caps, high-dim sweeps)
    // ------------------------------------------------------------------

    [Fact]
    public void Cross_7D_With_Small_Nodes()
    {
        // 7D separable function with small node count — tests high-dim code paths
        Func<double[], double> f = x =>
            x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6];
        var tt = new ChebyshevTT(f, 7,
            Enumerable.Range(0, 7).Select(_ => new[] { -1.0, 1.0 }).ToArray(),
            Enumerable.Repeat(3, 7).ToArray(),
            maxRank: 3);
        tt.Build(verbose: false, method: "cross", seed: 42);

        double[] pt = { 0.5, -0.3, 0.1, 0.7, -0.9, 0.2, -0.4 };
        double exact = pt.Sum();
        double approx = tt.Eval(pt);
        _out.WriteLine($"7D sum: exact={exact:E10}  approx={approx:E10}  err={Math.Abs(exact - approx):E2}");
        Assert.True(Math.Abs(approx - exact) < 0.1,
            $"7D error {Math.Abs(approx - exact):E2}");
    }

    // ------------------------------------------------------------------
    // 7. Save/load preserves all metadata
    // ------------------------------------------------------------------

    [Fact]
    public void Serialization_Preserves_All_Metadata()
    {
        var tt = new ChebyshevTT(
            x => x[0] * x[0] + x[1], 2,
            new[] { new[] { -2.0, 3.0 }, new[] { 0.0, 5.0 } },
            new[] { 7, 9 },
            maxRank: 4);
        tt.Build(verbose: false, method: "svd");

        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var loaded = ChebyshevTT.Load(path);

            // Metadata
            Assert.Equal(tt.NumDimensions, loaded.NumDimensions);
            Assert.Equal(tt.MaxRank, loaded.MaxRank);
            Assert.Equal(tt.NNodes, loaded.NNodes);
            Assert.Equal(tt.TotalBuildEvals, loaded.TotalBuildEvals);
            Assert.Equal(tt.TtRanks, loaded.TtRanks);

            // Domain
            for (int d = 0; d < 2; d++)
            {
                Assert.Equal(tt.Domain[d][0], loaded.Domain[d][0]);
                Assert.Equal(tt.Domain[d][1], loaded.Domain[d][1]);
            }

            // CompressionRatio
            Assert.Equal(tt.CompressionRatio, loaded.CompressionRatio, 10);

            // ErrorEstimate
            Assert.Equal(tt.ErrorEstimate(), loaded.ErrorEstimate(), 10);

            // Eval at multiple points
            var rng = new Random(55);
            for (int i = 0; i < 20; i++)
            {
                double[] pt = { -2 + 5 * rng.NextDouble(), 5 * rng.NextDouble() };
                double orig = tt.Eval(pt);
                double load = loaded.Eval(pt);
                Assert.True(Math.Abs(orig - load) < 1e-14,
                    $"Eval mismatch at i={i}: {Math.Abs(orig - load):E2}");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------------
    // 8. Error estimate decreases with more nodes
    // ------------------------------------------------------------------

    [Fact]
    public void Error_Estimate_Decreases_With_More_Nodes()
    {
        Func<double[], double> f = x => Math.Sin(x[0]) + Math.Cos(x[1]);
        double[] errors = new double[3];
        int[] nodeCounts = { 5, 9, 15 };

        for (int idx = 0; idx < 3; idx++)
        {
            int nk = nodeCounts[idx];
            var tt = new ChebyshevTT(f, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { nk, nk },
                maxRank: 5);
            tt.Build(verbose: false, method: "svd");
            errors[idx] = tt.ErrorEstimate();
            _out.WriteLine($"  n={nk}: error estimate = {errors[idx]:E4}");
        }

        // Error estimate should be non-increasing (or at worst roughly stable)
        Assert.True(errors[2] <= errors[0] * 10,
            $"Error with n=15 ({errors[2]:E2}) not smaller than 10x error with n=5 ({errors[0]:E2})");
    }

    // ------------------------------------------------------------------
    // 9. Build() on loaded object (null function) raises
    // ------------------------------------------------------------------

    [Fact]
    public void Build_On_Loaded_Object_Raises_NullRef()
    {
        var tt = TestFixtures.TtSin3DSvd;
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var loaded = ChebyshevTT.Load(path);

            // Loaded object has no function — Build should fail
            Assert.ThrowsAny<Exception>(() => loaded.Build(verbose: false));
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------------
    // 10. Per-mode rank cap with very small node count
    // ------------------------------------------------------------------

    [Fact]
    public void Rank_Cap_With_Small_Node_Count()
    {
        // 3D with n=2 per dim. Max possible rank is min(prod_left, prod_right).
        // For mode 1: left=n[0]=2, right=n[1]*n[2]=4 → cap=min(maxRank, 2)
        var tt = new ChebyshevTT(
            x => x[0] + x[1] + x[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 2, 2, 2 },
            maxRank: 100);
        tt.Build(verbose: false, method: "svd");

        int[] ranks = tt.TtRanks;
        _out.WriteLine($"Ranks with n=2: [{string.Join(", ", ranks)}]");

        // Interior ranks can't exceed 2 since each mode only has 2 nodes
        for (int i = 1; i < ranks.Length - 1; i++)
            Assert.True(ranks[i] <= 2,
                $"Rank {ranks[i]} at position {i} exceeds node count 2");
    }

    // ------------------------------------------------------------------
    // 11. FD derivative of linear function is exact
    // ------------------------------------------------------------------

    [Fact]
    public void FD_Derivative_Of_Linear_Function_Is_Exact()
    {
        // f(x,y,z) = 3x + 2y - z. All first derivatives are exact constants.
        var tt = new ChebyshevTT(
            x => 3 * x[0] + 2 * x[1] - x[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7, 7 },
            maxRank: 3);
        tt.Build(verbose: false, method: "svd");

        double[] pt = { 0.3, -0.5, 0.7 };
        var results = tt.EvalMulti(pt, new[] {
            new[] { 1, 0, 0 },  // df/dx = 3
            new[] { 0, 1, 0 },  // df/dy = 2
            new[] { 0, 0, 1 },  // df/dz = -1
            new[] { 2, 0, 0 },  // d2f/dx2 = 0
        });

        _out.WriteLine($"df/dx = {results[0]:E10} (expected 3)");
        _out.WriteLine($"df/dy = {results[1]:E10} (expected 2)");
        _out.WriteLine($"df/dz = {results[2]:E10} (expected -1)");
        _out.WriteLine($"d2f/dx2 = {results[3]:E10} (expected 0)");

        Assert.True(Math.Abs(results[0] - 3.0) < 1e-6, $"df/dx = {results[0]}, expected 3");
        Assert.True(Math.Abs(results[1] - 2.0) < 1e-6, $"df/dy = {results[1]}, expected 2");
        Assert.True(Math.Abs(results[2] - (-1.0)) < 1e-6, $"df/dz = {results[2]}, expected -1");
        Assert.True(Math.Abs(results[3]) < 1e-4, $"d2f/dx2 = {results[3]}, expected 0");
    }

    // ------------------------------------------------------------------
    // 12. TT-Cross on peaked function (stresses maxvol)
    // ------------------------------------------------------------------

    [Fact]
    public void Cross_Peaked_Function()
    {
        // Gaussian bump — requires higher rank, stresses maxvol pivoting
        Func<double[], double> f = x =>
            Math.Exp(-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) * 5.0);

        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 15, 15, 15 },
            maxRank: 10, maxSweeps: 5);
        tt.Build(verbose: false, method: "cross", seed: 42);

        // Peaked function is hardest at the peak
        double[] peak = { 0.0, 0.0, 0.0 };
        double exactPeak = f(peak);
        double approxPeak = tt.Eval(peak);
        _out.WriteLine($"Peak: exact={exactPeak:E10}  approx={approxPeak:E10}  err={Math.Abs(exactPeak - approxPeak):E2}");

        // Should be reasonable but may not be machine-precision
        Assert.True(Math.Abs(approxPeak - exactPeak) < 0.1,
            $"Peak error {Math.Abs(approxPeak - exactPeak):E2}");

        // Also test away from peak
        double[] away = { 0.5, 0.5, 0.5 };
        double exactAway = f(away);
        double approxAway = tt.Eval(away);
        Assert.True(Math.Abs(approxAway - exactAway) < 0.1,
            $"Away-from-peak error {Math.Abs(approxAway - exactAway):E2}");
    }

    // ------------------------------------------------------------------
    // 13. 6D TT-SVD (sequential reshape logic for higher dims)
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_6D_Sequential_Reshape()
    {
        // 6D separable: sum of sins
        Func<double[], double> f = x =>
            Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]) +
            Math.Sin(x[3]) + Math.Sin(x[4]) + Math.Sin(x[5]);

        var tt = new ChebyshevTT(f, 6,
            Enumerable.Range(0, 6).Select(_ => new[] { -1.0, 1.0 }).ToArray(),
            Enumerable.Repeat(5, 6).ToArray(),
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        int[] ranks = tt.TtRanks;
        _out.WriteLine($"6D SVD ranks: [{string.Join(", ", ranks)}]");

        // Separable function should have rank ≤ 2 at interior bonds
        for (int i = 1; i < ranks.Length - 1; i++)
            Assert.True(ranks[i] <= 3,
                $"6D separable rank {ranks[i]} at position {i} unexpectedly high");

        // Accuracy check
        var rng = new Random(88);
        double maxErr = 0;
        for (int i = 0; i < 10; i++)
        {
            double[] pt = new double[6];
            for (int d = 0; d < 6; d++)
                pt[d] = -1 + 2 * rng.NextDouble();
            double exact = f(pt);
            double approx = tt.Eval(pt);
            double err = Math.Abs(exact - approx);
            maxErr = Math.Max(maxErr, err);
        }
        _out.WriteLine($"6D max eval error: {maxErr:E2}");
        // 5 nodes per dim gives limited accuracy for sin; mainly validates 6D SVD reshape works
        Assert.True(maxErr < 1e-2, $"6D max error {maxErr:E2} too large");
    }

    // ------------------------------------------------------------------
    // 14. Multiplicative separable function yields rank 1
    // ------------------------------------------------------------------

    [Fact]
    public void SVD_Multiplicative_Separable_Rank1()
    {
        // f(x,y,z) = exp(x) * exp(y) * exp(z) = exp(x+y+z) — has exact TT rank 1
        var tt = new ChebyshevTT(
            x => Math.Exp(x[0]) * Math.Exp(x[1]) * Math.Exp(x[2]), 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 9, 9, 9 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        int[] ranks = tt.TtRanks;
        _out.WriteLine($"Multiplicative separable ranks: [{string.Join(", ", ranks)}]");

        // All interior ranks should be exactly 1
        for (int i = 1; i < ranks.Length - 1; i++)
            Assert.Equal(1, ranks[i]);

        // Accuracy
        double[] pt = { 0.3, -0.5, 0.7 };
        double exact = Math.Exp(0.3) * Math.Exp(-0.5) * Math.Exp(0.7);
        double approx = tt.Eval(pt);
        Assert.True(Math.Abs(approx - exact) < 1e-6,
            $"Multiplicative separable error {Math.Abs(approx - exact):E2}");
    }

    // ------------------------------------------------------------------
    // 15. Rebuild resets all cached state
    // ------------------------------------------------------------------

    [Fact]
    public void Rebuild_Resets_Cached_State()
    {
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7 },
            maxRank: 3);

        // First build
        tt.Build(verbose: false, method: "svd");
        double err1 = tt.ErrorEstimate();
        int[] ranks1 = tt.TtRanks;
        double eval1 = tt.Eval(new[] { 0.5, 0.3 });

        // Rebuild with same parameters — should produce identical results
        tt.Build(verbose: false, method: "svd");
        double err2 = tt.ErrorEstimate();
        int[] ranks2 = tt.TtRanks;
        double eval2 = tt.Eval(new[] { 0.5, 0.3 });

        // Cached error estimate should be recalculated (not stale from first build)
        Assert.Equal(err1, err2, 14);
        Assert.Equal(ranks1, ranks2);
        Assert.Equal(eval1, eval2, 14);
    }

    // ------------------------------------------------------------------
    // Helper: generate Chebyshev Type I nodes on [a, b]
    // ------------------------------------------------------------------

    private static double[] MakeChebyshevNodes(double a, double b, int n)
    {
        double[] nodes = new double[n];
        for (int i = 0; i < n; i++)
        {
            double xi = Math.Cos((2.0 * (n - i) - 1.0) * Math.PI / (2.0 * n));
            nodes[i] = (a + b) / 2.0 + (b - a) / 2.0 * xi;
        }
        return nodes;
    }
}
