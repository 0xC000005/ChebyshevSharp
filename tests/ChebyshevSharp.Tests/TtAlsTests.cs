using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_tensor_train.py classes
// TestALSInternals + TestALS + TestCompletion + TestCrossFeatureALS
// (PyChebyshev v0.13.0). Tests added incrementally across Phase 2 Tasks 5 and 6.
//
// IMPORTANT: ALS is seeded-stochastic (System.Random vs np.random.default_rng
// produce different streams). Every assertion must be tolerance-based.
// Never inline-literal expected values from Python tests for ALS-touched outputs.
public class TtAlsTests
{
}

public class AlsTests
{
    private static readonly double[][] UnitCube3D = new[]
    {
        new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
    };

    [Fact]
    public void Test_als_builds_and_reaches_tolerance_3d()
    {
        // f(x,y,z) = sin(x)*cos(y) + 0.3*z^2
        Func<double[], double> f = p => Math.Sin(p[0]) * Math.Cos(p[1]) + 0.3 * p[2] * p[2];
        var tt = new ChebyshevTT(f, 3, UnitCube3D, new[] { 10, 10, 10 },
            tolerance: 1e-4, maxRank: 6);
        tt.Build(verbose: false, seed: 42, method: "als");

        double[][] pts = new[]
        {
            new[] { 0.1, -0.2, 0.3 },
            new[] { 0.5, 0.5, -0.5 },
            new[] { -0.9, 0.1, 0.7 },
        };
        foreach (var p in pts)
        {
            double got = tt.Eval(p);
            double want = f(p);
            Assert.True(Math.Abs(got - want) < 1e-2,
                $"ALS eval at [{string.Join(", ", p)}]: got {got}, want {want}, err {Math.Abs(got - want):e3}");
        }
    }

    [Fact]
    public void Test_als_matches_cross_on_same_fixture()
    {
        Func<double[], double> f = p => Math.Exp(-p[0] * p[0]) * Math.Cos(p[1]);
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 10, 10 };
        var ttCross = new ChebyshevTT(f, 2, domain, nNodes, tolerance: 1e-6, maxRank: 8);
        ttCross.Build(verbose: false, seed: 1, method: "cross");
        var ttAls = new ChebyshevTT(f, 2, domain, nNodes, tolerance: 1e-4, maxRank: 8);
        ttAls.Build(verbose: false, seed: 1, method: "als");
        double[][] pts = new[]
        {
            new[] { 0.1, -0.2 }, new[] { 0.5, 0.5 }, new[] { -0.9, 0.7 },
        };
        foreach (var p in pts)
            Assert.True(Math.Abs(ttCross.Eval(p) - ttAls.Eval(p)) < 5e-2,
                $"ALS vs Cross diverged at [{string.Join(", ", p)}]");
    }

    [Fact]
    public void Test_als_respects_max_rank_cap()
    {
        // tanh(50*(x-y)) — nearly discontinuous, unreachable at low rank.
        Func<double[], double> f = p => Math.Tanh(50 * (p[0] - p[1]));
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 20, 20 }, tolerance: 1e-12, maxRank: 3);
        tt.Build(verbose: false, seed: 0, method: "als");
        foreach (int r in tt.TtRanks)
            Assert.True(r <= 3, $"rank {r} exceeds maxRank=3");
    }

    [Fact]
    public void Test_als_max_rank_cap_emits_build_warning()
    {
        Func<double[], double> f = p => Math.Tanh(50 * (p[0] - p[1]));
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 20, 20 }, tolerance: 1e-12, maxRank: 3);
        tt.Build(verbose: false, seed: 0, method: "als");
        Assert.NotNull(tt.BuildWarning);
        Assert.Contains("maxRank", tt.BuildWarning, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_als_deterministic_given_seed()
    {
        Func<double[], double> f = p => p[0] * p[1] + 0.5;
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var ttA = new ChebyshevTT(f, 2, domain, new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        var ttB = new ChebyshevTT(f, 2, domain, new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        ttA.Build(verbose: false, seed: 123, method: "als");
        ttB.Build(verbose: false, seed: 123, method: "als");
        TestFixtures.AssertClose(ttA.Eval(new[] { 0.3, -0.4 }), ttB.Eval(new[] { 0.3, -0.4 }),
            atol: 1e-12);
    }

    [Fact]
    public void Test_als_method_attribute_set()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 }, tolerance: 1e-2, maxRank: 3);
        tt.Build(verbose: false, method: "als");
        Assert.Equal("als", tt.Method);
    }

    [Fact]
    public void Test_als_total_build_evals_positive()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, tolerance: 1e-4, maxRank: 3);
        tt.Build(verbose: false, method: "als");
        Assert.True(tt.TotalBuildEvals > 0);
    }

    [Fact]
    public void Test_invalid_method_raises()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        var ex = Assert.Throws<ArgumentException>(() => tt.Build(verbose: false, method: "bogus"));
        Assert.Contains("als", ex.Message);
    }

    [Fact]
    public void Test_als_save_load_roundtrip()
    {
        Func<double[], double> f = p => Math.Sin(p[0]) + p[1] * p[1];
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        tt.Build(verbose: false, seed: 0, method: "als");
        double valBefore = tt.Eval(new[] { 0.3, -0.4 });
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var tt2 = ChebyshevTT.Load(path);
            double valAfter = tt2.Eval(new[] { 0.3, -0.4 });
            TestFixtures.AssertClose(valBefore, valAfter, atol: 1e-12);
            Assert.Equal("als", tt2.Method);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

public class AlsInternalsTests
{
    [Fact]
    public void Test_als_sweep_reduces_residual_on_rank1_target()
    {
        // Build an exactly-rank-1 target on an 8x8x8 grid.
        var rng = new Random(0);
        double[] u0 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        double[] u1 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        double[] u2 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();

        // Target tensor as flat row-major (i, j, k) → i*64 + j*8 + k
        var target = new double[8 * 8 * 8];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                for (int k = 0; k < 8; k++)
                    target[i * 64 + j * 8 + k] = u0[i] * u1[j] * u2[k];

        Func<int[], double> evalsAt = idx => target[idx[0] * 64 + idx[1] * 8 + idx[2]];

        // Random rank-1 initial cores
        var rng2 = new Random(1);
        var cores = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore[3];
        for (int d = 0; d < 3; d++)
        {
            var core = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore(1, 8, 1);
            for (int j = 0; j < 8; j++) core[0, j, 0] = rng2.NextDouble() * 2 - 1;
            cores[d] = core;
        }

        ChebyshevSharp.Internal.TensorTrainKernel.AlsFixedRankSweep(
            cores, evalsAt, new[] { 8, 8, 8 }, tolerance: 1e-12, maxIter: 5);

        // Reconstruct and compare
        double residual = ChebyshevSharp.Internal.TensorTrainKernel.GridResidual(cores, target, new[] { 8, 8, 8 });
        Assert.True(residual < 1e-8, $"rank-1 residual {residual} exceeds 1e-8");
    }
}
