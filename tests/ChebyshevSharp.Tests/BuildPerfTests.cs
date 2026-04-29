using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestParallelBuildHelpers (Phase 6 Task 2)
// ======================================================================

public class TestParallelBuildHelpers
{
    [Fact]
    public void Test_normalize_zero_throws()
    {
        Assert.Throws<ArgumentException>(() => ParallelBuild.NormalizeNWorkers(0));
    }

    [Fact]
    public void Test_normalize_minus_one_resolves_to_processor_count()
    {
        int? resolved = ParallelBuild.NormalizeNWorkers(-1);
        Assert.Equal(Environment.ProcessorCount, resolved);
    }

    [Fact]
    public void Test_evaluate_in_parallel_matches_sequential()
    {
        // f(point, _) = point[0]^2 + point[1]^2 — pure, thread-safe.
        static double F(double[] p, object? _) => p[0] * p[0] + p[1] * p[1];

        var points = new double[][]
        {
            new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
            new[] { 1.0, 1.0 }, new[] { 2.0, 3.0 }, new[] { -1.0, 4.0 },
            new[] { 0.5, 0.5 }, new[] { 1.5, 2.5 },
        };

        double[] sequential = ParallelBuild.EvaluateInParallel(F, points, null, null, null);
        double[] parallel = ParallelBuild.EvaluateInParallel(F, points, null, 4, null);

        Assert.Equal(points.Length, parallel.Length);
        for (int i = 0; i < points.Length; i++)
            TestFixtures.AssertClose(sequential[i], parallel[i], rtol: 0, atol: 0);  // bit-exact for pure F
    }
}

// ======================================================================
// TestApproxBuildPerf (Phase 6 Task 3)
// ======================================================================

public class TestApproxBuildPerf
{
    private static double F(double[] p, object? _) => Math.Sin(p[0]) * Math.Cos(p[1]);

    [Fact]
    public void Test_parallel_build_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 16, 16 };

        var seqApprox = new ChebyshevApproximation(F, 2, domain, nNodes);
        seqApprox.Build(verbose: false);

        var parApprox = new ChebyshevApproximation(F, 2, domain, nNodes, nWorkers: 4);
        parApprox.Build(verbose: false);

        Assert.Equal(seqApprox.TensorValues!.Length, parApprox.TensorValues!.Length);
        for (int i = 0; i < seqApprox.TensorValues.Length; i++)
            TestFixtures.AssertClose(seqApprox.TensorValues[i], parApprox.TensorValues[i],
                rtol: 0, atol: 0);  // bit-exact: pure deterministic F, identical points order
    }

    [Fact]
    public void Test_progress_count_matches_grid_size()
    {
        var counter = new ProgressCounter();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8 };

        var approx = new ChebyshevApproximation(F, 2, domain, nNodes, progress: counter);
        approx.Build(verbose: false);

        Assert.Equal(64, counter.LastValue);
        Assert.Equal(64, counter.CallCount);
    }

    [Fact]
    public void Test_progress_null_no_op()
    {
        static double F1D(double[] p, object? _) => Math.Sin(p[0]);
        var approx = new ChebyshevApproximation(F1D, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 }, progress: null);
        approx.Build(verbose: false);
        Assert.NotNull(approx.TensorValues);
    }

    [Fact]
    public void Test_nworkers_null_is_sequential()
    {
        var counter = new ProgressCounter();
        var domain = new[] { new[] { -1.0, 1.0 } };
        static double F1D(double[] p, object? _) => Math.Sin(p[0]);
        var approx = new ChebyshevApproximation(F1D, 1, domain, new[] { 5 },
            nWorkers: null, progress: counter);
        approx.Build(verbose: false);

        // Sequential path increments 1, 2, 3, 4, 5 in order — no race.
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, counter.AllValues);
    }

    [Fact]
    public void Test_thread_safety_smoke_with_lock_wrapped_counter()
    {
        // Function captures shared mutable state via lock — acts as a smoke check
        // that with a thread-safe wrapper, the parallel path produces identical
        // values to sequential.
        int sharedCounter = 0;
        object lockObj = new();
        double F2(double[] p, object? _)
        {
            lock (lockObj) { sharedCounter++; }
            return p[0] * p[0];
        }

        var domain = new[] { new[] { 0.0, 1.0 } };
        var approx = new ChebyshevApproximation(F2, 1, domain, new[] { 10 }, nWorkers: 4);
        approx.Build(verbose: false);
        Assert.Equal(10, sharedCounter);  // Each grid point evaluated exactly once.
    }

    /// <summary>Records every reported progress value for assertion.</summary>
    private sealed class ProgressCounter : IProgress<int>
    {
        private readonly object _lock = new();
        private readonly List<int> _values = new();
        public int LastValue { get; private set; }
        public int CallCount { get; private set; }
        public int[] AllValues
        {
            get { lock (_lock) return _values.ToArray(); }
        }

        public void Report(int value)
        {
            lock (_lock)
            {
                _values.Add(value);
                LastValue = value;
                CallCount++;
            }
        }
    }
}

// ======================================================================
// TestSplineBuildPerf, TestSliderBuildPerf, TestTtBuildPerf (Phase 6 Task 4)
// ======================================================================

public class TestSplineBuildPerf
{
    private static double F(double[] p, object? _) => Math.Sin(p[0]) * Math.Cos(p[1]);

    [Fact]
    public void Test_spline_parallel_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8 };
        var knots = new[] { new double[] { 0.0 }, new double[] { 0.0 } };

        var seq = new ChebyshevSpline(F, 2, domain, nNodes, knots);
        seq.Build(verbose: false);

        var par = new ChebyshevSpline(F, 2, domain, nNodes, knots, nWorkers: 4);
        par.Build(verbose: false);

        Assert.Equal(seq.Pieces.Length, par.Pieces.Length);
        for (int p = 0; p < seq.Pieces.Length; p++)
        {
            var sv = seq.Pieces[p]!.TensorValues!;
            var pv = par.Pieces[p]!.TensorValues!;
            Assert.Equal(sv.Length, pv.Length);
            for (int i = 0; i < sv.Length; i++)
                TestFixtures.AssertClose(sv[i], pv[i], rtol: 0, atol: 0);
        }
    }

    [Fact]
    public void Test_spline_progress_count_sums_across_pieces()
    {
        var counter = new CountingProgress();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 4, 4 };
        var knots = new[] { new double[] { 0.0 }, new double[] { 0.0 } };

        var spline = new ChebyshevSpline(F, 2, domain, nNodes, knots, progress: counter);
        spline.Build(verbose: false);

        // 4 pieces × 16 nodes each = 64 expected.
        Assert.Equal(64, counter.LastValue);
    }
}

public class TestSliderBuildPerf
{
    private static double F(double[] p, object? _) => p[0] + p[1] + p[2];

    [Fact]
    public void Test_slider_parallel_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 6, 6, 6 };
        var partition = new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } };
        var pivot = new[] { 0.0, 0.0, 0.0 };

        var seq = new ChebyshevSlider(F, 3, domain, nNodes, partition, pivot);
        seq.Build(verbose: false);

        var par = new ChebyshevSlider(F, 3, domain, nNodes, partition, pivot, nWorkers: 4);
        par.Build(verbose: false);

        var pt = new[] { 0.3, -0.2, 0.7 };
        TestFixtures.AssertClose(
            seq.Eval(pt, new int[3]), par.Eval(pt, new int[3]), rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void Test_slider_progress_count_sums_across_slides()
    {
        var counter = new CountingProgress();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 4, 4 };
        var partition = new[] { new[] { 0 }, new[] { 1 } };
        var pivot = new[] { 0.0, 0.0 };
        // Use a 2D function (F is 3D; safe local lambda avoids IndexOutOfRange).
        static double F2(double[] p, object? _) => p[0] + p[1];

        var slider = new ChebyshevSlider(F2, 2, domain, nNodes, partition, pivot, progress: counter);
        slider.Build(verbose: false);

        // 2 slides × 4 nodes each = 8 evaluations expected.
        Assert.Equal(8, counter.LastValue);
    }
}

public class TestTtBuildPerf
{
    [Fact]
    public void Test_tt_progress_per_sweep()
    {
        var counter = new CountingProgress();
        static double F(double[] p) => Math.Sin(p[0]) + p[1] * p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, maxRank: 5, maxSweeps: 3,
            progress: counter);
        tt.Build(verbose: false, seed: 42);
        Assert.True(counter.CallCount >= 1);
    }

    [Fact]
    public void Test_tt_nworkers_ignored_does_not_break_build()
    {
        // TT does not parallelize TT-Cross (D10); nWorkers != null must be a no-op.
        static double F(double[] p) => p[0] * p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, maxSweeps: 3, nWorkers: 4);
        tt.Build(verbose: false, seed: 42);
        Assert.Equal(2, tt.NumDimensions);
        var pt = new[] { 0.3, 0.4 };
        double v = tt.Eval(pt);
        TestFixtures.AssertClose(0.12, v, rtol: 1e-3, atol: 1e-3);
    }
}

/// <summary>Shared progress counter; Interlocked-safe.</summary>
internal sealed class CountingProgress : IProgress<int>
{
    private int _last;
    private int _calls;
    public int LastValue => System.Threading.Volatile.Read(ref _last);
    public int CallCount => System.Threading.Volatile.Read(ref _calls);
    public void Report(int value)
    {
        System.Threading.Interlocked.Exchange(ref _last, value);
        System.Threading.Interlocked.Increment(ref _calls);
    }
}
