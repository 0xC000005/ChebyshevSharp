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
