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
