using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class VectorizedEvalBatchPerfTests
{
    [Fact]
    public void Test_batch_results_match_loop_for_zero_derivative()
    {
        static double f(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var approx = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        approx.Build();

        // Random batch of 200 points.
        var rng = new Random(42);
        var points = new double[200][];
        for (int i = 0; i < 200; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] batch = approx.VectorizedEvalBatch(points, new[] { 0, 0 });
        for (int i = 0; i < 200; i++)
        {
            double single = approx.VectorizedEval(points[i], new[] { 0, 0 });
            Assert.Equal(single, batch[i], precision: 12);
        }
    }

    [Fact]
    public void Test_batch_results_match_loop_for_first_derivative()
    {
        static double f(double[] p, object? _) => Math.Exp(p[0]) * Math.Cos(p[1]);
        var approx = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        approx.Build();

        var rng = new Random(43);
        var points = new double[100][];
        for (int i = 0; i < 100; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] batch = approx.VectorizedEvalBatch(points, new[] { 1, 0 });
        for (int i = 0; i < 100; i++)
        {
            double single = approx.VectorizedEval(points[i], new[] { 1, 0 });
            Assert.Equal(single, batch[i], precision: 11);
        }
    }

    [Fact]
    public void Test_large_batch_correct_after_hoist()
    {
        // Large batch (1000 points) where the perf hoist amortization is most visible.
        static double f(double[] p, object? _) => p[0] + p[1] + p[2];
        var approx = new ChebyshevApproximation(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 });
        approx.Build();

        var rng = new Random(44);
        var points = new double[1000][];
        for (int i = 0; i < 1000; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] result = approx.VectorizedEvalBatch(points, new[] { 1, 0, 0 });
        // ∂f/∂x for f = x + y + z is 1 everywhere.
        for (int i = 0; i < 1000; i++)
            Assert.Equal(1.0, result[i], precision: 8);
    }
}
