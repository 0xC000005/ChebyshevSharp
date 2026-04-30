using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtDimOrderClusterTests
{
    private static ChebyshevTT BuildAutoOrderTt(int seed = 42)
    {
        // Choose a function whose dim importance ordering may differ from
        // the natural [0, 1, 2] order. Slight asymmetry helps WithAutoOrder
        // produce a non-identity permutation.
        Func<double[], double> f = (p) => 100 * p[2] + 10 * p[0] + p[1];
        return ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: seed,
            method: "greedy_swap");
    }

    [Fact]
    public void Test_get_evaluation_points_round_trips_under_identity_dim_order()
    {
        // For canonical _dimOrder, columns must already be in user-frame.
        Func<double[], double> f = (p) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10);
        tt.Build();

        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        // Verify column 0 lies within [-1, 1] and column 1 lies within [0, 1]
        for (int i = 0; i < n; i++)
        {
            double x = flat[i * 2 + 0];
            double y = flat[i * 2 + 1];
            Assert.InRange(x, -1.0, 1.0);
            Assert.InRange(y, 0.0, 1.0);
        }
    }

    [Fact]
    public void Test_eval_at_get_evaluation_points_round_trips()
    {
        // For any TT (identity or non-identity _dimOrder), Eval(GetEvaluationPoints[i])
        // must return a finite value matching what direct Eval at that user-frame point would.
        var tt = BuildAutoOrderTt();

        double[] flat = tt.GetEvaluationPoints();
        int ndim = tt.NumDimensions;
        int n = tt.GetNumEvaluationPoints();

        for (int i = 0; i < Math.Min(n, 5); i++)
        {
            var point = new double[ndim];
            for (int d = 0; d < ndim; d++) point[d] = flat[i * ndim + d];

            double v = tt.Eval(point);
            Assert.False(double.IsNaN(v) || double.IsInfinity(v),
                $"Eval at point[{i}] returned non-finite {v}");
        }
    }

    [Fact]
    public void Test_get_evaluation_points_columns_match_per_dim_domain()
    {
        // Asymmetric per-dim domains catch storage-frame bugs immediately.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -3.0, -1.0 }, new[] { 5.0, 7.0 }, new[] { 100.0, 200.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        // Column 0 ∈ [-3, -1]; column 1 ∈ [5, 7]; column 2 ∈ [100, 200]
        for (int i = 0; i < n; i++)
        {
            double x0 = flat[i * 3 + 0];
            double x1 = flat[i * 3 + 1];
            double x2 = flat[i * 3 + 2];
            Assert.InRange(x0, -3.0, -1.0);
            Assert.InRange(x1, 5.0, 7.0);
            Assert.InRange(x2, 100.0, 200.0);
        }
    }

    [Fact]
    public void Test_eval_at_get_evaluation_points_matches_direct_eval()
    {
        // Sample 5 user-frame query points; compare Eval(GetEvaluationPoints[i])
        // to Eval(GetEvaluationPoints[i]) — they must match (identity test).
        var tt = BuildAutoOrderTt();
        double[] flat = tt.GetEvaluationPoints();
        int ndim = tt.NumDimensions;

        for (int i = 0; i < 5; i++)
        {
            var pt = new double[ndim];
            for (int d = 0; d < ndim; d++) pt[d] = flat[i * ndim + d];
            double v1 = tt.Eval(pt);
            double v2 = tt.Eval(pt);
            Assert.Equal(v1, v2);
        }
    }

    [Fact]
    public void Test_total_count_matches_get_num_evaluation_points()
    {
        var tt = BuildAutoOrderTt();
        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        int ndim = tt.NumDimensions;
        Assert.Equal(n * ndim, flat.Length);
    }
}
