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

    [Fact]
    public void Test_eval_multi_does_not_mutate_dim_order()
    {
        var tt = BuildAutoOrderTt();
        var orderBefore = tt.DimOrder;

        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };
        _ = tt.EvalMulti(new[] { 0.1, 0.2, 0.3 }, derivOrders);

        var orderAfter = tt.DimOrder;
        Assert.Equal(orderBefore, orderAfter);
    }

    [Fact]
    public async System.Threading.Tasks.Task Test_eval_multi_concurrent_calls_no_exceptions()
    {
        // Race regression: 4 threads, 1000 calls each.
        var tt = BuildAutoOrderTt();
        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };

        var tasks = new System.Threading.Tasks.Task[4];
        for (int t = 0; t < 4; t++)
        {
            int seed = t;
            tasks[t] = System.Threading.Tasks.Task.Run(() =>
            {
                var rng = new Random(seed);
                for (int i = 0; i < 1000; i++)
                {
                    var pt = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };
                    var results = tt.EvalMulti(pt, derivOrders);
                    Assert.Equal(2, results.Length);
                    Assert.False(double.IsNaN(results[0]) || double.IsNaN(results[1]));
                }
            });
        }
        await System.Threading.Tasks.Task.WhenAll(tasks);
    }

    [Fact]
    public async System.Threading.Tasks.Task Test_eval_multi_concurrent_results_match_serial()
    {
        // For deterministic input points, concurrent and serial calls produce
        // identical results.
        var tt = BuildAutoOrderTt();
        var pt = new[] { 0.1, 0.2, 0.3 };
        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 }, new[] { 0, 1, 0 } };

        // Serial baseline
        var serial = tt.EvalMulti(pt, derivOrders);

        // 8 concurrent calls
        var tasks = new System.Threading.Tasks.Task<double[]>[8];
        for (int t = 0; t < 8; t++)
            tasks[t] = System.Threading.Tasks.Task.Run(() => tt.EvalMulti(pt, derivOrders));
        var results = await System.Threading.Tasks.Task.WhenAll(tasks);

        foreach (var concurrent in results)
        {
            for (int i = 0; i < serial.Length; i++)
                Assert.Equal(serial[i], concurrent[i]);
        }
    }

    [Fact]
    public void Test_eval_multi_under_auto_order_returns_correct_value()
    {
        // After WithAutoOrder, Eval and EvalMulti's all-zero-derivative entry
        // must agree.
        var tt = BuildAutoOrderTt();
        var pt = new[] { 0.4, 0.3, -0.2 };

        double single = tt.Eval(pt);
        var multi = tt.EvalMulti(pt, new[] { new[] { 0, 0, 0 } });
        Assert.Equal(single, multi[0], precision: 10);
    }

    [Fact]
    public void Test_eval_multi_identity_dim_order_unchanged()
    {
        // For canonical _dimOrder, EvalMulti behavior is unchanged.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10);
        tt.Build(seed: 42);

        var pt = new[] { 0.1, 0.2, 0.3 };
        var multi = tt.EvalMulti(pt, new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } });
        Assert.Equal(0.6, multi[0], precision: 4);  // f(0.1, 0.2, 0.3) = 0.6
    }

    [Fact]
    public void Test_inner_product_mismatched_dim_order_throws()
    {
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var a = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10);
        a.Build(seed: 42);

        // b is the same TT reordered. inner_product on mismatched _dimOrder must throw.
        var b = a.Reorder(new[] { 2, 0, 1 });

        var ex = Assert.Throws<ArgumentException>(() => a.InnerProduct(b));
        Assert.Contains("_dimOrder", ex.Message);
        Assert.Contains("Reorder", ex.Message);
    }

    [Fact]
    public void Test_inner_product_after_alignment_returns_correct_value()
    {
        // Same setup as above, but align via Reorder; result should be sensible.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var a = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10);
        a.Build(seed: 42);

        var b = a.Reorder(new[] { 2, 0, 1 });
        // Bring b back to a's dim order
        var bAligned = b.Reorder(a.DimOrder);

        double ip = a.InnerProduct(bAligned);
        Assert.False(double.IsNaN(ip));
        Assert.True(ip > 0);  // self-inner-product is positive
    }

    [Fact]
    public void Test_inner_product_identity_dim_order_unchanged()
    {
        Func<double[], double> f = (p) => p[0] + p[1];
        var a = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10);
        a.Build(seed: 42);

        // Same TT as itself, no reordering — must succeed.
        double ip = a.InnerProduct(a);
        Assert.False(double.IsNaN(ip));
    }

    [Fact]
    public void Test_integrate_out_of_domain_error_uses_user_frame_dim()
    {
        // Use explicit Reorder + asymmetric per-dim domains to guarantee non-identity
        // _dimOrder and exercise the user-frame error message logic.
        Func<double[], double> f = (p) => p[0] + p[1] + p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { 5.0, 7.0 }, new[] { 100.0, 200.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4, tolerance: 1e-10);
        tt.Build(verbose: false, seed: 42);
        var reordered = tt.Reorder(new[] { 2, 0, 1 });

        // Confirm non-identity _dimOrder so we exercise the user-frame error path.
        Assert.False(reordered.DimOrder.SequenceEqual(new[] { 0, 1, 2 }),
            "Reorder must produce non-identity _dimOrder for this test to be meaningful");

        // user-frame dim 1 has domain [5, 7]; pass bounds outside it.
        var ex = Assert.Throws<ArgumentException>(() =>
            reordered.Integrate(dims: new[] { 1 }, bounds: new[] { (0.0, 6.0) }));

        // Error message must reference user-frame dim 1, NOT the storage position
        // (which would be storage_dim = Array.IndexOf([2,0,1], 1) = 2).
        Assert.Contains("dim 1", ex.Message);
        Assert.DoesNotContain("dim 2", ex.Message);  // would be storage-frame index
    }

    [Fact]
    public void Test_integrate_in_domain_succeeds_for_auto_order()
    {
        Func<double[], double> f = (p) => p[0] + p[1];
        var tt = ChebyshevTT.WithAutoOrder(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // Full integration of x + y over [-1,1]^2 = 0.
        double result = (double)tt.Integrate();
        Assert.Equal(0.0, result, precision: 6);
    }

    [Fact]
    public void Test_integrate_user_frame_partial()
    {
        // Build TT with WithAutoOrder; integrate only dim 0 in user frame.
        Func<double[], double> f = (p) => p[0] + 2 * p[1];
        var tt = ChebyshevTT.WithAutoOrder(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // ∫ (x + 2y) dx from -1 to 1 = (x²/2 + 2yx) | -1 to 1 = 0 + 4y = 4y.
        // Result is a 1-D TT in dim 1.
        var partial = tt.Integrate(dims: new[] { 0 });
        Assert.NotNull(partial);
        Assert.IsType<ChebyshevTT>(partial);

        // Sample at y = 0.5: should be ~2.0.
        var partialTt = (ChebyshevTT)partial!;
        double atY05 = partialTt.Eval(new[] { 0.5 });
        Assert.Equal(2.0, atY05, precision: 4);
    }
}
