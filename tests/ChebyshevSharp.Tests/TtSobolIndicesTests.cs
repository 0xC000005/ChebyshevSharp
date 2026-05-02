using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtSobolIndicesTests
{
    private static ChebyshevTT Build3DTt(Func<double, double, double, double> f,
        int n = 8, int rank = 6, int seed = 42)
    {
        Func<double[], double> wrapper = p => f(p[0], p[1], p[2]);
        var tt = new ChebyshevTT(wrapper, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n, n },
            maxRank: rank,
            tolerance: 1e-10);
        tt.Build(verbose: false, seed: seed);
        return tt;
    }

    [Fact]
    public void Test_separable_function_first_order_sums_to_one()
    {
        // f(x, y, z) = x + y + z is purely additive (no coupling).
        // First-order indices should sum to ~1.0; total-order = first-order for additive.
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();

        double sumFirst = 0;
        foreach (double v in result.FirstOrder) sumFirst += v;
        Assert.True(sumFirst > 0.99 && sumFirst < 1.01,
            $"FirstOrder sum {sumFirst} should be near 1.0 for additive function");
    }

    [Fact]
    public void Test_first_order_le_total_order()
    {
        var tt = Build3DTt((x, y, z) => Math.Exp(x * y) + z);
        var result = tt.SobolIndices();

        for (int d = 0; d < 3; d++)
            Assert.True(result.FirstOrder[d] <= result.TotalOrder[d] + 1e-10,
                $"FirstOrder[{d}]={result.FirstOrder[d]} > TotalOrder[{d}]={result.TotalOrder[d]}");
    }

    [Fact]
    public void Test_constant_function_zero_variance()
    {
        // f(x, y, z) = 5.0 (constant). Variance should be ~0.
        var tt = Build3DTt((x, y, z) => 5.0);
        var result = tt.SobolIndices();

        Assert.True(result.Variance < 1e-15,
            $"Variance={result.Variance} should be near 0 for constant function");
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        Func<double[], double> f = p => p[0];
        var tt = new ChebyshevTT(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.Throws<InvalidOperationException>(() => tt.SobolIndices());
    }

    [Fact]
    public void Test_only_dim0_matters_first_order_concentrated()
    {
        // f(x, y, z) = x. Only dim 0 contributes.
        var tt = Build3DTt((x, y, z) => x);
        var result = tt.SobolIndices();

        Assert.True(result.FirstOrder[0] > 0.99,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 1.0");
        Assert.True(result.FirstOrder[1] < 0.01,
            $"FirstOrder[1]={result.FirstOrder[1]} should be near 0");
        Assert.True(result.FirstOrder[2] < 0.01,
            $"FirstOrder[2]={result.FirstOrder[2]} should be near 0");
    }

    [Fact]
    public void Test_pure_coupling_zero_first_order()
    {
        // f(x, y, z) = x * y has zero first-order energy in dims 0 and 1
        // (under the Chebyshev orthogonality measure, both x and y have mean 0,
        // so the additive parts integrate to zero).
        var tt = Build3DTt((x, y, z) => x * y);
        var result = tt.SobolIndices();

        Assert.True(result.FirstOrder[0] < 0.05,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 0 for pure coupling");
        Assert.True(result.FirstOrder[1] < 0.05,
            $"FirstOrder[1]={result.FirstOrder[1]} should be near 0 for pure coupling");
        // Total-order on dims 0 and 1 should be near 1 (both contribute fully through the coupling)
        Assert.True(result.TotalOrder[0] > 0.95,
            $"TotalOrder[0]={result.TotalOrder[0]} should be near 1 for pure coupling");
        Assert.True(result.TotalOrder[1] > 0.95,
            $"TotalOrder[1]={result.TotalOrder[1]} should be near 1 for pure coupling");
    }

    [Fact]
    public void Test_under_with_auto_order_keys_user_frame()
    {
        // Build a TT where dim 0 has the largest variance contribution.
        // After WithAutoOrder, _dimOrder may be non-identity, but result keys
        // must remain user-frame: index 0 should still report dim 0's importance.
        Func<double[], double> f = p => 100 * p[0] + p[1] + 0.01 * p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        var result = tt.SobolIndices();

        // Dim 0 dominates (contribution ~10000 of total ~10000.0001), so its
        // first-order should be near 1.
        Assert.True(result.FirstOrder[0] > 0.99,
            $"FirstOrder[0]={result.FirstOrder[0]} should dominate");
    }

    [Fact]
    public void Test_cross_validation_against_dense_path()
    {
        // Cross-validate TT-native against dense (ChebyshevApproximation.SobolIndices)
        // to within 1e-3 on a coupled function.
        Func<double[], double> f = p => Math.Exp(0.5 * p[0] * p[1]) + 0.3 * p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 8,
            tolerance: 1e-12);
        tt.Build(verbose: false, seed: 42);

        var ttNative = tt.SobolIndices();

        // Build the dense oracle via ChebyshevApproximation on the same f.
        Func<double[], object?, double> fApprox = (p, _) => Math.Exp(0.5 * p[0] * p[1]) + 0.3 * p[2];
        var approx = new ChebyshevApproximation(fApprox, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 });
        approx.Build(verbose: false);
        var dense = approx.SobolIndices();

        for (int d = 0; d < 3; d++)
        {
            Assert.True(Math.Abs(ttNative.FirstOrder[d] - dense.FirstOrder[d]) < 1e-3,
                $"FirstOrder[{d}]: TT={ttNative.FirstOrder[d]} vs dense={dense.FirstOrder[d]}");
            Assert.True(Math.Abs(ttNative.TotalOrder[d] - dense.TotalOrder[d]) < 1e-3,
                $"TotalOrder[{d}]: TT={ttNative.TotalOrder[d]} vs dense={dense.TotalOrder[d]}");
        }
    }

    [Fact]
    public void Test_total_order_sums_ge_one()
    {
        // For functions with coupling, total-order indices sum to >= 1 (they
        // double-count interaction energy).
        var tt = Build3DTt((x, y, z) => x * y + y * z + x * z);
        var result = tt.SobolIndices();

        double sumTotal = 0;
        foreach (double v in result.TotalOrder) sumTotal += v;
        Assert.True(sumTotal >= 1.0 - 1e-10,
            $"TotalOrder sum {sumTotal} should be >= 1 for coupled function");
    }

    [Fact]
    public void Test_indices_in_unit_interval()
    {
        var tt = Build3DTt((x, y, z) => x + y * y + z * z * z);
        var result = tt.SobolIndices();

        for (int d = 0; d < 3; d++)
        {
            Assert.True(result.FirstOrder[d] >= -1e-10 && result.FirstOrder[d] <= 1 + 1e-10,
                $"FirstOrder[{d}]={result.FirstOrder[d]} outside [0, 1]");
            Assert.True(result.TotalOrder[d] >= -1e-10 && result.TotalOrder[d] <= 1 + 1e-10,
                $"TotalOrder[{d}]={result.TotalOrder[d]} outside [0, 1]");
        }
    }

    [Fact]
    public void Test_variance_positive_for_non_constant()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();
        Assert.True(result.Variance > 0,
            $"Variance={result.Variance} should be positive for non-constant function");
    }

    [Fact]
    public void Test_returns_correct_array_lengths()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();
        Assert.Equal(3, result.FirstOrder.Length);
        Assert.Equal(3, result.TotalOrder.Length);
    }

    [Fact]
    public void Test_large_constant_plus_small_signal_recovers_dim()
    {
        // f(x, y, z) = 1.0 + 1e-12 * x. The signal in dim 0 is small relative
        // to the constant offset, but legitimate; the clamp must not silently
        // suppress it.
        var tt = new ChebyshevTT((p) => 1.0 + 1e-12 * p[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-14);
        tt.Build(verbose: false, method: "svd");

        var result = tt.SobolIndices();

        // Variance should be positive and small (not clamped to zero).
        Assert.True(result.Variance > 1e-25,
            $"Variance={result.Variance} should be positive (not clamped) for legitimate small signal");
        // Dim 0 carries all the signal; FirstOrder[0] should dominate.
        Assert.True(result.FirstOrder[0] > 0.95,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 1.0 for f = 1 + 1e-12*x");
    }

    [Fact]
    public void Test_cross_large_constant_plus_resolvable_small_signal_recovers_dim()
    {
        // Keep coverage on the TT-Cross build path with a signal large enough for
        // deterministic cross sampling, while the SVD test above covers the 1e-12 case.
        var tt = new ChebyshevTT((p) => 1.0 + 1e-6 * p[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-12);
        tt.Build(verbose: false, method: "cross", seed: 42);

        var result = tt.SobolIndices();

        Assert.True(result.Variance > 1e-14,
            $"Variance={result.Variance} should be positive for cross-resolvable small signal");
        Assert.True(result.FirstOrder[0] > 0.95,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 1.0 for f = 1 + 1e-6*x");
        Assert.True(result.TotalOrder[1] < 1e-5,
            $"TotalOrder[1]={result.TotalOrder[1]} should remain at numerical noise level");
    }
}
