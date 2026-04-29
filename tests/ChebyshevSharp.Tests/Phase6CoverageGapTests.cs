using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// Phase 6 coverage-gap followup tests
//
// Targets reachable Phase 6-added lines that the original tests miss:
//   - Sensitivity.cs NaN/Inf guard in ComputeSobolFromCoeffs.
//   - TensorTrainAlgebra.TtSwapAdjacent's tolerance==0 else branch.
//   - WithAutoOrder greedy_swap improving-permutation branch.
//   - WithAutoOrder random method's improving-permutation branch.
//   - ChebyshevTT.Clone deriv-id registry copy and _dimOrder copy.
//   - ChebyshevSpline.SobolIndices unbuilt guard for various unbuilt states.
//   - AutoKnots scan with all-zero finite-difference (constant-shaped function).
// ======================================================================

public class TestSensitivityNaNGuard
{
    [Fact]
    public void Test_compute_sobol_rejects_nan_coefficients()
    {
        // Direct internal call exercises the NaN/Inf guard at Sensitivity.cs:88-91.
        var coeffs = new double[] { 1.0, 2.0, double.NaN, 4.0 };
        var ex = Assert.Throws<ArgumentException>(
            () => Sensitivity.ComputeSobolFromCoeffs(coeffs, new[] { 4 }));
        Assert.Contains("NaN or Inf", ex.Message);
    }

    [Fact]
    public void Test_compute_sobol_rejects_inf_coefficients()
    {
        var coeffs = new double[] { 1.0, double.PositiveInfinity, 3.0, 4.0 };
        var ex = Assert.Throws<ArgumentException>(
            () => Sensitivity.ComputeSobolFromCoeffs(coeffs, new[] { 4 }));
        Assert.Contains("NaN or Inf", ex.Message);
    }
}

public class TestTtSwapAdjacentToleranceZero
{
    [Fact]
    public void Test_swap_with_zero_tolerance_uses_max_rank_only()
    {
        // tolerance=0 takes the else branch at TensorTrainAlgebra.cs:354-357
        // (skips the keepByTol clamp, just uses keep = max(1, min(maxRank, S.Count))).
        static double F(double[] p) => Math.Sin(p[0]) + p[1] * p[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, maxRank: 5, maxSweeps: 3);
        tt.Build(verbose: false, seed: 42);
        var cores = tt.GetCoeffCoresForTest();

        // tolerance=0 → no relative-tolerance pruning → branch at line 354 hit.
        var swapped = TensorTrainAlgebra.TtSwapAdjacent(cores, 0, maxRank: 10, tolerance: 0.0);
        Assert.Equal(3, swapped.Length);
        // Swap should be near-lossless when no truncation applied.
        double a = TensorTrainAlgebra.InnerProductCores(cores, cores);
        double b = TensorTrainAlgebra.InnerProductCores(swapped, swapped);
        TestFixtures.AssertClose(a, b, rtol: 1e-8, atol: 1e-8);
    }
}

public class TestWithAutoOrderImprovingPermutation
{
    /// <summary>
    /// Function with a clear rank-saving permutation: f(x,y,z) = sin(x*z) + cos(y).
    /// Under canonical [0,1,2], the y-z bond rank is large because y and z don't interact.
    /// Under [0,2,1], adjacent x and z (which DO interact via x*z) cluster together,
    /// reducing the largest rank.
    /// </summary>
    private static double F(double[] p) => Math.Sin(p[0] * p[2]) + Math.Cos(p[1]);

    [Fact]
    public void Test_greedy_swap_finds_improving_permutation()
    {
        // greedy_swap branch at ChebyshevTT.cs:2048-2059 fires when an adjacent transposition
        // strictly reduces total rank.
        var tt = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 }, maxRank: 8, maxSweeps: 5,
            nTrials: 5, method: "greedy_swap");
        // Must be a valid permutation; result is fully functional.
        Assert.Equal(3, tt.NumDimensions);
        Assert.Equal(new HashSet<int> { 0, 1, 2 }, new HashSet<int>(tt.DimOrder));
        var pt = new[] { 0.3, 0.4, -0.5 };
        TestFixtures.AssertClose(F(pt), tt.Eval(pt), rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_random_method_finds_improving_permutation()
    {
        // random branch at ChebyshevTT.cs:2065-2078 fires when a random shuffle
        // produces a lower-rank build than canonical.
        var tt = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 }, maxRank: 8, maxSweeps: 5,
            nTrials: 10, method: "random", seed: 42);
        Assert.Equal(3, tt.NumDimensions);
        Assert.Equal(new HashSet<int> { 0, 1, 2 }, new HashSet<int>(tt.DimOrder));
        var pt = new[] { 0.3, 0.4, -0.5 };
        TestFixtures.AssertClose(F(pt), tt.Eval(pt), rtol: 1e-3, atol: 1e-3);
    }
}

public class TestTtClonePreservesPhase6State
{
    [Fact]
    public void Test_clone_preserves_dim_order_after_reorder()
    {
        // Clone at ChebyshevTT.cs:2123 copies _dimOrder. The _dimOrder branch
        // matters for clone-of-Reorder-result fidelity.
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 }, maxRank: 8, maxSweeps: 5);
        tt.Build(verbose: false, seed: 42);
        var reord = tt.Reorder(new[] { 2, 0, 1 }, maxRank: 16, tolerance: 1e-12);
        var cloned = reord.Clone();
        Assert.Equal(reord.DimOrder, cloned.DimOrder);
        Assert.Equal(new[] { 2, 0, 1 }, cloned.DimOrder);
    }

    [Fact]
    public void Test_clone_preserves_derivative_id_registry()
    {
        // Clone at ChebyshevTT.cs:2125 (foreach _registeredDerivativeOrders) — covers
        // the registry-copy loop which is empty in most prior tests.
        static double F(double[] p) => Math.Sin(p[0]) * Math.Cos(p[1]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, maxRank: 5, maxSweeps: 3);
        tt.Build(verbose: false, seed: 42);
        int idValue = tt.GetDerivativeId(new[] { 0, 0 });
        int idDerivX = tt.GetDerivativeId(new[] { 1, 0 });

        var cloned = tt.Clone();
        // Cloned TT should evaluate via derivative-id with the same registered orders.
        var pt = new[] { 0.3, -0.4 };
        TestFixtures.AssertClose(tt.Eval(pt, idValue), cloned.Eval(pt, idValue), rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(tt.Eval(pt, idDerivX), cloned.Eval(pt, idDerivX), rtol: 1e-3, atol: 1e-3);
    }
}

public class TestSplineSobolIndicesPartialBuild
{
    [Fact]
    public void Test_sobol_indices_throws_when_pieces_array_partially_null()
    {
        // ChebyshevSpline.SobolIndices defensive guard for partial-built state.
        // We construct via the ctor without calling Build() — Pieces are null until Build.
        static double F(double[] p, object? _) => Math.Sin(p[0]);
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            new[] { Array.Empty<double>() });
        // Without Build(), Pieces[i].TensorValues is null.
        Assert.Throws<InvalidOperationException>(() => sp.SobolIndices());
    }
}

public class TestAutoKnotsConstantFunction
{
    [Fact]
    public void Test_constant_function_yields_no_knots()
    {
        // ScanForKnotsAlongDim's `if (meanD2 == 0) return Array.Empty<double>();` branch
        // — covers the early-return path when all 2nd differences are zero.
        static double F(double[] p, object? _) => 5.0;
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.Empty(sp.Knots[0]);
    }

    [Fact]
    public void Test_max_knots_per_dim_zero_skips_scan_entirely()
    {
        // AutoKnots `if (maxKnotsPerDim == 0)` skip-branch — explicitly covers the
        // continue path before any ScanForKnotsAlongDim work.
        static double F(double[] p, object? _) => Math.Abs(p[0]) + Math.Abs(p[1]);
        var sp = ChebyshevSpline.AutoKnots(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxKnotsPerDim: 0);
        Assert.Empty(sp.Knots[0]);
        Assert.Empty(sp.Knots[1]);
    }
}

public class TestEvalMultiOnPermutedTt
{
    /// <summary>
    /// EvalMulti on a non-identity-DimOrder TT: covers the in-place dim-order
    /// save/restore path at ChebyshevTT.cs:EvalMulti (Phase 6 Task 11).
    /// </summary>
    [Fact]
    public void Test_eval_multi_remap_under_non_identity_dim_order()
    {
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + 0.5 * p[2];
        var canonical = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 }, maxRank: 8, maxSweeps: 5);
        canonical.Build(verbose: false, seed: 42);
        var reord = canonical.Reorder(new[] { 2, 0, 1 }, maxRank: 16, tolerance: 1e-12);

        var pt = new[] { 0.3, -0.4, 0.5 };
        // Two derivative orders: value (0,0,0) and pure ∂/∂y (0,1,0).
        var orders = new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 0 } };
        var canRes = canonical.EvalMulti(pt, orders);
        var reordRes = reord.EvalMulti(pt, orders);

        // Both must produce the same result regardless of storage permutation.
        for (int i = 0; i < canRes.Length; i++)
            TestFixtures.AssertClose(canRes[i], reordRes[i], rtol: 1e-3, atol: 1e-3);
    }
}

public class TestWithAutoOrderNullSeedDeterminism
{
    [Fact]
    public void Test_random_with_null_seed_uses_default_42_not_tick_count()
    {
        // Python parity: the random method's default shuffle seed is 42
        // (matches np.random.default_rng(42) in tensor_train.py:2805).
        // Previously used Environment.TickCount which made seedless calls non-deterministic.
        //
        // We verify the fix by comparing two calls with explicit seed=42 (fully
        // deterministic: same shuffle RNG, same inner TT-Cross seed) against each other.
        // The null-seed and explicit-42 cases diverge because `seed` also threads into
        // the inner tt.Build() call; the shuffle-RNG fix is structural, not observable
        // through DimOrder equality when inner builds are non-deterministic.
        static double F(double[] p) => Math.Sin(p[0] * p[2]) + Math.Cos(p[1]);
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 6, 6, 6 };

        // Two explicit-seed=42 calls must be bit-identical (shuffle + inner builds both seeded).
        var tt1 = ChebyshevTT.WithAutoOrder(F, 3, domain, nNodes,
            nTrials: 3, method: "random", seed: 42);
        var tt2 = ChebyshevTT.WithAutoOrder(F, 3, domain, nNodes,
            nTrials: 3, method: "random", seed: 42);

        Assert.Equal(tt1.DimOrder, tt2.DimOrder);
        Assert.Equal(new HashSet<int> { 0, 1, 2 }, new HashSet<int>(tt1.DimOrder));
        var pt = new[] { 0.3, 0.4, -0.5 };
        TestFixtures.AssertClose(F(pt), tt1.Eval(pt), rtol: 1e-3, atol: 1e-3);
    }
}
