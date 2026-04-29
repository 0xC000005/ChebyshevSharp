using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestTtDimOrderThreading (Phase 6 Task 11)
// ======================================================================

public class TestTtDimOrderThreading
{
    private static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + 0.5 * p[2];

    private static (ChebyshevTT canonical, ChebyshevTT reord) Pair()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8, 8 };
        var canonical = new ChebyshevTT(F, 3, domain, nNodes, maxRank: 8, maxSweeps: 5);
        canonical.Build(verbose: false, seed: 42);
        var reord = canonical.Reorder(new[] { 2, 0, 1 }, maxRank: 16, tolerance: 1e-12);
        return (canonical, reord);
    }

    [Fact]
    public void Test_eval_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pt = new[] { 0.3, -0.4, 0.5 };
        TestFixtures.AssertClose(canonical.Eval(pt), reord.Eval(pt), rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_eval_batch_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pts = new double[,]
        {
            { 0.3, -0.4, 0.5 },
            { -0.7, 0.1, 0.2 },
            { 0.0, 0.0, 0.0 },
        };
        var canonicalRes = canonical.EvalBatch(pts);
        var reordRes = reord.EvalBatch(pts);
        for (int i = 0; i < canonicalRes.Length; i++)
            TestFixtures.AssertClose(canonicalRes[i], reordRes[i], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_eval_multi_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pt = new[] { 0.3, -0.4, 0.5 };
        var orders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };  // value + ∂/∂x[0]
        var canonicalRes = canonical.EvalMulti(pt, orders);
        var reordRes = reord.EvalMulti(pt, orders);
        for (int i = 0; i < canonicalRes.Length; i++)
            TestFixtures.AssertClose(canonicalRes[i], reordRes[i], rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_slice_updates_dimorder()
    {
        var (_, reord) = Pair();
        // reord has DimOrder = [2, 0, 1]; slice user-dim 0 → drop original dim 0.
        var sliced = reord.Slice(0, 0.5);
        Assert.Equal(2, sliced.NumDimensions);
        // sliced.DimOrder should be the surviving original dims renumbered to a permutation of [0, 1].
        Assert.Equal(new HashSet<int> { 0, 1 }, new HashSet<int>(sliced.DimOrder));
    }

    [Fact]
    public void Test_extrude_updates_dimorder()
    {
        var (_, reord) = Pair();
        var extruded = reord.Extrude(0, (-1.0, 1.0), 4);
        Assert.Equal(4, extruded.NumDimensions);
        // Result DimOrder: a permutation of [0..3].
        Assert.Equal(new HashSet<int> { 0, 1, 2, 3 }, new HashSet<int>(extruded.DimOrder));
    }

    [Fact]
    public void Test_to_dense_produces_tensor_in_original_dim_order()
    {
        var (canonical, reord) = Pair();
        var canonicalDense = canonical.ToDense();
        var reordDense = reord.ToDense();
        // Both should produce arrays of the same shape (since both have the same original-dim
        // axes and node counts). For a 3D 8x8x8 grid:
        Assert.Equal(canonicalDense.Length, reordDense.Length);
        // Numerical match (best effort given SVD truncation in reorder):
        for (int i = 0; i < canonicalDense.Length; i++)
            TestFixtures.AssertClose(canonicalDense[i], reordDense[i], rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_partial_integrate_updates_dimorder()
    {
        var (_, reord) = Pair();
        // Partial integrate over user-dim 1 → result is 2D.
        var integrated = (ChebyshevTT)reord.Integrate(dims: new[] { 1 });
        Assert.Equal(2, integrated.NumDimensions);
        Assert.Equal(new HashSet<int> { 0, 1 }, new HashSet<int>(integrated.DimOrder));
    }

    [Fact]
    public void Test_unary_negation_preserves_dimorder()
    {
        var (_, reord) = Pair();
        var neg = -reord;
        Assert.Equal(reord.DimOrder, neg.DimOrder);
    }

    [Fact]
    public void Test_binary_add_matching_dimorder()
    {
        var (canonical, reord) = Pair();
        // Add reord + reord — same DimOrder, succeeds.
        var sum = reord + reord;
        Assert.Equal(reord.DimOrder, sum.DimOrder);
    }

    [Fact]
    public void Test_binary_add_mismatched_dimorder_throws()
    {
        var (canonical, reord) = Pair();
        var ex = Assert.Throws<ArgumentException>(() => canonical + reord);
        Assert.Contains("dim_order", ex.Message);
        Assert.Contains("Reorder", ex.Message);
    }

    [Fact]
    public void Test_partial_integrate_with_bounds_respects_dimorder()
    {
        // Latent bug: when both _dimOrder ≠ identity AND bounds is non-null,
        // the bounds-to-storage pairing must preserve user order.
        // f(x, y, z) = sin(x) + cos(y) + z
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2];
        var canonical = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 }, maxRank: 8, maxSweeps: 5);
        canonical.Build(verbose: false, seed: 42);
        var reord = canonical.Reorder(new[] { 2, 0, 1 }, maxRank: 16, tolerance: 1e-12);

        // Integrate over user dim 0 with explicit bounds.
        var canIntegrated = (ChebyshevTT)canonical.Integrate(
            dims: new[] { 0 },
            bounds: new[] { (-0.5, 0.5) });
        var reordIntegrated = (ChebyshevTT)reord.Integrate(
            dims: new[] { 0 },
            bounds: new[] { (-0.5, 0.5) });

        // Both must Eval to the same value at any (y, z) point.
        var pt = new[] { 0.3, 0.4 };  // 2D after dim 0 integrated.
        TestFixtures.AssertClose(
            canIntegrated.Eval(pt), reordIntegrated.Eval(pt),
            rtol: 1e-3, atol: 1e-3);
    }
}
