using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTAddition + TestTTScalarMul + TestTTCrossFeatures (PyChebyshev v0.18.0).
// Tests added incrementally across Phase 2 Tasks 9 and 10.
public class TtAlgebraTests
{
}

public class BinaryAlgebraTests
{
    [Fact]
    public void Test_add_two_tts_returns_tt()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        var result = ttF + ttG;
        Assert.IsType<ChebyshevTT>(result);
    }

    [Fact]
    public void Test_add_eval_matches_sum_of_evals()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        var result = ttF + ttG;
        foreach (double[] p in new[] { new[] { 0.3, 0.4 }, new[] { -0.2, 0.5 }, new[] { 0.0, 0.0 } })
            TestFixtures.AssertClose(ttF.Eval(p) + ttG.Eval(p), result.Eval(p), atol: 1e-6);
    }

    [Fact]
    public void Test_subtract_returns_tt()
    {
        var ttA = TestFixtures.TtAlgebraF;
        var ttB = TestFixtures.TtAlgebraF;
        var result = ttA - ttB;
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.3, 0.4 }), atol: 1e-6);
    }

    [Fact]
    public void Test_add_incompatible_domain_raises()
    {
        var ttF = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttG = new ChebyshevTT(p => p[0], 1, new[] { new[] { 0.0, 2.0 } }, new[] { 4 });
        ttF.Build(verbose: false);
        ttG.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => { var _ = ttF + ttG; });
    }

    [Fact]
    public void Test_add_incompatible_n_nodes_raises()
    {
        var ttF = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttG = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttF.Build(verbose: false);
        ttG.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => { var _ = ttF + ttG; });
    }

    [Fact]
    public void Test_add_function_is_null_on_result()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var result = ttA + ttB;
        Assert.Throws<InvalidOperationException>((Action)(() => result.RunCompletion()));
    }

    [Fact]
    public void Test_chained_adds_respect_max_rank()
    {
        ChebyshevTT MakeTt(double coef)
        {
            var tt = new ChebyshevTT(p => coef * (p[0] + p[1]), 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 6, 6 }, maxRank: 4);
            tt.Build(verbose: false);
            return tt;
        }
        var result = MakeTt(1.0) + MakeTt(2.0) + MakeTt(3.0);
        foreach (int r in result.TtRanks)
            Assert.True(r <= 4, $"max_rank=4 violated; got rank {r}");
    }

    [Fact]
    public void Test_linearity_eval()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        // (a*f + b*g).eval(x) ≈ a*f(x) + b*g(x)
        double a = 2.0, b = -1.5;
        var combo = a * ttF + b * ttG;
        foreach (double[] p in new[] { new[] { 0.1, 0.2 }, new[] { -0.3, 0.5 } })
        {
            double expected = a * ttF.Eval(p) + b * ttG.Eval(p);
            TestFixtures.AssertClose(expected, combo.Eval(p), atol: 1e-6);
        }
    }
}

public class BinaryInPlaceTests
{
    [Fact]
    public void Test_add_in_place_matches_functional()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        double[] xs = { -0.5, 0.0, 0.5 };
        var functional = ttA + ttB;
        ttA.AddInPlace(ttB);
        foreach (double x in xs)
            TestFixtures.AssertClose(functional.Eval(new[] { x }), ttA.Eval(new[] { x }), atol: 1e-10);
    }

    [Fact]
    public void Test_sub_in_place_matches_functional()
    {
        var ttA = new ChebyshevTT(p => p[0] + 1.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var functional = ttA - ttB;
        ttA.SubInPlace(ttB);
        TestFixtures.AssertClose(functional.Eval(new[] { 0.3 }), ttA.Eval(new[] { 0.3 }), atol: 1e-10);
    }

    [Fact]
    public void Test_add_in_place_grid_mismatch_raises()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => ttA.AddInPlace(ttB));
    }
}

public class RoundingTests
{
    [Fact]
    public void Test_round_in_place_shrinks_rank_without_losing_accuracy()
    {
        // Build a sum that has artificially high rank, then round.
        var ttA = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 }, maxRank: 8);
        var ttB = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 }, maxRank: 8);
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        // Block-diag sum has rank doubled.
        var sum = ttA + ttB;
        double evalBefore = sum.Eval(new[] { 0.3, -0.4 });
        sum.RoundInPlace(1e-10);
        double evalAfter = sum.Eval(new[] { 0.3, -0.4 });
        TestFixtures.AssertClose(evalBefore, evalAfter, atol: 1e-8);
    }

    [Fact]
    public void Test_round_in_place_idempotent()
    {
        var ttA = TestFixtures.TtAlgebraF;
        var sum = ttA + ttA;
        sum.RoundInPlace(1e-10);
        var ranksBefore = sum.TtRanks;
        sum.RoundInPlace(1e-10);
        var ranksAfter = sum.TtRanks;
        Assert.Equal(ranksBefore, ranksAfter);
    }
}

public class ScalarAlgebraTests
{
    [Fact]
    public void Test_scalar_mul_returns_tt()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt * 2.5;
        Assert.IsType<ChebyshevTT>(result);
    }

    [Fact]
    public void Test_scalar_mul_eval_scales()
    {
        var tt = TestFixtures.TtAlgebraF;
        var result = tt * 3.0;
        foreach (double[] p in new[] { new[] { -0.5, 0.0 }, new[] { 0.5, 0.5 }, new[] { 0.0, -0.7 } })
            TestFixtures.AssertClose(3.0 * tt.Eval(p), result.Eval(p), atol: 1e-10);
    }

    [Fact]
    public void Test_rmul_works()
    {
        var tt = TestFixtures.TtAlgebraF;
        var lhs = 2.5 * tt;
        var rhs = tt * 2.5;
        foreach (double[] p in new[] { new[] { 0.3, -0.4 } })
            TestFixtures.AssertClose(lhs.Eval(p), rhs.Eval(p), atol: 1e-12);
    }

    [Fact]
    public void Test_truediv_scalar()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt / 2.0;
        TestFixtures.AssertClose(0.2, result.Eval(new[] { 0.4 }), atol: 1e-10);
    }

    [Fact]
    public void Test_truediv_by_zero_raises()
    {
        var tt = TestFixtures.TtAlgebraF;
        Assert.Throws<DivideByZeroException>(() => tt / 0.0);
    }

    [Fact]
    public void Test_unary_neg()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        tt.Build(verbose: false);
        var neg = -tt;
        TestFixtures.AssertClose(-tt.Eval(new[] { 0.5 }), neg.Eval(new[] { 0.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_mul_zero_yields_zero_tt()
    {
        var tt = TestFixtures.TtAlgebraF;
        var zero = tt * 0.0;
        foreach (double[] p in new[] { new[] { 0.3, -0.4 }, new[] { -0.7, 0.1 } })
            TestFixtures.AssertClose(0.0, zero.Eval(p), atol: 1e-12);
    }

    [Fact]
    public void Test_scalar_mul_function_null_on_result()
    {
        var tt = TestFixtures.TtAlgebraF;
        var result = tt * 2.0;
        Assert.Throws<InvalidOperationException>((Action)(() => result.RunCompletion()));
    }

    [Fact]
    public void Test_unbuilt_scalar_operations_raise_invalid_operation()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });

        Assert.Throws<InvalidOperationException>(() => { var _ = tt * 2.0; });
        Assert.Throws<InvalidOperationException>(() => { var _ = 2.0 * tt; });
        Assert.Throws<InvalidOperationException>(() => { var _ = tt / 2.0; });
        Assert.Throws<InvalidOperationException>(() => { var _ = -tt; });
        Assert.Throws<InvalidOperationException>(() => tt.ScalarMulInPlace(2.0));
        Assert.Throws<InvalidOperationException>(() => tt.ScalarDivInPlace(2.0));
        Assert.Throws<InvalidOperationException>(() => tt.NegateInPlace());
    }
}

public class ScalarInPlaceTests
{
    [Fact]
    public void Test_scalar_mul_in_place_mutates()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        double before = tt.Eval(new[] { 0.5 });
        tt.ScalarMulInPlace(2.0);
        TestFixtures.AssertClose(2.0 * before, tt.Eval(new[] { 0.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_div_in_place()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        tt.ScalarDivInPlace(4.0);
        TestFixtures.AssertClose(0.2, tt.Eval(new[] { 0.8 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_div_in_place_by_zero_raises()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<DivideByZeroException>(() => tt.ScalarDivInPlace(0.0));
    }

    [Fact]
    public void Test_negate_in_place()
    {
        var tt = new ChebyshevTT(p => p[0] + 1.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false);
        double before = tt.Eval(new[] { 0.3 });
        tt.NegateInPlace();
        TestFixtures.AssertClose(-before, tt.Eval(new[] { 0.3 }), atol: 1e-10);
    }
}

public class TtAlgebraCoverageTests
{
    [Fact]
    public void Test_3d_addition_linearity_covers_block_diag_middle_core()
    {
        // 3D sum exercises AddCores' interior-core block-diagonal branch (k != 0 && k != d-1).
        // Verifies both the algorithm AND that the represented function adds linearly.
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8, 8 };
        var ttA = new ChebyshevTT(p => Math.Sin(p[0]) + 0.5 * p[1], 3, domain, nNodes,
            tolerance: 1e-8, maxRank: 6);
        var ttB = new ChebyshevTT(p => Math.Cos(p[2]) * p[0], 3, domain, nNodes,
            tolerance: 1e-8, maxRank: 6);
        ttA.Build(verbose: false, seed: 11);
        ttB.Build(verbose: false, seed: 13);

        var sum = ttA + ttB;
        var pts = new[]
        {
            new[] { 0.1, -0.2, 0.3 },
            new[] { 0.5, 0.5, -0.5 },
            new[] { -0.7, 0.4, 0.9 },
        };
        foreach (var p in pts)
        {
            double expected = ttA.Eval(p) + ttB.Eval(p);
            double actual = sum.Eval(p);
            Assert.True(Math.Abs(expected - actual) < 1e-9,
                $"3D + linearity broke at point [{string.Join(", ", p)}]: {expected} vs {actual}");
        }
    }

    [Fact]
    public void Test_binary_op_raises_on_num_dim_mismatch()
    {
        var tt2D = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        var tt3D = new ChebyshevTT(p => p[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 }, tolerance: 1e-4, maxRank: 3);
        tt2D.Build(verbose: false, seed: 1);
        tt3D.Build(verbose: false, seed: 2);
        Assert.Throws<ArgumentException>(() => tt2D + tt3D);
    }
}
