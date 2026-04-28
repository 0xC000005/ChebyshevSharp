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
