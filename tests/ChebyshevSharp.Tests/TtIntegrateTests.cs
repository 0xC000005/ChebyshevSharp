using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestTtFullIntegrate (Phase 5)
// ======================================================================

public class TestTtFullIntegrate
{
    [Fact]
    public void Test_separable_function_sin_times_cos()
    {
        // f(x, y) = sin(x) * cos(y) over [-1, 1]^2.
        // ∫∫ = (∫sin) (∫cos) = 0 * (2 sin 1) = 0.
        static double F(double[] x) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_constant_function_volume()
    {
        // f = 7 over [0, 2] x [0, 3] integrates to 7 * 6 = 42.
        static double F(double[] x) => 7.0;
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 3.0 } },
            new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(42.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_5d_separable_against_analytical()
    {
        // f(x) = exp(-Σ x_i^2) over [-1, 1]^5.
        // ∫_{-1}^{1} exp(-x^2) dx = sqrt(pi) * erf(1) ≈ 1.49364826562485.
        // Total = (sqrt(pi) * erf(1))^5.
        static double F(double[] x)
        {
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += x[i] * x[i];
            return Math.Exp(-s);
        }
        var domain = new[] {
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }
        };
        var tt = new ChebyshevTT(F, 5, domain, new[] { 10, 10, 10, 10, 10 });
        tt.Build(verbose: false, seed: 42);
        double cheb = (double)tt.Integrate();
        // sqrt(pi) * erf(1)
        double oneD = Math.Sqrt(Math.PI) * Erf(1.0);
        double expected = Math.Pow(oneD, 5);
        TestFixtures.AssertClose(expected, cheb, rtol: 1e-4, atol: 1e-4);
    }

    [Fact]
    public void Test_works_after_method_svd()
    {
        // f(x, y) = x * y; ∫∫ = 0 over [-1, 1]^2.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, method: "svd");
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_works_after_method_als()
    {
        // f(x, y) = x*y + sin(x) over [-1, 1]^2.
        // ∫ x*y dx dy = 0; ∫ sin(x) dx dy = 0 (sin odd, then * 2). Total ≈ 0.
        static double F(double[] x) => x[0] * x[1] + Math.Sin(x[0]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        tt.Build(verbose: false, method: "als", seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_dims_order_invariance()
    {
        // integrate(dims=[0, 1]) == integrate(dims=[1, 0]) (full integration).
        static double F(double[] x) => Math.Sin(x[0]) + Math.Cos(x[1]);
        var ttA = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        ttA.Build(verbose: false, seed: 42);
        double a = (double)ttA.Integrate(dims: new[] { 0, 1 });

        var ttB = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        ttB.Build(verbose: false, seed: 42);
        double b = (double)ttB.Integrate(dims: new[] { 1, 0 });
        TestFixtures.AssertClose(a, b, rtol: 1e-10, atol: 1e-10);
    }

    // Abramowitz & Stegun 7.1.26 erf approximation (sufficient for 1e-4 tolerance).
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        double absX = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * absX);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
            + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
            * Math.Exp(-absX * absX);
        return sign * y;
    }
}

// ======================================================================
// TestTtPartialIntegrate (Phase 5)
// ======================================================================

public class TestTtPartialIntegrate
{
    [Fact]
    public void Test_returns_tt_with_correct_shape()
    {
        static double F(double[] x) => x[0] + x[1] + x[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 });
        tt.Build(verbose: false, seed: 42);
        var result = tt.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevTT>(result);
        var resultTt = (ChebyshevTT)result;
        Assert.Equal(2, resultTt.NumDimensions);
        Assert.Equal(new[] { 6, 6 }, resultTt.NNodes);
    }

    [Fact]
    public void Test_endpoint_dim_left()
    {
        // Integrate over dim 0 (no left neighbor). f(x, y) = x * y.
        // ∫_{-1}^{1} x dx = 0, so result(y) ≈ 0 for all y.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 6 });
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_endpoint_dim_right()
    {
        // Integrate over last dim (no right neighbor). f(x, y) = x * y.
        // ∫_{-1}^{1} y dy = 0, so result(x) ≈ 0.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 1 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_partial_consistent_with_consecutive()
    {
        // integrate([0, 1]) on a 3D TT should equal integrate([1]).integrate([0]).
        // Note: after integrate([1]) returns a 2D TT over (orig_dim_0, orig_dim_2),
        // its "dim 0" is original dim 0 — so chained integrate([0]) is correct.
        static double F(double[] x) => Math.Sin(x[0]) * Math.Cos(x[1]) * (1 + x[2] * x[2]);
        var ttA = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 });
        ttA.Build(verbose: false, seed: 42);
        var ttB = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 });
        ttB.Build(verbose: false, seed: 42);

        var joint = (ChebyshevTT)ttA.Integrate(dims: new[] { 0, 1 });
        var step1 = (ChebyshevTT)ttB.Integrate(dims: new[] { 1 });
        var step2 = (ChebyshevTT)step1.Integrate(dims: new[] { 0 });

        double xTest = 0.3;
        TestFixtures.AssertClose(
            joint.Eval(new[] { xTest }),
            step2.Eval(new[] { xTest }),
            rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_with_sub_interval_bounds()
    {
        // ∫_0^1 x dx = 0.5 over [-1, 1].
        static double F(double[] x) => x[0];
        var tt = new ChebyshevTT(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate(
            dims: new[] { 0 },
            bounds: new[] { (0.0, 1.0) });
        TestFixtures.AssertClose(0.5, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_descriptor_preserved_on_partial_result()
    {
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        tt.SetDescriptor("source");
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Equal("source", result.GetDescriptor());
    }

    [Fact]
    public void Test_additional_data_preserved_on_partial_result()
    {
        var sentinel = new Dictionary<string, int> { ["k"] = 42 };
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            additionalData: sentinel);
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Same(sentinel, result.GetAdditionalData());
    }

    [Fact]
    public void Test_partial_eval_works_recursively()
    {
        // Result of partial integrate must be a fully-functional TT:
        // Eval, EvalBatch, and recursive Integrate all work.
        static double F(double[] x) => x[0] * x[1] + x[0];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, seed: 42);
        // ∫_{-1}^{1} (x*y + x) dx = 0 + 0 = 0 → result(y) = 0 for all y.
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
        // Recursive integrate over the surviving 1D TT yields ∫_{-1}^{1} 0 dy = 0.
        double recursiveScalar = (double)result.Integrate();
        TestFixtures.AssertClose(0.0, recursiveScalar, rtol: 1e-10, atol: 1e-10);
    }
}
