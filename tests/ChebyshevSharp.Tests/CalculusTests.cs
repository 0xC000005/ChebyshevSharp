using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestIntegrateApprox
// ======================================================================

public class TestIntegrateApprox
{
    [Fact]
    public void Test_integrate_constant()
    {
        // Integral of constant 5 on [2, 5] = 15.
        static double f(double[] x, object? _) => 5.0;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { 2.0, 5.0 } }, new[] { 4 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        TestFixtures.AssertClose(15.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_x_squared()
    {
        // Integral of x^2 on [-1, 1] = 2/3 (exact for polynomial).
        static double f(double[] x, object? _) => x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        TestFixtures.AssertClose(2.0 / 3.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_x_cubed()
    {
        // Integral of x^3 on [-1, 1] = 0 (odd function, exact).
        static double f(double[] x, object? _) => x[0] * x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        Assert.True(Math.Abs(result) < 1e-10, $"Expected 0.0, got {result}");
    }

    [Fact]
    public void Test_integrate_sin()
    {
        // Integral of sin(x) on [0, pi] = 2 (spectral accuracy). Uses fixture.
        var cheb = TestFixtures.CalculusChebSin1D;
        var result = (double)cheb.Integrate();
        TestFixtures.AssertClose(2.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_cos()
    {
        // Integral of cos(x) on [-1, 1] = 2*sin(1).
        static double f(double[] x, object? _) => Math.Cos(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        double expected = 2.0 * Math.Sin(1.0);
        TestFixtures.AssertClose(expected, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_exp()
    {
        // Integral of exp(x) on [-1, 1] = e - 1/e.
        static double f(double[] x, object? _) => Math.Exp(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        double expected = Math.E - 1.0 / Math.E;
        TestFixtures.AssertClose(expected, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_2d_full()
    {
        // Integral of sin(x)+cos(y) on [-1,1]^2 = 4*sin(1).
        // int sin(x) dx from -1 to 1 = 0 (odd function).
        // int cos(y) dy from -1 to 1 = 2*sin(1).
        // Double integral = 0*2 + 2*2*sin(1) = 4*sin(1).
        var cheb = TestFixtures.CalculusCheb2D;
        var result = (double)cheb.Integrate();
        double expected = 4.0 * Math.Sin(1.0);
        TestFixtures.AssertClose(expected, result, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_integrate_2d_partial_dim0()
    {
        // Integrate out dim 0 from sin(x)+cos(y): result is 2*cos(y).
        var cheb = TestFixtures.CalculusCheb2D;
        var result = (ChebyshevApproximation)cheb.Integrate(dims: new[] { 0 });
        Assert.IsType<ChebyshevApproximation>(result);
        Assert.Equal(1, result.NumDimensions);
        foreach (double y in new[] { -0.7, 0.0, 0.3, 0.9 })
        {
            double val = result.VectorizedEval(new[] { y }, new[] { 0 });
            double expected = 2.0 * Math.Cos(y);
            TestFixtures.AssertClose(expected, val, rtol: 1e-8, atol: 1e-8);
        }
    }

    [Fact]
    public void Test_integrate_2d_partial_dim1()
    {
        // Integrate out dim 1 from sin(x)+cos(y): result is 2*sin(x) + 2*sin(1).
        var cheb = TestFixtures.CalculusCheb2D;
        var result = (ChebyshevApproximation)cheb.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevApproximation>(result);
        Assert.Equal(1, result.NumDimensions);
        foreach (double x in new[] { -0.7, 0.0, 0.3, 0.9 })
        {
            double val = result.VectorizedEval(new[] { x }, new[] { 0 });
            double expected = 2.0 * Math.Sin(x) + 2.0 * Math.Sin(1.0);
            TestFixtures.AssertClose(expected, val, rtol: 1e-8, atol: 1e-8);
        }
    }

    [Fact]
    public void Test_integrate_scaled_domain()
    {
        // Integral of x on [2, 5] = (25-4)/2 = 10.5.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { 2.0, 5.0 } }, new[] { 4 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate();
        TestFixtures.AssertClose(10.5, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_integrate_before_build_raises()
    {
        // integrate() raises InvalidOperationException if build() has not been called.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        Assert.Throws<InvalidOperationException>(() => cheb.Integrate());
    }
}

// ======================================================================
// TestRootsApprox
// ======================================================================

public class TestRootsApprox
{
    [Fact]
    public void Test_roots_sin()
    {
        // Roots of sin(x) on [-4, 4] should include {-pi, 0, pi}.
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -4.0, 4.0 } }, new[] { 25 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        double[] expected = new[] { -Math.PI, 0.0, Math.PI };
        Array.Sort(expected);
        Array.Sort(roots);
        Assert.Equal(3, roots.Length);
        for (int i = 0; i < roots.Length; i++)
        {
            TestFixtures.AssertClose(expected[i], roots[i], rtol: 0, atol: 1e-8);
        }
    }

    [Fact]
    public void Test_roots_quadratic()
    {
        // Roots of x^2 - 0.25 on [-1, 1] should be {-0.5, 0.5}.
        static double f(double[] x, object? _) => x[0] * x[0] - 0.25;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        Array.Sort(roots);
        Assert.Equal(2, roots.Length);
        TestFixtures.AssertClose(-0.5, roots[0], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(0.5, roots[1], rtol: 0, atol: 1e-10);
    }

    [Fact]
    public void Test_roots_no_roots()
    {
        // exp(x) on [0, 1] has no roots.
        static double f(double[] x, object? _) => Math.Exp(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        Assert.Empty(roots);
    }

    [Fact]
    public void Test_roots_constant_nonzero()
    {
        // Constant 5.0 has no roots.
        static double f(double[] x, object? _) => 5.0;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        Assert.Empty(roots);
    }

    [Fact]
    public void Test_roots_boundary()
    {
        // Function x on [0, 1] has root at 0.
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        Assert.True(roots.Length >= 1, $"Expected root near 0, got {roots.Length} roots");
        Assert.True(Array.Exists(roots, r => Math.Abs(r) < 1e-8),
            $"No root near 0 in [{string.Join(", ", roots)}]");
    }

    [Fact]
    public void Test_roots_2d_fixed()
    {
        // 2D roots: f(x,y) = x - y, with y fixed at 0.3, root at x=0.3.
        static double f(double[] x, object? _) => x[0] - x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.3 } });
        Assert.Single(roots);
        TestFixtures.AssertClose(0.3, roots[0], rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_roots_missing_fixed_raises()
    {
        // Multi-D without full fixed dict raises ArgumentException.
        static double f(double[] x, object? _) => x[0] + x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        cheb.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => cheb.Roots(dim: 0));
    }

    [Fact]
    public void Test_roots_fixed_out_of_domain_raises()
    {
        // Fixed value outside domain raises ArgumentException.
        static double f(double[] x, object? _) => x[0] + x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        cheb.Build(verbose: false);
        Assert.Throws<ArgumentException>(() =>
            cheb.Roots(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 5.0 } }));
    }

    [Fact]
    public void Test_roots_before_build_raises()
    {
        // roots() raises InvalidOperationException before build().
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        Assert.Throws<InvalidOperationException>(() => cheb.Roots());
    }

    [Fact]
    public void Test_roots_linear_n2()
    {
        // Linear function x - 0.3 on [-1,1] with n=2 nodes, root at 0.3.
        static double f(double[] x, object? _) => x[0] - 0.3;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 2 });
        cheb.Build(verbose: false);
        double[] roots = cheb.Roots();
        Assert.Single(roots);
        TestFixtures.AssertClose(0.3, roots[0], rtol: 0, atol: 1e-10);
    }
}

// ======================================================================
// TestMinMaxApprox
// ======================================================================

public class TestMinMaxApprox
{
    [Fact]
    public void Test_minimize_x_squared()
    {
        // Min of x^2 on [-1, 1] -> (0, 0).
        static double f(double[] x, object? _) => x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Minimize();
        TestFixtures.AssertClose(0.0, val, rtol: 0, atol: 1e-8);
        TestFixtures.AssertClose(0.0, loc, rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_maximize_x_squared()
    {
        // Max of x^2 on [-1, 1] -> (1, -1 or 1) (at boundary).
        static double f(double[] x, object? _) => x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Maximize();
        TestFixtures.AssertClose(1.0, val, rtol: 0, atol: 1e-8);
        Assert.True(Math.Abs(Math.Abs(loc) - 1.0) < 1e-8,
            $"Max location {loc} not at +/-1");
    }

    [Fact]
    public void Test_maximize_sin()
    {
        // Max of sin(x) on [0, pi] -> (1, pi/2).
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1,
            new[] { new[] { 0.0, Math.PI } }, new[] { 15 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Maximize();
        TestFixtures.AssertClose(1.0, val, rtol: 0, atol: 1e-8);
        TestFixtures.AssertClose(Math.PI / 2.0, loc, rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_minimize_sin_wide()
    {
        // Min of sin(x) on [0, 3*pi] -> (-1, 3*pi/2).
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1,
            new[] { new[] { 0.0, 3.0 * Math.PI } }, new[] { 25 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Minimize();
        TestFixtures.AssertClose(-1.0, val, rtol: 0, atol: 1e-6);
        TestFixtures.AssertClose(3.0 * Math.PI / 2.0, loc, rtol: 0, atol: 1e-6);
    }

    [Fact]
    public void Test_minimize_2d_fixed()
    {
        // 2D min: f(x,y) = x^2 + y, fix y=0.5, min at x=0, value=0.5.
        static double f(double[] x, object? _) => x[0] * x[0] + x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Minimize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.5 } });
        TestFixtures.AssertClose(0.5, val, rtol: 0, atol: 1e-8);
        TestFixtures.AssertClose(0.0, loc, rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_maximize_2d_fixed()
    {
        // 2D max: f(x,y) = -x^2 + y, fix y=0.5, max at x=0, value=0.5.
        static double f(double[] x, object? _) => -x[0] * x[0] + x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        var (val, loc) = cheb.Maximize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.5 } });
        TestFixtures.AssertClose(0.5, val, rtol: 0, atol: 1e-8);
        TestFixtures.AssertClose(0.0, loc, rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_minimize_constant()
    {
        // Constant function: min = max = constant.
        static double f(double[] x, object? _) => 3.14;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        cheb.Build(verbose: false);
        var (minVal, _) = cheb.Minimize();
        var (maxVal, _) = cheb.Maximize();
        TestFixtures.AssertClose(3.14, minVal, rtol: 0, atol: 1e-8);
        TestFixtures.AssertClose(3.14, maxVal, rtol: 0, atol: 1e-8);
    }

    [Fact]
    public void Test_minimize_before_build_raises()
    {
        // minimize() raises InvalidOperationException before build().
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        Assert.Throws<InvalidOperationException>(() => cheb.Minimize());
    }

    [Fact]
    public void Test_maximize_missing_fixed_raises()
    {
        // Multi-D maximize without full fixed dict raises ArgumentException.
        static double f(double[] x, object? _) => x[0] + x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        cheb.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => cheb.Maximize(dim: 0));
    }

    [Fact]
    public void Test_minimize_returns_tuple()
    {
        // Verify minimize() return type is a tuple of two doubles.
        static double f(double[] x, object? _) => x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 11 });
        cheb.Build(verbose: false);
        var result = cheb.Minimize();
        // In C#, the result is a named tuple (double value, double location)
        Assert.IsType<double>(result.value);
        Assert.IsType<double>(result.location);
    }
}

// ======================================================================
// TestSubIntervalIntegrateApprox
// ======================================================================

public class TestSubIntervalIntegrateApprox
{
    [Fact]
    public void Test_constant_half_domain()
    {
        // Integral of constant 5 on [0, 1] sub-interval of [-1, 1] = 5.
        static double f(double[] x, object? _) => 5.0;
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (0.0, 1.0) });
        TestFixtures.AssertClose(5.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_x_squared_sub_interval()
    {
        // Integral of x^2 on [0, 1] sub-interval of [-1, 1] = 1/3.
        static double f(double[] x, object? _) => x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (0.0, 1.0) });
        TestFixtures.AssertClose(1.0 / 3.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_sin_half_period()
    {
        // Integral of sin(x) on [0, pi] sub-interval of [0, 2*pi] = 2.
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1,
            new[] { new[] { 0.0, 2.0 * Math.PI } }, new[] { 25 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (0.0, Math.PI) });
        TestFixtures.AssertClose(2.0, result, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_exp_sub_interval()
    {
        // Integral of exp(x) on [0, 1] sub-interval of [-1, 1] = e - 1.
        static double f(double[] x, object? _) => Math.Exp(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (0.0, 1.0) });
        double expected = Math.E - 1.0;
        TestFixtures.AssertClose(expected, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_bounds_equal_domain()
    {
        // bounds == domain should match integrate() with no bounds.
        static double f(double[] x, object? _) => Math.Sin(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 15 });
        cheb.Build(verbose: false);
        var noBounds = (double)cheb.Integrate();
        var withBounds = (double)cheb.Integrate(bounds: new[] { (-1.0, 1.0) });
        TestFixtures.AssertClose(noBounds, withBounds, rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void Test_degenerate_lo_eq_hi()
    {
        // bounds lo == hi should give integral = 0.
        static double f(double[] x, object? _) => Math.Exp(x[0]);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (0.5, 0.5) });
        Assert.True(Math.Abs(result) < 1e-14, $"Expected 0.0, got {result}");
    }

    [Fact]
    public void Test_polynomial_exactness()
    {
        // Integral of x^4 on [-0.5, 0.5] sub-interval of [-1, 1] = 1/80.
        static double f(double[] x, object? _) => x[0] * x[0] * x[0] * x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(bounds: new[] { (-0.5, 0.5) });
        double expected = 1.0 / 80.0;
        TestFixtures.AssertClose(expected, result, rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void Test_adjacent_sub_intervals_sum()
    {
        // Adjacent sub-intervals sum to full integral.
        static double f(double[] x, object? _) => Math.Sin(x[0]) * Math.Exp(x[0] / 3.0);
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -2.0, 2.0 } }, new[] { 20 });
        cheb.Build(verbose: false);
        var full = (double)cheb.Integrate();
        var left = (double)cheb.Integrate(bounds: new[] { (-2.0, 0.5) });
        var right = (double)cheb.Integrate(bounds: new[] { (0.5, 2.0) });
        TestFixtures.AssertClose(full, left + right, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_2d_partial_sub_interval()
    {
        // 2D: integrate dim 0 on sub-interval [0, 1], result is 1D approx 1/3 + cos(y).
        static double f(double[] x, object? _) => x[0] * x[0] + Math.Cos(x[1]);
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 11, 11 });
        cheb.Build(verbose: false);
        var result = (ChebyshevApproximation)cheb.Integrate(
            dims: new[] { 0 }, bounds: new[] { (0.0, 1.0) });
        Assert.IsType<ChebyshevApproximation>(result);
        Assert.Equal(1, result.NumDimensions);
        foreach (double y in new[] { -0.5, 0.0, 0.7 })
        {
            double val = result.VectorizedEval(new[] { y }, new[] { 0 });
            double expected = 1.0 / 3.0 + Math.Cos(y);
            TestFixtures.AssertClose(expected, val, rtol: 1e-8, atol: 1e-8);
        }
    }

    [Fact]
    public void Test_2d_mixed_bounds()
    {
        // 2D: dims=[0,1], bounds on dim 0 = [0,1], dim 1 = full domain [-1,1].
        // int_0^1 x^2 dx * int_{-1}^1 1 dy + int_0^1 1 dx * int_{-1}^1 y^2 dy
        // = (1/3)*2 + 1*(2/3) = 2/3 + 2/3 = 4/3
        static double f(double[] x, object? _) => x[0] * x[0] + x[1] * x[1];
        var cheb = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 10, 10 });
        cheb.Build(verbose: false);
        var result = (double)cheb.Integrate(
            dims: new[] { 0, 1 }, bounds: new[] { (0.0, 1.0), (-1.0, 1.0) });
        double expected = 4.0 / 3.0;
        TestFixtures.AssertClose(expected, result, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_bounds_out_of_domain_raises()
    {
        // bounds outside domain raises ArgumentException with "outside domain".
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        cheb.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() =>
            cheb.Integrate(bounds: new[] { (-2.0, 0.5) }));
        Assert.Contains("outside domain", ex.Message);
    }

    [Fact]
    public void Test_bounds_lo_gt_hi_raises()
    {
        // bounds lo > hi raises ArgumentException with "lo=".
        static double f(double[] x, object? _) => x[0];
        var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        cheb.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() =>
            cheb.Integrate(bounds: new[] { (0.5, -0.5) }));
        Assert.Contains("lo=", ex.Message);
    }
}
