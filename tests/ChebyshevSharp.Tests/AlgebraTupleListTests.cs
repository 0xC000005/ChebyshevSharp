using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class AlgebraTupleListTests
{
    [Fact]
    public void Test_mixed_allocations_with_identical_bounds_compose()
    {
        // Two ChebyshevApproximations with bounds expressed via different
        // double[][] allocations but numerically identical: + must succeed.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        Func<double[], object?, double> g = (p, _) => p[0] - p[1];

        var d1 = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var d2 = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        Assert.NotSame(d1, d2);  // distinct allocations

        var a = new ChebyshevApproximation(f, 2, d1, new[] { 6, 6 });
        var b = new ChebyshevApproximation(g, 2, d2, new[] { 6, 6 });
        a.Build();
        b.Build();

        var c = a + b;
        Assert.NotNull(c);
    }

    [Fact]
    public void Test_genuinely_different_domain_still_throws()
    {
        // Domains differ by 0.5; must still throw "Domain mismatch".
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.5 } }, new[] { 6 });
        a.Build();
        b.Build();

        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Contains("Domain mismatch", ex.Message);
    }

    [Fact]
    public void Test_tiny_floating_difference_still_compose()
    {
        // Two operands constructed with bounds that differ by IEEE-754 noise.
        // Use rtol=1e-5, atol=1e-8 (np.allclose defaults).
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 + 5e-9 } }, new[] { 6 });
        a.Build();
        b.Build();

        var c = a + b;  // should succeed: difference is below atol
        Assert.NotNull(c);
    }

    [Fact]
    public void Test_difference_above_tolerance_throws()
    {
        // Difference > rtol * 1.0 = 1e-5; should throw.
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.001 } }, new[] { 6 });
        a.Build();
        b.Build();

        Assert.Throws<ArgumentException>(() => { var _ = a + b; });
    }

    [Fact]
    public void Test_node_count_mismatch_still_exact()
    {
        // n_nodes is int[], stays exact comparison.
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 7 });
        a.Build();
        b.Build();

        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Contains("Node count mismatch", ex.Message);
    }
}
