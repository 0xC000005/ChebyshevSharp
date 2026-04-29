// tests/ChebyshevSharp.Tests/RecordTypesTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class RecordTypesTests
{
    [Fact]
    public void Domain_implicit_conversion_both_directions()
    {
        double[][] raw = new[] { new[] { 0.0, 1.0 }, new[] { -1.0, 2.0 } };

        // raw -> Domain
        Domain d = raw;
        Assert.Equal(2, d.Bounds.Length);
        Assert.Equal(0.0, d.Bounds[0][0]);
        Assert.Equal(1.0, d.Bounds[0][1]);

        // Domain -> raw
        double[][] back = d;
        Assert.Same(raw, back);
    }

    [Fact]
    public void Ns_implicit_conversion_both_directions()
    {
        int[] raw = new[] { 5, 7 };

        Ns n = raw;
        Assert.Equal(new[] { 5, 7 }, n.Counts);

        int[] back = n;
        Assert.Same(raw, back);
    }

    [Fact]
    public void SpecialPoints_implicit_conversion_both_directions()
    {
        double[][] raw = new[] { new[] { 0.5 }, new[] { 0.7, 0.9 } };

        SpecialPoints sp = raw;
        Assert.Equal(2, sp.Points.Length);
        Assert.Equal(0.5, sp.Points[0][0]);

        double[][] back = sp;
        Assert.Same(raw, back);
    }
}
