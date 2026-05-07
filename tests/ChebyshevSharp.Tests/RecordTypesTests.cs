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
        Assert.NotSame(raw, back);
        Assert.Equal(raw, back);
    }

    [Fact]
    public void Ns_implicit_conversion_both_directions()
    {
        int[] raw = new[] { 5, 7 };

        Ns n = raw;
        Assert.Equal(new[] { 5, 7 }, n.Counts);

        int[] back = n;
        Assert.NotSame(raw, back);
        Assert.Equal(raw, back);
    }

    [Fact]
    public void SpecialPoints_implicit_conversion_both_directions()
    {
        double[][] raw = new[] { new[] { 0.5 }, new[] { 0.7, 0.9 } };

        SpecialPoints sp = raw;
        Assert.Equal(2, sp.Points.Length);
        Assert.Equal(0.5, sp.Points[0][0]);

        double[][] back = sp;
        Assert.NotSame(raw, back);
        Assert.Equal(raw, back);
    }

    [Fact]
    public void Domain_Ns_and_SpecialPoints_snapshot_constructor_inputs_and_properties()
    {
        double[][] rawDomain = new[] { new[] { 0.0, 1.0 }, new[] { -1.0, 2.0 } };
        int[] rawNs = new[] { 5, 7 };
        double[][] rawSpecialPoints = new[] { new[] { 0.5 }, new[] { 0.7, 0.9 } };

        Domain domain = rawDomain;
        Ns ns = rawNs;
        SpecialPoints specialPoints = rawSpecialPoints;

        rawDomain[0][0] = 10.0;
        rawNs[0] = 99;
        rawSpecialPoints[1][0] = -5.0;

        Assert.Equal(0.0, domain.Bounds[0][0]);
        Assert.Equal(new[] { 5, 7 }, ns.Counts);
        Assert.Equal(0.7, specialPoints.Points[1][0]);

        double[][] domainOut = domain.Bounds;
        int[] nsOut = ns.Counts;
        double[][] specialOut = specialPoints.Points;
        domainOut[0][0] = 20.0;
        nsOut[0] = 100;
        specialOut[1][0] = -10.0;

        Assert.Equal(0.0, domain.Bounds[0][0]);
        Assert.Equal(new[] { 5, 7 }, ns.Counts);
        Assert.Equal(0.7, specialPoints.Points[1][0]);
    }

    [Fact]
    public void SobolResult_arrays_return_snapshots()
    {
        var firstOrder = new[] { 0.25, 0.75 };
        var totalOrder = new[] { 0.5, 1.0 };
        var result = new SobolResult(firstOrder, totalOrder, 1.25);

        firstOrder[0] = 9.0;
        totalOrder[1] = 9.0;

        Assert.Equal(0.25, result.FirstOrder[0]);
        Assert.Equal(1.0, result.TotalOrder[1]);

        double[] firstOut = result.FirstOrder;
        double[] totalOut = result.TotalOrder;
        firstOut[0] = 10.0;
        totalOut[1] = 10.0;

        Assert.Equal(0.25, result.FirstOrder[0]);
        Assert.Equal(1.0, result.TotalOrder[1]);
        Assert.Equal(1.25, result.Variance);
    }

    [Fact]
    public void Record_type_deconstruction_returns_snapshots()
    {
        var domain = new Domain(new[] { new[] { 0.0, 1.0 } });
        var ns = new Ns(new[] { 5, 7 });
        var specialPoints = new SpecialPoints(new[] { new[] { 0.5, 0.75 } });
        var sobolResult = new SobolResult(new[] { 0.25, 0.75 }, new[] { 0.5, 1.0 }, 1.25);

        domain.Deconstruct(out double[][] domainBounds);
        ns.Deconstruct(out int[] counts);
        specialPoints.Deconstruct(out double[][] points);
        var (firstOrder, totalOrder, variance) = sobolResult;

        domainBounds[0][0] = 10.0;
        counts[0] = 99;
        points[0][0] = 9.0;
        firstOrder[0] = 9.0;
        totalOrder[0] = 9.0;

        Assert.Equal(0.0, domain.Bounds[0][0]);
        Assert.Equal(new[] { 5, 7 }, ns.Counts);
        Assert.Equal(0.5, specialPoints.Points[0][0]);
        Assert.Equal(0.25, sobolResult.FirstOrder[0]);
        Assert.Equal(0.5, sobolResult.TotalOrder[0]);
        Assert.Equal(1.25, variance);
    }

    [Fact]
    public void With_expressions_snapshot_init_values()
    {
        var updatedDomain = new[] { new[] { -1.0, 1.0 } };
        var updatedCounts = new[] { 9, 11 };
        var updatedPoints = new[] { new[] { 0.25, 0.75 } };
        var updatedFirstOrder = new[] { 0.4, 0.6 };
        var updatedTotalOrder = new[] { 0.8, 1.0 };

        var domain = new Domain(new[] { new[] { 0.0, 1.0 } }) with { Bounds = updatedDomain };
        var ns = new Ns(new[] { 5, 7 }) with { Counts = updatedCounts };
        var specialPoints = new SpecialPoints(new[] { new[] { 0.5 } }) with { Points = updatedPoints };
        var sobolResult = new SobolResult(new[] { 0.25, 0.75 }, new[] { 0.5, 1.0 }, 1.25)
        {
            FirstOrder = updatedFirstOrder,
            TotalOrder = updatedTotalOrder,
            Variance = 2.5,
        };

        updatedDomain[0][0] = 9.0;
        updatedCounts[0] = 99;
        updatedPoints[0][0] = 9.0;
        updatedFirstOrder[0] = 9.0;
        updatedTotalOrder[0] = 9.0;

        Assert.Equal(-1.0, domain.Bounds[0][0]);
        Assert.Equal(new[] { 9, 11 }, ns.Counts);
        Assert.Equal(0.25, specialPoints.Points[0][0]);
        Assert.Equal(0.4, sobolResult.FirstOrder[0]);
        Assert.Equal(0.8, sobolResult.TotalOrder[0]);
        Assert.Equal(2.5, sobolResult.Variance);
    }
}
