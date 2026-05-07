using Xunit;

namespace ChebyshevSharp.Tests;

public class TtPublicStateOwnershipTests
{
    [Fact]
    public void Domain_property_returns_deep_snapshot()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { 10.0, 20.0 } },
            nNodes: new[] { 5, 5 });
        tt.Build(verbose: false, method: "svd");

        double[][] first = tt.Domain;
        first[0][0] = double.NaN;
        first[1] = new[] { -100.0, -50.0 };

        double[][] second = tt.Domain;
        Assert.Equal(-1.0, second[0][0]);
        Assert.Equal(10.0, second[1][0]);
        Assert.NotSame(first, second);
        Assert.NotSame(first[0], second[0]);
    }

    [Fact]
    public void NNodes_property_returns_snapshot()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 7 });

        int[] first = tt.NNodes;
        first[0] = 999;

        int[] second = tt.NNodes;
        Assert.Equal(new[] { 5, 7 }, second);
        Assert.NotSame(first, second);
    }

    [Fact]
    public void Mutating_domain_snapshot_does_not_change_eval_contract()
    {
        var tt = new ChebyshevTT(
            p => p[0] + 2.0 * p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 7, 7 },
            maxRank: 4);
        tt.Build(verbose: false, method: "svd");

        double before = tt.Eval(new[] { 0.25, -0.5 });
        double[][] domain = tt.Domain;
        domain[0][0] = 10.0;
        domain[0][1] = 20.0;

        double after = tt.Eval(new[] { 0.25, -0.5 });
        Assert.Equal(before, after, precision: 12);
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.Eval(new[] { 2.0, 0.0 }));
    }

    [Fact]
    public void Extrude_result_in_place_mutation_does_not_change_source_tt()
    {
        var source = new ChebyshevTT(
            p => p[0] + 2.0 * p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 6, 6 },
            maxRank: 4);
        source.Build(verbose: false, method: "svd");

        double[] sourcePoint = [0.25, -0.4];
        double before = source.Eval(sourcePoint);

        ChebyshevTT extruded = source.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 4);
        extruded.ScalarMulInPlace(2.0);

        Assert.Equal(before, source.Eval(sourcePoint), precision: 12);
    }

    [Fact]
    public void Slice_result_in_place_mutation_does_not_change_source_tt()
    {
        var source = new ChebyshevTT(
            p => p[0] + 2.0 * p[1] - 0.5 * p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            maxRank: 4);
        source.Build(verbose: false, method: "svd");

        double[] sourcePoint = [0.2, 0.1, -0.3];
        double before = source.Eval(sourcePoint);

        ChebyshevTT sliced = source.Slice(dim: 1, value: 0.25);
        sliced.NegateInPlace();

        Assert.Equal(before, source.Eval(sourcePoint), precision: 12);
    }
}
