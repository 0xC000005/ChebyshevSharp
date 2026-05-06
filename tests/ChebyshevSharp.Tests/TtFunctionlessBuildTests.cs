using Xunit;
using ChebyshevSharp;

namespace ChebyshevSharp.Tests;

public class TtFunctionlessBuildTests
{
    [Fact]
    public void Build_OnFromValues_RaisesInvalidOperation()
    {
        var dense = new double[9];
        var tt = ChebyshevTT.FromValues(
            dense,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 3, 3 },
            maxRank: 2);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            tt.Build(verbose: false, method: "svd"));

        Assert.Contains("Function", ex.Message);
        Assert.True(tt.IsConstructionFinished());
    }
}
