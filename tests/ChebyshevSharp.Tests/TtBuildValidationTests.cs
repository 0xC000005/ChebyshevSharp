using Xunit;
using ChebyshevSharp;

namespace ChebyshevSharp.Tests;

public class TtBuildValidationTests
{
    [Theory]
    [InlineData("cross")]
    [InlineData("svd")]
    [InlineData("als")]
    public void Build_rejects_non_finite_function_values(string method)
    {
        var tt = new ChebyshevTT(
            _ => double.NaN,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 3, 3 },
            maxRank: 2);

        var ex = Assert.Throws<ArgumentException>(() =>
            tt.Build(verbose: false, seed: 42, method: method));

        Assert.Equal("function", ex.ParamName);
        Assert.Contains("non-finite", ex.Message);
        Assert.False(tt.IsConstructionFinished());
    }

    [Fact]
    public void RunCompletion_rejects_non_finite_function_values()
    {
        bool returnNonFinite = false;
        double F(double[] x) => returnNonFinite ? double.PositiveInfinity : x[0] + x[1];

        var tt = new ChebyshevTT(
            F,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 3, 3 },
            maxRank: 2);
        tt.Build(verbose: false, method: "svd");

        returnNonFinite = true;
        var ex = Assert.Throws<ArgumentException>(() =>
            tt.RunCompletion(tolerance: 1e-8, maxIter: 1, verbose: false));

        Assert.Equal("function", ex.ParamName);
        Assert.Contains("non-finite", ex.Message);
        Assert.True(tt.IsConstructionFinished());
    }
}
