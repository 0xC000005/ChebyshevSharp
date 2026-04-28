// tests/ChebyshevSharp.Tests/DeferBuildTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class DeferBuildTests
{
    [Fact]
    public void Approx_DeferBuild_then_SetValues_matches_FromValues()
    {
        var values = new double[5 * 5];
        for (int i = 0; i < 25; i++) values[i] = i * 0.1;
        var fromValues = ChebyshevApproximation.FromValues(
            values,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });

        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            deferBuild: true);
        Assert.False(deferred.IsConstructionFinished());
        deferred.SetOriginalFunctionValues(values);
        Assert.True(deferred.IsConstructionFinished());

        double[] pt = { 0.3, 0.5 };
        Assert.Equal(fromValues.Eval(pt), deferred.Eval(pt), precision: 12);
        Assert.Equal("from_values", deferred.GetConstructorType());
    }

    [Fact]
    public void Approx_DeferBuild_Eval_before_SetValues_throws()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);
        Assert.Throws<InvalidOperationException>(() => deferred.Eval(new[] { 0.5 }));
    }

    [Fact]
    public void Spline_DeferBuild_then_SetValues_works()
    {
        var values = new double[5];
        for (int i = 0; i < 5; i++) values[i] = i * 0.1;
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);
        Assert.False(deferred.IsConstructionFinished());
        deferred.SetOriginalFunctionValues(values);
        Assert.True(deferred.IsConstructionFinished());
    }

    [Fact]
    public void Spline_DeferBuild_Save_before_SetValues_throws()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);
        string tmp = System.IO.Path.GetTempFileName();
        try
        {
            Assert.Throws<InvalidOperationException>(() => deferred.Save(tmp));
        }
        finally
        {
            System.IO.File.Delete(tmp);
        }
    }
}
