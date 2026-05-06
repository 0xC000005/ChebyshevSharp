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
    public void Approx_SetValues_null_throws_argument_null()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);

        Assert.Throws<ArgumentNullException>(() => deferred.SetOriginalFunctionValues(null!));
        Assert.False(deferred.IsConstructionFinished());
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Approx_SetValues_non_finite_throws_and_remains_deferred(double badValue)
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);
        double[] values = [0.0, 1.0, badValue, 3.0, 4.0];

        Assert.Throws<ArgumentException>(() => deferred.SetOriginalFunctionValues(values));
        Assert.False(deferred.IsConstructionFinished());
    }

    [Fact]
    public void Approx_SetValues_wrong_length_throws_and_remains_deferred()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);

        Assert.Throws<ArgumentException>(() => deferred.SetOriginalFunctionValues([0.0, 1.0]));
        Assert.False(deferred.IsConstructionFinished());
    }

    [Fact]
    public void Approx_SetValues_after_construction_throws_invalid_operation()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);
        deferred.SetOriginalFunctionValues([0.0, 1.0, 2.0, 3.0, 4.0]);

        Assert.Throws<InvalidOperationException>(() =>
            deferred.SetOriginalFunctionValues([4.0, 3.0, 2.0, 1.0, 0.0]));
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

    [Fact]
    public void Spline_SetValues_null_throws_argument_null()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);

        Assert.Throws<ArgumentNullException>(() => deferred.SetOriginalFunctionValues(null!));
        Assert.False(deferred.IsConstructionFinished());
    }

    [Fact]
    public void Spline_SetValues_non_finite_failure_does_not_partially_construct_pieces()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { new[] { 0.0 } },
            deferBuild: true);
        double[] values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, double.NaN, 7.0, 8.0, 9.0];

        Assert.Throws<ArgumentException>(() => deferred.SetOriginalFunctionValues(values));
        Assert.False(deferred.IsConstructionFinished());
        Assert.Equal(0, deferred.GetNumEvaluationPoints());
    }

    [Fact]
    public void Spline_SetValues_wrong_length_throws_and_remains_deferred()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { new[] { 0.0 } },
            deferBuild: true);

        Assert.Throws<ArgumentException>(() => deferred.SetOriginalFunctionValues([0.0, 1.0]));
        Assert.False(deferred.IsConstructionFinished());
        Assert.Equal(0, deferred.GetNumEvaluationPoints());
    }

    [Fact]
    public void Spline_SetValues_after_construction_throws_invalid_operation()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);
        deferred.SetOriginalFunctionValues([0.0, 1.0, 2.0, 3.0, 4.0]);

        Assert.Throws<InvalidOperationException>(() =>
            deferred.SetOriginalFunctionValues([4.0, 3.0, 2.0, 1.0, 0.0]));
    }
}
