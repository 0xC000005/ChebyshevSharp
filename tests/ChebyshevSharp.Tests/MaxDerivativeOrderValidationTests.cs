namespace ChebyshevSharp.Tests;

public class MaxDerivativeOrderValidationTests
{
    private static readonly double[][] OneDimDomain = [[-1.0, 1.0]];
    private static readonly int[] OneDimNodes = [5];
    private static readonly int?[] OneDimNullableNodes = [5];

    [Fact]
    public void Approximation_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevApproximation(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Approximation_adaptive_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevApproximation(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNullableNodes,
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Approximation_from_values_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ChebyshevApproximation.FromValues(
                new double[5],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Spline_flat_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSpline(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                knots: [Array.Empty<double>()],
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Spline_adaptive_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSpline(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNullableNodes,
                knots: [Array.Empty<double>()],
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Spline_nested_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSpline(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodesNested: [OneDimNodes],
                knots: [Array.Empty<double>()],
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Spline_from_values_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ChebyshevSpline.FromValues(
                [new double[5]],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                knots: [Array.Empty<double>()],
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Slider_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSlider(
                (p, _) => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                partition: [[0]],
                pivotPoint: [0.0],
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

    [Fact]
    public void Tt_constructor_rejects_negative_max_derivative_order()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevTT(
                p => p[0],
                numDimensions: 1,
                domain: OneDimDomain,
                nNodes: OneDimNodes,
                maxDerivativeOrder: -1));

        Assert.Equal("maxDerivativeOrder", ex.ParamName);
    }

}
