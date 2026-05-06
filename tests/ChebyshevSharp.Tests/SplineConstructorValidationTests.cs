using Xunit;

namespace ChebyshevSharp.Tests;

public class SplineConstructorValidationTests
{
    private static readonly Func<double[], object?, double> F = (x, _) => x[0];

    [Fact]
    public void FlatConstructor_NullFunction_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSpline(
                null!, 1, new[] { new[] { -1.0, 1.0 } },
                new[] { 5 }, new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_DomainLengthMismatch_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 2, new[] { new[] { -1.0, 1.0 } },
                new[] { 5, 5 }, new[] { Array.Empty<double>(), Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_NNodesLengthMismatch_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5 }, new[] { Array.Empty<double>(), Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_NonPositiveNNodes_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                new[] { 0 }, new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_NonPositiveDimensionCount_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 0, Array.Empty<double[]>(),
                Array.Empty<int>(), Array.Empty<double[]>()));
    }

    [Fact]
    public void FlatConstructor_NullDomainRow_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new double[][] { null! },
                new[] { 5 }, new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_MalformedDomainRow_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0 } },
                new[] { 5 }, new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_NonFiniteDomain_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { double.NaN, 1.0 } },
                new[] { 5 }, new[] { Array.Empty<double>() }));
    }

    [Fact]
    public void FlatConstructor_NullKnotsRow_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                new[] { 5 }, new double[][] { null! }));
    }

    [Fact]
    public void FlatConstructor_NonFiniteKnot_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                new[] { 5 }, new[] { new[] { double.NaN } }));
    }

    [Fact]
    public void OptionalConstructor_NNodesLengthMismatch_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: new int?[] { 5 }, knots: null));
    }

    [Fact]
    public void OptionalConstructor_NonPositiveProvidedNNodes_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                nNodes: new int?[] { 0 }, knots: null));
    }

    [Fact]
    public void NestedConstructor_NullNNodesRow_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                nNodesNested: new int[][] { null! },
                knots: new[] { new[] { 0.0 } }));
    }

    [Fact]
    public void NestedConstructor_NonPositiveNNodes_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 5, 0 } },
                knots: new[] { new[] { 0.0 } }));
    }

    [Fact]
    public void WithSpecialPoints_NullSpecialPoints_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                F, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: null!,
                nNodes: new[] { 5 }));
    }
}
