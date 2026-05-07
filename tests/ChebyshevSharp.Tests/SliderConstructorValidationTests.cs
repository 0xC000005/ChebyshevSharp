using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderConstructorValidationTests
{
    private static double F(double[] p, object? _) => p.Sum();

    private static readonly double[][] Domain =
    [
        [-1.0, 1.0],
        [-2.0, 2.0]
    ];

    private static readonly int[] NNodes = [5, 6];
    private static readonly int[][] Partition = [[0], [1]];
    private static readonly double[] PivotPoint = [0.0, 0.0];

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_rejects_non_positive_numDimensions(int numDimensions)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSlider(
                F,
                numDimensions,
                [],
                [],
                [],
                []));
    }

    [Fact]
    public void Constructor_rejects_null_function()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSlider(null!, 2, Domain, NNodes, Partition, PivotPoint));
    }

    [Fact]
    public void Constructor_rejects_null_arrays()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSlider(F, 2, null!, NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSlider(F, 2, Domain, null!, Partition, PivotPoint));
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, null!, PivotPoint));
        Assert.Throws<ArgumentNullException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, Partition, null!));
    }

    [Fact]
    public void Constructor_rejects_shape_mismatches()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [[-1.0, 1.0]], NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, [5], Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, Partition, [0.0]));
    }

    [Fact]
    public void Constructor_rejects_malformed_domain_entries()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [null!, [-2.0, 2.0]], NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [[-1.0], [-2.0, 2.0]], NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [[1.0, -1.0], [-2.0, 2.0]], NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [[double.NaN, 1.0], [-2.0, 2.0]], NNodes, Partition, PivotPoint));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, [[-1.0, double.PositiveInfinity], [-2.0, 2.0]], NNodes, Partition, PivotPoint));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_rejects_non_positive_node_counts(int nodeCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSlider(F, 2, Domain, [nodeCount, 6], Partition, PivotPoint));
    }

    [Fact]
    public void Constructor_rejects_empty_partition_groups()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, [[], [0, 1]], PivotPoint));
    }

    [Fact]
    public void Constructor_rejects_null_partition_groups()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, [null!, [0, 1]], PivotPoint));
    }

    [Fact]
    public void Constructor_rejects_non_finite_or_out_of_domain_pivots()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, Partition, [double.NaN, 0.0]));
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, Partition, [0.0, double.PositiveInfinity]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevSlider(F, 2, Domain, NNodes, Partition, [2.0, 0.0]));
    }
}
