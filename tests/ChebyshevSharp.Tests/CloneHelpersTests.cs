// tests/ChebyshevSharp.Tests/CloneHelpersTests.cs
using ChebyshevSharp.Internal;
using Xunit;

namespace ChebyshevSharp.Tests;

public class CloneHelpersTests
{
    [Fact]
    public void DeepCopy_double_array_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((double[]?)null));
    }

    [Fact]
    public void DeepCopy_double_jagged_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((double[][]?)null));
    }

    [Fact]
    public void DeepCopy_double_2d_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((double[,]?)null));
    }

    [Fact]
    public void DeepCopy_double_2d_actually_deep_copies()
    {
        var src = new double[2, 2] { { 1, 2 }, { 3, 4 } };
        var copy = CloneHelpers.DeepCopy(src);
        Assert.NotNull(copy);
        Assert.False(ReferenceEquals(src, copy));
        Assert.Equal(src[0, 0], copy![0, 0]);
        Assert.Equal(src[1, 1], copy[1, 1]);
    }

    [Fact]
    public void DeepCopy_double_3d_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((double[][,]?)null));
    }

    [Fact]
    public void DeepCopy_double_3d_actually_deep_copies()
    {
        var src = new double[][,]
        {
            new double[,] { { 1, 2 }, { 3, 4 } },
            new double[,] { { 5, 6 }, { 7, 8 } }
        };
        var copy = CloneHelpers.DeepCopy(src);
        Assert.NotNull(copy);
        Assert.Equal(2, copy!.Length);
        Assert.False(ReferenceEquals(src[0], copy[0]));
        Assert.Equal(src[1][1, 1], copy[1][1, 1]);
    }

    [Fact]
    public void DeepCopy_int_array_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((int[]?)null));
    }

    [Fact]
    public void DeepCopy_int_jagged_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((int[][]?)null));
    }

    [Fact]
    public void DeepCopy_nullable_int_array_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopy((int?[]?)null));
    }

    [Fact]
    public void DeepCopy_intervals_null_input_returns_null()
    {
        Assert.Null(CloneHelpers.DeepCopyIntervals(null));
    }

    [Fact]
    public void DeepCopy_intervals_actually_deep_copies()
    {
        var src = new (double, double)[][]
        {
            new[] { (0.0, 1.0), (1.0, 2.0) },
            new[] { (2.0, 3.0) }
        };
        var copy = CloneHelpers.DeepCopyIntervals(src);
        Assert.NotNull(copy);
        Assert.False(ReferenceEquals(src[0], copy![0]));
        Assert.Equal((1.0, 2.0), copy[0][1]);
    }
}
