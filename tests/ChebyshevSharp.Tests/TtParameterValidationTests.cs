using Xunit;

namespace ChebyshevSharp.Tests;

public class TtParameterValidationTests
{
    private static readonly double[][] Domain =
    [
        [-1.0, 1.0],
        [-1.0, 1.0]
    ];

    private static readonly int[] NNodes = [5, 5];

    private static ChebyshevTT CreateBuiltTt()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2, Domain, NNodes, maxRank: 3, tolerance: 1e-8);
        tt.Build(verbose: false, seed: 0);
        return tt;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_rejects_non_positive_maxRank(int maxRank)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevTT(p => p[0], 2, Domain, NNodes, maxRank: maxRank));

        Assert.Equal("maxRank", ex.ParamName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Constructor_rejects_non_positive_or_non_finite_tolerance(double tolerance)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevTT(p => p[0], 2, Domain, NNodes, tolerance: tolerance));

        Assert.Equal("tolerance", ex.ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_rejects_non_positive_maxSweeps(int maxSweeps)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChebyshevTT(p => p[0], 2, Domain, NNodes, maxSweeps: maxSweeps));

        Assert.Equal("maxSweeps", ex.ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void FromValues_rejects_non_positive_maxRank(int maxRank)
    {
        var values = new double[25];

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ChebyshevTT.FromValues(values, 2, Domain, NNodes, maxRank: maxRank));

        Assert.Equal("maxRank", ex.ParamName);
    }

    [Theory]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void FromValues_rejects_negative_or_non_finite_tolerance(double tolerance)
    {
        var values = new double[25];

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ChebyshevTT.FromValues(values, 2, Domain, NNodes, tolerance: tolerance));

        Assert.Equal("tolerance", ex.ParamName);
    }

    [Fact]
    public void FromValues_allows_zero_tolerance_for_rank_only_svd_truncation()
    {
        var values = Enumerable.Range(0, 25).Select(i => Math.Sin(i)).ToArray();

        var tt = ChebyshevTT.FromValues(values, 2, Domain, NNodes, maxRank: 3, tolerance: 0.0);

        Assert.Equal(2, tt.NumDimensions);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Reorder_rejects_non_positive_maxRank(int maxRank)
    {
        var tt = CreateBuiltTt();

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.Reorder([1, 0], maxRank: maxRank));

        Assert.Equal("maxRank", ex.ParamName);
    }

    [Theory]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Reorder_rejects_negative_or_non_finite_tolerance(double tolerance)
    {
        var tt = CreateBuiltTt();

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.Reorder([1, 0], tolerance: tolerance));

        Assert.Equal("tolerance", ex.ParamName);
    }

    [Fact]
    public void Reorder_allows_zero_tolerance_for_rank_only_svd_truncation()
    {
        var tt = CreateBuiltTt();

        var reordered = tt.Reorder([1, 0], tolerance: 0.0);

        Assert.Equal([1, 0], reordered.DimOrder);
    }

    [Theory]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void RoundInPlace_rejects_negative_or_non_finite_tolerance(double tolerance)
    {
        var tt = CreateBuiltTt();

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.RoundInPlace(tolerance));

        Assert.Equal("tolerance", ex.ParamName);
    }

    [Fact]
    public void RoundInPlace_allows_zero_tolerance_for_rank_only_svd_truncation()
    {
        var tt = CreateBuiltTt();

        tt.RoundInPlace(0.0);

        Assert.Equal(2, tt.NumDimensions);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1e-6)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void RunCompletion_rejects_non_positive_or_non_finite_tolerance(double tolerance)
    {
        var tt = CreateBuiltTt();

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.RunCompletion(tolerance: tolerance));

        Assert.Equal("tolerance", ex.ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void RunCompletion_rejects_non_positive_maxIter(int maxIter)
    {
        var tt = CreateBuiltTt();

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.RunCompletion(maxIter: maxIter));

        Assert.Equal("maxIter", ex.ParamName);
    }

    [Fact]
    public void WithAutoOrder_rejects_invalid_trial_parameters()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ChebyshevTT.WithAutoOrder(
                p => p[0] + p[1],
                2,
                Domain,
                NNodes,
                maxSweeps: 0,
                verbose: false));

        Assert.Equal("maxSweeps", ex.ParamName);
    }
}
