namespace ChebyshevSharp.Tests;

/// <summary>
/// Regression tests for ChebyshevSpline paths that threw on valid input.
/// </summary>
public class SplineRegressionTests
{
    [Theory]
    [InlineData(3.7)]
    [InlineData(0.1)]
    [InlineData(-1234.5678)]
    [InlineData(1e6 + 0.3)]
    [InlineData(1.0 / 3.0)]
    public void SobolIndices_constant_spline_reports_zero_variance(double constant)
    {
        var spline = new ChebyshevSpline(
            (_, _) => constant,
            2,
            new[] { new[] { 0.0, 1.0 }, new[] { -2.0, 5.0 } },
            new[] { 7, 9 },
            new[] { new[] { 0.3, 0.7 }, new[] { 0.1, 1.9, 3.3 } });
        spline.Build(verbose: false);

        var result = spline.SobolIndices();

        Assert.Equal(0.0, result.Variance);
        Assert.All(result.FirstOrder, s => Assert.Equal(0.0, s));
        Assert.All(result.TotalOrder, s => Assert.Equal(0.0, s));
    }

    [Fact]
    public void SobolIndices_small_signal_on_large_offset_is_not_lost()
    {
        // Only x[0] matters; the signal is ~1e-9 of the offset.
        var spline = new ChebyshevSpline(
            (x, _) => 1e6 + 1e-3 * x[0],
            2,
            new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } },
            new[] { 5, 5 },
            new[] { new[] { 0.5 }, new[] { 0.4 } });
        spline.Build(verbose: false);

        var result = spline.SobolIndices();

        Assert.True(result.Variance > 0.0);
        Assert.Equal(1.0, result.FirstOrder[0], 3);
        Assert.Equal(0.0, result.FirstOrder[1], 3);
        Assert.Equal(1.0, result.TotalOrder[0], 3);
        Assert.Equal(0.0, result.TotalOrder[1], 3);
    }

    [Fact]
    public void Integrate_bounds_covering_one_dim_fully_and_another_partially()
    {
        // f(x, y) = x * y on [0, 2] x [0, 2], knot at x = 1.
        var spline = new ChebyshevSpline(
            (x, _) => x[0] * x[1],
            2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 2.0 } },
            new[] { 6, 6 },
            new[] { new[] { 1.0 }, Array.Empty<double>() });
        spline.Build(verbose: false);

        // x spans every piece exactly; y is a strict sub-interval.
        double result = (double)spline.Integrate(bounds: new[] { (0.0, 2.0), (0.5, 1.5) });

        // (int_0^2 x dx) * (int_0.5^1.5 y dy) = 2 * 1
        Assert.Equal(2.0, result, 12);
    }

    [Fact]
    public void Integrate_bounds_partial_in_knotted_dim_and_full_in_other()
    {
        var spline = new ChebyshevSpline(
            (x, _) => x[0] * x[1],
            2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 2.0 } },
            new[] { 6, 6 },
            new[] { new[] { 1.0 }, Array.Empty<double>() });
        spline.Build(verbose: false);

        double result = (double)spline.Integrate(bounds: new[] { (0.5, 1.5), (0.0, 2.0) });

        Assert.Equal(2.0, result, 12);
    }
}
