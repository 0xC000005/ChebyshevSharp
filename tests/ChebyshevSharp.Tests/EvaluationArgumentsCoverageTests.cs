using ChebyshevSharp.Internal;
using Xunit;

namespace ChebyshevSharp.Tests;

public class EvaluationArgumentsCoverageTests
{
    private static readonly double[][] UnitSquare =
    [
        [-1.0, 1.0],
        [-1.0, 1.0]
    ];

    [Fact]
    public void Validate_point_in_domain_rejects_low_and_high_coordinates()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointInDomain([-1.1, 0.0], 2, UnitSquare));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointInDomain([0.0, 1.1], 2, UnitSquare));
    }

    [Fact]
    public void Validate_points_in_domain_rejects_null_rows_and_out_of_domain_coordinates()
    {
        Assert.Throws<ArgumentException>(() =>
            EvaluationArguments.ValidatePointsInDomain([null!], 2, UnitSquare));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointsInDomain([[0.0, 0.0], [-1.1, 0.0]], 2, UnitSquare));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointsInDomain([[0.0, 0.0], [0.0, 1.1]], 2, UnitSquare));
    }

    [Fact]
    public void Validate_point_batch_in_domain_rejects_low_and_high_coordinates()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointBatchInDomain(
                new double[,] { { 0.0, 0.0 }, { -1.1, 0.0 } },
                2,
                UnitSquare));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EvaluationArguments.ValidatePointBatchInDomain(
                new double[,] { { 0.0, 0.0 }, { 0.0, 1.1 } },
                2,
                UnitSquare));
    }

    [Fact]
    public void Domain_validation_rejects_malformed_domains()
    {
        Assert.Throws<ArgumentNullException>(() =>
            EvaluationArguments.ValidatePointInDomain([0.0, 0.0], 2, null!));

        Assert.Throws<ArgumentException>(() =>
            EvaluationArguments.ValidatePointInDomain([0.0, 0.0], 2, [[-1.0, 1.0]]));

        Assert.Throws<ArgumentException>(() =>
            EvaluationArguments.ValidatePointInDomain([0.0, 0.0], 2, [[-1.0, 1.0], null!]));

        Assert.Throws<ArgumentException>(() =>
            EvaluationArguments.ValidatePointInDomain([0.0, 0.0], 2, [[-1.0, 1.0], [-1.0, 0.0, 1.0]]));
    }
}
