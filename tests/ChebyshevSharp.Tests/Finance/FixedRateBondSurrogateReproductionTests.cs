using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondSurrogateReproductionTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Default_experiment_reports_tensor_train_and_slider_models()
    {
        SurrogateExperimentReport report = FixedRateBondSurrogateExperiment.RunDefault(Pricer);

        Assert.Equal("fed-nominal-yield-curve-2026-05-15", report.FixtureId);
        Assert.Equal(5, report.Dimensions.Count);
        Assert.All(report.Dimensions, dimension => Assert.False(string.IsNullOrWhiteSpace(dimension.Unit)));
        Assert.Equal(["TensorTrain", "Slider"], report.Models.Select(model => model.ModelName).ToArray());
        Assert.All(report.Models, model =>
        {
            Assert.True(model.BuildEvaluations > 0);
            Assert.True(model.BuildSeconds >= 0.0);
            Assert.All(model.Metrics, AssertFiniteMetric);
        });
    }

    [Fact]
    public void Default_experiment_uses_compact_supported_curve_input_set()
    {
        SurrogateExperimentReport report = FixedRateBondSurrogateExperiment.RunDefault(Pricer);

        Assert.Equal([1, 5, 10], report.CurvePillarYears);
        Assert.DoesNotContain(20, report.CurvePillarYears);
        Assert.DoesNotContain(30, report.CurvePillarYears);
        Assert.All(report.ValidationPoints, point =>
        {
            Assert.False(string.IsNullOrWhiteSpace(point.Name));
            Assert.Equal(report.Dimensions.Count, point.Coordinates.Length);
            for (int i = 0; i < point.Coordinates.Length; i++)
            {
                SurrogateInputDimension dimension = report.Dimensions[i];
                Assert.InRange(point.Coordinates[i], dimension.LowerBound, dimension.UpperBound);
            }
        });
    }

    [Fact]
    public void Surrogate_reproduction_mode_writes_metric_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--surrogate-reproduction"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond surrogate reproduction", output);
        Assert.Contains("TensorTrain", output);
        Assert.Contains("Slider", output);
        Assert.Contains("PV rel max", output);
        Assert.Contains("rate-coupon mixed", output);
    }

    private static void AssertFiniteMetric(SurrogateMetricSummary metric)
    {
        Assert.False(string.IsNullOrWhiteSpace(metric.Name));
        Assert.True(double.IsFinite(metric.MeanAbsoluteError));
        Assert.True(double.IsFinite(metric.MaxAbsoluteError));
        Assert.True(double.IsFinite(metric.MeanRelativeError));
        Assert.True(double.IsFinite(metric.MaxRelativeError));
    }
}
