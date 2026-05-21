using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondNaiveSurrogateDiscoveryTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Default_report_uses_dense_baseline_and_records_full_tensor_infeasibility()
    {
        NaiveSurrogateDiscoveryReport report = NaiveSurrogateDiscovery.RunDefault(Pricer);

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal(new DateTime(2026, 5, 15), report.CurveDate);
        Assert.Equal(60, report.CurvePillarMonths.Count);
        Assert.Equal(6, report.CurvePillarMonths[0]);
        Assert.Equal(360, report.CurvePillarMonths[^1]);
        Assert.Equal(60, report.Feasibility.CurveBumpDimensions);
        Assert.Equal(62, report.Feasibility.SurrogateDimensionsExcludingNotional);
        Assert.Equal(63, report.Feasibility.SurrogateDimensionsIncludingNotional);
        Assert.Equal("3^62", report.Feasibility.ThreeNodeDenseGridLabel);
        Assert.Contains("too large", report.Feasibility.DenseTensorConclusion);
    }

    [Fact]
    public void Default_report_builds_naive_tensor_train_and_slider_metric_summaries()
    {
        NaiveSurrogateDiscoveryReport report = NaiveSurrogateDiscovery.RunDefault(Pricer);

        Assert.Equal(62, report.Dimensions.Count);
        Assert.Equal(60, report.Dimensions.Count(dimension => dimension.Name.EndsWith("zero-rate bump", StringComparison.Ordinal)));
        Assert.Equal(["TensorTrain", "Slider"], report.Models.Select(model => model.ModelName).ToArray());
        Assert.All(report.Models, model =>
        {
            Assert.True(model.BuildEvaluations > 0);
            Assert.True(model.BuildSeconds >= 0.0);
            Assert.Equal(62, model.InputDimensionCount);
            Assert.Contains(model.Metrics, metric => metric.Name == "PV");
            Assert.Contains(model.Metrics, metric => metric.Name == "10Y zero-pillar DV01");
            Assert.Contains(model.Metrics, metric => metric.Name == "30Y zero-pillar DV01");
            Assert.Contains(model.Metrics, metric => metric.Name == "maturity slope");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon-maturity mixed");
            Assert.All(model.Metrics, AssertFiniteMetric);
        });
    }

    [Fact]
    public void Default_report_includes_baseline_maturity_smoothness_evidence()
    {
        NaiveSurrogateDiscoveryReport report = NaiveSurrogateDiscovery.RunDefault(Pricer);

        Assert.NotEmpty(report.TopMaturitySpikeCandidates);
        Assert.All(report.TopMaturitySpikeCandidates, point =>
        {
            Assert.True(point.AbsoluteSecondDifference > 0.0);
            Assert.InRange(point.MaturityDate, report.CurveDate.AddYears(2).AddDays(-7), report.CurveDate.AddYears(30).AddDays(7));
            Assert.True(point.CashflowCount > 0);
            Assert.True(double.IsFinite(point.LeftSlopePerYear));
            Assert.True(double.IsFinite(point.RightSlopePerYear));
        });
    }

    [Fact]
    public void Naive_surrogate_discovery_mode_writes_case_study_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--naive-surrogate-discovery"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond naive surrogate discovery", output);
        Assert.Contains("Dense full tensor", output);
        Assert.Contains("TensorTrain", output);
        Assert.Contains("Slider", output);
        Assert.Contains("coupon-maturity mixed", output);
        Assert.Contains("Top maturity spike candidates", output);
    }

    private static void AssertFiniteMetric(NaiveSurrogateMetricSummary metric)
    {
        Assert.False(string.IsNullOrWhiteSpace(metric.Name));
        Assert.True(double.IsFinite(metric.MeanAbsoluteError));
        Assert.True(double.IsFinite(metric.MaxAbsoluteError));
        Assert.True(double.IsFinite(metric.MeanRelativeError));
        Assert.True(double.IsFinite(metric.MaxRelativeError));
        Assert.False(string.IsNullOrWhiteSpace(metric.WorstPointName));
        Assert.True(double.IsFinite(metric.ExpectedAtWorstPoint));
        Assert.True(double.IsFinite(metric.ActualAtWorstPoint));
    }
}
