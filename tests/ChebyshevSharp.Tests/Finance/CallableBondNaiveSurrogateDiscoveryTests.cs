using CallableBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class CallableBondNaiveSurrogateDiscoveryTests
{
    private static readonly ICallableBondReferencePricer Pricer = new QlNetCallableBondReferencePricer();
    private static readonly Lazy<CallableNaiveSurrogateDiscoveryReport> DefaultReport =
        new(() => CallableNaiveSurrogateDiscovery.RunDefault(Pricer));

    [Fact]
    public void Default_report_records_full_65d_tensor_infeasibility()
    {
        CallableNaiveSurrogateDiscoveryReport report = DefaultReport.Value;

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal(new DateTime(2026, 5, 15), report.CurveDate);
        Assert.Equal(60, report.Feasibility.CurveBumpDimensions);
        Assert.Equal(65, report.Feasibility.SurrogateDimensions);
        Assert.Equal("3^65", report.Feasibility.ThreeNodeDenseGridLabel);
        Assert.Contains("too large", report.Feasibility.DenseTensorConclusion);
        Assert.Equal(65, report.DimensionLabels.Count);
    }

    [Fact]
    public void Default_report_builds_naive_tensor_train_and_slider_metric_summaries()
    {
        CallableNaiveSurrogateDiscoveryReport report = DefaultReport.Value;

        Assert.Equal(["TensorTrain", "Slider"], report.Models.Select(model => model.ModelName).ToArray());
        Assert.All(report.Models, model =>
        {
            Assert.Equal(65, model.InputDimensionCount);
            Assert.True(model.BuildEvaluations > 0);
            Assert.True(model.BuildSeconds >= 0.0);
            Assert.Contains(model.Metrics, metric => metric.Name == "PV");
            Assert.Contains(model.Metrics, metric => metric.Name == "10Y zero-pillar DV01");
            Assert.Contains(model.Metrics, metric => metric.Name == "30Y zero-pillar DV01");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon derivative");
            Assert.Contains(model.Metrics, metric => metric.Name == "sigma sensitivity");
            Assert.Contains(model.Metrics, metric => metric.Name == "call-price sensitivity");
            Assert.Contains(model.Metrics, metric => metric.Name == "10Y rate-sigma mixed");
            Assert.Contains(model.Metrics, metric => metric.Name == "call-price-sigma mixed");
            Assert.All(model.Metrics, AssertFiniteMetric);
        });
    }

    [Fact]
    public void Naive_surrogate_discovery_mode_writes_case_study_summary()
    {
        using var writer = new StringWriter();

        CallableBondExample.Run(["--naive-surrogate-discovery"], writer);

        string output = writer.ToString();
        Assert.Contains("Callable bond naive surrogate discovery", output);
        Assert.Contains("65D public wrapper", output);
        Assert.Contains("Dense full tensor", output);
        Assert.Contains("TensorTrain", output);
        Assert.Contains("Slider", output);
        Assert.Contains("sigma sensitivity", output);
        Assert.Contains("call-price-sigma mixed", output);
    }

    private static void AssertFiniteMetric(CallableNaiveSurrogateMetricSummary metric)
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
