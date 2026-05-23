using CallableBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class CallableBondStructuredAlternativesTests
{
    private static readonly Lazy<CallableStructuredAlternativesReport> Report = new(
        () => CallableStructuredAlternatives.RunDefault(new QlNetCallableBondReferencePricer()));

    [Fact]
    public void Curve_factor_surrogate_preserves_65d_public_contract_but_labels_internal_compression()
    {
        CallableStructuredAlternativesReport report = Report.Value;
        CallableStructuredAlternativeModelSummary model = Assert.Single(
            report.Models,
            model => model.ModelName == "Curve-factor tensor");

        Assert.Equal("Curve-factor tensor", model.ModelName);
        Assert.Equal("factor-risk surrogate", model.ApproximationType);
        Assert.Equal(65, model.PublicInputDimensionCount);
        Assert.Equal(8, model.InternalDimensionCount);
        Assert.True(model.BuildEvaluations > 0);
        Assert.True(model.BuildSeconds >= 0.0);
        Assert.True(model.BaselineEvalMicroseconds > 0.0);
        Assert.True(model.SurrogateEvalMicroseconds > 0.0);
        Assert.True(double.IsFinite(model.BreakEvenEvaluations) || double.IsPositiveInfinity(model.BreakEvenEvaluations));
    }

    [Fact]
    public void Curve_factor_surrogate_reports_factor_aligned_and_arbitrary_bump_metrics()
    {
        CallableStructuredAlternativeModelSummary model = Assert.Single(
            Report.Value.Models,
            model => model.ModelName == "Curve-factor tensor");

        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "PV");
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "level-factor sensitivity");
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "slope-factor sensitivity");
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "curvature-factor sensitivity");
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "10Y zero-pillar DV01");
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "sigma sensitivity");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "PV");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "10Y zero-pillar DV01");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "sigma sensitivity");
        Assert.All(model.FactorAlignedMetrics, AssertFiniteMetric);
        Assert.All(model.ArbitraryBumpMetrics, AssertFiniteMetric);
    }

    [Fact]
    public void Embedded_option_decomposition_preserves_65d_contract_and_reports_metrics()
    {
        CallableStructuredAlternativeModelSummary model = Assert.Single(
            Report.Value.Models,
            model => model.ModelName == "Embedded-option curve-factor tensor");

        Assert.Equal("formula-aware factor-risk surrogate", model.ApproximationType);
        Assert.Equal(65, model.PublicInputDimensionCount);
        Assert.Equal(8, model.InternalDimensionCount);
        Assert.True(model.BuildEvaluations > 0);
        Assert.True(model.BaselineEvalMicroseconds > 0.0);
        Assert.True(model.SurrogateEvalMicroseconds > 0.0);
        Assert.True(double.IsFinite(model.BreakEvenEvaluations) || double.IsPositiveInfinity(model.BreakEvenEvaluations));
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "PV");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "PV");
        Assert.All(model.FactorAlignedMetrics, AssertFiniteMetric);
        Assert.All(model.ArbitraryBumpMetrics, AssertFiniteMetric);
    }

    [Fact]
    public void Embedded_option_full_pillar_tt_keeps_internal_dimension_full_size()
    {
        CallableStructuredAlternativeModelSummary model = Assert.Single(
            Report.Value.Models,
            model => model.ModelName == "Embedded-option full-pillar TT");

        Assert.Equal("formula-aware faithful full-pillar candidate", model.ApproximationType);
        Assert.Equal(65, model.PublicInputDimensionCount);
        Assert.Equal(65, model.InternalDimensionCount);
        Assert.True(model.BuildEvaluations > 0);
        Assert.True(model.BaselineEvalMicroseconds > 0.0);
        Assert.True(model.SurrogateEvalMicroseconds > 0.0);
        Assert.True(double.IsFinite(model.BreakEvenEvaluations) || double.IsPositiveInfinity(model.BreakEvenEvaluations));
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "PV");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "PV");
        Assert.All(model.FactorAlignedMetrics, AssertFiniteMetric);
        Assert.All(model.ArbitraryBumpMetrics, AssertFiniteMetric);
    }

    [Fact]
    public void Curve_factor_tt_reports_low_rank_internal_compression()
    {
        CallableStructuredAlternativeModelSummary model = Assert.Single(
            Report.Value.Models,
            model => model.ModelName == "Curve-factor TT");

        Assert.Equal("factor-risk surrogate", model.ApproximationType);
        Assert.Equal(65, model.PublicInputDimensionCount);
        Assert.Equal(8, model.InternalDimensionCount);
        Assert.True(model.BuildEvaluations > 0);
        Assert.True(model.BaselineEvalMicroseconds > 0.0);
        Assert.True(model.SurrogateEvalMicroseconds > 0.0);
        Assert.Contains(model.FactorAlignedMetrics, metric => metric.Name == "PV");
        Assert.Contains(model.ArbitraryBumpMetrics, metric => metric.Name == "PV");
    }

    [Fact]
    public void Structured_alternatives_mode_writes_case_study_summary()
    {
        using var writer = new StringWriter();

        CallableBondExample.Run(["--structured-alternatives"], writer);

        string output = writer.ToString();
        Assert.Contains("Callable bond structured alternatives", output);
        Assert.Contains("Curve-factor tensor", output);
        Assert.Contains("Curve-factor TT", output);
        Assert.Contains("Embedded-option curve-factor tensor", output);
        Assert.Contains("Embedded-option full-pillar TT", output);
        Assert.Contains("factor-risk surrogate", output);
        Assert.Contains("formula-aware factor-risk surrogate", output);
        Assert.Contains("formula-aware faithful full-pillar candidate", output);
        Assert.Contains("factor-aligned scenarios", output);
        Assert.Contains("arbitrary pillar shocks", output);
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
