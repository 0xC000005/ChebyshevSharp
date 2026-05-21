using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondStructuredAlternativesTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Phase7_report_keeps_full_wrapper_contract_for_every_candidate()
    {
        StructuredAlternativesReport report = StructuredAlternativesBenchmark.RunDefault(Pricer);

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Contains("MoCaX", report.ResearchBasis);
        Assert.Contains("Chebfun", report.ResearchBasis);
        Assert.NotEmpty(report.Models);
        Assert.All(report.Models, model =>
        {
            Assert.Equal(62, model.PublicInputDimensionCount);
            Assert.False(string.IsNullOrWhiteSpace(model.InternalMethod));
            Assert.False(string.IsNullOrWhiteSpace(model.Interpretation));
            Assert.Contains(model.Metrics, metric => metric.Name == "PV");
            Assert.Contains(model.Metrics, metric => metric.Name == "maturity sensitivity");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon-maturity mixed");
            Assert.All(model.Metrics, AssertFiniteMetric);
        });
    }

    [Fact]
    public void Phase7_report_compares_common_high_dimensional_modelling_practices()
    {
        StructuredAlternativesReport report = StructuredAlternativesBenchmark.RunDefault(Pricer);

        string[] modelNames = report.Models.Select(model => model.ModelName).ToArray();
        Assert.Contains("Stronger global TT", modelNames);
        Assert.Contains("Auto-ordered global TT", modelNames);
        Assert.Contains("Grouped Slider", modelNames);
        Assert.Contains("Curve-factor tensor", modelNames);
        Assert.Contains("Bucketed curve-factor tensor", modelNames);
        Assert.Contains("Semiannual bucketed curve-factor tensor", modelNames);

        StructuredAlternativeModelSummary groupedSlider = Assert.Single(
            report.Models,
            model => model.ModelName == "Grouped Slider");
        Assert.Contains("coupon", groupedSlider.InternalMethod, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("maturity", groupedSlider.InternalMethod, StringComparison.OrdinalIgnoreCase);

        StructuredAlternativeModelSummary bucketed = Assert.Single(
            report.Models,
            model => model.ModelName == "Bucketed curve-factor tensor");
        Assert.True(bucketed.BucketCount > 1);
        Assert.Contains("piecewise", bucketed.InternalMethod, StringComparison.OrdinalIgnoreCase);

        StructuredAlternativeModelSummary semiannualBucketed = Assert.Single(
            report.Models,
            model => model.ModelName == "Semiannual bucketed curve-factor tensor");
        Assert.True(semiannualBucketed.BucketCount > bucketed.BucketCount);
    }

    [Fact]
    public void Structured_alternatives_mode_writes_phase7_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--structured-alternatives"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond structured alternatives", output);
        Assert.Contains("full wrapper", output);
        Assert.Contains("Stronger global TT", output);
        Assert.Contains("Grouped Slider", output);
        Assert.Contains("Bucketed curve-factor tensor", output);
        Assert.Contains("Semiannual bucketed curve-factor tensor", output);
        Assert.Contains("Interpretation", output);
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
