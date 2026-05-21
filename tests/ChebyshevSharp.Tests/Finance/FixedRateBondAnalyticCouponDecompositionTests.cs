using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondAnalyticCouponDecompositionTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Phase8_report_preserves_wrapper_and_validates_coupon_linearity()
    {
        AnalyticCouponDecompositionReport report = AnalyticCouponDecompositionBenchmark.RunDefault(Pricer);

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.Dimensions.Count);
        Assert.Equal(12, report.CloneValidationPoints.Count);
        Assert.Equal(7, report.FactorAlignedValidationPoints.Count);
        Assert.True(report.Identity.MaxAbsoluteError < 1e-8);
        Assert.True(report.Identity.MaxRelativeError < 1e-10);
        Assert.NotEmpty(report.Models);
        Assert.All(report.Models, model =>
        {
            Assert.Equal(62, model.PublicInputDimensionCount);
            Assert.Contains(model.Metrics, metric => metric.Name == "PV");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon derivative");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon-maturity mixed");
            Assert.All(model.Metrics, AssertFiniteMetric);
        });
    }

    [Fact]
    public void Phase8_report_compares_decomposed_model_families()
    {
        AnalyticCouponDecompositionReport report = AnalyticCouponDecompositionBenchmark.RunDefault(Pricer);

        string[] modelNames = report.Models.Select(model => model.ModelName).ToArray();
        Assert.Contains("Exact coupon decomposition oracle", modelNames);
        Assert.Contains("Global decomposed TT", modelNames);
        Assert.Contains("Curve-factor decomposed tensor", modelNames);
        Assert.Contains("Bucketed decomposed curve-factor tensor", modelNames);
        Assert.Contains("Semiannual bucketed decomposed curve-factor tensor", modelNames);

        AnalyticCouponModelSummary exact = Assert.Single(
            report.Models,
            model => model.ModelName == "Exact coupon decomposition oracle");
        NaiveSurrogateMetricSummary exactCoupon = Assert.Single(
            exact.Metrics,
            metric => metric.Name == "coupon derivative");
        Assert.True(exactCoupon.MaxAbsoluteError < 1e-6);

        AnalyticCouponModelSummary factor = Assert.Single(
            report.Models,
            model => model.ModelName == "Curve-factor decomposed tensor");
        Assert.Equal(4, factor.InternalDimensionCount);
    }

    [Fact]
    public void Analytic_coupon_decomposition_mode_writes_phase8_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--analytic-coupon-decomposition"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond analytic coupon decomposition", output);
        Assert.Contains("full wrapper", output);
        Assert.Contains("coupon-linearity max abs", output);
        Assert.Contains("Global decomposed TT", output);
        Assert.Contains("Curve-factor decomposed tensor", output);
        Assert.Contains("Semiannual bucketed decomposed curve-factor tensor", output);
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
