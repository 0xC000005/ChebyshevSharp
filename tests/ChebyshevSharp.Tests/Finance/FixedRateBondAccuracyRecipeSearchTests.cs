using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondAccuracyRecipeSearchTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();
    private static readonly Lazy<AccuracyRecipeSearchReport> Report =
        new(() => AccuracyRecipeSearch.RunDefault(Pricer));

    [Fact]
    public void Phase12_report_preserves_full_wrapper_and_records_oracles()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.NotEmpty(report.CloneValidationPoints);
        Assert.NotEmpty(report.ProjectionOracle.Points);
        Assert.NotEmpty(report.DerivativeOracle.RateStepDiagnostics);
        Assert.NotEmpty(report.DerivativeOracle.MaturityStepDiagnostics);
        Assert.NotEmpty(report.ScheduleDispatch.Diagnostics);
        Assert.Contains("projection", report.Decision, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Projection_oracle_separates_arbitrary_clone_error_from_factor_aligned_points()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.True(report.ProjectionOracle.MaxClonePvAbsoluteError > 1e-6);
        Assert.True(report.ProjectionOracle.MaxClonePvRelativeError > 1e-6);
        Assert.True(report.ProjectionOracle.MaxFactorAlignedPvAbsoluteError < report.ProjectionOracle.MaxClonePvAbsoluteError);
        Assert.True(report.ProjectionOracle.MaxFactorAlignedPvRelativeError < report.ProjectionOracle.MaxClonePvRelativeError);
    }

    [Fact]
    public void Projection_oracle_compares_richer_deterministic_curve_basis()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        AccuracyProjectionBasisSummary richer = Assert.Single(
            report.ProjectionOracle.AlternativeBases,
            basis => basis.Name == "Five-factor deterministic curve basis");

        Assert.Equal(5, richer.FactorCount);
        Assert.True(double.IsFinite(richer.MaxClonePvAbsoluteError));
        Assert.True(double.IsFinite(richer.MaxClonePvRelativeError));
        Assert.True(richer.MaxClonePvRelativeError > 1e-6);
    }

    [Fact]
    public void Derivative_oracle_records_step_sensitivity_and_post_maturity_support()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.All(report.DerivativeOracle.RateStepDiagnostics, diagnostic =>
        {
            Assert.True(double.IsFinite(diagnostic.Step));
            Assert.True(double.IsFinite(diagnostic.Value));
        });
        Assert.All(report.DerivativeOracle.MaturityStepDiagnostics, diagnostic =>
        {
            Assert.True(double.IsFinite(diagnostic.Step));
            Assert.True(double.IsFinite(diagnostic.Value));
        });
        Assert.True(Math.Abs(report.DerivativeOracle.PostMaturityUnsupportedPillarDv01) < 1e-10);
    }

    [Fact]
    public void Active_support_oracle_preserves_price_when_post_maturity_pillars_are_removed()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.NotEmpty(report.ActiveSupport.Points);
        Assert.True(report.ActiveSupport.MaxPvAbsoluteError < 1e-8);
        Assert.Contains(report.ActiveSupport.Points, point => point.ActiveCurveBumpDimensions < 60);
        Assert.Contains(report.ActiveSupport.Points, point => point.ActiveCurveBumpDimensions == 60);
    }

    [Fact]
    public void Active_pillar_candidate_keeps_full_wrapper_with_smaller_local_dimension()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        AccuracyRecipeModelSummary activeTt = Assert.Single(
            report.CandidateModels,
            model => model.ModelName == "10Y active-pillar TT");
        Assert.Contains(report.CandidateModels, model => model.ModelName == "10Y narrow active-pillar TT");

        Assert.Equal(62, activeTt.PublicInputDimensionCount);
        Assert.InRange(activeTt.InternalDimensionCount, 1, 61);
        Assert.True(activeTt.BuildEvaluations > 0);
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "PV");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "10Y DV01");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "coupon derivative");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "maturity sensitivity");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "coupon-maturity mixed");
        Assert.All(activeTt.Metrics, metric =>
        {
            Assert.True(double.IsFinite(metric.MaxAbsoluteError));
            Assert.True(double.IsFinite(metric.MaxRelativeError));
        });
    }

    [Fact]
    public void Accuracy_recipe_search_mode_writes_phase12_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--accuracy-recipe-search"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond accuracy recipe search", output);
        Assert.Contains("Projection oracle", output);
        Assert.Contains("Derivative oracle", output);
        Assert.Contains("Schedule dispatch", output);
        Assert.Contains("Next decision", output);
    }
}
