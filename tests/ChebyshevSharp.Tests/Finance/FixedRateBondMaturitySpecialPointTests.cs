using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondMaturitySpecialPointTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();
    private static readonly Lazy<MaturitySpecialPointsReport> Report =
        new(() => MaturitySpecialPointsBenchmark.RunDefault(Pricer));

    [Fact]
    public void Phase9_report_preserves_full_public_wrapper()
    {
        MaturitySpecialPointsReport report = Report.Value;

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.NotEmpty(report.BreakpointInventory);
        Assert.Contains(report.Candidates, candidate => candidate.Name == "Schedule-aware special points");
        Assert.Contains(report.Candidates, candidate => candidate.Name == "Automatic detector candidates");
        Assert.Contains(report.Models, model => model.ModelName == "Schedule-aware special-point decomposed factor tensor");
        Assert.Contains(report.Models, model => model.ModelName == "Automatic-detector special-point decomposed factor tensor");
    }

    [Fact]
    public void Phase9_report_contains_finite_breakpoint_inventory()
    {
        MaturitySpecialPointsReport report = Report.Value;

        Assert.All(report.BreakpointInventory, point =>
        {
            Assert.True(point.MaturityDate > report.CurveDate);
            Assert.InRange(point.MaturityYears, 2.0, 30.0);
            Assert.True(point.CashflowCount > 0);
            Assert.True(point.CouponCashflowCount > 0);
            Assert.True(double.IsFinite(point.DirtyPrice));
            Assert.True(double.IsFinite(point.LeftSlopePerYear));
            Assert.True(double.IsFinite(point.RightSlopePerYear));
            Assert.True(double.IsFinite(point.CentralSlopePerYear));
            Assert.True(double.IsFinite(point.SecondDifference));
            if (point.ScheduleRegimeChanged)
            {
                Assert.False(string.IsNullOrWhiteSpace(point.ScheduleRegimeReason));
                Assert.NotEqual("none", point.ScheduleRegimeReason);
            }
        });
    }

    [Fact]
    public void Phase9_report_summarizes_schedule_sensitive_maturity_spikes()
    {
        MaturitySpecialPointsReport report = Report.Value;

        Assert.Equal(report.BreakpointInventory.Count, report.InventorySummary.PointCount);
        Assert.True(report.InventorySummary.ScheduleRegimeChangeCount > 0);
        Assert.True(report.InventorySummary.MaxAbsSecondDifference > 1e-3);
        Assert.True(report.InventorySummary.MaxAbsSlopeJump > 1.0);
        Assert.True(report.InventorySummary.WorstMaturityDate > report.CurveDate);
        Assert.False(string.IsNullOrWhiteSpace(report.InventorySummary.Interpretation));
    }

    [Fact]
    public void Phase9_candidates_are_sorted_unique_and_inside_domain()
    {
        MaturitySpecialPointsReport report = Report.Value;

        Assert.All(report.Candidates, candidate =>
        {
            Assert.Equal(candidate.CandidateCount, candidate.MaturityYears.Count);
            Assert.True(candidate.CandidateCount > 0);
            Assert.Equal(
                candidate.MaturityYears.Select(years => Math.Round(years, 6)).Distinct().Count(),
                candidate.CandidateCount);
            Assert.Equal(candidate.MaturityYears.Order().ToArray(), candidate.MaturityYears.ToArray());
            Assert.All(candidate.MaturityYears, years => Assert.InRange(years, 2.0, 30.0));
            Assert.False(string.IsNullOrWhiteSpace(candidate.Interpretation));
        });
    }

    [Fact]
    public void Phase9_candidates_record_their_inventory_provenance()
    {
        MaturitySpecialPointsReport report = Report.Value;
        MaturitySpecialPointCandidateSummary schedule = Assert.Single(
            report.Candidates,
            candidate => candidate.Name == "Schedule-aware special points");
        MaturitySpecialPointCandidateSummary detector = Assert.Single(
            report.Candidates,
            candidate => candidate.Name == "Automatic detector candidates");

        HashSet<double> zeroOffsetScheduleChanges = report.BreakpointInventory
            .Where(point => point.OffsetDays == 0 && point.ScheduleRegimeChanged)
            .Select(point => Math.Round(point.MaturityYears, 6))
            .ToHashSet();
        Assert.All(
            schedule.MaturityYears,
            years => Assert.Contains(Math.Round(years, 6), zeroOffsetScheduleChanges));

        double strongestSpike = Math.Round(report.BreakpointInventory
            .OrderByDescending(point => Math.Abs(point.SecondDifference))
            .First()
            .MaturityYears, 3);
        Assert.Contains(strongestSpike, detector.MaturityYears.Select(years => Math.Round(years, 3)));
        Assert.InRange(detector.CandidateCount, 1, 32);
        for (int i = 1; i < detector.MaturityYears.Count; i++)
        {
            Assert.True(detector.MaturityYears[i] - detector.MaturityYears[i - 1] >= 0.20);
        }
    }

    [Fact]
    public void Phase9_model_summaries_are_finite_and_keep_full_wrapper()
    {
        MaturitySpecialPointsReport report = Report.Value;

        Assert.NotEmpty(report.Models);
        Assert.All(report.Models, model =>
        {
            Assert.Equal(62, model.PublicInputDimensionCount);
            Assert.True(model.InternalDimensionCount > 0);
            Assert.True(model.BucketCount > 0);
            Assert.True(model.BuildEvaluations >= 0);
            Assert.True(model.BuildSeconds >= 0.0);
            Assert.False(string.IsNullOrWhiteSpace(model.InternalMethod));
            Assert.False(string.IsNullOrWhiteSpace(model.Interpretation));
            Assert.Contains(model.Metrics, metric => metric.Name == "PV");
            Assert.Contains(model.Metrics, metric => metric.Name == "maturity sensitivity");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon-maturity mixed");
            Assert.All(model.Metrics.Concat(model.FactorAlignedMetrics), AssertFiniteMetric);
        });
    }

    [Fact]
    public void Phase9_decision_prefers_schedule_aware_routing_over_current_controls()
    {
        MaturitySpecialPointsReport report = Report.Value;

        AnalyticCouponModelSummary uniform = Assert.Single(
            report.Models,
            model => model.ModelName == "Semiannual uniform bucketed decomposed factor tensor");
        AnalyticCouponModelSummary schedule = Assert.Single(
            report.Models,
            model => model.ModelName == "Schedule-aware special-point decomposed factor tensor");

        Assert.True(MaxRelativeError(schedule, "maturity sensitivity") < MaxRelativeError(uniform, "maturity sensitivity"));
        Assert.True(MaxRelativeError(schedule, "coupon-maturity mixed") < MaxRelativeError(uniform, "coupon-maturity mixed"));
        Assert.Contains("schedule-aware", report.Decision.Recommendation, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("library", report.Decision.LibraryEnhancementDecision, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Maturity_special_points_mode_writes_phase9_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--maturity-special-points"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond maturity special points", output);
        Assert.Contains("full wrapper", output);
        Assert.Contains("Breakpoint inventory", output);
        Assert.Contains("Schedule-aware special points", output);
        Assert.Contains("Automatic detector candidates", output);
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

    private static double MaxRelativeError(AnalyticCouponModelSummary model, string metricName)
        => model.Metrics.Single(metric => metric.Name == metricName).MaxRelativeError;
}
