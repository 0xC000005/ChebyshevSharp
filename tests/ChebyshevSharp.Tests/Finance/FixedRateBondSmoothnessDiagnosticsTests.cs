using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondSmoothnessDiagnosticsTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Coupon_slice_has_near_zero_second_difference()
    {
        SmoothnessDiagnosticReport report = SmoothnessDiagnostics.RunDefault(Pricer);

        Assert.Equal(5, report.CouponSlice.Count);
        Assert.True(report.MaxAbsCouponSecondDifference < 1e-8);
        Assert.All(report.CouponSlice, point => Assert.True(double.IsFinite(point.DirtyPrice)));
    }

    [Fact]
    public void Rate_bump_slice_is_finite_and_locally_smooth_for_supported_pillars()
    {
        SmoothnessDiagnosticReport report = SmoothnessDiagnostics.RunDefault(Pricer);

        RateSensitivityPoint tenYear = Assert.Single(report.RateSensitivities, point => point.PillarYears == 10);

        Assert.Equal(25, report.RateBumpSlice.Count);
        Assert.True(Math.Abs(tenYear.ZeroPillarDv01) > 1e-5);
        Assert.All(report.RateBumpSlice, point => Assert.True(double.IsFinite(point.DirtyPrice)));
        Assert.All(report.RateSensitivities, point =>
        {
            Assert.True(double.IsFinite(point.Derivative));
            Assert.True(double.IsFinite(point.ZeroPillarDv01));
            Assert.True(double.IsFinite(point.LocalSecondDifference));
        });
    }

    [Fact]
    public void Pillars_without_cashflow_interpolation_support_have_zero_dv01()
    {
        SmoothnessDiagnosticReport report = SmoothnessDiagnostics.RunDefault(Pricer);
        RateSensitivityPoint twentyYear = Assert.Single(report.RateSensitivities, point => point.PillarYears == 20);
        RateSensitivityPoint thirtyYear = Assert.Single(report.RateSensitivities, point => point.PillarYears == 30);

        Assert.Equal(0.0, twentyYear.ZeroPillarDv01, precision: 10);
        Assert.Equal(0.0, thirtyYear.ZeroPillarDv01, precision: 10);
    }

    [Fact]
    public void Maturity_slice_records_schedule_count_changes_near_boundaries()
    {
        SmoothnessDiagnosticReport report = SmoothnessDiagnostics.RunDefault(Pricer);

        Assert.NotEmpty(report.MaturitySlice);
        Assert.NotEmpty(report.TopMaturitySpikeCandidates);
        Assert.True(report.MaturitySlice.Select(point => point.CashflowCount).Distinct().Count() > 1);
        Assert.Contains(report.MaturitySlice, point => point.IsScheduleBoundaryCandidate);
    }

    [Fact]
    public void Diagnostics_mode_writes_summary_without_live_downloads()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--diagnostics"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond smoothness diagnostics", output);
        Assert.Contains("Coupon second-difference max", output);
        Assert.Contains("Largest absolute zero-pillar DV01", output);
        Assert.Contains("Top maturity spike candidates", output);
    }
}
