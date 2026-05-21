using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondMaturitySpecialPointTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Phase9_report_preserves_full_public_wrapper()
    {
        MaturitySpecialPointsReport report = MaturitySpecialPointsBenchmark.RunDefault(Pricer);

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", report.FixtureId);
        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.NotEmpty(report.BreakpointInventory);
        Assert.Contains(report.Candidates, candidate => candidate.Name == "Schedule-aware special points");
        Assert.Contains(report.Candidates, candidate => candidate.Name == "Automatic detector candidates");
    }

    [Fact]
    public void Phase9_report_contains_finite_breakpoint_inventory()
    {
        MaturitySpecialPointsReport report = MaturitySpecialPointsBenchmark.RunDefault(Pricer);

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
        });
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
}
