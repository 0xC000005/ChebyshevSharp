using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondRealisticBaselineTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Dense_curve_fixture_has_semiannual_public_zero_rates()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();

        Assert.Equal("fed-nominal-yield-curve-semiannual-2026-05-15", fixture.FixtureId);
        Assert.Equal(new DateTime(2026, 5, 15), fixture.Source.CurveDate.Date);
        Assert.Equal(60, fixture.Points.Count);
        Assert.Equal(6, fixture.Points[0].MaturityMonths);
        Assert.Equal(360, fixture.Points[^1].MaturityMonths);
        Assert.All(fixture.Points, point => Assert.Equal(0, point.MaturityMonths % 6));
        Assert.All(fixture.Points, point => Assert.InRange(point.ZeroYieldPercent, 0.0, 10.0));

        double maxAdjacentJump = fixture.Points.Zip(fixture.Points.Skip(1))
            .Max(pair => Math.Abs(pair.First.ZeroYieldPercent - pair.Second.ZeroYieldPercent));
        Assert.True(maxAdjacentJump < 0.08);

        AssertClose(3.8925, YieldAtMonths(fixture, 12), 1.0e-3);
        AssertClose(3.9893, YieldAtMonths(fixture, 24), 1.0e-3);
        AssertClose(4.2728, YieldAtMonths(fixture, 60), 1.0e-3);
        AssertClose(4.6898, YieldAtMonths(fixture, 120), 1.0e-3);
        AssertClose(5.3322, YieldAtMonths(fixture, 360), 1.0e-3);
    }

    [Fact]
    public void Dense_curve_fixture_uses_actual365_year_fractions_for_pillar_sampling()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        DateTime valuationDate = fixture.Source.CurveDate.Date;

        foreach (YieldCurvePoint point in fixture.Points)
        {
            DateTime pillarDate = valuationDate.AddMonths(point.MaturityMonths);
            double actual365Years = (pillarDate - valuationDate).TotalDays / 365.0;

            Assert.Equal(actual365Years, point.MaturityYears, precision: 12);
        }
    }

    [Fact]
    public void Dense_curve_fixture_converts_to_valuation_anchor_plus_semiannual_pillars()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();

        IReadOnlyList<ZeroRatePillar> pillars = FixedRateBondMarketData.ToZeroRatePillars(
            fixture,
            fixture.Source.CurveDate);

        Assert.Equal(61, pillars.Count);
        Assert.Equal(fixture.Source.CurveDate.Date, pillars[0].Date);
        Assert.Equal(fixture.Source.CurveDate.Date.AddMonths(6), pillars[1].Date);
        Assert.Equal(fixture.Source.CurveDate.Date.AddYears(30), pillars[^1].Date);
        Assert.Equal(fixture.Points[0].ZeroYieldPercent / 100.0, pillars[0].ZeroRate, precision: 12);
        Assert.Equal(fixture.Points[^1].ZeroYieldPercent / 100.0, pillars[^1].ZeroRate, precision: 12);

        Assert.True(pillars.Zip(pillars.Skip(1)).All(pair => pair.Second.Date > pair.First.Date));
    }

    [Fact]
    public void Reference_pricer_exposes_the_supported_finance_conventions()
    {
        FixedRateBondConventionSummary conventions = QlNetFixedRateBondReferencePricer.SupportedConventions;

        Assert.Equal("UnitedStates.GovernmentBond", conventions.Calendar);
        Assert.Equal("Semiannual", conventions.ScheduleFrequency);
        Assert.Equal("30/360 USA", conventions.CouponDayCount);
        Assert.Equal("Actual/365 Fixed", conventions.CurveDayCount);
        Assert.Equal("ModifiedFollowing", conventions.BusinessDayConvention);
        Assert.Equal("Backward", conventions.DateGeneration);
        Assert.False(conventions.EndOfMonth);
        Assert.Equal("linear zero-rate interpolation", conventions.CurveInterpolation);
        Assert.Equal("continuous annual", conventions.CurveCompounding);
        Assert.Equal(100.0, conventions.Redemption);
    }

    [Fact]
    public void Zero_coupon_price_matches_manual_linear_zero_rate_interpolation()
    {
        DateTime valuationDate = new(2026, 5, 15);
        DateTime oneYear = valuationDate.AddYears(1);
        DateTime twoYears = valuationDate.AddYears(2);
        DateTime maturity = valuationDate.AddMonths(18);
        const double oneYearRate = 0.04;
        const double twoYearRate = 0.06;
        var request = new FixedRateBondRequest(
            ValuationDate: valuationDate,
            EffectiveDate: valuationDate,
            MaturityDate: maturity,
            Coupon: 0.0,
            Notional: 100.0,
            ZeroCurve:
            [
                new ZeroRatePillar(valuationDate, oneYearRate),
                new ZeroRatePillar(oneYear, oneYearRate),
                new ZeroRatePillar(twoYears, twoYearRate),
            ]);

        FixedRateBondResult result = Pricer.Price(request);

        double t0 = Actual365(valuationDate, oneYear);
        double t1 = Actual365(valuationDate, twoYears);
        double t = Actual365(valuationDate, maturity);
        double interpolatedZeroRate = oneYearRate + (twoYearRate - oneYearRate) * ((t - t0) / (t1 - t0));
        double expectedDirtyPrice = 100.0 * Math.Exp(-interpolatedZeroRate * t);

        Assert.Equal(expectedDirtyPrice, result.DirtyPrice, precision: 8);
    }

    [Fact]
    public void Actual_treasury_auction_price_is_near_flat_auction_yield_price()
    {
        // Treasury auction result R_20260513_2:
        // https://www.treasurydirect.gov/instit/annceresult/press/preanre/2026/R_20260513_2.pdf
        // CUSIP 912810UU0, 5% coupon, 2056-05-15 maturity, 5.046% high yield,
        // price 99.292811.
        DateTime issueDate = new(2026, 5, 15);
        DateTime maturityDate = new(2056, 5, 15);
        const double auctionHighYield = 0.05046;
        const double auctionPrice = 99.292811;
        double continuousEquivalentRate = 2.0 * Math.Log(1.0 + (auctionHighYield / 2.0));
        var request = new FixedRateBondRequest(
            ValuationDate: issueDate,
            EffectiveDate: issueDate,
            MaturityDate: maturityDate,
            Coupon: 0.05,
            Notional: 100.0,
            ZeroCurve:
            [
                new ZeroRatePillar(issueDate, continuousEquivalentRate),
                new ZeroRatePillar(maturityDate, continuousEquivalentRate),
            ]);

        FixedRateBondResult result = Pricer.Price(request);

        Assert.Equal(0.0, result.AccruedAmount, precision: 12);
        AssertClose(auctionPrice, result.DirtyPrice, 0.10);
    }

    [Fact]
    public void Thirty_year_dense_baseline_has_regular_cashflows_and_sensible_price()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(
            fixture,
            coupon: 0.045,
            notional: 100.0);

        FixedRateBondResult result = Pricer.Price(request);
        CashflowInfo[] coupons = result.Cashflows.Where(cashflow => cashflow.IsCoupon).ToArray();

        Assert.Equal(fixture.Source.CurveDate.Date, request.ValuationDate);
        Assert.Equal(fixture.Source.CurveDate.Date, request.EffectiveDate);
        Assert.Equal(fixture.Source.CurveDate.Date.AddYears(30), request.MaturityDate);
        Assert.Equal(60, coupons.Length);
        Assert.Equal(61, result.Cashflows.Count);
        Assert.All(coupons, coupon => Assert.InRange(coupon.AccrualPeriod!.Value, 0.49, 0.51));
        Assert.Equal(0.0, result.AccruedAmount, precision: 10);
        Assert.Equal(result.DirtyPrice, result.CleanPrice, precision: 10);
        Assert.InRange(result.DirtyPrice, 70.0, 100.0);
        Assert.True(result.NetPresentValue > 0.0);
    }

    [Fact]
    public void Dense_baseline_preserves_coupon_ordering_and_notional_scaling()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();

        FixedRateBondResult zeroCoupon = Pricer.Price(
            FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture, coupon: 0.0));
        FixedRateBondResult marketCoupon = Pricer.Price(
            FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture, coupon: 0.045));
        FixedRateBondResult highCoupon = Pricer.Price(
            FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture, coupon: 0.08));
        FixedRateBondResult doubleNotional = Pricer.Price(
            FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture, coupon: 0.045, notional: 200.0));

        Assert.True(zeroCoupon.DirtyPrice < marketCoupon.DirtyPrice);
        Assert.True(marketCoupon.DirtyPrice < highCoupon.DirtyPrice);
        Assert.Equal(2.0 * marketCoupon.NetPresentValue, doubleNotional.NetPresentValue, precision: 8);
    }

    private static double YieldAtMonths(YieldCurveFixture fixture, int months)
        => fixture.Points.Single(point => point.MaturityMonths == months).ZeroYieldPercent;

    private static double Actual365(DateTime start, DateTime end)
        => (end.Date - start.Date).TotalDays / 365.0;

    private static void AssertClose(double expected, double actual, double tolerance)
        => Assert.True(
            Math.Abs(expected - actual) <= tolerance,
            $"Expected {actual} to be within {tolerance} of {expected}.");
}
