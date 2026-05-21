using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondReferencePricerTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Price_returns_finite_outputs_for_regular_fixed_rate_bond()
    {
        FixedRateBondResult result = Pricer.Price(FixedRateBondScenarios.RegularTenYear());

        Assert.True(double.IsFinite(result.DirtyPrice));
        Assert.True(double.IsFinite(result.CleanPrice));
        Assert.True(double.IsFinite(result.AccruedAmount));
        Assert.True(double.IsFinite(result.NetPresentValue));
        Assert.True(double.IsFinite(result.SettlementValue));
        Assert.Equal(21, result.Cashflows.Count);
        Assert.Contains(result.Cashflows, cf => cf.IsCoupon);
        Assert.Contains(result.Cashflows, cf => !cf.IsCoupon);
    }

    [Fact]
    public void Coupon_dependence_is_linear_for_dirty_price()
    {
        FixedRateBondResult lowCoupon = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 0.02));
        FixedRateBondResult midCoupon = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 0.04));
        FixedRateBondResult highCoupon = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 0.06));

        double lowToMid = midCoupon.DirtyPrice - lowCoupon.DirtyPrice;
        double midToHigh = highCoupon.DirtyPrice - midCoupon.DirtyPrice;

        Assert.Equal(lowToMid, midToHigh, precision: 10);
    }

    [Fact]
    public void Principal_and_annuity_recombine_to_coupon_price()
    {
        const double coupon = 0.0475;

        FixedRateBondResult principal = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 0.0));
        FixedRateBondResult unitCoupon = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 1.0));
        FixedRateBondResult target = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: coupon));

        double annuity = unitCoupon.DirtyPrice - principal.DirtyPrice;
        double recombined = principal.DirtyPrice + coupon * annuity;

        Assert.Equal(target.DirtyPrice, recombined, precision: 9);
    }

    [Fact]
    public void Zero_coupon_case_matches_principal_component()
    {
        FixedRateBondResult zeroCoupon = Pricer.Price(FixedRateBondScenarios.RegularTenYear(coupon: 0.0));

        double couponCashflowTotal = zeroCoupon.Cashflows
            .Where(cf => cf.IsCoupon)
            .Sum(cf => Math.Abs(cf.Amount));

        Assert.Equal(0.0, couponCashflowTotal, precision: 12);
        Assert.True(zeroCoupon.DirtyPrice > 0.0);
    }

    [Fact]
    public void Matured_bond_has_zero_value_and_rate_sensitivity()
    {
        FixedRateBondRequest matured = MaturedBondRequest(rateBump: 0.0);
        FixedRateBondRequest bumped = MaturedBondRequest(rateBump: 0.01);

        FixedRateBondResult baseResult = Pricer.Price(matured);
        FixedRateBondResult bumpedResult = Pricer.Price(bumped);

        Assert.Equal(0.0, baseResult.DirtyPrice, precision: 12);
        Assert.Equal(0.0, baseResult.CleanPrice, precision: 12);
        Assert.Equal(0.0, baseResult.NetPresentValue, precision: 12);
        Assert.Equal(baseResult.DirtyPrice, bumpedResult.DirtyPrice, precision: 12);
    }

    private static FixedRateBondRequest MaturedBondRequest(double rateBump)
    {
        DateTime valuationDate = new(2026, 5, 20);

        return new FixedRateBondRequest(
            ValuationDate: valuationDate,
            EffectiveDate: new DateTime(2010, 5, 20),
            MaturityDate: new DateTime(2020, 5, 20),
            Coupon: 0.05,
            Notional: 100.0,
            ZeroCurve:
            [
                new ZeroRatePillar(valuationDate, 0.03 + rateBump),
                new ZeroRatePillar(valuationDate.AddYears(1), 0.031 + rateBump),
                new ZeroRatePillar(valuationDate.AddYears(5), 0.034 + rateBump),
                new ZeroRatePillar(valuationDate.AddYears(10), 0.038 + rateBump),
            ]);
    }
}
