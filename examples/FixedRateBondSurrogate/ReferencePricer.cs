using QLNet;
using QDate = QLNet.Date;

namespace FixedRateBondSurrogate;

public sealed record ZeroRatePillar(DateTime Date, double ZeroRate);

public sealed record FixedRateBondRequest(
    DateTime ValuationDate,
    DateTime EffectiveDate,
    DateTime MaturityDate,
    double Coupon,
    double Notional,
    IReadOnlyList<ZeroRatePillar> ZeroCurve,
    int SettlementDays = 0);

public sealed record CashflowInfo(
    DateTime PaymentDate,
    double Amount,
    bool HasOccurred,
    bool IsCoupon,
    double? CouponRate,
    double? AccrualPeriod);

public sealed record FixedRateBondResult(
    double DirtyPrice,
    double CleanPrice,
    double AccruedAmount,
    double NetPresentValue,
    double SettlementValue,
    IReadOnlyList<CashflowInfo> Cashflows);

public sealed record FixedRateBondConventionSummary(
    string Calendar,
    string ScheduleFrequency,
    string CouponDayCount,
    string CurveDayCount,
    string BusinessDayConvention,
    string DateGeneration,
    bool EndOfMonth,
    string CurveInterpolation,
    string CurveCompounding,
    double Redemption);

public interface IFixedRateBondReferencePricer
{
    FixedRateBondResult Price(FixedRateBondRequest request);
}

public sealed class QlNetFixedRateBondReferencePricer : IFixedRateBondReferencePricer
{
    public static FixedRateBondConventionSummary SupportedConventions { get; } = new(
        Calendar: "UnitedStates.GovernmentBond",
        ScheduleFrequency: "Semiannual",
        CouponDayCount: "30/360 USA",
        CurveDayCount: "Actual/365 Fixed",
        BusinessDayConvention: "ModifiedFollowing",
        DateGeneration: "Backward",
        EndOfMonth: false,
        CurveInterpolation: "linear zero-rate interpolation",
        CurveCompounding: "continuous annual",
        Redemption: 100.0);

    public FixedRateBondResult Price(FixedRateBondRequest request)
    {
        Validate(request);

        QDate valuationDate = ToQlDate(request.ValuationDate);
        QDate effectiveDate = ToQlDate(request.EffectiveDate);
        QDate maturityDate = ToQlDate(request.MaturityDate);

        Settings.setEvaluationDate(valuationDate);

        Calendar calendar = new UnitedStates(UnitedStates.Market.GovernmentBond);
        DayCounter curveDayCounter = new Actual365Fixed();
        DayCounter couponDayCounter = new Thirty360(Thirty360.Thirty360Convention.USA);

        var schedule = new Schedule(
            effectiveDate,
            maturityDate,
            new Period(Frequency.Semiannual),
            calendar,
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.ModifiedFollowing,
            DateGeneration.Rule.Backward,
            false);

        var bond = new FixedRateBond(
            request.SettlementDays,
            request.Notional,
            schedule,
            [request.Coupon],
            couponDayCounter,
            BusinessDayConvention.ModifiedFollowing,
            redemption: 100.0,
            issueDate: effectiveDate);

        var curveDates = request.ZeroCurve.Select(p => ToQlDate(p.Date)).ToList();
        var curveRates = request.ZeroCurve.Select(p => p.ZeroRate).ToList();

        YieldTermStructure zeroCurve = new InterpolatedZeroCurve<Linear>(
            curveDates,
            curveRates,
            curveDayCounter,
            calendar,
            new Linear(),
            Compounding.Continuous,
            Frequency.Annual);

        var curveHandle = new Handle<YieldTermStructure>(zeroCurve);
        bond.setPricingEngine(new DiscountingBondEngine(curveHandle));

        DateTime valuation = request.ValuationDate.Date;
        var cashflows = bond.cashflows()
            .Select(cf => ToCashflowInfo(cf, valuation))
            .ToArray();

        return new FixedRateBondResult(
            DirtyPrice: bond.dirtyPrice(),
            CleanPrice: bond.cleanPrice(),
            AccruedAmount: bond.accruedAmount(valuationDate),
            NetPresentValue: bond.NPV(),
            SettlementValue: bond.settlementValue(),
            Cashflows: cashflows);
    }

    private static CashflowInfo ToCashflowInfo(CashFlow cashflow, DateTime valuationDate)
    {
        QDate paymentDate = cashflow.date();
        bool isCoupon = cashflow is Coupon;
        double? rate = null;
        double? accrualPeriod = null;

        if (cashflow is Coupon coupon)
        {
            rate = coupon.rate();
            accrualPeriod = coupon.accrualPeriod();
        }

        return new CashflowInfo(
            PaymentDate: paymentDate.ToDateTime(),
            Amount: cashflow.amount(),
            HasOccurred: paymentDate.ToDateTime().Date <= valuationDate.Date,
            IsCoupon: isCoupon,
            CouponRate: rate,
            AccrualPeriod: accrualPeriod);
    }

    private static void Validate(FixedRateBondRequest request)
    {
        if (request.Notional <= 0.0 || !double.IsFinite(request.Notional))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Notional must be finite and positive.");
        }

        if (!double.IsFinite(request.Coupon))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Coupon must be finite.");
        }

        if (request.EffectiveDate >= request.MaturityDate)
        {
            throw new ArgumentException("Effective date must be before maturity date.", nameof(request));
        }

        if (request.ZeroCurve.Count < 2)
        {
            throw new ArgumentException("At least two zero-rate pillars are required.", nameof(request));
        }

        DateTime previous = DateTime.MinValue;
        foreach (ZeroRatePillar pillar in request.ZeroCurve)
        {
            if (pillar.Date.Date <= previous.Date)
            {
                throw new ArgumentException("Zero-rate pillars must be strictly increasing by date.", nameof(request));
            }

            if (!double.IsFinite(pillar.ZeroRate))
            {
                throw new ArgumentException("Zero rates must be finite.", nameof(request));
            }

            previous = pillar.Date.Date;
        }

        if (request.ZeroCurve[0].Date.Date != request.ValuationDate.Date)
        {
            throw new ArgumentException("The first zero-rate pillar must be the valuation date.", nameof(request));
        }

        if (request.ZeroCurve[^1].Date.Date < request.MaturityDate.Date)
        {
            throw new ArgumentException("The zero curve must cover the maturity date.", nameof(request));
        }
    }

    private static QDate ToQlDate(DateTime date)
        => new(date.Day, date.Month, date.Year);
}

public static class FixedRateBondScenarios
{
    public static FixedRateBondRequest RegularTenYear(double coupon = 0.045, double notional = 100.0)
    {
        DateTime valuationDate = new(2026, 5, 20);

        return new FixedRateBondRequest(
            ValuationDate: valuationDate,
            EffectiveDate: new DateTime(2026, 5, 20),
            MaturityDate: new DateTime(2036, 5, 20),
            Coupon: coupon,
            Notional: notional,
            ZeroCurve:
            [
                new ZeroRatePillar(valuationDate, 0.0380),
                new ZeroRatePillar(valuationDate.AddYears(1), 0.0365),
                new ZeroRatePillar(valuationDate.AddYears(2), 0.0358),
                new ZeroRatePillar(valuationDate.AddYears(5), 0.0372),
                new ZeroRatePillar(valuationDate.AddYears(10), 0.0395),
                new ZeroRatePillar(valuationDate.AddYears(20), 0.0410),
                new ZeroRatePillar(valuationDate.AddYears(30), 0.0418),
            ]);
    }
}
