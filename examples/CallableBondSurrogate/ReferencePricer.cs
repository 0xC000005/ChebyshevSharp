using FixedRateBondSurrogate;
using QLNet;
using QDate = QLNet.Date;

namespace CallableBondSurrogate;

public sealed record CallableBondRequest(
    DateTime ValuationDate,
    DateTime EffectiveDate,
    DateTime MaturityDate,
    DateTime FirstCallDate,
    double Coupon,
    double Notional,
    double CallPrice,
    double HullWhiteMeanReversion,
    double HullWhiteSigma,
    int TreeTimeSteps,
    IReadOnlyList<ZeroRatePillar> ZeroCurve,
    int SettlementDays = 0);

public sealed record CallableBondResult(
    double DirtyPrice,
    double CleanPrice,
    double AccruedAmount,
    double NetPresentValue,
    double StraightDirtyPrice,
    double EmbeddedCallValue,
    int CallabilityCount);

public interface ICallableBondReferencePricer
{
    CallableBondResult Price(CallableBondRequest request);
}

public sealed class QlNetCallableBondReferencePricer : ICallableBondReferencePricer
{
    private static readonly IFixedRateBondReferencePricer StraightBondPricer =
        new QlNetFixedRateBondReferencePricer();

    public CallableBondResult Price(CallableBondRequest request)
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

        CallabilitySchedule callability = BuildCallabilitySchedule(request, calendar);

        var callableBond = new CallableFixedRateBond(
            request.SettlementDays,
            request.Notional,
            schedule,
            [request.Coupon],
            couponDayCounter,
            BusinessDayConvention.ModifiedFollowing,
            redemption: 100.0,
            issueDate: effectiveDate,
            putCallSchedule: callability);

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
        var model = new HullWhite(
            curveHandle,
            request.HullWhiteMeanReversion,
            request.HullWhiteSigma);

        callableBond.setPricingEngine(
            new TreeCallableFixedRateBondEngine(model, request.TreeTimeSteps, curveHandle));

        FixedRateBondResult straight = StraightBondPricer.Price(new FixedRateBondRequest(
            request.ValuationDate,
            request.EffectiveDate,
            request.MaturityDate,
            request.Coupon,
            request.Notional,
            request.ZeroCurve,
            request.SettlementDays));

        double dirtyPrice = callableBond.dirtyPrice();
        double straightDirtyPrice = straight.DirtyPrice;

        return new CallableBondResult(
            DirtyPrice: dirtyPrice,
            CleanPrice: callableBond.cleanPrice(),
            AccruedAmount: callableBond.accruedAmount(valuationDate),
            NetPresentValue: callableBond.NPV(),
            StraightDirtyPrice: straightDirtyPrice,
            EmbeddedCallValue: straightDirtyPrice - dirtyPrice,
            CallabilityCount: callability.Count);
    }

    private static CallabilitySchedule BuildCallabilitySchedule(
        CallableBondRequest request,
        Calendar calendar)
    {
        var schedule = new CallabilitySchedule();
        var price = new Bond.Price(request.CallPrice, Bond.Price.Type.Clean);

        DateTime callDate = request.FirstCallDate.Date;
        while (callDate < request.MaturityDate.Date)
        {
            QDate adjusted = calendar.adjust(ToQlDate(callDate), BusinessDayConvention.ModifiedFollowing);
            if (adjusted < ToQlDate(request.MaturityDate))
            {
                schedule.Add(new Callability(price, Callability.Type.Call, adjusted));
            }

            callDate = callDate.AddMonths(6);
        }

        return schedule;
    }

    private static void Validate(CallableBondRequest request)
    {
        if (request.Notional <= 0.0 || !double.IsFinite(request.Notional))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Notional must be finite and positive.");
        }

        if (!double.IsFinite(request.Coupon))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Coupon must be finite.");
        }

        if (!double.IsFinite(request.CallPrice) || request.CallPrice <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Call price must be finite and positive.");
        }

        if (!double.IsFinite(request.HullWhiteMeanReversion) || request.HullWhiteMeanReversion <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Hull-White mean reversion must be finite and positive.");
        }

        if (!double.IsFinite(request.HullWhiteSigma) || request.HullWhiteSigma < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Hull-White sigma must be finite and non-negative.");
        }

        if (request.TreeTimeSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Tree time steps must be positive.");
        }

        if (request.EffectiveDate >= request.MaturityDate)
        {
            throw new ArgumentException("Effective date must be before maturity date.", nameof(request));
        }

        if (request.FirstCallDate <= request.EffectiveDate || request.FirstCallDate >= request.MaturityDate)
        {
            throw new ArgumentException("First call date must be after effective date and before maturity.", nameof(request));
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

public static class CallableBondScenarios
{
    public static CallableBondRequest StandardThirtyYear(
        double coupon = 0.06,
        double callPrice = 100.0,
        double hullWhiteSigma = 0.01)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        DateTime curveDate = fixture.Source.CurveDate.Date;

        return new CallableBondRequest(
            ValuationDate: curveDate,
            EffectiveDate: curveDate,
            MaturityDate: curveDate.AddYears(30),
            FirstCallDate: curveDate.AddYears(5),
            Coupon: coupon,
            Notional: 100.0,
            CallPrice: callPrice,
            HullWhiteMeanReversion: 0.03,
            HullWhiteSigma: hullWhiteSigma,
            TreeTimeSteps: 80,
            ZeroCurve: FixedRateBondMarketData.ToZeroRatePillars(fixture, curveDate));
    }
}
