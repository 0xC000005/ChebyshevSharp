using ChebyshevSharp;
using FixedRateBondSurrogate;
using QLNet;
using QDate = QLNet.Date;

namespace CallableBondSurrogate;

public sealed class DynamicChebyshevCallablePricer
{
    private const int DefaultStateNodes = 21;
    private const double RelativeTimeTolerance = 1e-10;

    private static readonly double[] HermiteNodes =
    [
        -2.930637420257244,
        -1.981656756695843,
        -1.157193712446780,
        -0.3811869902073221,
        0.3811869902073221,
        1.157193712446780,
        1.981656756695843,
        2.930637420257244,
    ];

    private static readonly double[] HermiteWeights =
    [
        0.0001996040722113676,
        0.01707798300741348,
        0.2078023258148919,
        0.6611470125582413,
        0.6611470125582413,
        0.2078023258148919,
        0.01707798300741348,
        0.0001996040722113676,
    ];

    private readonly CallableBondFullDimensionalWrapper _wrapper;
    private readonly IFixedRateBondReferencePricer _straightPricer;
    private readonly int _stateNodes;
    private readonly bool _includeTreeTimeGrid;

    public DynamicChebyshevCallablePricer(
        CallableBondFullDimensionalWrapper wrapper,
        int stateNodes = DefaultStateNodes,
        bool includeTreeTimeGrid = false)
    {
        _wrapper = wrapper;
        _straightPricer = new QlNetFixedRateBondReferencePricer();
        _stateNodes = stateNodes;
        _includeTreeTimeGrid = includeTreeTimeGrid;
    }

    public double Price(double[] fullPoint)
    {
        CallableBondRequest request = _wrapper.ToRequest(fullPoint);
        FixedRateBondResult straight = _straightPricer.Price(new FixedRateBondRequest(
            request.ValuationDate,
            request.EffectiveDate,
            request.MaturityDate,
            request.Coupon,
            request.Notional,
            request.ZeroCurve,
            request.SettlementDays));

        EventPoint[] events = BuildEvents(request, straight, _includeTreeTimeGrid);
        if (events.Length == 0)
        {
            return 0.0;
        }

        double horizon = events[^1].Time;
        double xStd = request.HullWhiteSigma * Math.Sqrt((1.0 - Math.Exp(-2.0 * request.HullWhiteMeanReversion * horizon))
            / (2.0 * request.HullWhiteMeanReversion));
        double xMax = Math.Max(0.01, 5.0 * xStd);
        double[] stateDomain = [-xMax, xMax];

        Func<double, double> nextValue = _ => events[^1].CashflowAmount;
        double nextTime = events[^1].Time;

        for (int eventIndex = events.Length - 2; eventIndex >= 0; eventIndex--)
        {
            EventPoint current = events[eventIndex];
            Func<double, double> valueAtEvent = x =>
            {
                double continuation = ContinuationValue(request, current.Time, nextTime, x, nextValue);
                double value = current.CashflowAmount + continuation;
                return current.IsCallable
                    ? current.CashflowAmount + Math.Min(request.CallPrice, continuation)
                    : value;
            };

            ChebyshevApproximation approximation = BuildStateApproximation(valueAtEvent, stateDomain);
            nextValue = x => EvalStateApproximation(approximation, x, stateDomain);
            nextTime = current.Time;
        }

        return ContinuationValue(request, 0.0, nextTime, 0.0, nextValue);
    }

    private ChebyshevApproximation BuildStateApproximation(
        Func<double, double> valueAtEvent,
        double[] stateDomain)
    {
        double Function(double[] point, object? _)
            => valueAtEvent(point[0]);

        var approximation = new ChebyshevApproximation(
            Function,
            numDimensions: 1,
            domain: [stateDomain],
            nNodes: [_stateNodes]);
        approximation.Build(verbose: false);
        return approximation;
    }

    private static double EvalStateApproximation(
        ChebyshevApproximation approximation,
        double x,
        double[] stateDomain)
    {
        double clamped = Math.Clamp(x, stateDomain[0], stateDomain[1]);
        return approximation.VectorizedEval([clamped], [0]);
    }

    private static double ContinuationValue(
        CallableBondRequest request,
        double time,
        double nextTime,
        double x,
        Func<double, double> nextValue)
    {
        double dt = nextTime - time;
        if (dt <= RelativeTimeTolerance)
        {
            return nextValue(x);
        }

        double a = request.HullWhiteMeanReversion;
        double sigma = request.HullWhiteSigma;
        double mean = x * Math.Exp(-a * dt);
        double variance = sigma * sigma * (1.0 - Math.Exp(-2.0 * a * dt)) / (2.0 * a);
        double std = Math.Sqrt(Math.Max(variance, 0.0));
        double phi0 = Phi(request, time);
        double phi1 = Phi(request, nextTime);

        double sum = 0.0;
        for (int i = 0; i < HermiteNodes.Length; i++)
        {
            double xNext = mean + (Math.Sqrt(2.0) * std * HermiteNodes[i]);
            double r0 = x + phi0;
            double r1 = xNext + phi1;
            double discount = Math.Exp(-0.5 * (r0 + r1) * dt);
            sum += HermiteWeights[i] * discount * nextValue(xNext);
        }

        return sum / Math.Sqrt(Math.PI);
    }

    private static EventPoint[] BuildEvents(
        CallableBondRequest request,
        FixedRateBondResult straight,
        bool includeTreeTimeGrid)
    {
        var byTime = new SortedDictionary<double, EventPointBuilder>();
        foreach (CashflowInfo cashflow in straight.Cashflows.Where(cashflow => !cashflow.HasOccurred))
        {
            EventPointBuilder builder = GetBuilder(byTime, request, cashflow.PaymentDate.Date);
            builder.CashflowAmount += cashflow.Amount;
        }

        DateTime[] couponDates = straight.Cashflows
            .Where(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon)
            .Select(cashflow => cashflow.PaymentDate.Date)
            .Distinct()
            .Order()
            .ToArray();

        foreach (DateTime callDate in BuildCallDates(request))
        {
            DateTime adjustedCallDate = NearestCouponDate(callDate, couponDates);
            EventPointBuilder builder = GetBuilder(byTime, request, adjustedCallDate);
            builder.IsCallable = true;
        }

        if (includeTreeTimeGrid)
        {
            double maturityTime = YearFraction(request.ValuationDate, request.MaturityDate);
            for (int step = 1; step < request.TreeTimeSteps; step++)
            {
                double time = maturityTime * step / request.TreeTimeSteps;
                DateTime date = request.ValuationDate.AddDays(time * 365.0).Date;
                _ = GetBuilder(byTime, request, date, time);
            }
        }

        return byTime.Values
            .Where(builder => builder.Time > RelativeTimeTolerance)
            .Select(builder => new EventPoint(builder.Date, builder.Time, builder.CashflowAmount, builder.IsCallable))
            .OrderBy(point => point.Time)
            .ToArray();
    }

    private static EventPointBuilder GetBuilder(
        SortedDictionary<double, EventPointBuilder> byTime,
        CallableBondRequest request,
        DateTime date)
        => GetBuilder(byTime, request, date, YearFraction(request.ValuationDate, date));

    private static EventPointBuilder GetBuilder(
        SortedDictionary<double, EventPointBuilder> byTime,
        CallableBondRequest request,
        DateTime date,
        double time)
    {
        double key = TimeKey(time);
        if (!byTime.TryGetValue(key, out EventPointBuilder? builder))
        {
            builder = new EventPointBuilder(date, time);
            byTime.Add(key, builder);
        }

        return builder;
    }

    private static double TimeKey(double time)
        => Math.Round(time, 10);

    private static IEnumerable<DateTime> BuildCallDates(CallableBondRequest request)
    {
        Calendar calendar = new UnitedStates(UnitedStates.Market.GovernmentBond);
        DateTime callDate = request.FirstCallDate.Date;
        QDate maturity = ToQlDate(request.MaturityDate);
        while (callDate < request.MaturityDate.Date)
        {
            QDate adjusted = calendar.adjust(ToQlDate(callDate), BusinessDayConvention.ModifiedFollowing);
            if (adjusted < maturity)
            {
                yield return adjusted.ToDateTime().Date;
            }

            callDate = callDate.AddMonths(6);
        }
    }

    private static DateTime NearestCouponDate(DateTime callDate, IReadOnlyList<DateTime> couponDates)
    {
        if (couponDates.Count == 0)
        {
            return callDate;
        }

        return couponDates
            .OrderBy(date => Math.Abs((date.Date - callDate.Date).TotalDays))
            .First();
    }

    private static double Phi(CallableBondRequest request, double time)
    {
        double a = request.HullWhiteMeanReversion;
        double sigma = request.HullWhiteSigma;
        double convexity = sigma * sigma * Math.Pow(1.0 - Math.Exp(-a * time), 2.0) / (2.0 * a * a);
        return InstantaneousForward(request, time) + convexity;
    }

    private static double InstantaneousForward(CallableBondRequest request, double time)
    {
        double h = Math.Max(1.0 / 365.0, time * 1e-4);
        if (time <= h)
        {
            return (ZeroRateAtTime(request, h) * h) / h;
        }

        double left = time - h;
        double right = time + h;
        double leftIntegrated = ZeroRateAtTime(request, left) * left;
        double rightIntegrated = ZeroRateAtTime(request, right) * right;
        return (rightIntegrated - leftIntegrated) / (right - left);
    }

    private static double ZeroRateAtTime(CallableBondRequest request, double targetTime)
    {
        IReadOnlyList<ZeroRatePillar> curve = request.ZeroCurve;
        if (targetTime <= 0.0)
        {
            return curve[0].ZeroRate;
        }

        for (int i = 1; i < curve.Count; i++)
        {
            double rightTime = YearFraction(request.ValuationDate, curve[i].Date);
            if (targetTime <= rightTime)
            {
                double leftTime = YearFraction(request.ValuationDate, curve[i - 1].Date);
                double width = Math.Max(rightTime - leftTime, 1e-12);
                double weight = (targetTime - leftTime) / width;
                return ((1.0 - weight) * curve[i - 1].ZeroRate) + (weight * curve[i].ZeroRate);
            }
        }

        return curve[^1].ZeroRate;
    }

    private static double YearFraction(DateTime valuationDate, DateTime date)
        => (date.Date - valuationDate.Date).TotalDays / 365.0;

    private static QDate ToQlDate(DateTime date)
        => new(date.Day, date.Month, date.Year);

    private sealed record EventPoint(
        DateTime Date,
        double Time,
        double CashflowAmount,
        bool IsCallable);

    private sealed class EventPointBuilder
    {
        public EventPointBuilder(DateTime date, double time)
        {
            Date = date;
            Time = time;
        }

        public DateTime Date { get; }
        public double Time { get; }
        public double CashflowAmount { get; set; }
        public bool IsCallable { get; set; }
    }
}
