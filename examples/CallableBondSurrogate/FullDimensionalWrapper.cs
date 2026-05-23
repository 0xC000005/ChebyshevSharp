using System.Globalization;
using FixedRateBondSurrogate;

namespace CallableBondSurrogate;

public sealed class CallableBondFullDimensionalWrapper
{
    public const int CurveBumpCount = 60;
    public const int DimensionCount = 65;

    private readonly ICallableBondReferencePricer _pricer;
    private readonly DateTime _valuationDate;
    private readonly IReadOnlyList<ZeroRatePillar> _baseCurve;

    private CallableBondFullDimensionalWrapper(
        ICallableBondReferencePricer pricer,
        DateTime valuationDate,
        IReadOnlyList<ZeroRatePillar> baseCurve)
    {
        _pricer = pricer;
        _valuationDate = valuationDate.Date;
        _baseCurve = baseCurve;

        DimensionLabels =
        [
            .. Enumerable.Range(1, CurveBumpCount).Select(CurveBumpLabel),
            "coupon",
            "maturityYears",
            "firstCallYears",
            "callPrice",
            "hullWhiteSigma",
        ];
    }

    public IReadOnlyList<string> DimensionLabels { get; }

    public static CallableBondFullDimensionalWrapper CreateDefault(ICallableBondReferencePricer pricer)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        DateTime valuationDate = fixture.Source.CurveDate.Date;
        IReadOnlyList<ZeroRatePillar> curve = FixedRateBondMarketData.ToZeroRatePillars(fixture, valuationDate);
        if (curve.Count != CurveBumpCount + 1)
        {
            throw new InvalidDataException(
                $"Expected valuation anchor plus {CurveBumpCount} curve pillars, but fixture produced {curve.Count}.");
        }

        return new CallableBondFullDimensionalWrapper(pricer, valuationDate, curve);
    }

    public double[] CreateBasePoint()
    {
        var point = new double[DimensionCount];
        point[60] = 0.06;
        point[61] = 30.0;
        point[62] = 5.0;
        point[63] = 100.0;
        point[64] = 0.01;
        return point;
    }

    public CallableBondRequest ToRequest(double[] point)
    {
        ValidatePoint(point);

        var bumpedCurve = new ZeroRatePillar[_baseCurve.Count];
        bumpedCurve[0] = _baseCurve[0];
        for (int i = 0; i < CurveBumpCount; i++)
        {
            ZeroRatePillar pillar = _baseCurve[i + 1];
            bumpedCurve[i + 1] = pillar with { ZeroRate = pillar.ZeroRate + point[i] * 1e-4 };
        }

        double maturityYears = point[61];
        double firstCallYears = point[62];

        return new CallableBondRequest(
            ValuationDate: _valuationDate,
            EffectiveDate: _valuationDate,
            MaturityDate: AddYearsCoordinate(_valuationDate, maturityYears),
            FirstCallDate: AddYearsCoordinate(_valuationDate, firstCallYears),
            Coupon: point[60],
            Notional: 100.0,
            CallPrice: point[63],
            HullWhiteMeanReversion: 0.03,
            HullWhiteSigma: point[64],
            TreeTimeSteps: 80,
            ZeroCurve: bumpedCurve);
    }

    public double Price(double[] point)
        => _pricer.Price(ToRequest(point)).DirtyPrice;

    private static string CurveBumpLabel(int pillarNumber)
    {
        double years = pillarNumber * 0.5;
        string formatted = Math.Abs(years - Math.Round(years)) < 1e-12
            ? ((int)Math.Round(years)).ToString(CultureInfo.InvariantCulture)
            : years.ToString("0.0", CultureInfo.InvariantCulture);

        return $"curveBump_{formatted}Y_bp";
    }

    private static DateTime AddYearsCoordinate(DateTime valuationDate, double years)
    {
        double months = years * 12.0;
        double roundedMonths = Math.Round(months);
        if (Math.Abs(months - roundedMonths) < 1e-10)
        {
            return valuationDate.AddMonths(checked((int)roundedMonths));
        }

        return valuationDate.AddDays(years * 365.25);
    }

    private static void ValidatePoint(double[] point)
    {
        if (point.Length != DimensionCount)
        {
            throw new ArgumentException(
                $"Expected a {DimensionCount}D callable-bond point, but received {point.Length} coordinates.",
                nameof(point));
        }

        for (int i = 0; i < point.Length; i++)
        {
            if (!double.IsFinite(point[i]))
            {
                throw new ArgumentException($"Coordinate {i} must be finite.", nameof(point));
            }
        }
    }
}
