namespace FixedRateBondSurrogate;

public sealed record CouponSlicePoint(
    double Coupon,
    double DirtyPrice,
    double CleanPrice,
    double AccruedAmount,
    double CouponDerivative,
    double SecondDifference);

public sealed record RateSensitivityPoint(
    int PillarIndex,
    int PillarYears,
    DateTime PillarDate,
    double Derivative,
    double ZeroPillarDv01,
    double LocalSecondDifference);

public sealed record RateBumpSlicePoint(
    int PillarYears,
    double BumpBasisPoints,
    double DirtyPrice,
    double CleanPrice,
    double AccruedAmount);

public sealed record MaturitySlicePoint(
    DateTime BoundaryDate,
    int OffsetDays,
    DateTime MaturityDate,
    bool IsScheduleBoundaryCandidate,
    int CashflowCount,
    int CouponCashflowCount,
    double DirtyPrice,
    double CleanPrice,
    double AccruedAmount,
    DateTime? FirstFutureCashflowDate,
    DateTime? FinalCashflowDate,
    double? SlopePerYear,
    double? SecondDifference);

public sealed record SmoothnessDiagnosticReport(
    string FixtureId,
    DateTime CurveDate,
    IReadOnlyList<CouponSlicePoint> CouponSlice,
    IReadOnlyList<RateSensitivityPoint> RateSensitivities,
    IReadOnlyList<RateBumpSlicePoint> RateBumpSlice,
    IReadOnlyList<MaturitySlicePoint> MaturitySlice,
    IReadOnlyList<MaturitySlicePoint> TopMaturitySpikeCandidates,
    double MaxAbsCouponSecondDifference);

public static class SmoothnessDiagnostics
{
    public const double RateStep = 1e-4;
    public const double CouponStep = 1e-4;

    private static readonly double[] CouponSamples = [0.0, 0.02, 0.045, 0.08, 0.12];
    private static readonly int[] RateBumpBasisPoints = [-150, -75, 0, 75, 150];
    private static readonly int[] SelectedPillarYears = [1, 5, 10, 20, 30];

    public static SmoothnessDiagnosticReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDefaultCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularTenYearFromFixture(fixture);

        IReadOnlyList<CouponSlicePoint> couponSlice = BuildCouponSlice(pricer, request);
        IReadOnlyList<RateSensitivityPoint> rateSensitivities = BuildRateSensitivities(pricer, request);
        IReadOnlyList<RateBumpSlicePoint> rateBumpSlice = BuildRateBumpSlice(pricer, request);
        IReadOnlyList<MaturitySlicePoint> maturitySlice = BuildMaturitySlice(pricer, request);
        IReadOnlyList<MaturitySlicePoint> topSpikes = maturitySlice
            .Where(point => point.SecondDifference.HasValue)
            .OrderByDescending(point => Math.Abs(point.SecondDifference!.Value))
            .Take(5)
            .ToArray();

        return new SmoothnessDiagnosticReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            CouponSlice: couponSlice,
            RateSensitivities: rateSensitivities,
            RateBumpSlice: rateBumpSlice,
            MaturitySlice: maturitySlice,
            TopMaturitySpikeCandidates: topSpikes,
            MaxAbsCouponSecondDifference: couponSlice.Max(point => Math.Abs(point.SecondDifference)));
    }

    public static FixedRateBondRequest BumpZeroRate(
        FixedRateBondRequest request,
        int pillarIndex,
        double bump)
    {
        if (pillarIndex < 0 || pillarIndex >= request.ZeroCurve.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(pillarIndex));
        }

        ZeroRatePillar[] bumped = request.ZeroCurve.ToArray();
        ZeroRatePillar pillar = bumped[pillarIndex];
        bumped[pillarIndex] = pillar with { ZeroRate = pillar.ZeroRate + bump };
        return request with { ZeroCurve = bumped };
    }

    public static FixedRateBondRequest WithCoupon(FixedRateBondRequest request, double coupon)
        => request with { Coupon = coupon };

    public static FixedRateBondRequest WithMaturity(FixedRateBondRequest request, DateTime maturityDate)
    {
        if (maturityDate.Date <= request.EffectiveDate.Date)
        {
            throw new ArgumentException("Maturity date must be after the effective date.", nameof(maturityDate));
        }

        return request with { MaturityDate = maturityDate.Date };
    }

    public static double RateDerivative(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request,
        int pillarIndex,
        double step = RateStep)
    {
        double up = pricer.Price(BumpZeroRate(request, pillarIndex, step)).DirtyPrice;
        double down = pricer.Price(BumpZeroRate(request, pillarIndex, -step)).DirtyPrice;
        return (up - down) / (2.0 * step);
    }

    public static double CouponDerivative(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request,
        double step = CouponStep)
    {
        double up = pricer.Price(WithCoupon(request, request.Coupon + step)).DirtyPrice;
        double down = pricer.Price(WithCoupon(request, request.Coupon - step)).DirtyPrice;
        return (up - down) / (2.0 * step);
    }

    public static double RateCouponMixedDerivative(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request,
        int pillarIndex,
        double rateStep = RateStep,
        double couponStep = CouponStep)
    {
        double upUp = pricer.Price(WithCoupon(BumpZeroRate(request, pillarIndex, rateStep), request.Coupon + couponStep)).DirtyPrice;
        double upDown = pricer.Price(WithCoupon(BumpZeroRate(request, pillarIndex, rateStep), request.Coupon - couponStep)).DirtyPrice;
        double downUp = pricer.Price(WithCoupon(BumpZeroRate(request, pillarIndex, -rateStep), request.Coupon + couponStep)).DirtyPrice;
        double downDown = pricer.Price(WithCoupon(BumpZeroRate(request, pillarIndex, -rateStep), request.Coupon - couponStep)).DirtyPrice;

        return (upUp - upDown - downUp + downDown) / (4.0 * rateStep * couponStep);
    }

    private static IReadOnlyList<CouponSlicePoint> BuildCouponSlice(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        return CouponSamples
            .Select(coupon =>
            {
                FixedRateBondRequest current = WithCoupon(request, coupon);
                FixedRateBondResult result = pricer.Price(current);
                double derivative = CouponDerivative(pricer, current);
                double up = pricer.Price(WithCoupon(request, coupon + CouponStep)).DirtyPrice;
                double down = pricer.Price(WithCoupon(request, coupon - CouponStep)).DirtyPrice;
                double secondDifference = up - 2.0 * result.DirtyPrice + down;

                return new CouponSlicePoint(
                    Coupon: coupon,
                    DirtyPrice: result.DirtyPrice,
                    CleanPrice: result.CleanPrice,
                    AccruedAmount: result.AccruedAmount,
                    CouponDerivative: derivative,
                    SecondDifference: secondDifference);
            })
            .ToArray();
    }

    private static IReadOnlyList<RateSensitivityPoint> BuildRateSensitivities(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        double baseDirty = pricer.Price(request).DirtyPrice;
        var points = new List<RateSensitivityPoint>();

        foreach (int year in SelectedPillarYears)
        {
            int index = FindPillarIndex(request, year);
            double up = pricer.Price(BumpZeroRate(request, index, RateStep)).DirtyPrice;
            double down = pricer.Price(BumpZeroRate(request, index, -RateStep)).DirtyPrice;
            double derivative = (up - down) / (2.0 * RateStep);
            double localSecondDifference = up - 2.0 * baseDirty + down;

            points.Add(new RateSensitivityPoint(
                PillarIndex: index,
                PillarYears: year,
                PillarDate: request.ZeroCurve[index].Date,
                Derivative: derivative,
                ZeroPillarDv01: derivative * 1e-4,
                LocalSecondDifference: localSecondDifference));
        }

        return points;
    }

    private static IReadOnlyList<RateBumpSlicePoint> BuildRateBumpSlice(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        var points = new List<RateBumpSlicePoint>();

        foreach (int year in SelectedPillarYears)
        {
            int index = FindPillarIndex(request, year);

            foreach (int bumpBasisPoints in RateBumpBasisPoints)
            {
                FixedRateBondResult result = pricer.Price(
                    BumpZeroRate(request, index, bumpBasisPoints * 1e-4));

                points.Add(new RateBumpSlicePoint(
                    PillarYears: year,
                    BumpBasisPoints: bumpBasisPoints,
                    DirtyPrice: result.DirtyPrice,
                    CleanPrice: result.CleanPrice,
                    AccruedAmount: result.AccruedAmount));
            }
        }

        return points;
    }

    private static IReadOnlyList<MaturitySlicePoint> BuildMaturitySlice(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        var allPoints = new List<MaturitySlicePoint>();

        for (int months = 24; months <= 60; months += 6)
        {
            DateTime boundary = request.EffectiveDate.Date.AddMonths(months);
            var window = new List<MaturitySlicePoint>();

            for (int offset = -7; offset <= 7; offset++)
            {
                DateTime maturity = boundary.AddDays(offset);
                FixedRateBondResult result = pricer.Price(WithMaturity(request, maturity));
                CashflowInfo[] futureCashflows = result.Cashflows
                    .Where(cashflow => !cashflow.HasOccurred)
                    .OrderBy(cashflow => cashflow.PaymentDate)
                    .ToArray();

                window.Add(new MaturitySlicePoint(
                    BoundaryDate: boundary,
                    OffsetDays: offset,
                    MaturityDate: maturity,
                    IsScheduleBoundaryCandidate: offset == 0,
                    CashflowCount: result.Cashflows.Count,
                    CouponCashflowCount: result.Cashflows.Count(cashflow => cashflow.IsCoupon),
                    DirtyPrice: result.DirtyPrice,
                    CleanPrice: result.CleanPrice,
                    AccruedAmount: result.AccruedAmount,
                    FirstFutureCashflowDate: futureCashflows.FirstOrDefault()?.PaymentDate,
                    FinalCashflowDate: futureCashflows.LastOrDefault()?.PaymentDate,
                    SlopePerYear: null,
                    SecondDifference: null));
            }

            allPoints.AddRange(AddFiniteDifferences(window));
        }

        return allPoints;
    }

    private static IReadOnlyList<MaturitySlicePoint> AddFiniteDifferences(
        IReadOnlyList<MaturitySlicePoint> window)
    {
        var points = new MaturitySlicePoint[window.Count];

        for (int i = 0; i < window.Count; i++)
        {
            MaturitySlicePoint current = window[i];
            double? slope = null;
            double? secondDifference = null;

            if (i > 0 && i < window.Count - 1)
            {
                MaturitySlicePoint previous = window[i - 1];
                MaturitySlicePoint next = window[i + 1];
                slope = (next.DirtyPrice - previous.DirtyPrice) / (2.0 / 365.0);
                secondDifference = next.DirtyPrice - 2.0 * current.DirtyPrice + previous.DirtyPrice;
            }

            points[i] = current with
            {
                SlopePerYear = slope,
                SecondDifference = secondDifference,
            };
        }

        return points;
    }

    private static int FindPillarIndex(FixedRateBondRequest request, int pillarYears)
    {
        DateTime expected = request.ValuationDate.Date.AddYears(pillarYears);

        for (int i = 0; i < request.ZeroCurve.Count; i++)
        {
            if (request.ZeroCurve[i].Date.Date == expected)
            {
                return i;
            }
        }

        throw new InvalidOperationException($"No {pillarYears}Y zero-rate pillar exists on the request curve.");
    }
}
