namespace FixedRateBondSurrogate;

public sealed record MaturityBreakpointInventoryPoint(
    DateTime BoundaryDate,
    int OffsetDays,
    DateTime MaturityDate,
    double MaturityYears,
    int CashflowCount,
    int CouponCashflowCount,
    DateTime FinalCashflowDate,
    double FinalCouponAccrualPeriod,
    double DirtyPrice,
    double LeftSlopePerYear,
    double RightSlopePerYear,
    double CentralSlopePerYear,
    double SecondDifference,
    bool ScheduleRegimeChanged);

public sealed record MaturitySpecialPointCandidateSummary(
    string Name,
    string Source,
    int CandidateCount,
    IReadOnlyList<double> MaturityYears,
    string Interpretation);

public sealed record MaturitySpecialPointsReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    int PublicInputDimensionCount,
    IReadOnlyList<MaturityBreakpointInventoryPoint> BreakpointInventory,
    IReadOnlyList<MaturitySpecialPointCandidateSummary> Candidates,
    string Interpretation);

public static class MaturitySpecialPointsBenchmark
{
    private const int CurveBumpDimensionCount = 60;
    private const int TotalDimensionCount = CurveBumpDimensionCount + 2;

    public static MaturitySpecialPointsReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        if (request.ZeroCurve.Count != CurveBumpDimensionCount + 1)
        {
            throw new InvalidOperationException("Phase 9 expects the dense semiannual curve fixture.");
        }

        IReadOnlyList<MaturityBreakpointInventoryPoint> inventory = BuildBreakpointInventory(pricer, request);
        IReadOnlyList<MaturitySpecialPointCandidateSummary> candidates = BuildCandidates(inventory);

        return new MaturitySpecialPointsReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            WrapperContract: "curve bumps[60], coupon, maturity -> dirty PV",
            PublicInputDimensionCount: TotalDimensionCount,
            BreakpointInventory: inventory,
            Candidates: candidates,
            Interpretation:
                "Phase 9 tests maturity special points before introducing a reusable high-dimensional piecewise API.");
    }

    private static IReadOnlyList<MaturityBreakpointInventoryPoint> BuildBreakpointInventory(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        var points = new List<MaturityBreakpointInventoryPoint>();
        for (int months = 24; months <= 354; months += 6)
        {
            DateTime boundaryDate = request.ValuationDate.Date.AddMonths(months);
            for (int offsetDays = -1; offsetDays <= 1; offsetDays++)
            {
                DateTime maturityDate = boundaryDate.AddDays(offsetDays);
                double maturityYears = (maturityDate.Date - request.ValuationDate.Date).TotalDays / 365.25;
                if (maturityYears < 2.0 || maturityYears > 30.0)
                {
                    continue;
                }

                points.Add(BuildInventoryPoint(pricer, request, boundaryDate, offsetDays, maturityDate));
            }
        }

        return points;
    }

    private static MaturityBreakpointInventoryPoint BuildInventoryPoint(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request,
        DateTime boundaryDate,
        int offsetDays,
        DateTime maturityDate)
    {
        FixedRateBondResult previous = PriceAt(pricer, request, maturityDate.AddDays(-1));
        FixedRateBondResult current = PriceAt(pricer, request, maturityDate);
        FixedRateBondResult next = PriceAt(pricer, request, maturityDate.AddDays(1));

        double leftSlope = (current.DirtyPrice - previous.DirtyPrice) * 365.25;
        double rightSlope = (next.DirtyPrice - current.DirtyPrice) * 365.25;
        double centralSlope = (next.DirtyPrice - previous.DirtyPrice) * 365.25 / 2.0;
        double secondDifference = next.DirtyPrice - (2.0 * current.DirtyPrice) + previous.DirtyPrice;
        CashflowInfo? finalCoupon = current.Cashflows
            .Where(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon)
            .OrderBy(cashflow => cashflow.PaymentDate)
            .LastOrDefault();
        DateTime finalCashflowDate = current.Cashflows
            .Where(cashflow => !cashflow.HasOccurred)
            .Select(cashflow => cashflow.PaymentDate.Date)
            .DefaultIfEmpty(maturityDate.Date)
            .Max();
        int cashflowCount = current.Cashflows.Count(cashflow => !cashflow.HasOccurred);
        int couponCashflowCount = current.Cashflows.Count(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon);
        bool scheduleRegimeChanged = cashflowCount != previous.Cashflows.Count(cashflow => !cashflow.HasOccurred)
            || cashflowCount != next.Cashflows.Count(cashflow => !cashflow.HasOccurred)
            || Math.Abs(secondDifference) > 1e-10;

        return new MaturityBreakpointInventoryPoint(
            BoundaryDate: boundaryDate.Date,
            OffsetDays: offsetDays,
            MaturityDate: maturityDate.Date,
            MaturityYears: (maturityDate.Date - request.ValuationDate.Date).TotalDays / 365.25,
            CashflowCount: cashflowCount,
            CouponCashflowCount: couponCashflowCount,
            FinalCashflowDate: finalCashflowDate,
            FinalCouponAccrualPeriod: finalCoupon?.AccrualPeriod ?? 0.0,
            DirtyPrice: current.DirtyPrice,
            LeftSlopePerYear: leftSlope,
            RightSlopePerYear: rightSlope,
            CentralSlopePerYear: centralSlope,
            SecondDifference: secondDifference,
            ScheduleRegimeChanged: scheduleRegimeChanged);
    }

    private static FixedRateBondResult PriceAt(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request,
        DateTime maturityDate)
    {
        FixedRateBondRequest maturityRequest = request with { MaturityDate = maturityDate.Date };
        return pricer.Price(maturityRequest);
    }

    private static IReadOnlyList<MaturitySpecialPointCandidateSummary> BuildCandidates(
        IReadOnlyList<MaturityBreakpointInventoryPoint> inventory)
    {
        double[] scheduleMaturities = inventory
            .Where(point => point.OffsetDays == 0 && point.ScheduleRegimeChanged)
            .Select(point => point.MaturityYears)
            .DistinctBy(years => Math.Round(years, 6))
            .Order()
            .ToArray();
        double[] detectorMaturities = inventory
            .OrderByDescending(point => Math.Abs(point.SecondDifference))
            .Select(point => point.MaturityYears)
            .DistinctBy(years => Math.Round(years, 3))
            .Take(12)
            .Order()
            .ToArray();
        double[] hybridMaturities = scheduleMaturities
            .Concat(detectorMaturities)
            .DistinctBy(years => Math.Round(years, 6))
            .Order()
            .ToArray();

        return
        [
            new MaturitySpecialPointCandidateSummary(
                Name: "Schedule-aware special points",
                Source: "semiannual schedule boundary inventory",
                CandidateCount: scheduleMaturities.Length,
                MaturityYears: scheduleMaturities,
                Interpretation: "Declared points come from maturity dates where the local schedule diagnostics change."),
            new MaturitySpecialPointCandidateSummary(
                Name: "Automatic detector candidates",
                Source: "largest maturity-axis second differences",
                CandidateCount: detectorMaturities.Length,
                MaturityYears: detectorMaturities,
                Interpretation: "Detector points are evidence candidates and still require held-out validation."),
            new MaturitySpecialPointCandidateSummary(
                Name: "Hybrid special points",
                Source: "union of schedule-aware and detector candidates",
                CandidateCount: hybridMaturities.Length,
                MaturityYears: hybridMaturities,
                Interpretation: "The hybrid is only worth modelling if both inputs independently improve validation."),
        ];
    }
}
