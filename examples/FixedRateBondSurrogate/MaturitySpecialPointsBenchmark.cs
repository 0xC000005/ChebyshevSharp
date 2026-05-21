using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using ChebyshevSharp;

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
    bool ScheduleRegimeChanged,
    string ScheduleRegimeReason);

public sealed record MaturitySpecialPointCandidateSummary(
    string Name,
    string Source,
    int CandidateCount,
    IReadOnlyList<double> MaturityYears,
    string Interpretation);

public sealed record MaturityBreakpointInventorySummary(
    int PointCount,
    int ScheduleRegimeChangeCount,
    double MaxAbsSecondDifference,
    double MaxAbsSlopeJump,
    DateTime WorstMaturityDate,
    string Interpretation);

public sealed record MaturitySpecialPointDecision(
    string Recommendation,
    string LibraryEnhancementDecision,
    string Evidence);

public sealed record MaturitySpecialPointsReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    int PublicInputDimensionCount,
    IReadOnlyList<MaturityBreakpointInventoryPoint> BreakpointInventory,
    MaturityBreakpointInventorySummary InventorySummary,
    IReadOnlyList<MaturitySpecialPointCandidateSummary> Candidates,
    IReadOnlyList<AnalyticCouponModelSummary> Models,
    MaturitySpecialPointDecision Decision,
    string Interpretation);

public static class MaturitySpecialPointsBenchmark
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int TotalDimensionCount = CurveBumpDimensionCount + 2;
    private const int FactorDimensionCount = 3;
    private const int FactorMaturityDimension = FactorDimensionCount;
    private const int FactorNoCouponDimensionCount = FactorDimensionCount + 1;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double AnnuityCouponStep = 0.12;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;
    private const double DetectorMinimumDistanceYears = 0.20;
    private const int DetectorCandidateLimit = 32;
    private static readonly int[] PieceFactorNoCouponNNodes = [3, 3, 3, 3];
    private static readonly double[][] FullDomain = BuildFullDomain();
    private static readonly double[][] FactorNoCouponDomain =
    [
        [-300.0, 300.0],
        [-300.0, 300.0],
        [-300.0, 300.0],
        [2.0, 30.0],
    ];

    public static MaturitySpecialPointsReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        EnsureDenseFixtureShape(request);

        IReadOnlyList<MaturityBreakpointInventoryPoint> inventory = BuildBreakpointInventory(pricer, request);
        IReadOnlyList<MaturitySpecialPointCandidateSummary> candidates = BuildCandidates(inventory);
        var adapter = new RequestAdapter(pricer, request);
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount);
        IReadOnlyList<SurrogateValidationPoint> clonePoints = BuildCloneValidationPoints(adapter);
        IReadOnlyList<SurrogateValidationPoint> factorPoints = BuildFactorAlignedValidationPoints(adapter, factorBasis);
        IReadOnlyList<AnalyticCouponModelSummary> models = BuildModels(
            adapter,
            factorBasis,
            candidates,
            clonePoints,
            factorPoints);

        return new MaturitySpecialPointsReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            WrapperContract: "curve bumps[60], coupon, maturity -> dirty PV",
            PublicInputDimensionCount: TotalDimensionCount,
            BreakpointInventory: inventory,
            InventorySummary: SummarizeInventory(inventory),
            Candidates: candidates,
            Models: models,
            Decision: BuildDecision(models),
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

    [ExcludeFromCodeCoverage(Justification = "Default fixture shape is covered by RunDefault; this is a defensive guard for edited fixtures.")]
    private static void EnsureDenseFixtureShape(FixedRateBondRequest request)
    {
        if (request.ZeroCurve.Count != CurveBumpDimensionCount + 1)
        {
            throw new InvalidOperationException("Phase 9 expects the dense semiannual curve fixture.");
        }
    }

    private static MaturityBreakpointInventorySummary SummarizeInventory(
        IReadOnlyList<MaturityBreakpointInventoryPoint> inventory)
    {
        MaturityBreakpointInventoryPoint worst = inventory
            .OrderByDescending(point => Math.Abs(point.SecondDifference))
            .First();

        return new MaturityBreakpointInventorySummary(
            PointCount: inventory.Count,
            ScheduleRegimeChangeCount: inventory.Count(point => point.ScheduleRegimeChanged),
            MaxAbsSecondDifference: inventory.Max(point => Math.Abs(point.SecondDifference)),
            MaxAbsSlopeJump: inventory.Max(point => Math.Abs(point.LeftSlopePerYear - point.RightSlopePerYear)),
            WorstMaturityDate: worst.MaturityDate,
            Interpretation:
                "Large one-day slope jumps and second differences identify maturity regions where a single smooth global surrogate is a poor modelling assumption.");
    }

    private static IReadOnlyList<AnalyticCouponModelSummary> BuildModels(
        RequestAdapter adapter,
        CurveFactorBasis factorBasis,
        IReadOnlyList<MaturitySpecialPointCandidateSummary> candidates,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        double[] scheduleBreakpoints = CandidateBreakpoints(candidates, "Schedule-aware special points");
        double[] detectorBreakpoints = CandidateBreakpoints(candidates, "Automatic detector candidates");
        double[] hybridBreakpoints = CandidateBreakpoints(candidates, "Hybrid special points");

        return
        [
            BuildPiecewiseModelSummary(
                "Global decomposed curve-factor tensor",
                "Two dense 4D decomposed factor tensors over level/slope/curvature/maturity with one global maturity piece.",
                adapter,
                factorBasis,
                [],
                clonePoints,
                factorPoints,
                "Control: this is the Phase 8 factor model without maturity splitting."),
            BuildPiecewiseModelSummary(
                "Semiannual uniform bucketed decomposed factor tensor",
                "Two dense 4D decomposed factor tensors routed through uniform 0.5Y maturity buckets.",
                adapter,
                factorBasis,
                UniformSemiannualBreakpoints(),
                clonePoints,
                factorPoints,
                "Control: this tests simple fixed-cadence maturity splitting before special-point routing."),
            BuildPiecewiseModelSummary(
                "Schedule-aware special-point decomposed factor tensor",
                "Two dense 4D decomposed factor tensors routed through schedule-derived maturity special points.",
                adapter,
                factorBasis,
                scheduleBreakpoints,
                clonePoints,
                factorPoints,
                "This tests whether declared schedule-regime evidence is better than a single smooth maturity axis."),
            BuildPiecewiseModelSummary(
                "Automatic-detector special-point decomposed factor tensor",
                "Two dense 4D decomposed factor tensors routed through the largest maturity-axis second-difference candidates.",
                adapter,
                factorBasis,
                detectorBreakpoints,
                clonePoints,
                factorPoints,
                "This tests whether numerical kink detection finds useful split points without schedule metadata."),
            BuildPiecewiseModelSummary(
                "Hybrid special-point decomposed factor tensor",
                "Two dense 4D decomposed factor tensors routed through the union of schedule-derived and detected candidates.",
                adapter,
                factorBasis,
                hybridBreakpoints,
                clonePoints,
                factorPoints,
                "This tests whether schedule metadata and numerical detection are complementary."),
        ];
    }

    private static AnalyticCouponModelSummary BuildPiecewiseModelSummary(
        string modelName,
        string internalMethod,
        RequestAdapter adapter,
        CurveFactorBasis factorBasis,
        IReadOnlyList<double> maturityBreakpoints,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints,
        string interpretation)
    {
        PiecewiseDecomposedFactorTensor model = PiecewiseDecomposedFactorTensor.Build(
            adapter,
            factorBasis,
            maturityBreakpoints);

        return new AnalyticCouponModelSummary(
            ModelName: modelName,
            PublicInputDimensionCount: TotalDimensionCount,
            InternalMethod: internalMethod,
            InternalDimensionCount: FactorNoCouponDimensionCount,
            BucketCount: model.PieceCount,
            BuildEvaluations: model.BuildEvaluations,
            BuildSeconds: model.BuildSeconds,
            Metrics: SummarizeMetrics(adapter.Price, model.Eval, clonePoints),
            FactorAlignedMetrics: SummarizeMetrics(adapter.Price, model.Eval, factorPoints),
            Interpretation: interpretation);
    }

    private static MaturitySpecialPointDecision BuildDecision(IReadOnlyList<AnalyticCouponModelSummary> models)
    {
        AnalyticCouponModelSummary uniform = models.Single(
            model => model.ModelName == "Semiannual uniform bucketed decomposed factor tensor");
        AnalyticCouponModelSummary schedule = models.Single(
            model => model.ModelName == "Schedule-aware special-point decomposed factor tensor");
        AnalyticCouponModelSummary detector = models.Single(
            model => model.ModelName == "Automatic-detector special-point decomposed factor tensor");
        AnalyticCouponModelSummary hybrid = models.Single(
            model => model.ModelName == "Hybrid special-point decomposed factor tensor");

        double uniformMaturity = Metric(uniform, "maturity sensitivity").MaxRelativeError;
        double scheduleMaturity = Metric(schedule, "maturity sensitivity").MaxRelativeError;
        double detectorMaturity = Metric(detector, "maturity sensitivity").MaxRelativeError;
        double hybridMaturity = Metric(hybrid, "maturity sensitivity").MaxRelativeError;
        double uniformCouponMaturity = Metric(uniform, "coupon-maturity mixed").MaxRelativeError;
        double scheduleCouponMaturity = Metric(schedule, "coupon-maturity mixed").MaxRelativeError;

        return new MaturitySpecialPointDecision(
            Recommendation:
                "Prefer schedule-aware maturity special-point routing over the current uniform bucket control for the next modelling iteration.",
            LibraryEnhancementDecision:
                "Open a follow-up library design only for a schedule-aware high-dimensional piecewise router; detector-only splitting needs stronger validation before becoming a ChebyshevSharp API.",
            Evidence:
                $"Schedule-aware routing reduced maturity relative error from {uniformMaturity:P2} to {scheduleMaturity:P2} " +
                $"and coupon-maturity mixed relative error from {uniformCouponMaturity:P2} to {scheduleCouponMaturity:P2}. " +
                $"Detector-only maturity relative error was {detectorMaturity:P2}; hybrid maturity relative error was {hybridMaturity:P2}.");
    }

    private static NaiveSurrogateMetricSummary Metric(AnalyticCouponModelSummary model, string name)
        => model.Metrics.Single(metric => metric.Name == name);

    private static double[] CandidateBreakpoints(
        IReadOnlyList<MaturitySpecialPointCandidateSummary> candidates,
        string name)
        => candidates
            .Single(candidate => candidate.Name == name)
            .MaturityYears
            .Where(years => years > 2.0 + 1e-6 && years < 30.0 - 1e-6)
            .Order()
            .ToArray();

    private static double[] UniformSemiannualBreakpoints()
    {
        var breakpoints = new List<double>();
        for (double years = 2.5; years < 30.0 - 1e-12; years += 0.5)
        {
            breakpoints.Add(years);
        }

        return breakpoints.ToArray();
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
        CashflowInfo? finalCoupon = FinalFutureCoupon(current);
        DateTime finalCashflowDate = current.Cashflows
            .Where(cashflow => !cashflow.HasOccurred)
            .Select(cashflow => cashflow.PaymentDate.Date)
            .DefaultIfEmpty(maturityDate.Date)
            .Max();
        int cashflowCount = current.Cashflows.Count(cashflow => !cashflow.HasOccurred);
        int couponCashflowCount = current.Cashflows.Count(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon);
        string scheduleRegimeReason = BuildScheduleRegimeReason(previous, current, next);
        bool scheduleRegimeChanged = scheduleRegimeReason != "none";

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
            ScheduleRegimeChanged: scheduleRegimeChanged,
            ScheduleRegimeReason: scheduleRegimeReason);
    }

    private static string BuildScheduleRegimeReason(
        FixedRateBondResult previous,
        FixedRateBondResult current,
        FixedRateBondResult next)
    {
        var reasons = new List<string>();

        int currentCashflows = FutureCashflowCount(current);
        if (currentCashflows != FutureCashflowCount(previous) || currentCashflows != FutureCashflowCount(next))
        {
            reasons.Add("cashflow-count");
        }

        int currentCoupons = FutureCouponCount(current);
        if (currentCoupons != FutureCouponCount(previous) || currentCoupons != FutureCouponCount(next))
        {
            reasons.Add("coupon-count");
        }

        return reasons.Count == 0 ? "none" : string.Join("+", reasons);
    }

    private static int FutureCashflowCount(FixedRateBondResult result)
        => result.Cashflows.Count(cashflow => !cashflow.HasOccurred);

    private static int FutureCouponCount(FixedRateBondResult result)
        => result.Cashflows.Count(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon);

    private static CashflowInfo? FinalFutureCoupon(FixedRateBondResult result)
        => result.Cashflows
            .Where(cashflow => !cashflow.HasOccurred && cashflow.IsCoupon)
            .OrderBy(cashflow => cashflow.PaymentDate)
            .LastOrDefault();

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
        double[] detectorMaturities = SelectDetectorMaturities(inventory);
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

    private static double[] SelectDetectorMaturities(IReadOnlyList<MaturityBreakpointInventoryPoint> inventory)
    {
        var selected = new List<double>();
        foreach (MaturityBreakpointInventoryPoint point in inventory.OrderByDescending(point => Math.Abs(point.SecondDifference)))
        {
            if (selected.Any(existing => Math.Abs(existing - point.MaturityYears) < DetectorMinimumDistanceYears))
            {
                continue;
            }

            selected.Add(point.MaturityYears);
            if (selected.Count == DetectorCandidateLimit)
            {
                break;
            }
        }

        return selected.Order().ToArray();
    }

    private static IReadOnlyList<NaiveSurrogateMetricSummary> SummarizeMetrics(
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        var metricFunctions = new (string Name, Func<Func<double[], double>, double[], double> Compute)[]
        {
            ("PV", (f, point) => f(point)),
            ("1Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(12), RateBpStep)),
            ("5Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(60), RateBpStep)),
            ("10Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(120), RateBpStep)),
            ("20Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(240), RateBpStep)),
            ("30Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(360), RateBpStep)),
            ("coupon derivative", (f, point) => FirstDerivative(f, point, CouponDimension, CouponStep)),
            ("maturity sensitivity", (f, point) => FirstDerivative(f, point, MaturityDimension, MaturityYearStep)),
            ("10Y rate-coupon mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(120), RateBpStep, CouponDimension, CouponStep)),
            ("30Y rate-coupon mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(360), RateBpStep, CouponDimension, CouponStep)),
            ("10Y rate-maturity mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(120), RateBpStep, MaturityDimension, MaturityYearStep)),
            ("30Y rate-maturity mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(360), RateBpStep, MaturityDimension, MaturityYearStep)),
            ("20Y-30Y rate-rate mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(240), RateBpStep, CurveDimensionForMonths(360), RateBpStep)),
            ("coupon-maturity mixed", (f, point) => MixedDerivative(f, point, CouponDimension, CouponStep, MaturityDimension, MaturityYearStep)),
        };

        return metricFunctions
            .Select(metric => SummarizeMetric(metric.Name, metric.Compute, baseline, model, validationPoints))
            .ToArray();
    }

    private static NaiveSurrogateMetricSummary SummarizeMetric(
        string name,
        Func<Func<double[], double>, double[], double> compute,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double sumAbs = 0.0;
        double maxAbs = 0.0;
        double sumRel = 0.0;
        double maxRel = 0.0;
        string worstPointName = validationPoints[0].Name;
        double expectedAtWorst = 0.0;
        double actualAtWorst = 0.0;

        foreach (SurrogateValidationPoint point in validationPoints)
        {
            double expected = compute(baseline, point.Coordinates);
            double actual = compute(model, point.Coordinates);
            double abs = Math.Abs(actual - expected);
            double rel = abs / Math.Max(Math.Abs(expected), RelativeErrorFloor);

            sumAbs += abs;
            sumRel += rel;

            if (abs >= maxAbs)
            {
                maxAbs = abs;
                maxRel = Math.Max(maxRel, rel);
                worstPointName = point.Name;
                expectedAtWorst = expected;
                actualAtWorst = actual;
            }
            else
            {
                maxRel = Math.Max(maxRel, rel);
            }
        }

        return new NaiveSurrogateMetricSummary(
            Name: name,
            MeanAbsoluteError: sumAbs / validationPoints.Count,
            MaxAbsoluteError: maxAbs,
            MeanRelativeError: sumRel / validationPoints.Count,
            MaxRelativeError: maxRel,
            WorstPointName: worstPointName,
            ExpectedAtWorstPoint: expectedAtWorst,
            ActualAtWorstPoint: actualAtWorst);
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildCloneValidationPoints(RequestAdapter adapter)
    {
        double week = MaturityYearStep;
        double[][] coordinates =
        [
            FullPoint(0.045, 16.0, _ => 0.0),
            FullPoint(0.08, 10.0, _ => 100.0),
            FullPoint(0.02, 25.0, _ => -100.0),
            FullPoint(0.0, 30.0, _ => 0.0),
            FullPoint(0.12, 2.25, _ => 0.0),
            FullPoint(0.12, 29.5, index => index % 2 == 0 ? 150.0 : -150.0),
            FullPoint(0.005, 3.0, index => index % 2 == 0 ? -150.0 : 150.0),
            FullPoint(0.065, 15.5, index => -120.0 + 240.0 * index / (CurveBumpDimensionCount - 1)),
            FullPoint(0.035, 7.5, index => 120.0 - 240.0 * index / (CurveBumpDimensionCount - 1)),
            FullPoint(0.095, 20.25, index => 75.0 * Math.Sin((index + 1) * Math.PI / 8.0)),
            FullPoint(0.045, 10.0 - week, _ => 0.0),
            FullPoint(0.045, 10.0 + week, _ => 0.0),
        ];

        return BuildValidationPoints("c", adapter, coordinates);
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildFactorAlignedValidationPoints(
        RequestAdapter adapter,
        CurveFactorBasis factorBasis)
    {
        double week = MaturityYearStep;
        double[][] factorCoordinates =
        [
            [0.0, 0.0, 0.0, 16.0],
            [90.0, 0.0, 0.0, 10.0],
            [-90.0, 45.0, 0.0, 25.0],
            [45.0, -60.0, 45.0, 15.5],
            [50.0, -35.0, 25.0, 29.5],
            [0.0, 75.0, -50.0, 10.0 - week],
            [0.0, 75.0, -50.0, 10.0 + week],
        ];
        double[] coupons = [0.045, 0.08, 0.02, 0.065, 0.12, 0.045, 0.045];

        double[][] coordinates = factorCoordinates
            .Select((point, index) => FactorNoCouponToFullPoint(point, factorBasis, coupons[index]))
            .ToArray();
        return BuildValidationPoints("f", adapter, coordinates);
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildValidationPoints(
        string prefix,
        RequestAdapter adapter,
        IReadOnlyList<double[]> coordinates)
    {
        return coordinates
            .Select((point, index) => new SurrogateValidationPoint(
                Name: $"{prefix}{index + 1}",
                Coordinates: point,
                BaselineDirtyPrice: adapter.Price(point)))
            .ToArray();
    }

    private static double[][] BuildFullDomain()
    {
        var domain = new double[TotalDimensionCount][];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            domain[i] = [-150.0, 150.0];
        }

        domain[CouponDimension] = [0.0, 0.12];
        domain[MaturityDimension] = [2.0, 30.0];
        return domain;
    }

    private static double[] FullPoint(double coupon, double maturityYears, Func<int, double> bumpByCurveIndex)
    {
        var point = new double[TotalDimensionCount];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            point[i] = bumpByCurveIndex(i);
        }

        point[CouponDimension] = coupon;
        point[MaturityDimension] = maturityYears;
        return point;
    }

    private static double[] FactorNoCouponToFullPoint(
        double[] factorPoint,
        CurveFactorBasis factorBasis,
        double coupon)
    {
        var fullPoint = new double[TotalDimensionCount];
        double[] reconstructedBumps = factorBasis.Reconstruct(factorPoint);
        Array.Copy(reconstructedBumps, fullPoint, reconstructedBumps.Length);
        fullPoint[CouponDimension] = coupon;
        fullPoint[MaturityDimension] = factorPoint[FactorMaturityDimension];
        return fullPoint;
    }

    private static double[] FullToFactorNoCouponPoint(double[] fullPoint, CurveFactorBasis factorBasis)
    {
        var curveBumps = new double[CurveBumpDimensionCount];
        Array.Copy(fullPoint, curveBumps, CurveBumpDimensionCount);
        double[] factors = factorBasis.Project(curveBumps);
        return [factors[0], factors[1], factors[2], fullPoint[MaturityDimension]];
    }

    private static void ClampPointInPlace(double[] point, double[][] domain)
    {
        for (int i = 0; i < point.Length; i++)
        {
            point[i] = Math.Min(Math.Max(point[i], domain[i][0]), domain[i][1]);
        }
    }

    [ExcludeFromCodeCoverage(Justification = "Private helper is called only with fixed valid semiannual tenors in this benchmark.")]
    private static int CurveDimensionForMonths(int months)
    {
        if (months % 6 != 0 || months < 6 || months > 360)
        {
            throw new ArgumentOutOfRangeException(nameof(months), "Expected a dense semiannual tenor from 6M to 360M.");
        }

        return (months / 6) - 1;
    }

    private static double FirstDerivative(
        Func<double[], double> function,
        double[] point,
        int dimension,
        double step)
    {
        (double downStep, double upStep) = StepsInsideDomain(point[dimension], dimension, step);
        double[] down = Shift(point, dimension, -downStep);
        double[] up = Shift(point, dimension, upStep);
        return (function(up) - function(down)) / (upStep + downStep);
    }

    private static double MixedDerivative(
        Func<double[], double> function,
        double[] point,
        int firstDimension,
        double firstStep,
        int secondDimension,
        double secondStep)
    {
        double[] center = (double[])point.Clone();
        center[firstDimension] = ClampForCentralDifference(center[firstDimension], firstDimension, firstStep);
        center[secondDimension] = ClampForCentralDifference(center[secondDimension], secondDimension, secondStep);

        double[] upUp = Shift(Shift(center, firstDimension, firstStep), secondDimension, secondStep);
        double[] upDown = Shift(Shift(center, firstDimension, firstStep), secondDimension, -secondStep);
        double[] downUp = Shift(Shift(center, firstDimension, -firstStep), secondDimension, secondStep);
        double[] downDown = Shift(Shift(center, firstDimension, -firstStep), secondDimension, -secondStep);

        return (function(upUp) - function(upDown) - function(downUp) + function(downDown))
            / (4.0 * firstStep * secondStep);
    }

    private static (double DownStep, double UpStep) StepsInsideDomain(double value, int dimension, double requestedStep)
    {
        double lower = FullDomain[dimension][0];
        double upper = FullDomain[dimension][1];
        double down = value - requestedStep >= lower ? requestedStep : 0.0;
        double up = value + requestedStep <= upper ? requestedStep : 0.0;

        if (down > 0.0 && up > 0.0)
        {
            return (down, up);
        }

        return down > 0.0 ? (requestedStep, 0.0) : (0.0, requestedStep);
    }

    private static double ClampForCentralDifference(double value, int dimension, double step)
        => Math.Min(Math.Max(value, FullDomain[dimension][0] + step), FullDomain[dimension][1] - step);

    private static double[] Shift(double[] point, int dimension, double shift)
    {
        double[] shifted = (double[])point.Clone();
        shifted[dimension] += shift;
        return shifted;
    }

    private sealed class RequestAdapter
    {
        private readonly IFixedRateBondReferencePricer _pricer;
        private readonly FixedRateBondRequest _baseRequest;

        public RequestAdapter(
            IFixedRateBondReferencePricer pricer,
            FixedRateBondRequest baseRequest)
        {
            _pricer = pricer;
            _baseRequest = baseRequest;
        }

        public double Price(double[] point)
            => _pricer.Price(ToRequest(point, point[CouponDimension])).DirtyPrice;

        public double Principal(double[] point)
            => PriceWithCoupon(point, 0.0);

        public double Annuity(double[] point)
            => (PriceWithCoupon(point, AnnuityCouponStep) - Principal(point)) / AnnuityCouponStep;

        private double PriceWithCoupon(double[] point, double coupon)
            => _pricer.Price(ToRequest(point, coupon)).DirtyPrice;

        private FixedRateBondRequest ToRequest(double[] point, double coupon)
        {
            ZeroRatePillar[] curve = _baseRequest.ZeroCurve.ToArray();
            for (int i = 0; i < CurveBumpDimensionCount; i++)
            {
                int curveIndex = i + 1;
                ZeroRatePillar pillar = curve[curveIndex];
                curve[curveIndex] = pillar with { ZeroRate = pillar.ZeroRate + point[i] * 1e-4 };
            }

            DateTime maturityDate = _baseRequest.ValuationDate.Date.AddDays(
                (int)Math.Round(365.25 * point[MaturityDimension]));

            return _baseRequest with
            {
                Coupon = coupon,
                MaturityDate = maturityDate,
                ZeroCurve = curve,
            };
        }
    }

    private sealed class PiecewiseDecomposedFactorTensor
    {
        private readonly CurveFactorBasis _factorBasis;
        private readonly Piece[] _pieces;

        private PiecewiseDecomposedFactorTensor(CurveFactorBasis factorBasis, Piece[] pieces)
        {
            _factorBasis = factorBasis;
            _pieces = pieces;
        }

        public int PieceCount => _pieces.Length;

        public int BuildEvaluations => _pieces.Sum(piece => piece.Model.BuildEvaluations);

        public double BuildSeconds => _pieces.Sum(piece => piece.Model.BuildSeconds);

        public static PiecewiseDecomposedFactorTensor Build(
            RequestAdapter adapter,
            CurveFactorBasis factorBasis,
            IReadOnlyList<double> maturityBreakpoints)
        {
            double[] edges = BuildEdges(maturityBreakpoints);
            var pieces = new Piece[edges.Length - 1];
            for (int i = 0; i < pieces.Length; i++)
            {
                double lo = edges[i];
                double hi = edges[i + 1];
                double[][] domain =
                [
                    (double[])FactorNoCouponDomain[0].Clone(),
                    (double[])FactorNoCouponDomain[1].Clone(),
                    (double[])FactorNoCouponDomain[2].Clone(),
                    [lo, hi],
                ];
                DecomposedFactorTensor model = DecomposedFactorTensor.Build(
                    adapter,
                    factorBasis,
                    domain,
                    PieceFactorNoCouponNNodes);
                pieces[i] = new Piece(lo, hi, model);
            }

            return new PiecewiseDecomposedFactorTensor(factorBasis, pieces);
        }

        public double Eval(double[] fullPoint)
        {
            double maturity = fullPoint[MaturityDimension];
            Piece piece = _pieces.First(piece =>
                maturity >= piece.Lo && (maturity < piece.Hi || ReferenceEquals(piece, _pieces[^1])));
            return piece.Model.Eval(fullPoint, _factorBasis);
        }

        private static double[] BuildEdges(IReadOnlyList<double> maturityBreakpoints)
        {
            double[] interior = maturityBreakpoints
                .Where(years => years > 2.0 + 1e-6 && years < 30.0 - 1e-6)
                .Select(years => Math.Round(years, 8))
                .Distinct()
                .Order()
                .ToArray();
            return [2.0, .. interior, 30.0];
        }

        private sealed record Piece(double Lo, double Hi, DecomposedFactorTensor Model);
    }

    private sealed class DecomposedFactorTensor
    {
        private readonly ChebyshevApproximation _principal;
        private readonly ChebyshevApproximation _annuity;

        private DecomposedFactorTensor(ChebyshevApproximation principal, ChebyshevApproximation annuity)
        {
            _principal = principal;
            _annuity = annuity;
        }

        public int BuildEvaluations => _principal.NEvaluations + _annuity.NEvaluations;

        public double BuildSeconds => _principal.BuildTime + _annuity.BuildTime;

        public static DecomposedFactorTensor Build(
            RequestAdapter adapter,
            CurveFactorBasis factorBasis,
            double[][] domain,
            int[] nNodes)
        {
            double Principal(double[] factorPoint, object? _)
                => adapter.Principal(FactorNoCouponToFullPoint(factorPoint, factorBasis, coupon: 0.0));

            double Annuity(double[] factorPoint, object? _)
                => adapter.Annuity(FactorNoCouponToFullPoint(factorPoint, factorBasis, coupon: 0.0));

            var principal = new ChebyshevApproximation(
                Principal,
                numDimensions: FactorNoCouponDimensionCount,
                domain: domain,
                nNodes: nNodes);
            var annuity = new ChebyshevApproximation(
                Annuity,
                numDimensions: FactorNoCouponDimensionCount,
                domain: domain,
                nNodes: nNodes);

            Stopwatch sw = Stopwatch.StartNew();
            principal.Build(verbose: false);
            annuity.Build(verbose: false);
            sw.Stop();

            return new DecomposedFactorTensor(principal, annuity);
        }

        public double Eval(double[] fullPoint, CurveFactorBasis factorBasis)
        {
            double[] factorPoint = FullToFactorNoCouponPoint(fullPoint, factorBasis);
            ClampPointInPlace(factorPoint, _principal.Domain);
            return _principal.Eval(factorPoint) + fullPoint[CouponDimension] * _annuity.Eval(factorPoint);
        }
    }

    private sealed class CurveFactorBasis
    {
        private readonly double[][] _basis;
        private readonly double[,] _gramInverse;

        public CurveFactorBasis(int pointCount)
        {
            _basis = BuildBasis(pointCount);
            _gramInverse = Invert3x3(BuildGram(_basis));
        }

        public double[] Project(double[] curveBumps)
        {
            var rhs = new double[FactorDimensionCount];
            for (int factor = 0; factor < FactorDimensionCount; factor++)
            {
                for (int i = 0; i < curveBumps.Length; i++)
                {
                    rhs[factor] += _basis[factor][i] * curveBumps[i];
                }
            }

            return Multiply(_gramInverse, rhs);
        }

        public double[] Reconstruct(double[] factorPoint)
        {
            var curveBumps = new double[_basis[0].Length];
            for (int factor = 0; factor < FactorDimensionCount; factor++)
            {
                for (int i = 0; i < curveBumps.Length; i++)
                {
                    curveBumps[i] += factorPoint[factor] * _basis[factor][i];
                }
            }

            return curveBumps;
        }

        private static double[][] BuildBasis(int pointCount)
        {
            var level = new double[pointCount];
            var slope = new double[pointCount];
            var curvature = new double[pointCount];

            for (int i = 0; i < pointCount; i++)
            {
                double u = (double)i / (pointCount - 1);
                double x = 2.0 * u - 1.0;
                level[i] = 1.0;
                slope[i] = x;
                curvature[i] = 2.0 * x * x - 1.0;
            }

            return [level, slope, curvature];
        }

        private static double[,] BuildGram(double[][] basis)
        {
            var gram = new double[FactorDimensionCount, FactorDimensionCount];
            for (int i = 0; i < FactorDimensionCount; i++)
            {
                for (int j = 0; j < FactorDimensionCount; j++)
                {
                    for (int k = 0; k < basis[i].Length; k++)
                    {
                        gram[i, j] += basis[i][k] * basis[j][k];
                    }
                }
            }

            return gram;
        }

        private static double[] Multiply(double[,] matrix, double[] vector)
        {
            var result = new double[vector.Length];
            for (int row = 0; row < vector.Length; row++)
            {
                for (int col = 0; col < vector.Length; col++)
                {
                    result[row] += matrix[row, col] * vector[col];
                }
            }

            return result;
        }

        [ExcludeFromCodeCoverage(Justification = "Defensive singular-matrix guard is not reachable with the fixed level/slope/curvature basis.")]
        private static double[,] Invert3x3(double[,] matrix)
        {
            double a = matrix[0, 0], b = matrix[0, 1], c = matrix[0, 2];
            double d = matrix[1, 0], e = matrix[1, 1], f = matrix[1, 2];
            double g = matrix[2, 0], h = matrix[2, 1], i = matrix[2, 2];

            double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if (Math.Abs(det) < 1e-14)
            {
                throw new InvalidOperationException("Curve factor basis Gram matrix is singular.");
            }

            return new[,]
            {
                { (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det },
                { (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det },
                { (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det },
            };
        }
    }
}
