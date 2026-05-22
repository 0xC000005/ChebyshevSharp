using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record ScheduleAwareRouterPieceSummary(
    int Index,
    double MaturityLo,
    double MaturityHi,
    bool IncludesUpperBound,
    string Source);

public sealed record ScheduleAwareRouterDecision(
    string Recommendation,
    string LibraryEnhancementDecision,
    string Evidence);

public sealed record OneSidedMaturityDiagnostic(
    string Name,
    double BreakpointYears,
    double EpsilonYears,
    double BaselineLeftSlopePerYear,
    double BaselineRightSlopePerYear,
    double RouterLeftSlopePerYear,
    double RouterRightSlopePerYear,
    double LeftSlopeAbsoluteError,
    double RightSlopeAbsoluteError);

public sealed record ScheduleAwareRouterReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    int PublicInputDimensionCount,
    IReadOnlyList<double> Phase9ScheduleCandidateYears,
    IReadOnlyList<double> ScheduleBreakpoints,
    IReadOnlyList<ScheduleAwareRouterPieceSummary> Pieces,
    IReadOnlyList<OneSidedMaturityDiagnostic> OneSidedMaturityDiagnostics,
    IReadOnlyList<AnalyticCouponModelSummary> Models,
    ScheduleAwareRouterDecision Decision,
    string Interpretation);

public sealed class ScheduleAwarePiecewiseRouter
{
    private const int MaturityDimension = 61;
    private readonly ScheduleAwareRouterPieceSummary[] _pieces;
    private readonly Func<double[], int, double> _evalByPiece;

    public ScheduleAwarePiecewiseRouter(
        IReadOnlyList<ScheduleAwareRouterPieceSummary> pieces,
        Func<double[], int, double> evalByPiece,
        int buildEvaluations = 0,
        double buildSeconds = 0.0)
    {
        ArgumentNullException.ThrowIfNull(pieces);
        ArgumentNullException.ThrowIfNull(evalByPiece);
        if (pieces.Count == 0)
        {
            throw new ArgumentException("At least one maturity piece is required.", nameof(pieces));
        }

        _pieces = pieces.ToArray();
        _evalByPiece = evalByPiece;
        BuildEvaluations = buildEvaluations;
        BuildSeconds = buildSeconds;
        ValidatePieces(_pieces);
    }

    public int PieceCount => _pieces.Length;

    public int BuildEvaluations { get; }

    public double BuildSeconds { get; }

    public ScheduleAwareRouterPieceSummary Route(double maturityYears)
    {
        if (!double.IsFinite(maturityYears))
        {
            throw new ArgumentOutOfRangeException(nameof(maturityYears), "Maturity must be finite.");
        }

        foreach (ScheduleAwareRouterPieceSummary piece in _pieces)
        {
            bool inPiece = maturityYears >= piece.MaturityLo &&
                (maturityYears < piece.MaturityHi || (piece.IncludesUpperBound && maturityYears <= piece.MaturityHi));
            if (inPiece)
            {
                return piece;
            }
        }

        throw new ArgumentOutOfRangeException(nameof(maturityYears), "Maturity is outside the router domain.");
    }

    public double Eval(double[] fullPoint)
    {
        ArgumentNullException.ThrowIfNull(fullPoint);
        if (fullPoint.Length <= MaturityDimension)
        {
            throw new ArgumentException("The router expects the full 62-coordinate wrapper point.", nameof(fullPoint));
        }

        ScheduleAwareRouterPieceSummary piece = Route(fullPoint[MaturityDimension]);
        return _evalByPiece(fullPoint, piece.Index);
    }

    private static void ValidatePieces(IReadOnlyList<ScheduleAwareRouterPieceSummary> pieces)
    {
        for (int i = 0; i < pieces.Count; i++)
        {
            ScheduleAwareRouterPieceSummary piece = pieces[i];
            if (piece.Index != i)
            {
                throw new ArgumentException("Piece indices must be contiguous and zero-based.", nameof(pieces));
            }

            if (piece.MaturityHi <= piece.MaturityLo)
            {
                throw new ArgumentException("Each maturity piece must have positive width.", nameof(pieces));
            }

            if (i < pieces.Count - 1)
            {
                ScheduleAwareRouterPieceSummary next = pieces[i + 1];
                if (piece.IncludesUpperBound)
                {
                    throw new ArgumentException("Only the final maturity piece may include its upper bound.", nameof(pieces));
                }

                if (Math.Abs(piece.MaturityHi - next.MaturityLo) > 1e-10)
                {
                    throw new ArgumentException("Maturity pieces must be contiguous.", nameof(pieces));
                }
            }
            else if (!piece.IncludesUpperBound)
            {
                throw new ArgumentException("The final maturity piece must include its upper bound.", nameof(pieces));
            }
        }
    }
}

public static class ScheduleAwareRouterBenchmark
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int PublicInputDimensionCount = CurveBumpDimensionCount + 2;
    private const int FactorDimensionCount = 3;
    private const int FactorMaturityDimension = FactorDimensionCount;
    private const int FactorNoCouponDimensionCount = FactorDimensionCount + 1;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double AnnuityCouponStep = 0.12;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;
    private static readonly int[] PieceFactorNoCouponNNodes = [3, 3, 3, 3];
    private static readonly double[][] FullDomain = BuildFullDomain();
    private static readonly double[][] FactorNoCouponDomain =
    [
        [-300.0, 300.0],
        [-300.0, 300.0],
        [-300.0, 300.0],
        [2.0, 30.0],
    ];

    public static ScheduleAwareRouterReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        EnsureDenseFixtureShape(request);

        MaturitySpecialPointsReport phase9 = MaturitySpecialPointsBenchmark.RunDefault(pricer);
        double[] scheduleCandidates = phase9.Candidates
            .Single(candidate => candidate.Name == "Schedule-aware special points")
            .MaturityYears
            .Order()
            .ToArray();
        double[] scheduleBreakpoints = scheduleCandidates
            .Where(IsInteriorBreakpoint)
            .Order()
            .ToArray();
        var adapter = new RequestAdapter(pricer, request);
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount);
        IReadOnlyList<SurrogateValidationPoint> clonePoints = BuildCloneValidationPoints(adapter);
        IReadOnlyList<SurrogateValidationPoint> factorPoints = BuildFactorAlignedValidationPoints(adapter, factorBasis);
        RoutedDecomposedFactorTensor router = RoutedDecomposedFactorTensor.Build(
            adapter,
            factorBasis,
            BuildPieces(scheduleBreakpoints));
        AnalyticCouponModelSummary routerModel = BuildRouterModelSummary(
            adapter,
            router,
            clonePoints,
            factorPoints);

        return new ScheduleAwareRouterReport(
            FixtureId: phase9.FixtureId,
            CurveDate: phase9.CurveDate,
            WrapperContract: phase9.WrapperContract,
            PublicInputDimensionCount: PublicInputDimensionCount,
            Phase9ScheduleCandidateYears: scheduleCandidates,
            ScheduleBreakpoints: scheduleBreakpoints,
            Pieces: router.Pieces,
            OneSidedMaturityDiagnostics: BuildOneSidedMaturityDiagnostics(adapter.Price, router.Eval, scheduleBreakpoints),
            Models:
            [
                phase9.Models.Single(model => model.ModelName == "Global decomposed curve-factor tensor") with
                {
                    ModelName = "Phase 9 global decomposed factor control"
                },
                phase9.Models.Single(model => model.ModelName == "Semiannual uniform bucketed decomposed factor tensor") with
                {
                    ModelName = "Phase 9 uniform 0.5Y control"
                },
                phase9.Models.Single(model => model.ModelName == "Schedule-aware special-point decomposed factor tensor") with
                {
                    ModelName = "Phase 9 schedule-aware special-point control"
                },
                routerModel,
            ],
            Decision: BuildDecision(phase9, routerModel),
            Interpretation:
                "Phase 10 evaluates schedule-aware routing as a high-dimensional wrapper strategy, not as a generic kink detector.");
    }

    private static AnalyticCouponModelSummary BuildRouterModelSummary(
        RequestAdapter adapter,
        RoutedDecomposedFactorTensor router,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        return new AnalyticCouponModelSummary(
            ModelName: "Schedule-aware router decomposed factor tensor",
            PublicInputDimensionCount: PublicInputDimensionCount,
            InternalMethod:
                "Explicit maturity router over schedule-derived pieces; each piece owns decomposed principal and annuity curve-factor tensors.",
            InternalDimensionCount: FactorNoCouponDimensionCount,
            BucketCount: router.PieceCount,
            BuildEvaluations: router.BuildEvaluations,
            BuildSeconds: router.BuildSeconds,
            Metrics: SummarizeMetrics(adapter.Price, router.Eval, clonePoints),
            FactorAlignedMetrics: SummarizeMetrics(adapter.Price, router.Eval, factorPoints),
            Interpretation:
                "This is the Phase 10 candidate: a real router surface rather than only special-point breakpoints passed to the Phase 9 helper.");
    }

    private static ScheduleAwareRouterDecision BuildDecision(
        MaturitySpecialPointsReport phase9,
        AnalyticCouponModelSummary routerModel)
    {
        AnalyticCouponModelSummary scheduleControl = phase9.Models.Single(
            model => model.ModelName == "Schedule-aware special-point decomposed factor tensor");
        NaiveSurrogateMetricSummary scheduleMaturity = Metric(scheduleControl, "maturity sensitivity");
        NaiveSurrogateMetricSummary routerMaturity = Metric(routerModel, "maturity sensitivity");
        NaiveSurrogateMetricSummary routerCouponMaturity = Metric(routerModel, "coupon-maturity mixed");

        return new ScheduleAwareRouterDecision(
            Recommendation:
                "Keep the router example-local until Phase 10 documentation and PR review confirm the residual error source.",
            LibraryEnhancementDecision:
                "The evidence supports a follow-up design discussion for a schedule-aware piecewise router, not a generic automatic kink detector API yet.",
            Evidence:
                $"Phase 9 schedule-control maturity rel max was {scheduleMaturity.MaxRelativeError:P2}; " +
                $"the explicit router maturity rel max is {routerMaturity.MaxRelativeError:P2}, " +
                $"with coupon-maturity rel max {routerCouponMaturity.MaxRelativeError:P2}.");
    }

    private static NaiveSurrogateMetricSummary Metric(AnalyticCouponModelSummary model, string name)
        => model.Metrics.Single(metric => metric.Name == name);

    [ExcludeFromCodeCoverage(Justification = "Default Phase 9 candidate shape covers the source; this predicate only guards edited candidate lists.")]
    private static bool IsInteriorBreakpoint(double years)
        => years > 2.0 + 1e-6 && years < 30.0 - 1e-6;

    private static IReadOnlyList<ScheduleAwareRouterPieceSummary> BuildPieces(IReadOnlyList<double> breakpoints)
    {
        var pieces = new List<ScheduleAwareRouterPieceSummary>();
        double lo = 2.0;
        for (int i = 0; i < breakpoints.Count; i++)
        {
            pieces.Add(new ScheduleAwareRouterPieceSummary(
                Index: i,
                MaturityLo: lo,
                MaturityHi: breakpoints[i],
                IncludesUpperBound: false,
                Source: "Phase 9 schedule-aware candidate"));
            lo = breakpoints[i];
        }

        pieces.Add(new ScheduleAwareRouterPieceSummary(
            Index: pieces.Count,
            MaturityLo: lo,
            MaturityHi: 30.0,
            IncludesUpperBound: true,
            Source: "terminal maturity domain"));
        return pieces;
    }

    private static IReadOnlyList<OneSidedMaturityDiagnostic> BuildOneSidedMaturityDiagnostics(
        Func<double[], double> baseline,
        Func<double[], double> router,
        IReadOnlyList<double> breakpoints)
    {
        return breakpoints
            .Where(years => years > 2.0 + 2.0 * MaturityYearStep && years < 30.0 - 2.0 * MaturityYearStep)
            .Take(12)
            .Select((years, index) => BuildOneSidedMaturityDiagnostic(
                $"split-{index + 1}",
                years,
                baseline,
                router))
            .ToArray();
    }

    private static OneSidedMaturityDiagnostic BuildOneSidedMaturityDiagnostic(
        string name,
        double breakpointYears,
        Func<double[], double> baseline,
        Func<double[], double> router)
    {
        double baselineLeft = OneSidedLeftSlope(baseline, breakpointYears);
        double baselineRight = OneSidedRightSlope(baseline, breakpointYears);
        double routerLeft = OneSidedLeftSlope(router, breakpointYears);
        double routerRight = OneSidedRightSlope(router, breakpointYears);

        return new OneSidedMaturityDiagnostic(
            Name: name,
            BreakpointYears: breakpointYears,
            EpsilonYears: MaturityYearStep,
            BaselineLeftSlopePerYear: baselineLeft,
            BaselineRightSlopePerYear: baselineRight,
            RouterLeftSlopePerYear: routerLeft,
            RouterRightSlopePerYear: routerRight,
            LeftSlopeAbsoluteError: Math.Abs(routerLeft - baselineLeft),
            RightSlopeAbsoluteError: Math.Abs(routerRight - baselineRight));
    }

    private static double OneSidedLeftSlope(Func<double[], double> function, double breakpointYears)
        => (function(FullPoint(0.045, breakpointYears - MaturityYearStep, _ => 0.0)) -
            function(FullPoint(0.045, breakpointYears - 2.0 * MaturityYearStep, _ => 0.0))) /
            MaturityYearStep;

    private static double OneSidedRightSlope(Func<double[], double> function, double breakpointYears)
        => (function(FullPoint(0.045, breakpointYears + 2.0 * MaturityYearStep, _ => 0.0)) -
            function(FullPoint(0.045, breakpointYears + MaturityYearStep, _ => 0.0))) /
            MaturityYearStep;

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

    private static double[] FullPoint(double coupon, double maturityYears, Func<int, double> bumpByCurveIndex)
    {
        var point = new double[PublicInputDimensionCount];
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
        var fullPoint = new double[PublicInputDimensionCount];
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

    private static double[][] BuildFullDomain()
    {
        var domain = new double[PublicInputDimensionCount][];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            domain[i] = [-150.0, 150.0];
        }

        domain[CouponDimension] = [0.0, 0.12];
        domain[MaturityDimension] = [2.0, 30.0];
        return domain;
    }

    [ExcludeFromCodeCoverage(Justification = "Default fixture shape is covered by RunDefault; this guard exists for edited fixtures.")]
    private static void EnsureDenseFixtureShape(FixedRateBondRequest request)
    {
        if (request.ZeroCurve.Count != CurveBumpDimensionCount + 1)
        {
            throw new InvalidOperationException("Phase 10 expects the dense semiannual curve fixture.");
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

    private sealed class RoutedDecomposedFactorTensor
    {
        private readonly ScheduleAwarePiecewiseRouter _router;

        private RoutedDecomposedFactorTensor(
            IReadOnlyList<ScheduleAwareRouterPieceSummary> pieces,
            DecomposedFactorTensor[] models)
        {
            Pieces = pieces;
            _router = new ScheduleAwarePiecewiseRouter(
                pieces,
                (fullPoint, pieceIndex) => models[pieceIndex].Eval(fullPoint),
                models.Sum(model => model.BuildEvaluations),
                models.Sum(model => model.BuildSeconds));
        }

        public IReadOnlyList<ScheduleAwareRouterPieceSummary> Pieces { get; }

        public int PieceCount => _router.PieceCount;

        public int BuildEvaluations => _router.BuildEvaluations;

        public double BuildSeconds => _router.BuildSeconds;

        public static RoutedDecomposedFactorTensor Build(
            RequestAdapter adapter,
            CurveFactorBasis factorBasis,
            IReadOnlyList<ScheduleAwareRouterPieceSummary> pieces)
        {
            var models = new DecomposedFactorTensor[pieces.Count];
            for (int i = 0; i < pieces.Count; i++)
            {
                ScheduleAwareRouterPieceSummary piece = pieces[i];
                double[][] domain =
                [
                    (double[])FactorNoCouponDomain[0].Clone(),
                    (double[])FactorNoCouponDomain[1].Clone(),
                    (double[])FactorNoCouponDomain[2].Clone(),
                    [piece.MaturityLo, piece.MaturityHi],
                ];

                models[i] = DecomposedFactorTensor.Build(
                    adapter,
                    factorBasis,
                    domain,
                    PieceFactorNoCouponNNodes);
            }

            return new RoutedDecomposedFactorTensor(pieces, models);
        }

        public double Eval(double[] fullPoint)
            => _router.Eval(fullPoint);
    }

    private sealed class DecomposedFactorTensor
    {
        private readonly CurveFactorBasis _factorBasis;
        private readonly ChebyshevApproximation _principal;
        private readonly ChebyshevApproximation _annuity;

        private DecomposedFactorTensor(
            CurveFactorBasis factorBasis,
            ChebyshevApproximation principal,
            ChebyshevApproximation annuity)
        {
            _factorBasis = factorBasis;
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

            return new DecomposedFactorTensor(factorBasis, principal, annuity);
        }

        public double Eval(double[] fullPoint)
        {
            double[] factorPoint = FullToFactorNoCouponPoint(fullPoint, _factorBasis);
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
