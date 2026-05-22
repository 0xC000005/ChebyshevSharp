using System.Diagnostics;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record AccuracyProjectionPoint(
    string Name,
    string SetName,
    double OriginalDirtyPrice,
    double ReconstructedDirtyPrice,
    double AbsoluteError,
    double RelativeError);

public sealed record AccuracyProjectionBasisSummary(
    string Name,
    int FactorCount,
    double MaxClonePvAbsoluteError,
    double MaxClonePvRelativeError);

public sealed record AccuracyProjectionOracleSummary(
    IReadOnlyList<AccuracyProjectionPoint> Points,
    IReadOnlyList<AccuracyProjectionBasisSummary> AlternativeBases,
    double MaxClonePvAbsoluteError,
    double MaxClonePvRelativeError,
    double MaxFactorAlignedPvAbsoluteError,
    double MaxFactorAlignedPvRelativeError);

public sealed record AccuracyDerivativeStepDiagnostic(
    string Name,
    double Step,
    string StepUnit,
    double Value);

public sealed record AccuracyDerivativeOracleSummary(
    IReadOnlyList<AccuracyDerivativeStepDiagnostic> RateStepDiagnostics,
    IReadOnlyList<AccuracyDerivativeStepDiagnostic> MaturityStepDiagnostics,
    double PostMaturityUnsupportedPillarDv01);

public sealed record AccuracyScheduleDispatchDiagnostic(
    string Name,
    double MaturityYears,
    int PieceIndex,
    double PieceLo,
    double PieceHi);

public sealed record AccuracyScheduleDispatchSummary(
    IReadOnlyList<AccuracyScheduleDispatchDiagnostic> Diagnostics);

public sealed record AccuracyActiveSupportPoint(
    string Name,
    double MaturityYears,
    int ActiveCurveBumpDimensions,
    double OriginalDirtyPrice,
    double TruncatedDirtyPrice,
    double AbsoluteError,
    double RelativeError);

public sealed record AccuracyActiveSupportSummary(
    IReadOnlyList<AccuracyActiveSupportPoint> Points,
    double MaxPvAbsoluteError,
    double MaxPvRelativeError,
    int MinActiveCurveBumpDimensions,
    int MaxActiveCurveBumpDimensions);

public sealed record AccuracyRecipeMetricSummary(
    string Name,
    double MeanAbsoluteError,
    double MaxAbsoluteError,
    double MeanRelativeError,
    double MaxRelativeError);

public sealed record AccuracyRecipeModelSummary(
    string ModelName,
    int PublicInputDimensionCount,
    int InternalDimensionCount,
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<AccuracyRecipeMetricSummary> Metrics,
    string Interpretation);

public sealed record AccuracyRecipeSearchReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    int PublicInputDimensionCount,
    IReadOnlyList<SurrogateValidationPoint> CloneValidationPoints,
    IReadOnlyList<SurrogateValidationPoint> FactorAlignedValidationPoints,
    AccuracyProjectionOracleSummary ProjectionOracle,
    AccuracyDerivativeOracleSummary DerivativeOracle,
    AccuracyScheduleDispatchSummary ScheduleDispatch,
    AccuracyActiveSupportSummary ActiveSupport,
    IReadOnlyList<AccuracyRecipeModelSummary> CandidateModels,
    string Decision);

public static class AccuracyRecipeSearch
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int PublicInputDimensionCount = CurveBumpDimensionCount + 2;
    private const int FactorDimensionCount = 3;
    private const double RelativeErrorFloor = 1e-10;

    public static AccuracyRecipeSearchReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        if (fixture.Points.Count != CurveBumpDimensionCount)
        {
            throw new InvalidOperationException(
                $"Expected {CurveBumpDimensionCount} dense curve points, found {fixture.Points.Count}.");
        }

        var adapter = new RequestAdapter(pricer, request);
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount);
        IReadOnlyList<SurrogateValidationPoint> clonePoints = BuildCloneValidationPoints(adapter);
        IReadOnlyList<SurrogateValidationPoint> factorPoints = BuildFactorAlignedValidationPoints(adapter, factorBasis);

        AccuracyProjectionOracleSummary projection = BuildProjectionOracle(adapter, factorBasis, clonePoints, factorPoints);
        AccuracyDerivativeOracleSummary derivative = BuildDerivativeOracle(adapter, request);
        AccuracyScheduleDispatchSummary schedule = BuildScheduleDispatchSummary();
        AccuracyActiveSupportSummary activeSupport = BuildActiveSupportOracle(adapter, clonePoints);
        AccuracyRecipeModelSummary[] candidateModels =
        [
            BuildActivePillarTt(
                adapter,
                name: "10Y active-pillar TT",
                maturityLo: 9.75,
                maturityHi: 10.25,
                curveNodes: 3,
                couponNodes: 3,
                maturityNodes: 3,
                maxRank: 4,
                tolerance: 1e-4,
                maxSweeps: 3,
                interpretation:
                    "Local TT over active curve pillars, coupon, and maturity for a 10Y window; the public wrapper remains 62D."),
            BuildActivePillarTt(
                adapter,
                name: "10Y narrow active-pillar TT",
                maturityLo: 9.95,
                maturityHi: 10.05,
                curveNodes: 3,
                couponNodes: 5,
                maturityNodes: 7,
                maxRank: 6,
                tolerance: 1e-5,
                maxSweeps: 5,
                interpretation:
                    "Narrower 10Y active-pillar TT with higher coupon/maturity resolution and rank budget."),
        ];

        AccuracyRecipeModelSummary narrow = candidateModels.Single(model => model.ModelName == "10Y narrow active-pillar TT");
        AccuracyRecipeMetricSummary narrowPv = narrow.Metrics.Single(metric => metric.Name == "PV");
        AccuracyRecipeMetricSummary narrowMaturity = narrow.Metrics.Single(metric => metric.Name == "maturity sensitivity");
        string decision =
            "Projection oracle is material, and active support is exact on the validation bank. " +
            $"A narrowed 10Y active-pillar TT reduces local PV max relative error to {narrowPv.MaxRelativeError:P2}, " +
            $"but maturity-sensitivity max relative error remains {narrowMaturity.MaxRelativeError:P2}. " +
            "The next recipe must handle maturity derivatives explicitly before generalizing the router.";

        return new AccuracyRecipeSearchReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            WrapperContract: "curve bumps[60], coupon, maturity -> dirty PV",
            PublicInputDimensionCount: PublicInputDimensionCount,
            CloneValidationPoints: clonePoints,
            FactorAlignedValidationPoints: factorPoints,
            ProjectionOracle: projection,
            DerivativeOracle: derivative,
            ScheduleDispatch: schedule,
            ActiveSupport: activeSupport,
            CandidateModels: candidateModels,
            Decision: decision);
    }

    private static AccuracyProjectionOracleSummary BuildProjectionOracle(
        RequestAdapter adapter,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        AccuracyProjectionPoint[] points = clonePoints
            .Select(point => ProjectionPoint("clone", point, adapter, factorBasis))
            .Concat(factorPoints.Select(point => ProjectionPoint("factor-aligned", point, adapter, factorBasis)))
            .ToArray();

        AccuracyProjectionPoint[] clone = points.Where(point => point.SetName == "clone").ToArray();
        AccuracyProjectionPoint[] factor = points.Where(point => point.SetName == "factor-aligned").ToArray();

        return new AccuracyProjectionOracleSummary(
            Points: points,
            AlternativeBases:
            [
                BuildProjectionBasisSummary("Five-factor deterministic curve basis", 5, adapter, clonePoints),
            ],
            MaxClonePvAbsoluteError: clone.Max(point => point.AbsoluteError),
            MaxClonePvRelativeError: clone.Max(point => point.RelativeError),
            MaxFactorAlignedPvAbsoluteError: factor.Max(point => point.AbsoluteError),
            MaxFactorAlignedPvRelativeError: factor.Max(point => point.RelativeError));
    }

    private static AccuracyProjectionBasisSummary BuildProjectionBasisSummary(
        string name,
        int factorCount,
        RequestAdapter adapter,
        IReadOnlyList<SurrogateValidationPoint> clonePoints)
    {
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount, factorCount);
        AccuracyProjectionPoint[] points = clonePoints
            .Select(point => ProjectionPoint("clone", point, adapter, factorBasis))
            .ToArray();

        return new AccuracyProjectionBasisSummary(
            Name: name,
            FactorCount: factorCount,
            MaxClonePvAbsoluteError: points.Max(point => point.AbsoluteError),
            MaxClonePvRelativeError: points.Max(point => point.RelativeError));
    }

    private static AccuracyProjectionPoint ProjectionPoint(
        string setName,
        SurrogateValidationPoint point,
        RequestAdapter adapter,
        CurveFactorBasis factorBasis)
    {
        double[] factorPoint = ToFactorPoint(point.Coordinates, factorBasis);
        double[] reconstructedPoint = ToFullPoint(factorPoint, factorBasis);
        double reconstructed = adapter.Price(reconstructedPoint);
        double absolute = Math.Abs(reconstructed - point.BaselineDirtyPrice);

        return new AccuracyProjectionPoint(
            Name: point.Name,
            SetName: setName,
            OriginalDirtyPrice: point.BaselineDirtyPrice,
            ReconstructedDirtyPrice: reconstructed,
            AbsoluteError: absolute,
            RelativeError: RelativeError(absolute, point.BaselineDirtyPrice));
    }

    private static AccuracyDerivativeOracleSummary BuildDerivativeOracle(
        RequestAdapter adapter,
        FixedRateBondRequest baseRequest)
    {
        double[] basePoint = FullPoint(coupon: 0.045, maturityYears: 10.0, _ => 0.0);
        int tenYearPillar = CurveDimensionForMonths(120);
        AccuracyDerivativeStepDiagnostic[] rateDiagnostics = new[] { 1e-4, 5e-5, 1e-5 }
            .Select(step => new AccuracyDerivativeStepDiagnostic(
                Name: "10Y pillar DV01 central difference",
                Step: step,
                StepUnit: "bp-coordinate",
                Value: FirstDerivative(adapter.Price, basePoint, tenYearPillar, step)))
            .ToArray();

        AccuracyDerivativeStepDiagnostic[] maturityDiagnostics =
        [
            MaturityStep(adapter.Price, basePoint, days: 1),
            MaturityStep(adapter.Price, basePoint, days: 3),
            MaturityStep(adapter.Price, basePoint, days: 7),
        ];

        double unsupportedDv01 = UnsupportedPostMaturityDv01(adapter, baseRequest);

        return new AccuracyDerivativeOracleSummary(
            RateStepDiagnostics: rateDiagnostics,
            MaturityStepDiagnostics: maturityDiagnostics,
            PostMaturityUnsupportedPillarDv01: unsupportedDv01);
    }

    private static AccuracyDerivativeStepDiagnostic MaturityStep(
        Func<double[], double> price,
        double[] point,
        int days)
    {
        double step = days / 365.25;
        return new AccuracyDerivativeStepDiagnostic(
            Name: $"{days}d maturity central difference",
            Step: step,
            StepUnit: "years",
            Value: FirstDerivative(price, point, MaturityDimension, step));
    }

    private static double UnsupportedPostMaturityDv01(
        RequestAdapter adapter,
        FixedRateBondRequest baseRequest)
    {
        double maturityYears = (baseRequest.ValuationDate.AddYears(10) - baseRequest.ValuationDate).TotalDays / 365.25;
        double[] point = FullPoint(coupon: 0.045, maturityYears, _ => 0.0);
        return FirstDerivative(adapter.Price, point, CurveDimensionForMonths(360), 1e-4);
    }

    private static AccuracyScheduleDispatchSummary BuildScheduleDispatchSummary()
    {
        double[] breakpoints = [2.0, 2.5, 3.0, 3.5, 4.0, 30.0];
        double[] probes = [2.0, 2.5001, 3.25, 29.9];

        return new AccuracyScheduleDispatchSummary(
            Diagnostics: probes
                .Select((maturity, index) =>
                {
                    int pieceIndex = RouteHalfOpen(maturity, breakpoints);
                    return new AccuracyScheduleDispatchDiagnostic(
                        Name: $"dispatch-{index}",
                        MaturityYears: maturity,
                        PieceIndex: pieceIndex,
                        PieceLo: breakpoints[pieceIndex],
                        PieceHi: breakpoints[pieceIndex + 1]);
                })
                .ToArray());
    }

    private static AccuracyActiveSupportSummary BuildActiveSupportOracle(
        RequestAdapter adapter,
        IReadOnlyList<SurrogateValidationPoint> clonePoints)
    {
        AccuracyActiveSupportPoint[] points = clonePoints
            .Select(point =>
            {
                double maturityYears = point.Coordinates[MaturityDimension];
                int activeDimensions = ActiveCurveBumpDimensions(maturityYears);
                double[] truncated = TruncateInactiveCurveBumps(point.Coordinates, activeDimensions);
                double truncatedPrice = adapter.Price(truncated);
                double absolute = Math.Abs(truncatedPrice - point.BaselineDirtyPrice);

                return new AccuracyActiveSupportPoint(
                    Name: point.Name,
                    MaturityYears: maturityYears,
                    ActiveCurveBumpDimensions: activeDimensions,
                    OriginalDirtyPrice: point.BaselineDirtyPrice,
                    TruncatedDirtyPrice: truncatedPrice,
                    AbsoluteError: absolute,
                    RelativeError: RelativeError(absolute, point.BaselineDirtyPrice));
            })
            .ToArray();

        return new AccuracyActiveSupportSummary(
            Points: points,
            MaxPvAbsoluteError: points.Max(point => point.AbsoluteError),
            MaxPvRelativeError: points.Max(point => point.RelativeError),
            MinActiveCurveBumpDimensions: points.Min(point => point.ActiveCurveBumpDimensions),
            MaxActiveCurveBumpDimensions: points.Max(point => point.ActiveCurveBumpDimensions));
    }

    private static int ActiveCurveBumpDimensions(double maturityYears)
        => Math.Min(CurveBumpDimensionCount, (int)Math.Floor(maturityYears * 2.0) + 1);

    private static double[] TruncateInactiveCurveBumps(double[] point, int activeDimensions)
    {
        double[] truncated = (double[])point.Clone();
        for (int i = activeDimensions; i < CurveBumpDimensionCount; i++)
        {
            truncated[i] = 0.0;
        }

        return truncated;
    }

    private static int RouteHalfOpen(double maturity, double[] breakpoints)
    {
        for (int i = 0; i < breakpoints.Length - 1; i++)
        {
            bool last = i == breakpoints.Length - 2;
            if (maturity >= breakpoints[i] && (maturity < breakpoints[i + 1] || (last && maturity <= breakpoints[i + 1])))
            {
                return i;
            }
        }

        throw new ArgumentOutOfRangeException(nameof(maturity), "Maturity is outside the schedule dispatch domain.");
    }

    private static AccuracyRecipeModelSummary BuildActivePillarTt(
        RequestAdapter adapter,
        string name,
        double maturityLo,
        double maturityHi,
        int curveNodes,
        int couponNodes,
        int maturityNodes,
        int maxRank,
        double tolerance,
        int maxSweeps,
        string interpretation)
    {
        int activeCurveDimensions = ActiveCurveBumpDimensions(maturityHi);
        int internalDimensions = activeCurveDimensions + 2;
        double[][] domain = BuildActivePillarDomain(activeCurveDimensions, maturityLo, maturityHi);
        int[] nNodes = BuildActivePillarNodeCounts(activeCurveDimensions, curveNodes, couponNodes, maturityNodes);

        double Price(double[] internalPoint)
            => adapter.Price(ActiveInternalToFullPoint(internalPoint, activeCurveDimensions));

        var tt = new ChebyshevTT(
            Price,
            numDimensions: internalDimensions,
            domain: domain,
            nNodes: nNodes,
            maxRank: maxRank,
            tolerance: tolerance,
            maxSweeps: maxSweeps);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260522, method: "cross");
        sw.Stop();

        double Eval(double[] fullPoint)
            => tt.Eval(FullToActiveInternalPoint(fullPoint, activeCurveDimensions));

        IReadOnlyList<SurrogateValidationPoint> validationPoints =
            BuildTenYearActiveValidationPoints(adapter, activeCurveDimensions);

        return new AccuracyRecipeModelSummary(
            ModelName: name,
            PublicInputDimensionCount: PublicInputDimensionCount,
            InternalDimensionCount: internalDimensions,
            BuildEvaluations: tt.TotalBuildEvals,
            BuildSeconds: sw.Elapsed.TotalSeconds,
            Metrics:
            [
                SummarizeMetric("PV", adapter.Price, Eval, validationPoints),
                SummarizeMetric(
                    "10Y DV01",
                    point => FirstDerivative(adapter.Price, point, CurveDimensionForMonths(120), 1e-4),
                    point => FirstDerivative(Eval, point, CurveDimensionForMonths(120), 1e-4),
                    validationPoints),
                SummarizeMetric(
                    "coupon derivative",
                    point => FirstDerivative(adapter.Price, point, CouponDimension, 1e-4),
                    point => FirstDerivative(Eval, point, CouponDimension, 1e-4),
                    validationPoints),
                SummarizeMetric(
                    "maturity sensitivity",
                    point => FirstDerivative(adapter.Price, point, MaturityDimension, 7.0 / 365.25),
                    point => FirstDerivative(Eval, point, MaturityDimension, 7.0 / 365.25),
                    validationPoints),
                SummarizeMetric(
                    "maturity left sensitivity",
                    point => BackwardDerivative(adapter.Price, point, MaturityDimension, 7.0 / 365.25),
                    point => BackwardDerivative(Eval, point, MaturityDimension, 7.0 / 365.25),
                    validationPoints),
                SummarizeMetric(
                    "maturity right sensitivity",
                    point => ForwardDerivative(adapter.Price, point, MaturityDimension, 7.0 / 365.25),
                    point => ForwardDerivative(Eval, point, MaturityDimension, 7.0 / 365.25),
                    validationPoints),
                SummarizeMetric(
                    "coupon-maturity mixed",
                    point => MixedDerivative(adapter.Price, point, CouponDimension, 1e-4, MaturityDimension, 7.0 / 365.25),
                    point => MixedDerivative(Eval, point, CouponDimension, 1e-4, MaturityDimension, 7.0 / 365.25),
                    validationPoints),
            ],
            Interpretation: interpretation);
    }

    private static int[] BuildActivePillarNodeCounts(
        int activeCurveDimensions,
        int curveNodes,
        int couponNodes,
        int maturityNodes)
    {
        int[] nNodes = Enumerable.Repeat(curveNodes, activeCurveDimensions + 2).ToArray();
        nNodes[activeCurveDimensions] = couponNodes;
        nNodes[activeCurveDimensions + 1] = maturityNodes;
        return nNodes;
    }

    private static double[][] BuildActivePillarDomain(
        int activeCurveDimensions,
        double maturityLo,
        double maturityHi)
    {
        double[][] curveDomain = Enumerable
            .Range(0, activeCurveDimensions)
            .Select(_ => new[] { -150.0, 150.0 })
            .ToArray();
        return curveDomain
            .Append([0.0, 0.12])
            .Append([maturityLo, maturityHi])
            .ToArray();
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildTenYearActiveValidationPoints(
        RequestAdapter adapter,
        int activeCurveDimensions)
    {
        const double week = 7.0 / 365.25;
        double[][] points =
        [
            FullPoint(0.045, 10.0, _ => 0.0),
            FullPoint(0.08, 10.0, index => index < activeCurveDimensions ? 100.0 : 0.0),
            FullPoint(0.02, 10.0, index => index < activeCurveDimensions ? -100.0 : 0.0),
            FullPoint(
                0.065,
                10.0,
                index => index < activeCurveDimensions ? -120.0 + 240.0 * index / (activeCurveDimensions - 1) : 0.0),
            FullPoint(0.045, 10.0 - week, index => index < activeCurveDimensions && index % 2 == 0 ? 100.0 : 0.0),
            FullPoint(0.045, 10.0 + week, index => index < activeCurveDimensions && index % 2 == 1 ? -100.0 : 0.0),
        ];

        return points
            .Select((point, index) => new SurrogateValidationPoint($"active-10y-{index}", point, adapter.Price(point)))
            .ToArray();
    }

    private static AccuracyRecipeMetricSummary SummarizeMetric(
        string name,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double[] absoluteErrors = validationPoints
            .Select(point => Math.Abs(model(point.Coordinates) - baseline(point.Coordinates)))
            .ToArray();
        double[] relativeErrors = validationPoints
            .Zip(absoluteErrors, (point, absolute) => RelativeError(absolute, baseline(point.Coordinates)))
            .ToArray();

        return new AccuracyRecipeMetricSummary(
            Name: name,
            MeanAbsoluteError: absoluteErrors.Average(),
            MaxAbsoluteError: absoluteErrors.Max(),
            MeanRelativeError: relativeErrors.Average(),
            MaxRelativeError: relativeErrors.Max());
    }

    private static double[] ActiveInternalToFullPoint(double[] internalPoint, int activeCurveDimensions)
    {
        var fullPoint = new double[PublicInputDimensionCount];
        Array.Copy(internalPoint, fullPoint, activeCurveDimensions);
        fullPoint[CouponDimension] = internalPoint[activeCurveDimensions];
        fullPoint[MaturityDimension] = internalPoint[activeCurveDimensions + 1];
        return fullPoint;
    }

    private static double[] FullToActiveInternalPoint(double[] fullPoint, int activeCurveDimensions)
    {
        var internalPoint = new double[activeCurveDimensions + 2];
        Array.Copy(fullPoint, internalPoint, activeCurveDimensions);
        internalPoint[activeCurveDimensions] = fullPoint[CouponDimension];
        internalPoint[activeCurveDimensions + 1] = fullPoint[MaturityDimension];
        return internalPoint;
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildCloneValidationPoints(RequestAdapter adapter)
    {
        const double week = 7.0 / 365.25;
        double[][] points =
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

        return points
            .Select((point, index) => new SurrogateValidationPoint($"clone-{index}", point, adapter.Price(point)))
            .ToArray();
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildFactorAlignedValidationPoints(
        RequestAdapter adapter,
        CurveFactorBasis factorBasis)
    {
        double[][] factorPoints =
        [
            [0.0, 0.0, 0.0, 0.045, 16.0],
            [80.0, 0.0, 0.0, 0.075, 10.0],
            [-80.0, 0.0, 0.0, 0.025, 25.0],
            [0.0, 75.0, 0.0, 0.055, 12.5],
            [0.0, -75.0, 0.0, 0.085, 7.5],
            [0.0, 0.0, 75.0, 0.035, 20.0],
            [60.0, -40.0, 35.0, 0.095, 28.0],
        ];

        return factorPoints
            .Select(point => ToFullPoint(point, factorBasis))
            .Select((point, index) => new SurrogateValidationPoint($"factor-{index}", point, adapter.Price(point)))
            .ToArray();
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

    private static double[] ToFactorPoint(double[] fullPoint, CurveFactorBasis factorBasis)
    {
        var curveBumps = new double[CurveBumpDimensionCount];
        Array.Copy(fullPoint, curveBumps, CurveBumpDimensionCount);
        double[] factors = factorBasis.Project(curveBumps);
        var factorPoint = new double[factorBasis.FactorCount + 2];
        Array.Copy(factors, factorPoint, factors.Length);
        factorPoint[factorBasis.FactorCount] = fullPoint[CouponDimension];
        factorPoint[factorBasis.FactorCount + 1] = fullPoint[MaturityDimension];
        return factorPoint;
    }

    private static double[] ToFullPoint(double[] factorPoint, CurveFactorBasis factorBasis)
    {
        var fullPoint = new double[PublicInputDimensionCount];
        double[] reconstructedBumps = factorBasis.Reconstruct(factorPoint);
        Array.Copy(reconstructedBumps, fullPoint, reconstructedBumps.Length);
        fullPoint[CouponDimension] = factorPoint[factorBasis.FactorCount];
        fullPoint[MaturityDimension] = factorPoint[factorBasis.FactorCount + 1];
        return fullPoint;
    }

    private static double FirstDerivative(
        Func<double[], double> function,
        double[] point,
        int dimension,
        double step)
    {
        double[] down = Shift(point, dimension, -step);
        double[] up = Shift(point, dimension, step);
        return (function(up) - function(down)) / (2.0 * step);
    }

    private static double BackwardDerivative(
        Func<double[], double> function,
        double[] point,
        int dimension,
        double step)
    {
        double[] down = Shift(point, dimension, -step);
        return (function(point) - function(down)) / step;
    }

    private static double ForwardDerivative(
        Func<double[], double> function,
        double[] point,
        int dimension,
        double step)
    {
        double[] up = Shift(point, dimension, step);
        return (function(up) - function(point)) / step;
    }

    private static double MixedDerivative(
        Func<double[], double> function,
        double[] point,
        int firstDimension,
        double firstStep,
        int secondDimension,
        double secondStep)
    {
        double[] upUp = Shift(Shift(point, firstDimension, firstStep), secondDimension, secondStep);
        double[] upDown = Shift(Shift(point, firstDimension, firstStep), secondDimension, -secondStep);
        double[] downUp = Shift(Shift(point, firstDimension, -firstStep), secondDimension, secondStep);
        double[] downDown = Shift(Shift(point, firstDimension, -firstStep), secondDimension, -secondStep);

        return (function(upUp) - function(upDown) - function(downUp) + function(downDown))
            / (4.0 * firstStep * secondStep);
    }

    private static double[] Shift(double[] point, int dimension, double shift)
    {
        double[] shifted = (double[])point.Clone();
        shifted[dimension] += shift;
        return shifted;
    }

    private static int CurveDimensionForMonths(int months)
    {
        if (months % 6 != 0 || months < 6 || months > 360)
        {
            throw new ArgumentOutOfRangeException(nameof(months), "Expected a dense semiannual tenor from 6M to 360M.");
        }

        return (months / 6) - 1;
    }

    private static double RelativeError(double absoluteError, double expected)
        => absoluteError / Math.Max(Math.Abs(expected), RelativeErrorFloor);

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
            => _pricer.Price(ToRequest(point)).DirtyPrice;

        private FixedRateBondRequest ToRequest(double[] point)
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
                Coupon = point[CouponDimension],
                MaturityDate = maturityDate,
                ZeroCurve = curve,
            };
        }
    }

    private sealed class CurveFactorBasis
    {
        private readonly double[][] _basis;
        private readonly double[,] _gramInverse;

        public CurveFactorBasis(int pointCount, int factorCount = FactorDimensionCount)
        {
            _basis = BuildBasis(pointCount, factorCount);
            _gramInverse = Invert3x3(BuildGram(_basis));
        }

        public int FactorCount => _basis.Length;

        public double[] Project(double[] curveBumps)
        {
            var rhs = new double[FactorCount];
            for (int factor = 0; factor < FactorCount; factor++)
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
            for (int factor = 0; factor < FactorCount; factor++)
            {
                for (int i = 0; i < curveBumps.Length; i++)
                {
                    curveBumps[i] += factorPoint[factor] * _basis[factor][i];
                }
            }

            return curveBumps;
        }

        private static double[][] BuildBasis(int pointCount, int factorCount)
        {
            if (factorCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(factorCount), "At least one curve factor is required.");
            }

            double[][] basis = Enumerable
                .Range(0, factorCount)
                .Select(_ => new double[pointCount])
                .ToArray();
            for (int i = 0; i < pointCount; i++)
            {
                double u = pointCount == 1 ? 0.0 : (double)i / (pointCount - 1);
                double x = 2.0 * u - 1.0;
                basis[0][i] = 1.0;
                if (factorCount > 1)
                {
                    basis[1][i] = x;
                }

                if (factorCount > 2)
                {
                    basis[2][i] = 2.0 * x * x - 1.0;
                }

                if (factorCount > 3)
                {
                    basis[3][i] = 4.0 * x * x * x - 3.0 * x;
                }

                if (factorCount > 4)
                {
                    basis[4][i] = 8.0 * Math.Pow(x, 4.0) - 8.0 * x * x + 1.0;
                }

                for (int factor = 5; factor < factorCount; factor++)
                {
                    basis[factor][i] = Math.Cos(factor * Math.Acos(Math.Clamp(x, -1.0, 1.0)));
                }
            }

            return basis;
        }

        private static double[,] BuildGram(double[][] basis)
        {
            var gram = new double[basis.Length, basis.Length];
            for (int i = 0; i < basis.Length; i++)
            {
                for (int j = 0; j < basis.Length; j++)
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

        private static double[,] Invert3x3(double[,] matrix)
        {
            int n = matrix.GetLength(0);
            var augmented = new double[n, 2 * n];
            for (int row = 0; row < n; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    augmented[row, col] = matrix[row, col];
                }

                augmented[row, n + row] = 1.0;
            }

            for (int pivot = 0; pivot < n; pivot++)
            {
                int bestRow = pivot;
                for (int row = pivot + 1; row < n; row++)
                {
                    if (Math.Abs(augmented[row, pivot]) > Math.Abs(augmented[bestRow, pivot]))
                    {
                        bestRow = row;
                    }
                }

                if (Math.Abs(augmented[bestRow, pivot]) < 1e-14)
                {
                    throw new InvalidOperationException("Curve factor basis Gram matrix is singular.");
                }

                if (bestRow != pivot)
                {
                    for (int col = 0; col < 2 * n; col++)
                    {
                        (augmented[pivot, col], augmented[bestRow, col]) =
                            (augmented[bestRow, col], augmented[pivot, col]);
                    }
                }

                double scale = augmented[pivot, pivot];
                for (int col = 0; col < 2 * n; col++)
                {
                    augmented[pivot, col] /= scale;
                }

                for (int row = 0; row < n; row++)
                {
                    if (row == pivot)
                    {
                        continue;
                    }

                    double factor = augmented[row, pivot];
                    for (int col = 0; col < 2 * n; col++)
                    {
                        augmented[row, col] -= factor * augmented[pivot, col];
                    }
                }
            }

            var inverse = new double[n, n];
            for (int row = 0; row < n; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    inverse[row, col] = augmented[row, n + col];
                }
            }

            return inverse;
        }
    }
}
