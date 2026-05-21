using System.Diagnostics;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record StructuredAlternativeModelSummary(
    string ModelName,
    int PublicInputDimensionCount,
    string InternalMethod,
    int InternalDimensionCount,
    int BucketCount,
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<NaiveSurrogateMetricSummary> Metrics,
    IReadOnlyList<NaiveSurrogateMetricSummary> FactorAlignedMetrics,
    string Interpretation);

public sealed record StructuredAlternativesReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    string ResearchBasis,
    int PublicInputDimensionCount,
    IReadOnlyList<SurrogateInputDimension> Dimensions,
    IReadOnlyList<SurrogateValidationPoint> CloneValidationPoints,
    IReadOnlyList<SurrogateValidationPoint> FactorAlignedValidationPoints,
    IReadOnlyList<StructuredAlternativeModelSummary> Models);

public static class StructuredAlternativesBenchmark
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int TotalDimensionCount = CurveBumpDimensionCount + 2;
    private const int FactorDimensionCount = 3;
    private const int FactorCouponDimension = FactorDimensionCount;
    private const int FactorMaturityDimension = FactorDimensionCount + 1;
    private const int FactorInputDimensionCount = FactorDimensionCount + 2;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] FullNNodes = Enumerable.Repeat(3, TotalDimensionCount).ToArray();
    private static readonly int[] StrongGlobalTtNNodes = Enumerable.Repeat(4, TotalDimensionCount).ToArray();
    private static readonly int[] FactorNNodes = [3, 3, 3, 5, 5];
    private static readonly int[] BucketFactorNNodes = [3, 3, 3, 4, 4];
    private static readonly double[][] FullDomain = BuildFullDomain();
    private static readonly double[][] FactorDomain =
    [
        [-300.0, 300.0],
        [-300.0, 300.0],
        [-300.0, 300.0],
        [0.0, 0.12],
        [2.0, 30.0],
    ];

    public static StructuredAlternativesReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        if (fixture.Points.Count != CurveBumpDimensionCount)
        {
            throw new InvalidOperationException(
                $"Expected {CurveBumpDimensionCount} dense curve points, found {fixture.Points.Count}.");
        }

        var adapter = new RequestAdapter(pricer, baseRequest);
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount);
        Func<double[], double> baseline = point => adapter.Price(point);
        IReadOnlyList<SurrogateInputDimension> dimensions = BuildDimensions(fixture);
        IReadOnlyList<SurrogateValidationPoint> clonePoints = BuildCloneValidationPoints(adapter);
        IReadOnlyList<SurrogateValidationPoint> factorPoints = BuildFactorAlignedValidationPoints(adapter, factorBasis);

        StructuredAlternativeModelSummary[] models =
        [
            BuildStrongerGlobalTt(baseline, clonePoints, factorPoints),
            BuildAutoOrderedGlobalTt(baseline, clonePoints, factorPoints),
            BuildGroupedSlider(baseline, clonePoints, factorPoints),
            BuildCurveFactorTensor(baseline, factorBasis, clonePoints, factorPoints),
            BuildBucketedCurveFactorTensor(baseline, factorBasis, clonePoints, factorPoints),
            BuildSemiannualBucketedCurveFactorTensor(baseline, factorBasis, clonePoints, factorPoints),
        ];

        return new StructuredAlternativesReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            WrapperContract: "curve bumps[60], coupon, maturity -> dirty PV",
            ResearchBasis:
                "MoCaX-style model-space parameterisation, Chebyshev Slider, Tensor Train compression, " +
                "and Chebfun-style piecewise treatment of nonsmooth coordinates.",
            PublicInputDimensionCount: TotalDimensionCount,
            Dimensions: dimensions,
            CloneValidationPoints: clonePoints,
            FactorAlignedValidationPoints: factorPoints,
            Models: models);
    }

    private static StructuredAlternativeModelSummary BuildStrongerGlobalTt(
        Func<double[], double> baseline,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        var tt = new ChebyshevTT(
            baseline,
            numDimensions: TotalDimensionCount,
            domain: FullDomain,
            nNodes: StrongGlobalTtNNodes,
            maxRank: 8,
            tolerance: 1e-5,
            maxSweeps: 5);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260521, method: "cross");
        sw.Stop();

        return ModelSummary(
            "Stronger global TT",
            "Full 62D TT-Cross with the same wrapper and higher node/rank/sweep budget than Phase 6.",
            TotalDimensionCount,
            bucketCount: 1,
            buildEvaluations: tt.TotalBuildEvals,
            buildSeconds: sw.Elapsed.TotalSeconds,
            baseline,
            tt.Eval,
            clonePoints,
            factorPoints,
            "This tests whether simply trying harder globally fixes the failure before adding structure.");
    }

    private static StructuredAlternativeModelSummary BuildAutoOrderedGlobalTt(
        Func<double[], double> baseline,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        Stopwatch sw = Stopwatch.StartNew();
        ChebyshevTT tt = ChebyshevTT.WithAutoOrder(
            baseline,
            numDimensions: TotalDimensionCount,
            domain: FullDomain,
            numNodes: FullNNodes,
            maxRank: 6,
            tolerance: 1e-5,
            maxSweeps: 3,
            nTrials: 1,
            method: "random",
            seed: 20260521,
            verbose: false);
        sw.Stop();

        return ModelSummary(
            "Auto-ordered global TT",
            $"Full 62D TT-Cross with one random auto-order trial; retained order starts [{string.Join(", ", tt.DimOrder.Take(6))}, ...].",
            TotalDimensionCount,
            bucketCount: 1,
            buildEvaluations: tt.TotalBuildEvals,
            buildSeconds: sw.Elapsed.TotalSeconds,
            baseline,
            tt.Eval,
            clonePoints,
            factorPoints,
            "This measures whether dimension ordering changes rank and validation error without changing the public input.");
    }

    private static StructuredAlternativeModelSummary BuildGroupedSlider(
        Func<double[], double> baseline,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        double[] pivotPoint = FullPoint(0.045, 16.0, _ => 0.0);
        int[] interactionGroup =
        [
            CurveDimensionForMonths(60),
            CurveDimensionForMonths(120),
            CurveDimensionForMonths(240),
            CurveDimensionForMonths(360),
            CouponDimension,
            MaturityDimension,
        ];
        HashSet<int> grouped = interactionGroup.ToHashSet();
        int[][] partition = Enumerable
            .Range(0, TotalDimensionCount)
            .Where(dimension => !grouped.Contains(dimension))
            .Select(dimension => new[] { dimension })
            .Append(interactionGroup)
            .ToArray();

        var slider = new ChebyshevSlider(
            (point, _) => baseline(point),
            numDimensions: TotalDimensionCount,
            domain: FullDomain,
            nNodes: FullNNodes,
            partition: partition,
            pivotPoint: pivotPoint);

        slider.Build(verbose: false);
        double Eval(double[] point) => slider.Eval(point, new int[TotalDimensionCount]);

        return ModelSummary(
            "Grouped Slider",
            "Full 62D Slider with 5Y, 10Y, 20Y, 30Y, coupon, and maturity grouped in one slide; other curve pillars remain singleton slides.",
            TotalDimensionCount,
            bucketCount: 1,
            buildEvaluations: slider.TotalBuildEvals,
            buildSeconds: slider.BuildTime,
            baseline,
            Eval,
            clonePoints,
            factorPoints,
            "This tests whether a practitioner-selected interaction group recovers the reported coupon/maturity and key-rate mixed terms.");
    }

    private static StructuredAlternativeModelSummary BuildCurveFactorTensor(
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        double FactorPrice(double[] factorPoint, object? _)
            => baseline(ToFullPoint(factorPoint, factorBasis));

        var tensor = new ChebyshevApproximation(
            FactorPrice,
            numDimensions: FactorInputDimensionCount,
            domain: FactorDomain,
            nNodes: FactorNNodes);

        tensor.Build(verbose: false);

        double Eval(double[] fullPoint)
        {
            double[] factorPoint = ToFactorPoint(fullPoint, factorBasis);
            ClampFactorPointInPlace(factorPoint, FactorDomain);
            return tensor.Eval(factorPoint);
        }

        return ModelSummary(
            "Curve-factor tensor",
            "Full wrapper projected to level/slope/curvature curve factors plus coupon and maturity; dense 5D Chebyshev tensor inside.",
            FactorInputDimensionCount,
            bucketCount: 1,
            buildEvaluations: tensor.NEvaluations,
            buildSeconds: tensor.BuildTime,
            baseline,
            Eval,
            clonePoints,
            factorPoints,
            "This follows the common model-space/PCA-style compression idea; projection error is visible on arbitrary 60D bump vectors.");
    }

    private static StructuredAlternativeModelSummary BuildBucketedCurveFactorTensor(
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        BucketedFactorTensor bucketed = BucketedFactorTensor.Build(
            baseline,
            factorBasis,
            maturityLo: 2.0,
            maturityHi: 30.0,
            bucketWidth: 1.0);

        return ModelSummary(
            "Bucketed curve-factor tensor",
            "Full wrapper projected to curve factors, then routed to piecewise 1Y maturity buckets; dense 5D Chebyshev tensor per bucket.",
            FactorInputDimensionCount,
            bucketed.BucketCount,
            bucketed.BuildEvaluations,
            bucketed.BuildSeconds,
            baseline,
            bucketed.Eval,
            clonePoints,
            factorPoints,
            "This combines dimensional compression with the Phase 6 evidence that maturity is piecewise schedule-sensitive.");
    }

    private static StructuredAlternativeModelSummary BuildSemiannualBucketedCurveFactorTensor(
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        BucketedFactorTensor bucketed = BucketedFactorTensor.Build(
            baseline,
            factorBasis,
            maturityLo: 2.0,
            maturityHi: 30.0,
            bucketWidth: 0.5);

        return ModelSummary(
            "Semiannual bucketed curve-factor tensor",
            "Full wrapper projected to curve factors, then routed to piecewise 0.5Y maturity buckets aligned with the semiannual schedule cadence.",
            FactorInputDimensionCount,
            bucketed.BucketCount,
            bucketed.BuildEvaluations,
            bucketed.BuildSeconds,
            baseline,
            bucketed.Eval,
            clonePoints,
            factorPoints,
            "This tests whether narrower schedule-cadence buckets are enough before requiring true special-point or edge-detected library support.");
    }

    private static StructuredAlternativeModelSummary ModelSummary(
        string modelName,
        string internalMethod,
        int internalDimensionCount,
        int bucketCount,
        int buildEvaluations,
        double buildSeconds,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints,
        string interpretation)
    {
        return new StructuredAlternativeModelSummary(
            ModelName: modelName,
            PublicInputDimensionCount: TotalDimensionCount,
            InternalMethod: internalMethod,
            InternalDimensionCount: internalDimensionCount,
            BucketCount: bucketCount,
            BuildEvaluations: buildEvaluations,
            BuildSeconds: buildSeconds,
            Metrics: SummarizeMetrics(baseline, model, clonePoints),
            FactorAlignedMetrics: SummarizeMetrics(baseline, model, factorPoints),
            Interpretation: interpretation);
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

    private static IReadOnlyList<SurrogateInputDimension> BuildDimensions(YieldCurveFixture fixture)
    {
        var dimensions = new List<SurrogateInputDimension>();
        for (int i = 0; i < fixture.Points.Count; i++)
        {
            YieldCurvePoint point = fixture.Points[i];
            dimensions.Add(new SurrogateInputDimension(
                Name: $"{FormatTenor(point.MaturityMonths)} zero-rate bump",
                Unit: "basis points",
                LowerBound: FullDomain[i][0],
                UpperBound: FullDomain[i][1]));
        }

        dimensions.Add(new SurrogateInputDimension(
            "coupon",
            "decimal annual rate",
            FullDomain[CouponDimension][0],
            FullDomain[CouponDimension][1]));
        dimensions.Add(new SurrogateInputDimension(
            "maturity",
            "years from valuation date",
            FullDomain[MaturityDimension][0],
            FullDomain[MaturityDimension][1]));
        return dimensions;
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
            [0.0, 0.0, 0.0, 0.045, 16.0],
            [90.0, 0.0, 0.0, 0.08, 10.0],
            [-90.0, 45.0, 0.0, 0.02, 25.0],
            [45.0, -60.0, 45.0, 0.065, 15.5],
            [50.0, -35.0, 25.0, 0.12, 29.5],
            [0.0, 75.0, -50.0, 0.045, 10.0 - week],
            [0.0, 75.0, -50.0, 0.045, 10.0 + week],
        ];

        double[][] coordinates = factorCoordinates
            .Select(point => ToFullPoint(point, factorBasis))
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

    private static double[] ToFullPoint(double[] factorPoint, CurveFactorBasis factorBasis)
    {
        var fullPoint = new double[TotalDimensionCount];
        double[] reconstructedBumps = factorBasis.Reconstruct(factorPoint);
        Array.Copy(reconstructedBumps, fullPoint, reconstructedBumps.Length);
        fullPoint[CouponDimension] = factorPoint[FactorCouponDimension];
        fullPoint[MaturityDimension] = factorPoint[FactorMaturityDimension];
        return fullPoint;
    }

    private static double[] ToFactorPoint(double[] fullPoint, CurveFactorBasis factorBasis)
    {
        var curveBumps = new double[CurveBumpDimensionCount];
        Array.Copy(fullPoint, curveBumps, CurveBumpDimensionCount);
        double[] factors = factorBasis.Project(curveBumps);
        return
        [
            factors[0],
            factors[1],
            factors[2],
            fullPoint[CouponDimension],
            fullPoint[MaturityDimension],
        ];
    }

    private static void ClampFactorPointInPlace(double[] factorPoint, double[][] domain)
    {
        for (int i = 0; i < factorPoint.Length; i++)
        {
            factorPoint[i] = Math.Min(Math.Max(factorPoint[i], domain[i][0]), domain[i][1]);
        }
    }

    private static int CurveDimensionForMonths(int months)
    {
        if (months % 6 != 0 || months < 6 || months > 360)
        {
            throw new ArgumentOutOfRangeException(nameof(months), "Expected a dense semiannual tenor from 6M to 360M.");
        }

        return (months / 6) - 1;
    }

    private static string FormatTenor(int months)
        => months % 12 == 0 ? $"{months / 12}Y" : $"{months}M";

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

    private sealed class BucketedFactorTensor
    {
        private readonly CurveFactorBasis _factorBasis;
        private readonly Bucket[] _buckets;

        private BucketedFactorTensor(CurveFactorBasis factorBasis, Bucket[] buckets)
        {
            _factorBasis = factorBasis;
            _buckets = buckets;
        }

        public int BucketCount => _buckets.Length;

        public int BuildEvaluations => _buckets.Sum(bucket => bucket.Model.NEvaluations);

        public double BuildSeconds => _buckets.Sum(bucket => bucket.Model.BuildTime);

        public static BucketedFactorTensor Build(
            Func<double[], double> baseline,
            CurveFactorBasis factorBasis,
            double maturityLo,
            double maturityHi,
            double bucketWidth)
        {
            var buckets = new List<Bucket>();
            for (double lo = maturityLo; lo < maturityHi - 1e-12; lo += bucketWidth)
            {
                double hi = Math.Min(maturityHi, lo + bucketWidth);
                double[][] domain =
                [
                    (double[])FactorDomain[0].Clone(),
                    (double[])FactorDomain[1].Clone(),
                    (double[])FactorDomain[2].Clone(),
                    (double[])FactorDomain[3].Clone(),
                    [lo, hi],
                ];

                double Price(double[] factorPoint, object? _)
                    => baseline(ToFullPoint(factorPoint, factorBasis));

                var model = new ChebyshevApproximation(
                    Price,
                    numDimensions: FactorInputDimensionCount,
                    domain: domain,
                    nNodes: BucketFactorNNodes);
                model.Build(verbose: false);
                buckets.Add(new Bucket(lo, hi, model));
            }

            return new BucketedFactorTensor(factorBasis, buckets.ToArray());
        }

        public double Eval(double[] fullPoint)
        {
            double maturity = fullPoint[MaturityDimension];
            Bucket bucket = _buckets.FirstOrDefault(candidate =>
                maturity >= candidate.Lo && maturity < candidate.Hi) ?? _buckets[^1];

            double[] factorPoint = ToFactorPoint(fullPoint, _factorBasis);
            double[][] domain = bucket.Model.Domain;
            ClampFactorPointInPlace(factorPoint, domain);
            return bucket.Model.Eval(factorPoint);
        }

        private sealed record Bucket(double Lo, double Hi, ChebyshevApproximation Model);
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
                double u = pointCount == 1 ? 0.0 : (double)i / (pointCount - 1);
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
