using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record AnalyticCouponIdentitySummary(
    int ValidationPointCount,
    double MaxAbsoluteError,
    double MaxRelativeError,
    string Interpretation);

public sealed record AnalyticCouponModelSummary(
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

public sealed record AnalyticCouponDecompositionReport(
    string FixtureId,
    DateTime CurveDate,
    string WrapperContract,
    string Formula,
    int PublicInputDimensionCount,
    AnalyticCouponIdentitySummary Identity,
    IReadOnlyList<SurrogateInputDimension> Dimensions,
    IReadOnlyList<SurrogateValidationPoint> CloneValidationPoints,
    IReadOnlyList<SurrogateValidationPoint> FactorAlignedValidationPoints,
    IReadOnlyList<AnalyticCouponModelSummary> Models);

public static class AnalyticCouponDecompositionBenchmark
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int TotalDimensionCount = CurveBumpDimensionCount + 2;
    private const int NoCouponMaturityDimension = CurveBumpDimensionCount;
    private const int NoCouponDimensionCount = CurveBumpDimensionCount + 1;
    private const int FactorDimensionCount = 3;
    private const int FactorMaturityDimension = FactorDimensionCount;
    private const int FactorNoCouponDimensionCount = FactorDimensionCount + 1;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double AnnuityCouponStep = 0.12;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] NoCouponNNodes = Enumerable.Repeat(4, NoCouponDimensionCount).ToArray();
    private static readonly int[] FactorNoCouponNNodes = [3, 3, 3, 5];
    private static readonly int[] BucketFactorNoCouponNNodes = [3, 3, 3, 4];
    private static readonly double[][] FullDomain = BuildFullDomain();
    private static readonly double[][] NoCouponDomain = BuildNoCouponDomain();
    private static readonly double[][] FactorNoCouponDomain =
    [
        [-300.0, 300.0],
        [-300.0, 300.0],
        [-300.0, 300.0],
        [2.0, 30.0],
    ];

    public static AnalyticCouponDecompositionReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        EnsureDenseFixtureShape(fixture);

        var adapter = new RequestAdapter(pricer, baseRequest);
        var factorBasis = new CurveFactorBasis(CurveBumpDimensionCount);
        Func<double[], double> baseline = adapter.Price;
        Func<double[], double> exactDecomposition = point =>
            adapter.Principal(point) + point[CouponDimension] * adapter.Annuity(point);

        IReadOnlyList<SurrogateInputDimension> dimensions = BuildDimensions(fixture);
        IReadOnlyList<SurrogateValidationPoint> clonePoints = BuildCloneValidationPoints(adapter);
        IReadOnlyList<SurrogateValidationPoint> factorPoints = BuildFactorAlignedValidationPoints(adapter, factorBasis);
        SurrogateValidationPoint[] identityPoints = clonePoints.Concat(factorPoints).ToArray();

        AnalyticCouponModelSummary[] models =
        [
            BuildExactDecompositionOracle(baseline, exactDecomposition, clonePoints, factorPoints),
            BuildGlobalDecomposedTt(adapter, baseline, clonePoints, factorPoints),
            BuildCurveFactorDecomposedTensor(adapter, baseline, factorBasis, clonePoints, factorPoints),
            BuildBucketedCurveFactorDecomposedTensor(adapter, baseline, factorBasis, clonePoints, factorPoints),
            BuildSemiannualBucketedCurveFactorDecomposedTensor(adapter, baseline, factorBasis, clonePoints, factorPoints),
        ];

        return new AnalyticCouponDecompositionReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            WrapperContract: "curve bumps[60], coupon, maturity -> dirty PV",
            Formula: "PV(curve, coupon, T) = PrincipalPV(curve, T) + coupon * AnnuityPV(curve, T)",
            PublicInputDimensionCount: TotalDimensionCount,
            Identity: SummarizeIdentity(baseline, exactDecomposition, identityPoints),
            Dimensions: dimensions,
            CloneValidationPoints: clonePoints,
            FactorAlignedValidationPoints: factorPoints,
            Models: models);
    }

    private static AnalyticCouponModelSummary BuildExactDecompositionOracle(
        Func<double[], double> baseline,
        Func<double[], double> exactDecomposition,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        return ModelSummary(
            "Exact coupon decomposition oracle",
            "Reference-pricer principal and annuity calls recombined analytically through the full wrapper.",
            NoCouponDimensionCount,
            bucketCount: 1,
            buildEvaluations: 0,
            buildSeconds: 0.0,
            baseline,
            exactDecomposition,
            clonePoints,
            factorPoints,
            "This validates the coupon-linearity formula before fitting any Chebyshev object.");
    }

    private static AnalyticCouponModelSummary BuildGlobalDecomposedTt(
        RequestAdapter adapter,
        Func<double[], double> baseline,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        double Principal(double[] noCouponPoint)
            => adapter.Principal(NoCouponToFullPoint(noCouponPoint, coupon: 0.0));

        double Annuity(double[] noCouponPoint)
            => adapter.Annuity(NoCouponToFullPoint(noCouponPoint, coupon: 0.0));

        var principalTt = new ChebyshevTT(
            Principal,
            numDimensions: NoCouponDimensionCount,
            domain: NoCouponDomain,
            nNodes: NoCouponNNodes,
            maxRank: 8,
            tolerance: 1e-5,
            maxSweeps: 5);
        var annuityTt = new ChebyshevTT(
            Annuity,
            numDimensions: NoCouponDimensionCount,
            domain: NoCouponDomain,
            nNodes: NoCouponNNodes,
            maxRank: 8,
            tolerance: 1e-5,
            maxSweeps: 5);

        Stopwatch sw = Stopwatch.StartNew();
        principalTt.Build(verbose: false, seed: 20260521, method: "cross");
        annuityTt.Build(verbose: false, seed: 20260522, method: "cross");
        sw.Stop();

        double Eval(double[] fullPoint)
        {
            double[] noCouponPoint = FullToNoCouponPoint(fullPoint);
            return principalTt.Eval(noCouponPoint) + fullPoint[CouponDimension] * annuityTt.Eval(noCouponPoint);
        }

        return ModelSummary(
            "Global decomposed TT",
            "Two 61D TT-Cross models with n=4 and maxRank=8 over curve bumps[60] and maturity: one for principal, one for annuity.",
            NoCouponDimensionCount,
            bucketCount: 1,
            buildEvaluations: principalTt.TotalBuildEvals + annuityTt.TotalBuildEvals,
            buildSeconds: sw.Elapsed.TotalSeconds,
            baseline,
            Eval,
            clonePoints,
            factorPoints,
            "This tests whether removing coupon alone fixes the full global TT failure.");
    }

    private static AnalyticCouponModelSummary BuildCurveFactorDecomposedTensor(
        RequestAdapter adapter,
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        DecomposedFactorTensor model = DecomposedFactorTensor.Build(
            adapter,
            factorBasis,
            FactorNoCouponDomain,
            FactorNoCouponNNodes);

        return ModelSummary(
            "Curve-factor decomposed tensor",
            "Two dense 4D tensors over level/slope/curvature curve factors and maturity, recombined with coupon analytically.",
            FactorNoCouponDimensionCount,
            bucketCount: 1,
            buildEvaluations: model.BuildEvaluations,
            buildSeconds: model.BuildSeconds,
            baseline,
            model.Eval,
            clonePoints,
            factorPoints,
            "This removes coupon from the Phase 7 factor tensor and tests whether coupon-related Greeks improve.");
    }

    private static AnalyticCouponModelSummary BuildBucketedCurveFactorDecomposedTensor(
        RequestAdapter adapter,
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        BucketedDecomposedFactorTensor model = BucketedDecomposedFactorTensor.Build(
            adapter,
            factorBasis,
            maturityLo: 2.0,
            maturityHi: 30.0,
            bucketWidth: 1.0);

        return ModelSummary(
            "Bucketed decomposed curve-factor tensor",
            "Two dense 4D decomposed factor tensors routed through 1Y maturity buckets.",
            FactorNoCouponDimensionCount,
            model.BucketCount,
            model.BuildEvaluations,
            model.BuildSeconds,
            baseline,
            model.Eval,
            clonePoints,
            factorPoints,
            "This combines analytic coupon recombination with the simple Phase 7 maturity bucket idea.");
    }

    private static AnalyticCouponModelSummary BuildSemiannualBucketedCurveFactorDecomposedTensor(
        RequestAdapter adapter,
        Func<double[], double> baseline,
        CurveFactorBasis factorBasis,
        IReadOnlyList<SurrogateValidationPoint> clonePoints,
        IReadOnlyList<SurrogateValidationPoint> factorPoints)
    {
        BucketedDecomposedFactorTensor model = BucketedDecomposedFactorTensor.Build(
            adapter,
            factorBasis,
            maturityLo: 2.0,
            maturityHi: 30.0,
            bucketWidth: 0.5);

        return ModelSummary(
            "Semiannual bucketed decomposed curve-factor tensor",
            "Two dense 4D decomposed factor tensors routed through 0.5Y maturity buckets.",
            FactorNoCouponDimensionCount,
            model.BucketCount,
            model.BuildEvaluations,
            model.BuildSeconds,
            baseline,
            model.Eval,
            clonePoints,
            factorPoints,
            "This checks whether coupon decomposition plus schedule-cadence buckets is enough before Phase 9 kink detection.");
    }

    private static AnalyticCouponModelSummary ModelSummary(
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
        return new AnalyticCouponModelSummary(
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

    private static AnalyticCouponIdentitySummary SummarizeIdentity(
        Func<double[], double> baseline,
        Func<double[], double> exactDecomposition,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double maxAbs = 0.0;
        double maxRel = 0.0;
        foreach (SurrogateValidationPoint point in validationPoints)
        {
            double expected = baseline(point.Coordinates);
            double actual = exactDecomposition(point.Coordinates);
            double abs = Math.Abs(actual - expected);
            double rel = abs / Math.Max(Math.Abs(expected), RelativeErrorFloor);
            maxAbs = Math.Max(maxAbs, abs);
            maxRel = Math.Max(maxRel, rel);
        }

        return new AnalyticCouponIdentitySummary(
            ValidationPointCount: validationPoints.Count,
            MaxAbsoluteError: maxAbs,
            MaxRelativeError: maxRel,
            Interpretation:
                "The restricted fixed-rate bullet baseline is linear in coupon over the validation bank.");
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

    private static double[][] BuildNoCouponDomain()
    {
        var domain = new double[NoCouponDimensionCount][];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            domain[i] = [-150.0, 150.0];
        }

        domain[NoCouponMaturityDimension] = [2.0, 30.0];
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

    private static double[] FullToNoCouponPoint(double[] fullPoint)
    {
        var point = new double[NoCouponDimensionCount];
        Array.Copy(fullPoint, point, CurveBumpDimensionCount);
        point[NoCouponMaturityDimension] = fullPoint[MaturityDimension];
        return point;
    }

    private static double[] NoCouponToFullPoint(double[] noCouponPoint, double coupon)
    {
        var point = new double[TotalDimensionCount];
        Array.Copy(noCouponPoint, point, CurveBumpDimensionCount);
        point[CouponDimension] = coupon;
        point[MaturityDimension] = noCouponPoint[NoCouponMaturityDimension];
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

    [ExcludeFromCodeCoverage(Justification = "Defensive fixture-shape guard; default fixture shape is covered by RunDefault.")]
    private static void EnsureDenseFixtureShape(YieldCurveFixture fixture)
    {
        if (fixture.Points.Count != CurveBumpDimensionCount)
        {
            throw new InvalidOperationException(
                $"Expected {CurveBumpDimensionCount} dense curve points, found {fixture.Points.Count}.");
        }
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
            principal.Build(verbose: false);
            annuity.Build(verbose: false);
            return new DecomposedFactorTensor(factorBasis, principal, annuity);
        }

        public double Eval(double[] fullPoint)
        {
            double[] factorPoint = FullToFactorNoCouponPoint(fullPoint, _factorBasis);
            ClampPointInPlace(factorPoint, _principal.Domain);
            return _principal.Eval(factorPoint) + fullPoint[CouponDimension] * _annuity.Eval(factorPoint);
        }
    }

    private sealed class BucketedDecomposedFactorTensor
    {
        private readonly CurveFactorBasis _factorBasis;
        private readonly double _maturityLo;
        private readonly double _bucketWidth;
        private readonly Bucket[] _buckets;

        private BucketedDecomposedFactorTensor(
            CurveFactorBasis factorBasis,
            double maturityLo,
            double bucketWidth,
            Bucket[] buckets)
        {
            _factorBasis = factorBasis;
            _maturityLo = maturityLo;
            _bucketWidth = bucketWidth;
            _buckets = buckets;
        }

        public int BucketCount => _buckets.Length;

        public int BuildEvaluations => _buckets.Sum(bucket => bucket.Model.BuildEvaluations);

        public double BuildSeconds => _buckets.Sum(bucket => bucket.Model.BuildSeconds);

        public static BucketedDecomposedFactorTensor Build(
            RequestAdapter adapter,
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
                    (double[])FactorNoCouponDomain[0].Clone(),
                    (double[])FactorNoCouponDomain[1].Clone(),
                    (double[])FactorNoCouponDomain[2].Clone(),
                    [lo, hi],
                ];

                DecomposedFactorTensor model = DecomposedFactorTensor.Build(
                    adapter,
                    factorBasis,
                    domain,
                    BucketFactorNoCouponNNodes);
                buckets.Add(new Bucket(lo, hi, model));
            }

            return new BucketedDecomposedFactorTensor(factorBasis, maturityLo, bucketWidth, buckets.ToArray());
        }

        public double Eval(double[] fullPoint)
        {
            double maturity = fullPoint[MaturityDimension];
            int bucketIndex = (int)Math.Floor((maturity - _maturityLo) / _bucketWidth);
            bucketIndex = Math.Clamp(bucketIndex, 0, _buckets.Length - 1);
            Bucket bucket = _buckets[bucketIndex];
            return bucket.Model.Eval(fullPoint);
        }

        private sealed record Bucket(double Lo, double Hi, DecomposedFactorTensor Model);
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
