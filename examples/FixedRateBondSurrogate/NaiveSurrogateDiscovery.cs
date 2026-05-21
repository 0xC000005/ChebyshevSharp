using System.Diagnostics;
using System.Numerics;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record NaiveSurrogateFeasibility(
    int ConceptualInputBlocks,
    int CurveBumpDimensions,
    int SurrogateDimensionsExcludingNotional,
    int SurrogateDimensionsIncludingNotional,
    string ThreeNodeDenseGridLabel,
    string ThreeNodeDenseGridCount,
    string FiveNodeDenseGridLabel,
    string FiveNodeDenseGridCount,
    string DenseTensorConclusion);

public sealed record NaiveSurrogateMetricSummary(
    string Name,
    double MeanAbsoluteError,
    double MaxAbsoluteError,
    double MeanRelativeError,
    double MaxRelativeError,
    string WorstPointName,
    double ExpectedAtWorstPoint,
    double ActualAtWorstPoint);

public sealed record NaiveSurrogateModelSummary(
    string ModelName,
    int InputDimensionCount,
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<NaiveSurrogateMetricSummary> Metrics);

public sealed record NaiveSurrogateSanityModelValue(
    string ModelName,
    double Value,
    double AbsoluteError);

public sealed record NaiveSurrogateSanityCheck(
    string Name,
    string Category,
    string TestPoint,
    string ExpectedBehavior,
    double BaselineValue,
    double BaselineTolerance,
    IReadOnlyList<NaiveSurrogateSanityModelValue> ModelValues,
    string Interpretation);

public sealed record NaiveMaturityScanPoint(
    DateTime BoundaryDate,
    int OffsetDays,
    DateTime MaturityDate,
    int CashflowCount,
    double DirtyPrice,
    double? CentralSlopePerYear,
    double? SecondDifference,
    double? AbsoluteSecondDifference);

public sealed record NaiveMaturitySpikeCandidate(
    DateTime BoundaryDate,
    int OffsetDays,
    DateTime MaturityDate,
    int CashflowCount,
    double DirtyPrice,
    double LeftSlopePerYear,
    double RightSlopePerYear,
    double SecondDifference,
    double AbsoluteSecondDifference);

public sealed record NaiveSurrogateDiscoveryReport(
    string FixtureId,
    DateTime CurveDate,
    NaiveSurrogateFeasibility Feasibility,
    IReadOnlyList<SurrogateInputDimension> Dimensions,
    IReadOnlyList<int> CurvePillarMonths,
    IReadOnlyList<SurrogateValidationPoint> ValidationPoints,
    IReadOnlyList<NaiveSurrogateModelSummary> Models,
    IReadOnlyList<NaiveSurrogateSanityCheck> SanityChecks,
    IReadOnlyList<NaiveMaturityScanPoint> MaturityScan,
    IReadOnlyList<NaiveMaturitySpikeCandidate> TopMaturitySpikeCandidates);

public static class NaiveSurrogateDiscovery
{
    private const int CurveBumpDimensionCount = 60;
    private const int CouponDimension = CurveBumpDimensionCount;
    private const int MaturityDimension = CurveBumpDimensionCount + 1;
    private const int TotalDimensionCount = CurveBumpDimensionCount + 2;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] NNodes = Enumerable.Repeat(3, TotalDimensionCount).ToArray();
    private static readonly double[][] Domain = BuildDomain();

    public static IReadOnlyList<NaiveMaturityScanPoint> RunMaturityScanDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        return BuildMaturityScan(pricer, baseRequest);
    }

    public static NaiveSurrogateDiscoveryReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        if (fixture.Points.Count != CurveBumpDimensionCount)
        {
            throw new InvalidOperationException(
                $"Expected {CurveBumpDimensionCount} dense curve points, found {fixture.Points.Count}.");
        }

        int[] curvePillarMonths = fixture.Points.Select(point => point.MaturityMonths).ToArray();
        var adapter = new RequestAdapter(pricer, baseRequest);
        IReadOnlyList<SurrogateInputDimension> dimensions = BuildDimensions(fixture);
        IReadOnlyList<SurrogateValidationPoint> validationPoints = BuildValidationPoints(adapter);
        Func<double[], double> fullPv = point => adapter.Price(point);

        BuiltNaiveModel[] builtModels =
        [
            BuildTensorTrainModel(fullPv, validationPoints),
            BuildSliderModel(fullPv, validationPoints),
        ];
        IReadOnlyList<NaiveSurrogateSanityCheck> sanityChecks = BuildSanityChecks(
            fullPv,
            builtModels);
        IReadOnlyList<NaiveMaturityScanPoint> maturityScan = BuildMaturityScan(
            pricer,
            baseRequest);
        IReadOnlyList<NaiveMaturitySpikeCandidate> maturitySpikes = BuildMaturitySpikeCandidates(maturityScan);

        return new NaiveSurrogateDiscoveryReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            Feasibility: BuildFeasibility(fixture),
            Dimensions: dimensions,
            CurvePillarMonths: curvePillarMonths,
            ValidationPoints: validationPoints,
            Models: builtModels.Select(model => model.Summary).ToArray(),
            SanityChecks: sanityChecks,
            MaturityScan: maturityScan,
            TopMaturitySpikeCandidates: maturitySpikes);
    }

    private static double[][] BuildDomain()
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

    private static NaiveSurrogateFeasibility BuildFeasibility(YieldCurveFixture fixture)
    {
        int curveDimensions = fixture.Points.Count;
        int excludingNotional = curveDimensions + 2;
        int includingNotional = excludingNotional + 1;

        return new NaiveSurrogateFeasibility(
            ConceptualInputBlocks: 4,
            CurveBumpDimensions: curveDimensions,
            SurrogateDimensionsExcludingNotional: excludingNotional,
            SurrogateDimensionsIncludingNotional: includingNotional,
            ThreeNodeDenseGridLabel: $"3^{excludingNotional}",
            ThreeNodeDenseGridCount: BigInteger.Pow(new BigInteger(3), excludingNotional).ToString("N0"),
            FiveNodeDenseGridLabel: $"5^{excludingNotional}",
            FiveNodeDenseGridCount: BigInteger.Pow(new BigInteger(5), excludingNotional).ToString("N0"),
            DenseTensorConclusion:
                "The dense tensor is too large to build; this phase uses full-input TT/Slider only as naive probes.");
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
                LowerBound: Domain[i][0],
                UpperBound: Domain[i][1]));
        }

        dimensions.Add(new SurrogateInputDimension(
            "coupon",
            "decimal annual rate",
            Domain[CouponDimension][0],
            Domain[CouponDimension][1]));
        dimensions.Add(new SurrogateInputDimension(
            "maturity",
            "years from valuation date",
            Domain[MaturityDimension][0],
            Domain[MaturityDimension][1]));
        return dimensions;
    }

    private static string FormatTenor(int months)
        => months % 12 == 0 ? $"{months / 12}Y" : $"{months}M";

    private static IReadOnlyList<SurrogateValidationPoint> BuildValidationPoints(RequestAdapter adapter)
    {
        double week = MaturityYearStep;
        double[][] coordinates =
        [
            Point(0.045, 16.0, _ => 0.0),
            Point(0.08, 10.0, _ => 100.0),
            Point(0.02, 25.0, _ => -100.0),
            Point(0.0, 30.0, _ => 0.0),
            Point(0.12, 2.25, _ => 0.0),
            Point(0.12, 29.5, index => index % 2 == 0 ? 150.0 : -150.0),
            Point(0.005, 3.0, index => index % 2 == 0 ? -150.0 : 150.0),
            Point(0.065, 15.5, index => -120.0 + 240.0 * index / (CurveBumpDimensionCount - 1)),
            Point(0.035, 7.5, index => 120.0 - 240.0 * index / (CurveBumpDimensionCount - 1)),
            Point(0.095, 20.25, index => 75.0 * Math.Sin((index + 1) * Math.PI / 8.0)),
            Point(0.045, 10.0 - week, _ => 0.0),
            Point(0.045, 10.0 + week, _ => 0.0),
        ];

        return coordinates
            .Select((point, index) => new SurrogateValidationPoint(
                Name: $"n{index + 1}",
                Coordinates: point,
                BaselineDirtyPrice: adapter.Price(point)))
            .ToArray();
    }

    private static double[] Point(double coupon, double maturityYears, Func<int, double> bumpByCurveIndex)
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

    private static BuiltNaiveModel BuildTensorTrainModel(
        Func<double[], double> fullPv,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        var tt = new ChebyshevTT(
            fullPv,
            numDimensions: Domain.Length,
            domain: Domain,
            nNodes: NNodes,
            maxRank: 6,
            tolerance: 1e-5,
            maxSweeps: 4);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260521, method: "cross");
        sw.Stop();

        var summary = new NaiveSurrogateModelSummary(
            ModelName: "TensorTrain",
            InputDimensionCount: TotalDimensionCount,
            BuildEvaluations: tt.TotalBuildEvals,
            BuildSeconds: sw.Elapsed.TotalSeconds,
            Metrics: SummarizeMetrics(fullPv, tt.Eval, validationPoints));
        return new BuiltNaiveModel(summary, tt.Eval);
    }

    private static BuiltNaiveModel BuildSliderModel(
        Func<double[], double> fullPv,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double[] pivotPoint = Point(0.045, 16.0, _ => 0.0);
        int[][] partition = Enumerable
            .Range(0, TotalDimensionCount)
            .Select(dimension => new[] { dimension })
            .ToArray();
        var slider = new ChebyshevSlider(
            (point, _) => fullPv(point),
            numDimensions: Domain.Length,
            domain: Domain,
            nNodes: NNodes,
            partition: partition,
            pivotPoint: pivotPoint);

        slider.Build(verbose: false);

        double EvalSlider(double[] point) => slider.Eval(point, new int[TotalDimensionCount]);

        var summary = new NaiveSurrogateModelSummary(
            ModelName: "Slider",
            InputDimensionCount: TotalDimensionCount,
            BuildEvaluations: slider.TotalBuildEvals,
            BuildSeconds: slider.BuildTime,
            Metrics: SummarizeMetrics(fullPv, EvalSlider, validationPoints));
        return new BuiltNaiveModel(summary, EvalSlider);
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

    private static int CurveDimensionForMonths(int months)
    {
        if (months % 6 != 0 || months < 6 || months > 360)
        {
            throw new ArgumentOutOfRangeException(nameof(months), "Expected a dense semiannual tenor from 6M to 360M.");
        }

        return (months / 6) - 1;
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
            maxAbs = Math.Max(maxAbs, abs);
            sumRel += rel;
            maxRel = Math.Max(maxRel, rel);

            if (abs >= maxAbs)
            {
                worstPointName = point.Name;
                expectedAtWorst = expected;
                actualAtWorst = actual;
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

    private static IReadOnlyList<NaiveSurrogateSanityCheck> BuildSanityChecks(
        Func<double[], double> baseline,
        IReadOnlyList<BuiltNaiveModel> models)
    {
        double[] tenYearBond = Point(0.045, 10.0, _ => 0.0);

        return
        [
            BuildSanityCheck(
                "10Y bond / 30Y zero-pillar DV01",
                "post-maturity curve support",
                "coupon 4.5%, maturity 10Y, all curve bumps 0",
                "A 30Y zero-rate bump has no interpolation support for a bond maturing at the 10Y pillar.",
                f => FirstDerivative(f, tenYearBond, CurveDimensionForMonths(360), RateBpStep),
                baseline,
                models,
                baselineTolerance: 1e-10,
                interpretation:
                    "The baseline price scan must be flat. A nonzero naive surrogate value means the global approximation leaked long-end curve dependence into an unsupported tenor."),
            BuildSanityCheck(
                "10Y bond / 20Y-30Y rate-rate mixed",
                "post-maturity curve support",
                "coupon 4.5%, maturity 10Y, all curve bumps 0",
                "Two unsupported post-maturity zero-rate nodes should have zero second mixed sensitivity.",
                f => MixedDerivative(
                    f,
                    tenYearBond,
                    CurveDimensionForMonths(240),
                    RateBpStep,
                    CurveDimensionForMonths(360),
                    RateBpStep),
                baseline,
                models,
                baselineTolerance: 1e-10,
                interpretation:
                    "This is a structural support check, not a claim that every rate-rate cross term is zero. Active neighboring pillars can interact through interpolation and discount-factor curvature."),
        ];
    }

    private static NaiveSurrogateSanityCheck BuildSanityCheck(
        string name,
        string category,
        string testPoint,
        string expectedBehavior,
        Func<Func<double[], double>, double> compute,
        Func<double[], double> baseline,
        IReadOnlyList<BuiltNaiveModel> models,
        double baselineTolerance,
        string interpretation)
    {
        double baselineValue = compute(baseline);
        NaiveSurrogateSanityModelValue[] modelValues = models
            .Select(model =>
            {
                double value = compute(model.Eval);
                return new NaiveSurrogateSanityModelValue(
                    model.Summary.ModelName,
                    value,
                    Math.Abs(value - baselineValue));
            })
            .ToArray();

        return new NaiveSurrogateSanityCheck(
            Name: name,
            Category: category,
            TestPoint: testPoint,
            ExpectedBehavior: expectedBehavior,
            BaselineValue: baselineValue,
            BaselineTolerance: baselineTolerance,
            ModelValues: modelValues,
            Interpretation: interpretation);
    }

    private static IReadOnlyList<NaiveMaturityScanPoint> BuildMaturityScan(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        var scan = new List<NaiveMaturityScanPoint>();

        for (int months = 24; months <= 360; months += 6)
        {
            DateTime boundary = request.EffectiveDate.Date.AddMonths(months);
            var window = new List<(DateTime Maturity, int Offset, FixedRateBondResult Result)>();

            for (int offset = -7; offset <= 7; offset++)
            {
                DateTime maturity = boundary.AddDays(offset);
                if (maturity <= request.EffectiveDate.Date ||
                    maturity > request.ZeroCurve[^1].Date.Date)
                {
                    continue;
                }

                window.Add((
                    Maturity: maturity,
                    Offset: offset,
                    Result: pricer.Price(SmoothnessDiagnostics.WithMaturity(request, maturity))));
            }

            for (int i = 0; i < window.Count; i++)
            {
                double? centralSlope = null;
                double? secondDifference = null;
                double? absoluteSecondDifference = null;

                if (i > 0 && i < window.Count - 1)
                {
                    FixedRateBondResult previous = window[i - 1].Result;
                    FixedRateBondResult current = window[i].Result;
                    FixedRateBondResult next = window[i + 1].Result;
                    centralSlope = (next.DirtyPrice - previous.DirtyPrice) * 365.0 / 2.0;
                    secondDifference = next.DirtyPrice - 2.0 * current.DirtyPrice + previous.DirtyPrice;
                    absoluteSecondDifference = Math.Abs(secondDifference.Value);
                }

                scan.Add(new NaiveMaturityScanPoint(
                    BoundaryDate: boundary,
                    OffsetDays: window[i].Offset,
                    MaturityDate: window[i].Maturity,
                    CashflowCount: window[i].Result.Cashflows.Count,
                    DirtyPrice: window[i].Result.DirtyPrice,
                    CentralSlopePerYear: centralSlope,
                    SecondDifference: secondDifference,
                    AbsoluteSecondDifference: absoluteSecondDifference));
            }
        }

        return scan.ToArray();
    }

    private static IReadOnlyList<NaiveMaturitySpikeCandidate> BuildMaturitySpikeCandidates(
        IReadOnlyList<NaiveMaturityScanPoint> scan)
    {
        return scan
            .Where(point => point.SecondDifference.HasValue)
            .Select(point =>
            {
                double leftSlope = double.NaN;
                double rightSlope = double.NaN;
                NaiveMaturityScanPoint? previous = scan.FirstOrDefault(other =>
                    other.BoundaryDate == point.BoundaryDate &&
                    other.OffsetDays == point.OffsetDays - 1);
                NaiveMaturityScanPoint? next = scan.FirstOrDefault(other =>
                    other.BoundaryDate == point.BoundaryDate &&
                    other.OffsetDays == point.OffsetDays + 1);
                if (previous is not null)
                {
                    leftSlope = (point.DirtyPrice - previous.DirtyPrice) * 365.0;
                }

                if (next is not null)
                {
                    rightSlope = (next.DirtyPrice - point.DirtyPrice) * 365.0;
                }

                return new NaiveMaturitySpikeCandidate(
                    BoundaryDate: point.BoundaryDate,
                    OffsetDays: point.OffsetDays,
                    MaturityDate: point.MaturityDate,
                    CashflowCount: point.CashflowCount,
                    DirtyPrice: point.DirtyPrice,
                    LeftSlopePerYear: leftSlope,
                    RightSlopePerYear: rightSlope,
                    SecondDifference: point.SecondDifference!.Value,
                    AbsoluteSecondDifference: point.AbsoluteSecondDifference!.Value);
            })
            .OrderByDescending(candidate => candidate.AbsoluteSecondDifference)
            .Take(8)
            .ToArray();
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
        double lower = Domain[dimension][0];
        double upper = Domain[dimension][1];
        double down = value - requestedStep >= lower ? requestedStep : 0.0;
        double up = value + requestedStep <= upper ? requestedStep : 0.0;

        if (down > 0.0 && up > 0.0)
        {
            return (down, up);
        }

        return down > 0.0 ? (requestedStep, 0.0) : (0.0, requestedStep);
    }

    private static double ClampForCentralDifference(double value, int dimension, double step)
        => Math.Min(Math.Max(value, Domain[dimension][0] + step), Domain[dimension][1] - step);

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

    private sealed record BuiltNaiveModel(
        NaiveSurrogateModelSummary Summary,
        Func<double[], double> Eval);
}
