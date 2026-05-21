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
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<NaiveSurrogateMetricSummary> Metrics);

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
    IReadOnlyList<int> CurvePillarYears,
    IReadOnlyList<SurrogateValidationPoint> ValidationPoints,
    IReadOnlyList<NaiveSurrogateModelSummary> Models,
    IReadOnlyList<NaiveMaturitySpikeCandidate> TopMaturitySpikeCandidates);

public static class NaiveSurrogateDiscovery
{
    private const int CouponDimension = 5;
    private const int MaturityDimension = 6;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] SelectedCurvePillarYears = [1, 5, 10, 20, 30];
    private static readonly int[] NNodes = [5, 5, 5, 5, 5, 5, 5];
    private static readonly double[][] Domain =
    [
        [-150.0, 150.0],
        [-150.0, 150.0],
        [-150.0, 150.0],
        [-150.0, 150.0],
        [-150.0, 150.0],
        [0.0, 0.12],
        [2.0, 30.0],
    ];

    public static NaiveSurrogateDiscoveryReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        var adapter = new RequestAdapter(pricer, baseRequest, SelectedCurvePillarYears);
        IReadOnlyList<SurrogateInputDimension> dimensions = BuildDimensions();
        IReadOnlyList<SurrogateValidationPoint> validationPoints = BuildValidationPoints(adapter);
        Func<double[], double> fullPv = point => adapter.Price(point);

        NaiveSurrogateModelSummary ttSummary = BuildTensorTrainSummary(fullPv, validationPoints);
        NaiveSurrogateModelSummary sliderSummary = BuildSliderSummary(fullPv, validationPoints);
        IReadOnlyList<NaiveMaturitySpikeCandidate> maturitySpikes = BuildMaturitySpikeCandidates(
            pricer,
            baseRequest);

        return new NaiveSurrogateDiscoveryReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            Feasibility: BuildFeasibility(fixture),
            Dimensions: dimensions,
            CurvePillarYears: SelectedCurvePillarYears,
            ValidationPoints: validationPoints,
            Models: [ttSummary, sliderSummary],
            TopMaturitySpikeCandidates: maturitySpikes);
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
                "The dense tensor is too large to build; this phase uses TT/Slider only as limited naive probes.");
    }

    private static IReadOnlyList<SurrogateInputDimension> BuildDimensions()
    {
        var dimensions = new List<SurrogateInputDimension>();
        for (int i = 0; i < SelectedCurvePillarYears.Length; i++)
        {
            dimensions.Add(new SurrogateInputDimension(
                Name: $"{SelectedCurvePillarYears[i]}Y zero-rate bump",
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

    private static IReadOnlyList<SurrogateValidationPoint> BuildValidationPoints(RequestAdapter adapter)
    {
        double week = MaturityYearStep;
        double[][] coordinates =
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 16.0],
            [125.0, -75.0, 50.0, 25.0, -100.0, 0.08, 10.0],
            [-125.0, 75.0, -50.0, -25.0, 100.0, 0.02, 25.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 2.25],
            [150.0, -150.0, 150.0, -150.0, 150.0, 0.12, 29.5],
            [-150.0, 150.0, -150.0, 150.0, -150.0, 0.005, 3.0],
            [100.0, 50.0, -50.0, 75.0, -100.0, 0.065, 15.5],
            [-40.0, 120.0, 80.0, -90.0, 30.0, 0.035, 7.5],
            [25.0, -35.0, 110.0, -120.0, 80.0, 0.095, 20.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 10.0 - week],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 10.0 + week],
        ];

        return coordinates
            .Select((point, index) => new SurrogateValidationPoint(
                Name: $"n{index + 1}",
                Coordinates: point,
                BaselineDirtyPrice: adapter.Price(point)))
            .ToArray();
    }

    private static NaiveSurrogateModelSummary BuildTensorTrainSummary(
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

        return new NaiveSurrogateModelSummary(
            ModelName: "TensorTrain",
            BuildEvaluations: tt.TotalBuildEvals,
            BuildSeconds: sw.Elapsed.TotalSeconds,
            Metrics: SummarizeMetrics(fullPv, tt.Eval, validationPoints));
    }

    private static NaiveSurrogateModelSummary BuildSliderSummary(
        Func<double[], double> fullPv,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double[] pivotPoint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 16.0];
        int[][] partition = [[0, 1, 2, 3, 4], [5], [6]];
        var slider = new ChebyshevSlider(
            (point, _) => fullPv(point),
            numDimensions: Domain.Length,
            domain: Domain,
            nNodes: NNodes,
            partition: partition,
            pivotPoint: pivotPoint);

        slider.Build(verbose: false);

        return new NaiveSurrogateModelSummary(
            ModelName: "Slider",
            BuildEvaluations: slider.TotalBuildEvals,
            BuildSeconds: slider.BuildTime,
            Metrics: SummarizeMetrics(fullPv, point => slider.Eval(point, [0, 0, 0, 0, 0, 0, 0]), validationPoints));
    }

    private static IReadOnlyList<NaiveSurrogateMetricSummary> SummarizeMetrics(
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        var metricFunctions = new (string Name, Func<Func<double[], double>, double[], double> Compute)[]
        {
            ("PV", (f, point) => f(point)),
            ("1Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, 0, RateBpStep)),
            ("5Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, 1, RateBpStep)),
            ("10Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, 2, RateBpStep)),
            ("20Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, 3, RateBpStep)),
            ("30Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, 4, RateBpStep)),
            ("coupon derivative", (f, point) => FirstDerivative(f, point, CouponDimension, CouponStep)),
            ("maturity slope", (f, point) => FirstDerivative(f, point, MaturityDimension, MaturityYearStep)),
            ("10Y rate-coupon mixed", (f, point) => MixedDerivative(f, point, 2, RateBpStep, CouponDimension, CouponStep)),
            ("30Y rate-coupon mixed", (f, point) => MixedDerivative(f, point, 4, RateBpStep, CouponDimension, CouponStep)),
            ("10Y rate-maturity mixed", (f, point) => MixedDerivative(f, point, 2, RateBpStep, MaturityDimension, MaturityYearStep)),
            ("30Y rate-maturity mixed", (f, point) => MixedDerivative(f, point, 4, RateBpStep, MaturityDimension, MaturityYearStep)),
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

    private static IReadOnlyList<NaiveMaturitySpikeCandidate> BuildMaturitySpikeCandidates(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest request)
    {
        var candidates = new List<NaiveMaturitySpikeCandidate>();

        for (int months = 24; months <= 360; months += 6)
        {
            DateTime boundary = request.EffectiveDate.Date.AddMonths(months);
            for (int offset = -2; offset <= 2; offset++)
            {
                DateTime maturity = boundary.AddDays(offset);
                if (maturity <= request.EffectiveDate.Date ||
                    maturity.AddDays(1) > request.ZeroCurve[^1].Date.Date)
                {
                    continue;
                }

                FixedRateBondResult previous = pricer.Price(SmoothnessDiagnostics.WithMaturity(request, maturity.AddDays(-1)));
                FixedRateBondResult current = pricer.Price(SmoothnessDiagnostics.WithMaturity(request, maturity));
                FixedRateBondResult next = pricer.Price(SmoothnessDiagnostics.WithMaturity(request, maturity.AddDays(1)));
                double leftSlope = (current.DirtyPrice - previous.DirtyPrice) * 365.0;
                double rightSlope = (next.DirtyPrice - current.DirtyPrice) * 365.0;
                double secondDifference = next.DirtyPrice - 2.0 * current.DirtyPrice + previous.DirtyPrice;

                candidates.Add(new NaiveMaturitySpikeCandidate(
                    BoundaryDate: boundary,
                    OffsetDays: offset,
                    MaturityDate: maturity,
                    CashflowCount: current.Cashflows.Count,
                    DirtyPrice: current.DirtyPrice,
                    LeftSlopePerYear: leftSlope,
                    RightSlopePerYear: rightSlope,
                    SecondDifference: secondDifference,
                    AbsoluteSecondDifference: Math.Abs(secondDifference)));
            }
        }

        return candidates
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
        private readonly int[] _curvePillarIndices;

        public RequestAdapter(
            IFixedRateBondReferencePricer pricer,
            FixedRateBondRequest baseRequest,
            IReadOnlyList<int> curvePillarYears)
        {
            _pricer = pricer;
            _baseRequest = baseRequest;
            _curvePillarIndices = curvePillarYears
                .Select(years => FindPillarIndex(baseRequest, years))
                .ToArray();
        }

        public double Price(double[] point)
            => _pricer.Price(ToRequest(point)).DirtyPrice;

        private FixedRateBondRequest ToRequest(double[] point)
        {
            ZeroRatePillar[] curve = _baseRequest.ZeroCurve.ToArray();
            for (int i = 0; i < _curvePillarIndices.Length; i++)
            {
                int curveIndex = _curvePillarIndices[i];
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

        private static int FindPillarIndex(FixedRateBondRequest request, int pillarYears)
        {
            DateTime expected = request.ValuationDate.Date.AddYears(pillarYears);
            return Enumerable
                .Range(0, request.ZeroCurve.Count)
                .First(index => request.ZeroCurve[index].Date.Date == expected);
        }
    }
}
