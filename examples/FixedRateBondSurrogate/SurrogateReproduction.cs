using System.Diagnostics;
using ChebyshevSharp;

namespace FixedRateBondSurrogate;

public sealed record SurrogateInputDimension(
    string Name,
    string Unit,
    double LowerBound,
    double UpperBound);

public sealed record SurrogateValidationPoint(
    string Name,
    double[] Coordinates,
    double BaselineDirtyPrice);

public sealed record SurrogateMetricSummary(
    string Name,
    double MeanAbsoluteError,
    double MaxAbsoluteError,
    double MeanRelativeError,
    double MaxRelativeError);

public sealed record SurrogateModelSummary(
    string ModelName,
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<SurrogateMetricSummary> Metrics);

public sealed record SurrogateExperimentReport(
    string FixtureId,
    DateTime CurveDate,
    IReadOnlyList<SurrogateInputDimension> Dimensions,
    IReadOnlyList<int> CurvePillarYears,
    IReadOnlyList<SurrogateValidationPoint> ValidationPoints,
    IReadOnlyList<SurrogateModelSummary> Models);

public static class FixedRateBondSurrogateExperiment
{
    private const int CouponDimension = 3;
    private const int MaturityDimension = 4;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double MaturityYearStep = 7.0 / 365.25;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] SelectedCurvePillarYears = [1, 5, 10];
    private static readonly int[] NNodes = [5, 5, 5, 5, 5];
    private static readonly double[][] Domain =
    [
        [-150.0, 150.0],
        [-150.0, 150.0],
        [-150.0, 150.0],
        [0.0, 0.12],
        [8.0, 12.0],
    ];

    public static SurrogateExperimentReport RunDefault(IFixedRateBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDefaultCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularTenYearFromFixture(fixture);
        var adapter = new RequestAdapter(pricer, baseRequest, SelectedCurvePillarYears);
        IReadOnlyList<SurrogateInputDimension> dimensions = BuildDimensions();
        IReadOnlyList<SurrogateValidationPoint> validationPoints = BuildValidationPoints(adapter);

        Func<double[], double> fullPv = point => adapter.Price(point);
        SurrogateModelSummary ttSummary = BuildTensorTrainSummary(fullPv, validationPoints);
        SurrogateModelSummary sliderSummary = BuildSliderSummary(fullPv, validationPoints);

        return new SurrogateExperimentReport(
            FixtureId: fixture.FixtureId,
            CurveDate: fixture.Source.CurveDate.Date,
            Dimensions: dimensions,
            CurvePillarYears: SelectedCurvePillarYears,
            ValidationPoints: validationPoints,
            Models: [ttSummary, sliderSummary]);
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

        dimensions.Add(new SurrogateInputDimension("coupon", "decimal annual rate", Domain[CouponDimension][0], Domain[CouponDimension][1]));
        dimensions.Add(new SurrogateInputDimension("maturity", "years from valuation date", Domain[MaturityDimension][0], Domain[MaturityDimension][1]));
        return dimensions;
    }

    private static IReadOnlyList<SurrogateValidationPoint> BuildValidationPoints(RequestAdapter adapter)
    {
        double[][] coordinates =
        [
            [0.0, 0.0, 0.0, 0.045, 10.0],
            [125.0, -125.0, 75.0, 0.08, 9.25],
            [-125.0, 125.0, -75.0, 0.02, 11.75],
            [0.0, 0.0, 125.0, 0.115, 8.25],
            [75.0, 0.0, -125.0, 0.005, 11.5],
            [150.0, -150.0, 150.0, 0.12, 8.0],
            [-80.0, -40.0, 110.0, 0.065, 8.75],
            [45.0, 95.0, -60.0, 0.035, 10.5],
            [-115.0, 70.0, 20.0, 0.10, 9.75],
        ];

        return coordinates
            .Select((point, index) => new SurrogateValidationPoint(
                Name: $"v{index + 1}",
                Coordinates: point,
                BaselineDirtyPrice: adapter.Price(point)))
            .ToArray();
    }

    private static SurrogateModelSummary BuildTensorTrainSummary(
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

        return new SurrogateModelSummary(
            ModelName: "TensorTrain",
            BuildEvaluations: tt.TotalBuildEvals,
            BuildSeconds: sw.Elapsed.TotalSeconds,
            Metrics: SummarizeMetrics(fullPv, tt.Eval, validationPoints));
    }

    private static SurrogateModelSummary BuildSliderSummary(
        Func<double[], double> fullPv,
        IReadOnlyList<SurrogateValidationPoint> validationPoints)
    {
        double[] pivotPoint = [0.0, 0.0, 0.0, 0.045, 10.0];
        int[][] partition = [[0, 1, 2], [3], [4]];
        var slider = new ChebyshevSlider(
            (point, _) => fullPv(point),
            numDimensions: Domain.Length,
            domain: Domain,
            nNodes: NNodes,
            partition: partition,
            pivotPoint: pivotPoint);

        slider.Build(verbose: false);

        return new SurrogateModelSummary(
            ModelName: "Slider",
            BuildEvaluations: slider.TotalBuildEvals,
            BuildSeconds: slider.BuildTime,
            Metrics: SummarizeMetrics(fullPv, point => slider.Eval(point, [0, 0, 0, 0, 0]), validationPoints));
    }

    private static IReadOnlyList<SurrogateMetricSummary> SummarizeMetrics(
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
            ("coupon derivative", (f, point) => FirstDerivative(f, point, CouponDimension, CouponStep)),
            ("maturity slope", (f, point) => FirstDerivative(f, point, MaturityDimension, MaturityYearStep)),
            ("rate-coupon mixed", (f, point) => MixedDerivative(f, point, 2, RateBpStep, CouponDimension, CouponStep)),
            ("rate-maturity mixed", (f, point) => MixedDerivative(f, point, 2, RateBpStep, MaturityDimension, MaturityYearStep)),
        };

        return metricFunctions
            .Select(metric => SummarizeMetric(metric.Name, metric.Compute, baseline, model, validationPoints))
            .ToArray();
    }

    private static SurrogateMetricSummary SummarizeMetric(
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
        }

        return new SurrogateMetricSummary(
            Name: name,
            MeanAbsoluteError: sumAbs / validationPoints.Count,
            MaxAbsoluteError: maxAbs,
            MeanRelativeError: sumRel / validationPoints.Count,
            MaxRelativeError: maxRel);
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
