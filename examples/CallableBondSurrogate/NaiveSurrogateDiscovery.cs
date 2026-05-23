using System.Diagnostics;
using System.Numerics;
using ChebyshevSharp;

namespace CallableBondSurrogate;

public sealed record CallableNaiveSurrogateFeasibility(
    int CurveBumpDimensions,
    int SurrogateDimensions,
    string ThreeNodeDenseGridLabel,
    string ThreeNodeDenseGridCount,
    string DenseTensorConclusion);

public sealed record CallableNaiveSurrogateMetricSummary(
    string Name,
    double MeanAbsoluteError,
    double MaxAbsoluteError,
    double MeanRelativeError,
    double MaxRelativeError,
    string WorstPointName,
    double ExpectedAtWorstPoint,
    double ActualAtWorstPoint);

public sealed record CallableNaiveSurrogateModelSummary(
    string ModelName,
    int InputDimensionCount,
    int BuildEvaluations,
    double BuildSeconds,
    IReadOnlyList<CallableNaiveSurrogateMetricSummary> Metrics);

public sealed record CallableNaiveSurrogateDiscoveryReport(
    string FixtureId,
    DateTime CurveDate,
    CallableNaiveSurrogateFeasibility Feasibility,
    IReadOnlyList<string> DimensionLabels,
    IReadOnlyList<CallableNaiveSurrogateModelSummary> Models);

public static class CallableNaiveSurrogateDiscovery
{
    private const int CouponDimension = 60;
    private const int MaturityDimension = 61;
    private const int FirstCallDimension = 62;
    private const int CallPriceDimension = 63;
    private const int SigmaDimension = 64;
    private const double RateBpStep = 1.0;
    private const double CouponStep = 1e-4;
    private const double CallPriceStep = 0.01;
    private const double SigmaStep = 1e-4;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly int[] NNodes =
        Enumerable.Repeat(3, CallableBondFullDimensionalWrapper.DimensionCount).ToArray();

    private static readonly double[][] Domain = BuildDomain();

    public static CallableNaiveSurrogateDiscoveryReport RunDefault(ICallableBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        CallableBondFullDimensionalWrapper wrapper = CallableBondFullDimensionalWrapper.CreateDefault(pricer);
        IReadOnlyList<CallableValidationPoint> validationPoints = BuildValidationPoints(wrapper);
        Func<double[], double> fullPv = point => wrapper.Price(point);

        BuiltCallableNaiveModel[] builtModels =
        [
            BuildTensorTrainModel(fullPv, validationPoints),
            BuildSliderModel(fullPv, wrapper.CreateBasePoint(), validationPoints),
        ];

        CallableBondRequest baseRequest = wrapper.ToRequest(wrapper.CreateBasePoint());

        return new CallableNaiveSurrogateDiscoveryReport(
            FixtureId: "fed-nominal-yield-curve-semiannual-2026-05-15",
            CurveDate: baseRequest.ValuationDate,
            Feasibility: BuildFeasibility(),
            DimensionLabels: wrapper.DimensionLabels,
            Models: builtModels.Select(model => model.Summary).ToArray());
    }

    private static double[][] BuildDomain()
    {
        var domain = new double[CallableBondFullDimensionalWrapper.DimensionCount][];
        for (int i = 0; i < CallableBondFullDimensionalWrapper.CurveBumpCount; i++)
        {
            domain[i] = [-150.0, 150.0];
        }

        domain[CouponDimension] = [0.03, 0.09];
        domain[MaturityDimension] = [12.0, 30.0];
        domain[FirstCallDimension] = [3.0, 8.0];
        domain[CallPriceDimension] = [98.0, 104.0];
        domain[SigmaDimension] = [0.003, 0.025];
        return domain;
    }

    private static CallableNaiveSurrogateFeasibility BuildFeasibility()
        => new(
            CurveBumpDimensions: CallableBondFullDimensionalWrapper.CurveBumpCount,
            SurrogateDimensions: CallableBondFullDimensionalWrapper.DimensionCount,
            ThreeNodeDenseGridLabel: "3^65",
            ThreeNodeDenseGridCount: BigInteger.Pow(new BigInteger(3), CallableBondFullDimensionalWrapper.DimensionCount)
                .ToString("N0"),
            DenseTensorConclusion:
                "The dense tensor is too large to build; this phase uses full-input TT/Slider only as naive probes.");

    private static IReadOnlyList<CallableValidationPoint> BuildValidationPoints(
        CallableBondFullDimensionalWrapper wrapper)
    {
        double[][] points =
        [
            Point(coupon: 0.06, maturityYears: 24.0, firstCallYears: 5.0, callPrice: 100.0, sigma: 0.010, _ => 0.0),
            Point(coupon: 0.08, maturityYears: 20.0, firstCallYears: 4.0, callPrice: 102.0, sigma: 0.015, _ => 100.0),
            Point(coupon: 0.04, maturityYears: 25.0, firstCallYears: 7.0, callPrice: 99.0, sigma: 0.006, _ => -100.0),
            Point(coupon: 0.055, maturityYears: 16.0, firstCallYears: 3.5, callPrice: 101.0, sigma: 0.020,
                index => 80.0 * Math.Sin((index + 1) * Math.PI / 10.0)),
            Point(coupon: 0.07, maturityYears: 28.0, firstCallYears: 6.0, callPrice: 103.0, sigma: 0.012,
                index => -120.0 + 240.0 * index / (CallableBondFullDimensionalWrapper.CurveBumpCount - 1)),
        ];

        return points
            .Select((point, index) => new CallableValidationPoint(
                Name: $"c{index + 1}",
                Coordinates: point,
                BaselineDirtyPrice: wrapper.Price(point)))
            .ToArray();
    }

    private static double[] Point(
        double coupon,
        double maturityYears,
        double firstCallYears,
        double callPrice,
        double sigma,
        Func<int, double> bumpByCurveIndex)
    {
        var point = new double[CallableBondFullDimensionalWrapper.DimensionCount];
        for (int i = 0; i < CallableBondFullDimensionalWrapper.CurveBumpCount; i++)
        {
            point[i] = bumpByCurveIndex(i);
        }

        point[CouponDimension] = coupon;
        point[MaturityDimension] = maturityYears;
        point[FirstCallDimension] = firstCallYears;
        point[CallPriceDimension] = callPrice;
        point[SigmaDimension] = sigma;
        return point;
    }

    private static BuiltCallableNaiveModel BuildTensorTrainModel(
        Func<double[], double> fullPv,
        IReadOnlyList<CallableValidationPoint> validationPoints)
    {
        var tt = new ChebyshevTT(
            fullPv,
            numDimensions: Domain.Length,
            domain: Domain,
            nNodes: NNodes,
            maxRank: 4,
            tolerance: 1e-4,
            maxSweeps: 3);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260522, method: "cross");
        sw.Stop();

        var summary = new CallableNaiveSurrogateModelSummary(
            ModelName: "TensorTrain",
            InputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            BuildEvaluations: tt.TotalBuildEvals,
            BuildSeconds: sw.Elapsed.TotalSeconds,
            Metrics: SummarizeMetrics(fullPv, tt.Eval, validationPoints));
        return new BuiltCallableNaiveModel(summary, tt.Eval);
    }

    private static BuiltCallableNaiveModel BuildSliderModel(
        Func<double[], double> fullPv,
        double[] pivotPoint,
        IReadOnlyList<CallableValidationPoint> validationPoints)
    {
        int[][] partition = Enumerable
            .Range(0, CallableBondFullDimensionalWrapper.DimensionCount)
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

        double EvalSlider(double[] point) => slider.Eval(point, new int[CallableBondFullDimensionalWrapper.DimensionCount]);

        var summary = new CallableNaiveSurrogateModelSummary(
            ModelName: "Slider",
            InputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            BuildEvaluations: slider.TotalBuildEvals,
            BuildSeconds: slider.BuildTime,
            Metrics: SummarizeMetrics(fullPv, EvalSlider, validationPoints));
        return new BuiltCallableNaiveModel(summary, EvalSlider);
    }

    private static IReadOnlyList<CallableNaiveSurrogateMetricSummary> SummarizeMetrics(
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableValidationPoint> validationPoints)
    {
        var metricFunctions = new (string Name, Func<Func<double[], double>, double[], double> Compute)[]
        {
            ("PV", (f, point) => f(point)),
            ("10Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(120), RateBpStep)),
            ("30Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(360), RateBpStep)),
            ("coupon derivative", (f, point) => FirstDerivative(f, point, CouponDimension, CouponStep)),
            ("sigma sensitivity", (f, point) => FirstDerivative(f, point, SigmaDimension, SigmaStep)),
            ("call-price sensitivity", (f, point) => FirstDerivative(f, point, CallPriceDimension, CallPriceStep)),
            ("10Y rate-sigma mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(120), RateBpStep, SigmaDimension, SigmaStep)),
            ("call-price-sigma mixed", (f, point) => MixedDerivative(f, point, CallPriceDimension, CallPriceStep, SigmaDimension, SigmaStep)),
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

    private static CallableNaiveSurrogateMetricSummary SummarizeMetric(
        string name,
        Func<Func<double[], double>, double[], double> compute,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableValidationPoint> validationPoints)
    {
        double sumAbs = 0.0;
        double maxAbs = 0.0;
        double sumRel = 0.0;
        double maxRel = 0.0;
        string worstPointName = validationPoints[0].Name;
        double expectedAtWorst = 0.0;
        double actualAtWorst = 0.0;

        foreach (CallableValidationPoint point in validationPoints)
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

        return new CallableNaiveSurrogateMetricSummary(
            Name: name,
            MeanAbsoluteError: sumAbs / validationPoints.Count,
            MaxAbsoluteError: maxAbs,
            MeanRelativeError: sumRel / validationPoints.Count,
            MaxRelativeError: maxRel,
            WorstPointName: worstPointName,
            ExpectedAtWorstPoint: expectedAtWorst,
            ActualAtWorstPoint: actualAtWorst);
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

    private sealed record CallableValidationPoint(
        string Name,
        double[] Coordinates,
        double BaselineDirtyPrice);

    private sealed record BuiltCallableNaiveModel(
        CallableNaiveSurrogateModelSummary Summary,
        Func<double[], double> Eval);
}
