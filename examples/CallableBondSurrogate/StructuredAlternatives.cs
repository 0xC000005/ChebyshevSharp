using System.Diagnostics;
using ChebyshevSharp;
using FixedRateBondSurrogate;

namespace CallableBondSurrogate;

public sealed record CallableStructuredAlternativeModelSummary(
    string ModelName,
    string ApproximationType,
    int PublicInputDimensionCount,
    int InternalDimensionCount,
    int BuildEvaluations,
    double BuildSeconds,
    double BaselineEvalMicroseconds,
    double SurrogateEvalMicroseconds,
    double BreakEvenEvaluations,
    IReadOnlyList<CallableNaiveSurrogateMetricSummary> FactorAlignedMetrics,
    IReadOnlyList<CallableNaiveSurrogateMetricSummary> ArbitraryBumpMetrics);

public sealed record CallableStructuredAlternativesReport(
    string FixtureId,
    DateTime CurveDate,
    IReadOnlyList<CallableStructuredAlternativeModelSummary> Models);

public static class CallableStructuredAlternatives
{
    private const int PublicCurveBumpCount = 60;
    private const int InternalDimensionCount = 8;
    private const int CouponDimension = 60;
    private const int MaturityDimension = 61;
    private const int FirstCallDimension = 62;
    private const int CallPriceDimension = 63;
    private const int SigmaDimension = 64;
    private const double RateBpStep = 1.0;
    private const double SigmaStep = 1e-4;
    private const double RelativeErrorFloor = 1e-10;

    private static readonly double[][] InternalDomain =
    [
        [-100.0, 100.0], // level
        [-100.0, 100.0], // slope
        [-100.0, 100.0], // curvature
        [0.03, 0.09],
        [12.0, 30.0],
        [3.0, 8.0],
        [98.0, 104.0],
        [0.003, 0.025],
    ];

    private static readonly int[] NNodes = Enumerable.Repeat(3, InternalDimensionCount).ToArray();
    private static readonly double[][] PublicDomain = BuildPublicDomain();
    private static readonly int[] PublicNNodes =
        Enumerable.Repeat(3, CallableBondFullDimensionalWrapper.DimensionCount).ToArray();
    private static readonly double[] TenorCoordinates = BuildTenorCoordinates();
    private static readonly double[][] CurveBasis = BuildCurveBasis();

    public static CallableStructuredAlternativesReport RunDefault(ICallableBondReferencePricer pricer)
    {
        ArgumentNullException.ThrowIfNull(pricer);

        CallableBondFullDimensionalWrapper wrapper = CallableBondFullDimensionalWrapper.CreateDefault(pricer);
        CallableBondRequest baseRequest = wrapper.ToRequest(wrapper.CreateBasePoint());
        CurveFactorSurrogate directSurrogate = BuildCurveFactorSurrogate(wrapper);
        CurveFactorSurrogate curveFactorTt = BuildCurveFactorTtSurrogate(wrapper);
        CurveFactorSurrogate embeddedOptionSurrogate = BuildEmbeddedOptionSurrogate(wrapper);
        CurveFactorSurrogate embeddedOptionFullPillar = BuildEmbeddedOptionFullPillarTt(wrapper);

        IReadOnlyList<CallableValidationPoint> factorAligned = BuildFactorAlignedValidation(wrapper);
        IReadOnlyList<CallableValidationPoint> arbitrary = BuildArbitraryValidation(wrapper);
        double[] timingPoint = factorAligned[0].Coordinates;
        SpeedSummary directSpeed = MeasureSpeed(wrapper.Price, directSurrogate.EvalFullPoint, timingPoint, directSurrogate.BuildSeconds);
        SpeedSummary curveFactorTtSpeed = MeasureSpeed(wrapper.Price, curveFactorTt.EvalFullPoint, timingPoint, curveFactorTt.BuildSeconds);
        SpeedSummary embeddedOptionSpeed = MeasureSpeed(wrapper.Price, embeddedOptionSurrogate.EvalFullPoint, timingPoint, embeddedOptionSurrogate.BuildSeconds);
        SpeedSummary fullPillarSpeed = MeasureSpeed(wrapper.Price, embeddedOptionFullPillar.EvalFullPoint, timingPoint, embeddedOptionFullPillar.BuildSeconds);

        var directModel = new CallableStructuredAlternativeModelSummary(
            ModelName: "Curve-factor tensor",
            ApproximationType: "factor-risk surrogate",
            PublicInputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            InternalDimensionCount: InternalDimensionCount,
            BuildEvaluations: directSurrogate.BuildEvaluations,
            BuildSeconds: directSurrogate.BuildSeconds,
            BaselineEvalMicroseconds: directSpeed.BaselineEvalMicroseconds,
            SurrogateEvalMicroseconds: directSpeed.SurrogateEvalMicroseconds,
            BreakEvenEvaluations: directSpeed.BreakEvenEvaluations,
            FactorAlignedMetrics: SummarizeMetrics(wrapper.Price, directSurrogate.EvalFullPoint, factorAligned),
            ArbitraryBumpMetrics: SummarizeMetrics(wrapper.Price, directSurrogate.EvalFullPoint, arbitrary));

        var curveFactorTtModel = new CallableStructuredAlternativeModelSummary(
            ModelName: "Curve-factor TT",
            ApproximationType: "factor-risk surrogate",
            PublicInputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            InternalDimensionCount: InternalDimensionCount,
            BuildEvaluations: curveFactorTt.BuildEvaluations,
            BuildSeconds: curveFactorTt.BuildSeconds,
            BaselineEvalMicroseconds: curveFactorTtSpeed.BaselineEvalMicroseconds,
            SurrogateEvalMicroseconds: curveFactorTtSpeed.SurrogateEvalMicroseconds,
            BreakEvenEvaluations: curveFactorTtSpeed.BreakEvenEvaluations,
            FactorAlignedMetrics: SummarizeMetrics(wrapper.Price, curveFactorTt.EvalFullPoint, factorAligned),
            ArbitraryBumpMetrics: SummarizeMetrics(wrapper.Price, curveFactorTt.EvalFullPoint, arbitrary));

        var embeddedOptionModel = new CallableStructuredAlternativeModelSummary(
            ModelName: "Embedded-option curve-factor tensor",
            ApproximationType: "formula-aware factor-risk surrogate",
            PublicInputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            InternalDimensionCount: InternalDimensionCount,
            BuildEvaluations: embeddedOptionSurrogate.BuildEvaluations,
            BuildSeconds: embeddedOptionSurrogate.BuildSeconds,
            BaselineEvalMicroseconds: embeddedOptionSpeed.BaselineEvalMicroseconds,
            SurrogateEvalMicroseconds: embeddedOptionSpeed.SurrogateEvalMicroseconds,
            BreakEvenEvaluations: embeddedOptionSpeed.BreakEvenEvaluations,
            FactorAlignedMetrics: SummarizeMetrics(wrapper.Price, embeddedOptionSurrogate.EvalFullPoint, factorAligned),
            ArbitraryBumpMetrics: SummarizeMetrics(wrapper.Price, embeddedOptionSurrogate.EvalFullPoint, arbitrary));

        var embeddedOptionFullPillarModel = new CallableStructuredAlternativeModelSummary(
            ModelName: "Embedded-option full-pillar TT",
            ApproximationType: "formula-aware faithful full-pillar candidate",
            PublicInputDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            InternalDimensionCount: CallableBondFullDimensionalWrapper.DimensionCount,
            BuildEvaluations: embeddedOptionFullPillar.BuildEvaluations,
            BuildSeconds: embeddedOptionFullPillar.BuildSeconds,
            BaselineEvalMicroseconds: fullPillarSpeed.BaselineEvalMicroseconds,
            SurrogateEvalMicroseconds: fullPillarSpeed.SurrogateEvalMicroseconds,
            BreakEvenEvaluations: fullPillarSpeed.BreakEvenEvaluations,
            FactorAlignedMetrics: SummarizeMetrics(wrapper.Price, embeddedOptionFullPillar.EvalFullPoint, factorAligned),
            ArbitraryBumpMetrics: SummarizeMetrics(wrapper.Price, embeddedOptionFullPillar.EvalFullPoint, arbitrary));

        return new CallableStructuredAlternativesReport(
            FixtureId: "fed-nominal-yield-curve-semiannual-2026-05-15",
            CurveDate: baseRequest.ValuationDate,
            Models: [directModel, curveFactorTtModel, embeddedOptionModel, embeddedOptionFullPillarModel]);
    }

    private static CurveFactorSurrogate BuildCurveFactorSurrogate(
        CallableBondFullDimensionalWrapper wrapper)
    {
        double InternalFunction(double[] internalPoint, object? _)
            => wrapper.Price(ToFullPoint(internalPoint));

        var approximation = new ChebyshevApproximation(
            InternalFunction,
            numDimensions: InternalDimensionCount,
            domain: InternalDomain,
            nNodes: NNodes);

        Stopwatch sw = Stopwatch.StartNew();
        approximation.Build(verbose: false);
        sw.Stop();

        double EvalFullPoint(double[] fullPoint)
        {
            double[] internalPoint = ProjectFullPoint(fullPoint);
            return approximation.VectorizedEval(internalPoint, new int[InternalDimensionCount]);
        }

        return new CurveFactorSurrogate(
            EvalFullPoint,
            approximation.NEvaluations,
            sw.Elapsed.TotalSeconds);
    }

    private static CurveFactorSurrogate BuildEmbeddedOptionSurrogate(
        CallableBondFullDimensionalWrapper wrapper)
    {
        double InternalFunction(double[] internalPoint, object? _)
            => EmbeddedOptionValue(wrapper, ToFullPoint(internalPoint));

        var approximation = new ChebyshevApproximation(
            InternalFunction,
            numDimensions: InternalDimensionCount,
            domain: InternalDomain,
            nNodes: NNodes);

        Stopwatch sw = Stopwatch.StartNew();
        approximation.Build(verbose: false);
        sw.Stop();

        double EvalFullPoint(double[] fullPoint)
        {
            double[] internalPoint = ProjectFullPoint(fullPoint);
            double optionValue = approximation.VectorizedEval(internalPoint, new int[InternalDimensionCount]);
            return StraightDirtyPrice(wrapper, fullPoint) - optionValue;
        }

        return new CurveFactorSurrogate(
            EvalFullPoint,
            approximation.NEvaluations,
            sw.Elapsed.TotalSeconds);
    }

    private static CurveFactorSurrogate BuildCurveFactorTtSurrogate(
        CallableBondFullDimensionalWrapper wrapper)
    {
        int[] nNodes = Enumerable.Repeat(5, InternalDimensionCount).ToArray();
        var tt = new ChebyshevTT(
            point => wrapper.Price(ToFullPoint(point)),
            numDimensions: InternalDimensionCount,
            domain: InternalDomain,
            nNodes: nNodes,
            maxRank: 6,
            tolerance: 1e-5,
            maxSweeps: 4);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260522, method: "cross");
        sw.Stop();

        double EvalFullPoint(double[] fullPoint)
        {
            double[] internalPoint = ProjectFullPoint(fullPoint);
            return tt.Eval(internalPoint);
        }

        return new CurveFactorSurrogate(
            EvalFullPoint,
            tt.TotalBuildEvals,
            sw.Elapsed.TotalSeconds);
    }

    private static CurveFactorSurrogate BuildEmbeddedOptionFullPillarTt(
        CallableBondFullDimensionalWrapper wrapper)
    {
        var tt = new ChebyshevTT(
            point => EmbeddedOptionValue(wrapper, point),
            numDimensions: CallableBondFullDimensionalWrapper.DimensionCount,
            domain: PublicDomain,
            nNodes: PublicNNodes,
            maxRank: 4,
            tolerance: 1e-4,
            maxSweeps: 3);

        Stopwatch sw = Stopwatch.StartNew();
        tt.Build(verbose: false, seed: 20260522, method: "cross");
        sw.Stop();

        double EvalFullPoint(double[] fullPoint)
        {
            double optionValue = tt.Eval(fullPoint);
            return StraightDirtyPrice(wrapper, fullPoint) - optionValue;
        }

        return new CurveFactorSurrogate(
            EvalFullPoint,
            tt.TotalBuildEvals,
            sw.Elapsed.TotalSeconds);
    }

    private static double EmbeddedOptionValue(
        CallableBondFullDimensionalWrapper wrapper,
        double[] fullPoint)
        => StraightDirtyPrice(wrapper, fullPoint) - wrapper.Price(fullPoint);

    private static double StraightDirtyPrice(
        CallableBondFullDimensionalWrapper wrapper,
        double[] fullPoint)
    {
        var straightPricer = new QlNetFixedRateBondReferencePricer();
        CallableBondRequest callable = wrapper.ToRequest(fullPoint);
        FixedRateBondResult straight = straightPricer.Price(new FixedRateBondRequest(
            callable.ValuationDate,
            callable.EffectiveDate,
            callable.MaturityDate,
            callable.Coupon,
            callable.Notional,
            callable.ZeroCurve,
            callable.SettlementDays));
        return straight.DirtyPrice;
    }

    private static IReadOnlyList<CallableValidationPoint> BuildFactorAlignedValidation(
        CallableBondFullDimensionalWrapper wrapper)
    {
        double[][] internalPoints =
        [
            InternalPoint(level: 0.0, slope: 0.0, curvature: 0.0, coupon: 0.06, maturity: 24.0, firstCall: 5.0, callPrice: 100.0, sigma: 0.010),
            InternalPoint(level: 60.0, slope: 20.0, curvature: -15.0, coupon: 0.08, maturity: 20.0, firstCall: 4.0, callPrice: 102.0, sigma: 0.015),
            InternalPoint(level: -50.0, slope: -30.0, curvature: 20.0, coupon: 0.04, maturity: 25.0, firstCall: 7.0, callPrice: 99.0, sigma: 0.006),
        ];

        return internalPoints
            .Select((point, index) =>
            {
                double[] fullPoint = ToFullPoint(point);
                return new CallableValidationPoint($"factor{index + 1}", fullPoint, wrapper.Price(fullPoint));
            })
            .ToArray();
    }

    private static IReadOnlyList<CallableValidationPoint> BuildArbitraryValidation(
        CallableBondFullDimensionalWrapper wrapper)
    {
        double[][] fullPoints =
        [
            FullPoint(coupon: 0.06, maturityYears: 24.0, firstCallYears: 5.0, callPrice: 100.0, sigma: 0.010, _ => 0.0),
            FullPoint(coupon: 0.08, maturityYears: 20.0, firstCallYears: 4.0, callPrice: 102.0, sigma: 0.015,
                index => index % 2 == 0 ? 90.0 : -90.0),
            FullPoint(coupon: 0.055, maturityYears: 16.0, firstCallYears: 3.5, callPrice: 101.0, sigma: 0.020,
                index => 80.0 * Math.Sin((index + 1) * Math.PI / 10.0)),
        ];

        return fullPoints
            .Select((point, index) => new CallableValidationPoint($"arbitrary{index + 1}", point, wrapper.Price(point)))
            .ToArray();
    }

    private static IReadOnlyList<CallableNaiveSurrogateMetricSummary> SummarizeMetrics(
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableValidationPoint> validationPoints)
    {
        var metricFunctions = new (string Name, Func<Func<double[], double>, double[], double> Compute)[]
        {
            ("PV", (f, point) => f(point)),
            ("level-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[0], RateBpStep)),
            ("slope-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[1], RateBpStep)),
            ("curvature-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[2], RateBpStep)),
            ("10Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(120), RateBpStep)),
            ("30Y zero-pillar DV01", (f, point) => FirstDerivative(f, point, CurveDimensionForMonths(360), RateBpStep)),
            ("sigma sensitivity", (f, point) => FirstDerivative(f, point, SigmaDimension, SigmaStep)),
        };

        return metricFunctions
            .Select(metric => SummarizeMetric(metric.Name, metric.Compute, baseline, model, validationPoints))
            .ToArray();
    }

    private static SpeedSummary MeasureSpeed(
        Func<double[], double> baseline,
        Func<double[], double> surrogate,
        double[] point,
        double buildSeconds)
    {
        _ = baseline(point);
        _ = surrogate(point);

        double baselineMicroseconds = TimePerCallMicroseconds(baseline, point, iterations: 20);
        double surrogateMicroseconds = TimePerCallMicroseconds(surrogate, point, iterations: 2_000);
        double savedSeconds = (baselineMicroseconds - surrogateMicroseconds) / 1_000_000.0;
        double breakEven = savedSeconds > 0.0
            ? buildSeconds / savedSeconds
            : double.PositiveInfinity;

        return new SpeedSummary(baselineMicroseconds, surrogateMicroseconds, breakEven);
    }

    private static double TimePerCallMicroseconds(
        Func<double[], double> function,
        double[] point,
        int iterations)
    {
        Stopwatch sw = Stopwatch.StartNew();
        double sink = 0.0;
        for (int i = 0; i < iterations; i++)
        {
            sink += function(point);
        }

        sw.Stop();
        GC.KeepAlive(sink);
        return sw.Elapsed.TotalMilliseconds * 1_000.0 / iterations;
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

    private static double[] InternalPoint(
        double level,
        double slope,
        double curvature,
        double coupon,
        double maturity,
        double firstCall,
        double callPrice,
        double sigma)
        => [level, slope, curvature, coupon, maturity, firstCall, callPrice, sigma];

    private static double[] FullPoint(
        double coupon,
        double maturityYears,
        double firstCallYears,
        double callPrice,
        double sigma,
        Func<int, double> bumpByCurveIndex)
    {
        var point = new double[CallableBondFullDimensionalWrapper.DimensionCount];
        for (int i = 0; i < PublicCurveBumpCount; i++)
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

    private static double[] ToFullPoint(double[] internalPoint)
    {
        var full = new double[CallableBondFullDimensionalWrapper.DimensionCount];
        for (int i = 0; i < PublicCurveBumpCount; i++)
        {
            full[i] =
                internalPoint[0] * CurveBasis[0][i] +
                internalPoint[1] * CurveBasis[1][i] +
                internalPoint[2] * CurveBasis[2][i];
        }

        full[CouponDimension] = internalPoint[3];
        full[MaturityDimension] = internalPoint[4];
        full[FirstCallDimension] = internalPoint[5];
        full[CallPriceDimension] = internalPoint[6];
        full[SigmaDimension] = internalPoint[7];
        return full;
    }

    private static double[] ProjectFullPoint(double[] fullPoint)
    {
        if (fullPoint.Length != CallableBondFullDimensionalWrapper.DimensionCount)
        {
            throw new ArgumentException(
                $"Expected {CallableBondFullDimensionalWrapper.DimensionCount} coordinates.",
                nameof(fullPoint));
        }

        var internalPoint = new double[InternalDimensionCount];
        for (int basisIndex = 0; basisIndex < 3; basisIndex++)
        {
            double numerator = 0.0;
            double denominator = 0.0;
            for (int i = 0; i < PublicCurveBumpCount; i++)
            {
                double basis = CurveBasis[basisIndex][i];
                numerator += fullPoint[i] * basis;
                denominator += basis * basis;
            }

            internalPoint[basisIndex] = numerator / denominator;
        }

        internalPoint[3] = fullPoint[CouponDimension];
        internalPoint[4] = fullPoint[MaturityDimension];
        internalPoint[5] = fullPoint[FirstCallDimension];
        internalPoint[6] = fullPoint[CallPriceDimension];
        internalPoint[7] = fullPoint[SigmaDimension];
        return internalPoint;
    }

    private static double[] BuildTenorCoordinates()
        => Enumerable.Range(0, PublicCurveBumpCount)
            .Select(i => -1.0 + 2.0 * i / (PublicCurveBumpCount - 1))
            .ToArray();

    private static double[][] BuildPublicDomain()
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

    private static double[][] BuildCurveBasis()
        =>
        [
            Enumerable.Repeat(1.0, PublicCurveBumpCount).ToArray(),
            TenorCoordinates.ToArray(),
            TenorCoordinates.Select(x => 2.0 * x * x - 1.0).ToArray(),
        ];

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
        double[] down = Shift(point, dimension, -step);
        double[] up = Shift(point, dimension, step);
        return (function(up) - function(down)) / (2.0 * step);
    }

    private static double DirectionalCurveDerivative(
        Func<double[], double> function,
        double[] point,
        double[] direction,
        double step)
    {
        double[] down = ShiftCurve(point, direction, -step);
        double[] up = ShiftCurve(point, direction, step);
        return (function(up) - function(down)) / (2.0 * step);
    }

    private static double[] ShiftCurve(double[] point, double[] direction, double scale)
    {
        double[] shifted = (double[])point.Clone();
        for (int i = 0; i < PublicCurveBumpCount; i++)
        {
            shifted[i] += scale * direction[i];
        }

        return shifted;
    }

    private static double[] Shift(double[] point, int dimension, double shift)
    {
        double[] shifted = (double[])point.Clone();
        shifted[dimension] += shift;
        return shifted;
    }

    private sealed record CurveFactorSurrogate(
        Func<double[], double> EvalFullPoint,
        int BuildEvaluations,
        double BuildSeconds);

    private sealed record SpeedSummary(
        double BaselineEvalMicroseconds,
        double SurrogateEvalMicroseconds,
        double BreakEvenEvaluations);

    private sealed record CallableValidationPoint(
        string Name,
        double[] Coordinates,
        double BaselineDirtyPrice);
}
