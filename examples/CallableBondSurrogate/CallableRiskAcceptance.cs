namespace CallableBondSurrogate;

public sealed record CallableRiskValidationPoint(
    string Name,
    double[] Coordinates);

public sealed record CallableRiskScalarMetricSummary(
    string Name,
    double MeanAbsoluteError,
    double MaxAbsoluteError,
    double MeanRelativeError,
    double MaxRelativeError,
    string WorstPointName,
    double ExpectedAtWorstPoint,
    double ActualAtWorstPoint);

public sealed record CallableRiskVectorMetricSummary(
    string Name,
    double MeanMaxComponentAbsoluteError,
    double MaxComponentAbsoluteError,
    double MeanL1RelativeError,
    double MaxL1RelativeError,
    string WorstPointName,
    string WorstComponentLabel,
    double ExpectedWorstComponent,
    double ActualWorstComponent);

public sealed record CallableRiskAcceptanceMetrics(
    IReadOnlyList<CallableRiskScalarMetricSummary> ScalarMetrics,
    IReadOnlyList<CallableRiskVectorMetricSummary> VectorMetrics);

public static class CallableRiskAcceptance
{
    public const int CouponDimension = 60;
    public const int MaturityDimension = 61;
    public const int FirstCallDimension = 62;
    public const int CallPriceDimension = 63;
    public const int SigmaDimension = 64;
    public const double RateBpStep = 1.0;
    public const double CouponStep = 1e-4;
    public const double CallPriceStep = 0.01;
    public const double SigmaStep = 1e-4;
    public const double RelativeErrorFloor = 1e-10;

    private static readonly double[][] Domain = BuildDomain();
    private static readonly double[] TenorCoordinates = BuildTenorCoordinates();
    private static readonly double[][] CurveBasis = BuildCurveBasis();

    public static IReadOnlyList<CallableRiskValidationPoint> BuildDefaultValidationBank()
        =>
        [
            Point("base-30y-atm", coupon: 0.06, maturityYears: 30.0, firstCallYears: 5.0, callPrice: 100.0, sigma: 0.010, _ => 0.0),
            Point("factor-up-steep", coupon: 0.08, maturityYears: 20.0, firstCallYears: 4.0, callPrice: 102.0, sigma: 0.015,
                index => 60.0 + 20.0 * CurveBasis[1][index] - 15.0 * CurveBasis[2][index]),
            Point("factor-down-curved", coupon: 0.04, maturityYears: 25.0, firstCallYears: 7.0, callPrice: 99.0, sigma: 0.006,
                index => -50.0 - 30.0 * CurveBasis[1][index] + 20.0 * CurveBasis[2][index]),
            Point("alternating-local", coupon: 0.08, maturityYears: 20.0, firstCallYears: 4.0, callPrice: 102.0, sigma: 0.015,
                index => index % 2 == 0 ? 90.0 : -90.0),
            Point("sinusoidal-local", coupon: 0.055, maturityYears: 16.0, firstCallYears: 3.5, callPrice: 101.0, sigma: 0.020,
                index => 80.0 * Math.Sin((index + 1) * Math.PI / 10.0)),
            Point("high-vol-near-par-call", coupon: 0.06, maturityYears: 30.0, firstCallYears: 5.0, callPrice: 100.0, sigma: 0.024,
                index => -35.0 + 15.0 * CurveBasis[2][index]),
            Point("low-vol-high-call", coupon: 0.05, maturityYears: 18.0, firstCallYears: 6.0, callPrice: 104.0, sigma: 0.004,
                index => 40.0 * Math.Sin((index + 1) * Math.PI / 6.0)),
        ];

    public static double[][] BuildPublicDomain()
        => BuildDomain();

    public static CallableRiskAcceptanceMetrics Summarize(
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableRiskValidationPoint> validationPoints)
    {
        ArgumentNullException.ThrowIfNull(baseline);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(validationPoints);
        if (validationPoints.Count == 0)
        {
            throw new ArgumentException("At least one validation point is required.", nameof(validationPoints));
        }

        var scalarComputations = new (string Name, Func<Func<double[], double>, double[], double> Compute)[]
        {
            ("PV", (f, point) => f(point)),
            ("level-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[0], RateBpStep)),
            ("slope-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[1], RateBpStep)),
            ("curvature-factor sensitivity", (f, point) => DirectionalCurveDerivative(f, point, CurveBasis[2], RateBpStep)),
            ("coupon derivative", (f, point) => FirstDerivative(f, point, CouponDimension, CouponStep)),
            ("call-price sensitivity", (f, point) => FirstDerivative(f, point, CallPriceDimension, CallPriceStep)),
            ("sigma sensitivity", (f, point) => FirstDerivative(f, point, SigmaDimension, SigmaStep)),
            ("10Y rate-sigma mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(120), RateBpStep, SigmaDimension, SigmaStep)),
            ("call-price-sigma mixed", (f, point) => MixedDerivative(f, point, CallPriceDimension, CallPriceStep, SigmaDimension, SigmaStep)),
            ("10Y-10.5Y rate-rate mixed", (f, point) => MixedDerivative(f, point, CurveDimensionForMonths(120), RateBpStep, CurveDimensionForMonths(126), RateBpStep)),
        };

        var vectorComputations = new (string Name, Func<Func<double[], double>, double[], double[]> Compute)[]
        {
            ("full 60-pillar DV01 vector", FullDv01Vector),
        };

        return new CallableRiskAcceptanceMetrics(
            ScalarMetrics: scalarComputations
                .Select(metric => SummarizeScalar(metric.Name, metric.Compute, baseline, model, validationPoints))
                .ToArray(),
            VectorMetrics: vectorComputations
                .Select(metric => SummarizeVector(metric.Name, metric.Compute, baseline, model, validationPoints))
                .ToArray());
    }

    public static double[] BuildCurveFactorPoint(
        double level,
        double slope,
        double curvature,
        double coupon,
        double maturityYears,
        double firstCallYears,
        double callPrice,
        double sigma)
    {
        var point = new double[CallableBondFullDimensionalWrapper.DimensionCount];
        for (int i = 0; i < CallableBondFullDimensionalWrapper.CurveBumpCount; i++)
        {
            point[i] = level * CurveBasis[0][i] + slope * CurveBasis[1][i] + curvature * CurveBasis[2][i];
        }

        point[CouponDimension] = coupon;
        point[MaturityDimension] = maturityYears;
        point[FirstCallDimension] = firstCallYears;
        point[CallPriceDimension] = callPrice;
        point[SigmaDimension] = sigma;
        return point;
    }

    private static CallableRiskValidationPoint Point(
        string name,
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
        return new CallableRiskValidationPoint(name, point);
    }

    private static CallableRiskScalarMetricSummary SummarizeScalar(
        string name,
        Func<Func<double[], double>, double[], double> compute,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableRiskValidationPoint> validationPoints)
    {
        double sumAbs = 0.0;
        double maxAbs = 0.0;
        double sumRel = 0.0;
        double maxRel = 0.0;
        string worstPointName = validationPoints[0].Name;
        double expectedAtWorst = 0.0;
        double actualAtWorst = 0.0;

        foreach (CallableRiskValidationPoint point in validationPoints)
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

        return new CallableRiskScalarMetricSummary(
            name,
            MeanAbsoluteError: sumAbs / validationPoints.Count,
            MaxAbsoluteError: maxAbs,
            MeanRelativeError: sumRel / validationPoints.Count,
            MaxRelativeError: maxRel,
            WorstPointName: worstPointName,
            ExpectedAtWorstPoint: expectedAtWorst,
            ActualAtWorstPoint: actualAtWorst);
    }

    private static CallableRiskVectorMetricSummary SummarizeVector(
        string name,
        Func<Func<double[], double>, double[], double[]> compute,
        Func<double[], double> baseline,
        Func<double[], double> model,
        IReadOnlyList<CallableRiskValidationPoint> validationPoints)
    {
        double sumMaxAbs = 0.0;
        double maxComponentAbs = 0.0;
        double sumL1Rel = 0.0;
        double maxL1Rel = 0.0;
        string worstPointName = validationPoints[0].Name;
        string worstComponentLabel = CurveComponentLabel(0);
        double expectedWorstComponent = 0.0;
        double actualWorstComponent = 0.0;

        foreach (CallableRiskValidationPoint point in validationPoints)
        {
            double[] expected = compute(baseline, point.Coordinates);
            double[] actual = compute(model, point.Coordinates);
            if (expected.Length != actual.Length)
            {
                throw new InvalidOperationException("Baseline and model vector metric lengths differ.");
            }

            double pointMaxAbs = 0.0;
            double l1Error = 0.0;
            double l1Reference = 0.0;
            int pointWorstIndex = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double abs = Math.Abs(actual[i] - expected[i]);
                l1Error += abs;
                l1Reference += Math.Abs(expected[i]);
                if (abs >= pointMaxAbs)
                {
                    pointMaxAbs = abs;
                    pointWorstIndex = i;
                }
            }

            double l1Rel = l1Error / Math.Max(l1Reference, RelativeErrorFloor);
            sumMaxAbs += pointMaxAbs;
            sumL1Rel += l1Rel;
            maxL1Rel = Math.Max(maxL1Rel, l1Rel);

            if (pointMaxAbs >= maxComponentAbs)
            {
                maxComponentAbs = pointMaxAbs;
                worstPointName = point.Name;
                worstComponentLabel = CurveComponentLabel(pointWorstIndex);
                expectedWorstComponent = expected[pointWorstIndex];
                actualWorstComponent = actual[pointWorstIndex];
            }
        }

        return new CallableRiskVectorMetricSummary(
            name,
            MeanMaxComponentAbsoluteError: sumMaxAbs / validationPoints.Count,
            MaxComponentAbsoluteError: maxComponentAbs,
            MeanL1RelativeError: sumL1Rel / validationPoints.Count,
            MaxL1RelativeError: maxL1Rel,
            WorstPointName: worstPointName,
            WorstComponentLabel: worstComponentLabel,
            ExpectedWorstComponent: expectedWorstComponent,
            ActualWorstComponent: actualWorstComponent);
    }

    private static double[] FullDv01Vector(Func<double[], double> function, double[] point)
    {
        var vector = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = FirstDerivative(function, point, i, RateBpStep);
        }

        return vector;
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

    private static (double DownStep, double UpStep) StepsInsideDomain(
        double value,
        int dimension,
        double requestedStep)
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

    private static double[] ShiftCurve(double[] point, double[] direction, double scale)
    {
        double[] shifted = (double[])point.Clone();
        for (int i = 0; i < CallableBondFullDimensionalWrapper.CurveBumpCount; i++)
        {
            shifted[i] += scale * direction[i];
        }

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

    private static string CurveComponentLabel(int index)
    {
        double years = 0.5 * (index + 1);
        return Math.Abs(years - Math.Round(years)) < 1e-12
            ? $"{(int)Math.Round(years)}Y"
            : $"{years:0.0}Y";
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

    private static double[] BuildTenorCoordinates()
        => Enumerable.Range(0, CallableBondFullDimensionalWrapper.CurveBumpCount)
            .Select(i => -1.0 + 2.0 * i / (CallableBondFullDimensionalWrapper.CurveBumpCount - 1))
            .ToArray();

    private static double[][] BuildCurveBasis()
        =>
        [
            Enumerable.Repeat(1.0, CallableBondFullDimensionalWrapper.CurveBumpCount).ToArray(),
            TenorCoordinates.ToArray(),
            TenorCoordinates.Select(x => 2.0 * x * x - 1.0).ToArray(),
        ];
}
