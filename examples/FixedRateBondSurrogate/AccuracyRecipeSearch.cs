namespace FixedRateBondSurrogate;

public sealed record AccuracyProjectionPoint(
    string Name,
    string SetName,
    double OriginalDirtyPrice,
    double ReconstructedDirtyPrice,
    double AbsoluteError,
    double RelativeError);

public sealed record AccuracyProjectionOracleSummary(
    IReadOnlyList<AccuracyProjectionPoint> Points,
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

        string decision = projection.MaxClonePvAbsoluteError > projection.MaxFactorAlignedPvAbsoluteError * 10.0
            ? "Projection oracle is already material: factor compression must be separated from arbitrary 60-pillar clone accuracy before adding more TT complexity."
            : "Projection oracle is not dominant yet; continue with derivative and local-piece resolution diagnostics.";

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
            MaxClonePvAbsoluteError: clone.Max(point => point.AbsoluteError),
            MaxClonePvRelativeError: clone.Max(point => point.RelativeError),
            MaxFactorAlignedPvAbsoluteError: factor.Max(point => point.AbsoluteError),
            MaxFactorAlignedPvRelativeError: factor.Max(point => point.RelativeError));
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
        return
        [
            factors[0],
            factors[1],
            factors[2],
            fullPoint[CouponDimension],
            fullPoint[MaturityDimension],
        ];
    }

    private static double[] ToFullPoint(double[] factorPoint, CurveFactorBasis factorBasis)
    {
        var fullPoint = new double[PublicInputDimensionCount];
        double[] reconstructedBumps = factorBasis.Reconstruct(factorPoint);
        Array.Copy(reconstructedBumps, fullPoint, reconstructedBumps.Length);
        fullPoint[CouponDimension] = factorPoint[3];
        fullPoint[MaturityDimension] = factorPoint[4];
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
