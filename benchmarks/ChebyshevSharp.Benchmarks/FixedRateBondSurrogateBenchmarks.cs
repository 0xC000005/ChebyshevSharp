using BenchmarkDotNet.Attributes;
using FixedRateBondSurrogate;

namespace ChebyshevSharp.Benchmarks;

[MemoryDiagnoser]
public class FixedRateBondSurrogateBenchmarks
{
    private const int CurveBumpDimensionCount = ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount;
    private const int CouponDimension = ScheduleResolvedCashflowChebyshevBondPricer.CouponDimension;
    private const int MaturityDimension = ScheduleResolvedCashflowChebyshevBondPricer.MaturityDimension;

    private IFixedRateBondReferencePricer _referencePricer = null!;
    private FixedRateBondRequest _baseRequest = null!;
    private ScheduleResolvedCashflowChebyshevBondPricer _chebyshevPricer = null!;
    private CachedCashflowPricer _cachedCashflowPricer = null!;
    private double[] _point = null!;
    private double[][] _batchPoints = null!;
    private double[] _curveGradient = null!;
    private double[] _rateCouponMixed = null!;

    [GlobalSetup]
    public void Setup()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        _baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        _referencePricer = new QlNetFixedRateBondReferencePricer();
        _chebyshevPricer = new ScheduleResolvedCashflowChebyshevBondPricer(_referencePricer, _baseRequest);
        _point = FullPoint(0.0575, 10.25, index => 65.0 * Math.Sin((index + 1) * Math.PI / 13.0));
        _batchPoints = BuildBatchPoints();
        _curveGradient = new double[CurveBumpDimensionCount];
        _rateCouponMixed = new double[CurveBumpDimensionCount];
        _cachedCashflowPricer = CachedCashflowPricer.Build(_referencePricer, _baseRequest, MaturityDateFromPoint(_point));

        _ = ReferencePrice(_point);
        _ = _cachedCashflowPricer.Price(_point);
        _ = _chebyshevPricer.Eval(_point);
        _ = _chebyshevPricer.EvalRiskUnchecked(_point, _curveGradient, _rateCouponMixed, out _);
    }

    [Benchmark(Baseline = true)]
    public double QlNet_ValueOnly()
        => ReferencePrice(_point);

    [Benchmark]
    public double CachedCashflow_ValueOnly()
        => _cachedCashflowPricer.Price(_point);

    [Benchmark]
    public double ChebyshevKernel_ValueOnly()
        => _chebyshevPricer.Eval(_point);

    [Benchmark]
    public double ChebyshevKernel_ValueOnly_Unchecked()
        => _chebyshevPricer.EvalUnchecked(_point);

    [Benchmark]
    public double QlNet_AllPillarDv01_FiniteDifference()
    {
        double sum = 0.0;
        for (int dim = 0; dim < CurveBumpDimensionCount; dim++)
        {
            sum += FirstDerivative(ReferencePrice, _point, dim, 1e-4);
        }

        return sum;
    }

    [Benchmark]
    public double ChebyshevKernel_AllPillarRisk_Analytic()
    {
        double dirtyPrice = _chebyshevPricer.EvalRiskUnchecked(
            _point,
            _curveGradient,
            _rateCouponMixed,
            out double couponDerivative);

        double sink = dirtyPrice + couponDerivative;
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            sink += _curveGradient[i] + _rateCouponMixed[i];
        }

        return sink;
    }

    [Benchmark]
    public double QlNet_Batch32_ValueOnly()
    {
        double sum = 0.0;
        foreach (double[] point in _batchPoints)
        {
            sum += ReferencePrice(point);
        }

        return sum;
    }

    [Benchmark]
    public double ChebyshevKernel_Batch32_ValueOnly()
    {
        double sum = 0.0;
        foreach (double[] point in _batchPoints)
        {
            sum += _chebyshevPricer.EvalUnchecked(point);
        }

        return sum;
    }

    private double ReferencePrice(double[] point)
        => _referencePricer.Price(RequestFromPoint(point)).DirtyPrice;

    private FixedRateBondRequest RequestFromPoint(double[] point)
    {
        var curve = new ZeroRatePillar[_baseRequest.ZeroCurve.Count];
        curve[0] = _baseRequest.ZeroCurve[0];
        for (int i = 1; i < curve.Length; i++)
        {
            curve[i] = _baseRequest.ZeroCurve[i] with
            {
                ZeroRate = _baseRequest.ZeroCurve[i].ZeroRate + point[i - 1] * 1e-4,
            };
        }

        return _baseRequest with
        {
            Coupon = point[CouponDimension],
            MaturityDate = MaturityDateFromPoint(point),
            ZeroCurve = curve,
        };
    }

    private DateTime MaturityDateFromPoint(double[] point)
        => _baseRequest.ValuationDate.Date.AddDays((int)Math.Round(365.25 * point[MaturityDimension]));

    private static double[][] BuildBatchPoints()
    {
        var points = new double[32][];
        for (int i = 0; i < points.Length; i++)
        {
            double maturity = 2.25 + 27.0 * i / (points.Length - 1);
            double coupon = 0.015 + 0.09 * ((i % 9) / 8.0);
            points[i] = FullPoint(
                coupon,
                maturity,
                index => 55.0 * Math.Sin((index + 1) * (i + 1) * Math.PI / 37.0));
        }

        return points;
    }

    private static double[] FullPoint(double coupon, double maturityYears, Func<int, double> curveBump)
    {
        var point = new double[ScheduleResolvedCashflowChebyshevBondPricer.PublicInputDimensionCount];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            point[i] = curveBump(i);
        }

        point[CouponDimension] = coupon;
        point[MaturityDimension] = maturityYears;
        return point;
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

    private static double[] Shift(double[] point, int dimension, double delta)
    {
        double[] copy = (double[])point.Clone();
        copy[dimension] += delta;
        return copy;
    }

    private sealed class CachedCashflowPricer
    {
        private readonly double _scale;
        private readonly Component[] _components;

        private CachedCashflowPricer(double scale, Component[] components)
        {
            _scale = scale;
            _components = components;
        }

        public static CachedCashflowPricer Build(
            IFixedRateBondReferencePricer referencePricer,
            FixedRateBondRequest baseRequest,
            DateTime maturityDate)
        {
            FixedRateBondResult result = referencePricer.Price(baseRequest with
            {
                Coupon = 1.0,
                MaturityDate = maturityDate,
            });

            Component[] components = result.Cashflows
                .Where(cashflow => !cashflow.HasOccurred)
                .Select(cashflow => Component.Build(
                    baseRequest,
                    cashflow.PaymentDate.Date,
                    cashflow.IsCoupon ? cashflow.Amount : 0.0,
                    cashflow.IsCoupon ? 0.0 : cashflow.Amount))
                .ToArray();

            return new CachedCashflowPricer(100.0 / baseRequest.Notional, components);
        }

        public double Price(double[] point)
        {
            double coupon = point[CouponDimension];
            double pv = 0.0;
            foreach (Component component in _components)
            {
                pv += (component.PrincipalAmount + coupon * component.CouponMultiplier) * component.Discount(point);
            }

            return pv * _scale;
        }

        private sealed class Component
        {
            private readonly double _paymentTime;
            private readonly int _lowerPublicDimension;
            private readonly int _upperPublicDimension;
            private readonly double _lowerBaseRate;
            private readonly double _upperBaseRate;
            private readonly double _lowerWeight;
            private readonly double _upperWeight;

            private Component(
                double paymentTime,
                int lowerPublicDimension,
                int upperPublicDimension,
                double lowerBaseRate,
                double upperBaseRate,
                double lowerWeight,
                double upperWeight,
                double couponMultiplier,
                double principalAmount)
            {
                _paymentTime = paymentTime;
                _lowerPublicDimension = lowerPublicDimension;
                _upperPublicDimension = upperPublicDimension;
                _lowerBaseRate = lowerBaseRate;
                _upperBaseRate = upperBaseRate;
                _lowerWeight = lowerWeight;
                _upperWeight = upperWeight;
                CouponMultiplier = couponMultiplier;
                PrincipalAmount = principalAmount;
            }

            public double CouponMultiplier { get; }

            public double PrincipalAmount { get; }

            public static Component Build(
                FixedRateBondRequest baseRequest,
                DateTime paymentDate,
                double couponMultiplier,
                double principalAmount)
            {
                IReadOnlyList<ZeroRatePillar> curve = baseRequest.ZeroCurve;
                DateTime valuationDate = baseRequest.ValuationDate.Date;
                int upper = 0;
                while (upper < curve.Count && curve[upper].Date.Date < paymentDate.Date)
                {
                    upper++;
                }

                int lower = curve[upper].Date.Date == paymentDate.Date ? upper : Math.Max(0, upper - 1);
                double paymentTime = Actual365(valuationDate, paymentDate);
                double lowerTime = Actual365(valuationDate, curve[lower].Date);
                double upperTime = Actual365(valuationDate, curve[upper].Date);
                double upperWeight = lower == upper || Math.Abs(upperTime - lowerTime) < 1e-14
                    ? 1.0
                    : (paymentTime - lowerTime) / (upperTime - lowerTime);

                return new Component(
                    paymentTime,
                    lower - 1,
                    upper - 1,
                    curve[lower].ZeroRate,
                    curve[upper].ZeroRate,
                    lower == upper ? 0.0 : 1.0 - upperWeight,
                    upperWeight,
                    couponMultiplier,
                    principalAmount);
            }

            public double Discount(double[] point)
            {
                double lowerRate = BumpedRate(_lowerBaseRate, _lowerPublicDimension, point);
                double upperRate = BumpedRate(_upperBaseRate, _upperPublicDimension, point);
                double zeroRate = _lowerWeight * lowerRate + _upperWeight * upperRate;
                return Math.Exp(-zeroRate * _paymentTime);
            }

            private static double BumpedRate(double baseRate, int publicDimension, double[] point)
                => publicDimension < 0 ? baseRate : baseRate + point[publicDimension] * 1e-4;

            private static double Actual365(DateTime valuationDate, DateTime paymentDate)
                => (paymentDate.Date - valuationDate.Date).TotalDays / 365.0;
        }
    }
}
