using System.Diagnostics;

namespace FixedRateBondSurrogate;

public sealed record ScheduleResolvedRiskResult(
    double DirtyPrice,
    double[] CurveBumpGradient,
    double CouponDerivative,
    double[] RateCouponMixed);

public sealed class ScheduleResolvedCashflowChebyshevBondPricer
{
    public const int CurveBumpDimensionCount = 60;
    public const int PublicInputDimensionCount = CurveBumpDimensionCount + 2;
    public const int CouponDimension = CurveBumpDimensionCount;
    public const int MaturityDimension = CurveBumpDimensionCount + 1;

    private const int DiscountKernelNodes = 9;
    private const double CurveBumpDomainLo = -150.0;
    private const double CurveBumpDomainHi = 150.0;
    private const double CouponDomainLo = 0.0;
    private const double CouponDomainHi = 0.12;
    private const double MaturityDomainLo = 2.0;
    private const double MaturityDomainHi = 30.0;

    private readonly IFixedRateBondReferencePricer _pricer;
    private readonly FixedRateBondRequest _baseRequest;
    private readonly Dictionary<DateTime, CashflowTemplate> _scheduleCache = new();
    private readonly Dictionary<DateTime, DiscountKernel> _discountKernels = new();

    public ScheduleResolvedCashflowChebyshevBondPricer(
        IFixedRateBondReferencePricer pricer,
        FixedRateBondRequest baseRequest)
    {
        ArgumentNullException.ThrowIfNull(pricer);
        ArgumentNullException.ThrowIfNull(baseRequest);

        if (baseRequest.ZeroCurve.Count != CurveBumpDimensionCount + 1)
        {
            throw new ArgumentException(
                $"Expected {CurveBumpDimensionCount + 1} zero-curve pillars including the valuation date.",
                nameof(baseRequest));
        }

        _pricer = pricer;
        _baseRequest = baseRequest;
    }

    public int MaxKernelDimension { get; private set; }

    public int BuildEvaluations => _discountKernels.Values.Sum(kernel => kernel.BuildEvaluations);

    public double BuildSeconds => _discountKernels.Values.Sum(kernel => kernel.BuildSeconds);

    public double PriceDirty(FixedRateBondRequest request)
        => Eval(ToFullPoint(request));

    public double Eval(double[] fullPoint)
    {
        ValidateFullPoint(fullPoint);

        return EvalUnchecked(fullPoint);
    }

    public double EvalUnchecked(double[] fullPoint)
    {
        DateTime maturityDate = MaturityDateFromPoint(fullPoint);
        CashflowTemplate template = GetTemplate(maturityDate);
        double coupon = fullPoint[CouponDimension];
        double pv = 0.0;

        foreach (CashflowComponent component in template.Components)
        {
            double amount = component.PrincipalAmount + coupon * component.CouponMultiplier;
            pv += amount * component.Kernel.Eval(fullPoint);
        }

        return pv * 100.0 / _baseRequest.Notional;
    }

    public ScheduleResolvedRiskResult EvalRisk(double[] fullPoint)
    {
        var curveBumpGradient = new double[CurveBumpDimensionCount];
        var rateCouponMixed = new double[CurveBumpDimensionCount];
        double dirtyPrice = EvalRisk(fullPoint, curveBumpGradient, rateCouponMixed, out double couponDerivative);

        return new ScheduleResolvedRiskResult(
            dirtyPrice,
            curveBumpGradient,
            couponDerivative,
            rateCouponMixed);
    }

    public double EvalRisk(
        double[] fullPoint,
        Span<double> curveBumpGradient,
        Span<double> rateCouponMixed,
        out double couponDerivative)
    {
        ValidateFullPoint(fullPoint);
        ValidateRiskBuffers(curveBumpGradient, rateCouponMixed);

        return EvalRiskUnchecked(fullPoint, curveBumpGradient, rateCouponMixed, out couponDerivative);
    }

    public double EvalRiskUnchecked(
        double[] fullPoint,
        Span<double> curveBumpGradient,
        Span<double> rateCouponMixed,
        out double couponDerivative)
    {
        ValidateRiskBuffers(curveBumpGradient, rateCouponMixed);

        DateTime maturityDate = MaturityDateFromPoint(fullPoint);
        CashflowTemplate template = GetTemplate(maturityDate);
        double coupon = fullPoint[CouponDimension];
        double pv = 0.0;
        couponDerivative = 0.0;
        curveBumpGradient[..CurveBumpDimensionCount].Clear();
        rateCouponMixed[..CurveBumpDimensionCount].Clear();
        Span<int> localDimensions = stackalloc int[2];
        Span<double> localDerivatives = stackalloc double[2];

        foreach (CashflowComponent component in template.Components)
        {
            double amount = component.PrincipalAmount + coupon * component.CouponMultiplier;
            component.Kernel.EvalWithDerivatives(
                fullPoint,
                localDimensions,
                localDerivatives,
                out double discount,
                out int derivativeCount);

            pv += amount * discount;
            couponDerivative += component.CouponMultiplier * discount;

            for (int i = 0; i < derivativeCount; i++)
            {
                int dimension = localDimensions[i];
                double discountDerivative = localDerivatives[i];
                curveBumpGradient[dimension] += amount * discountDerivative;
                rateCouponMixed[dimension] += component.CouponMultiplier * discountDerivative;
            }
        }

        double scale = 100.0 / _baseRequest.Notional;
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            curveBumpGradient[i] *= scale;
            rateCouponMixed[i] *= scale;
        }

        couponDerivative *= scale;
        return pv * scale;
    }

    private double[] ToFullPoint(FixedRateBondRequest request)
    {
        ValidateCompatibleRequest(request);

        var point = new double[PublicInputDimensionCount];
        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            int curveIndex = i + 1;
            point[i] = (request.ZeroCurve[curveIndex].ZeroRate - _baseRequest.ZeroCurve[curveIndex].ZeroRate) * 1e4;
        }

        point[CouponDimension] = request.Coupon;
        point[MaturityDimension] = (request.MaturityDate.Date - request.ValuationDate.Date).TotalDays / 365.25;
        ValidateFullPoint(point);
        return point;
    }

    private void ValidateCompatibleRequest(FixedRateBondRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        if (request.ValuationDate.Date != _baseRequest.ValuationDate.Date)
        {
            throw new ArgumentException("The surrogate was built for a different valuation date.", nameof(request));
        }

        if (request.EffectiveDate.Date != _baseRequest.EffectiveDate.Date)
        {
            throw new ArgumentException("The surrogate was built for a different effective date.", nameof(request));
        }

        if (request.SettlementDays != _baseRequest.SettlementDays)
        {
            throw new ArgumentException("The surrogate was built for a different settlement convention.", nameof(request));
        }

        if (request.ZeroCurve.Count != _baseRequest.ZeroCurve.Count)
        {
            throw new ArgumentException("The request zero curve must use the same pillar count as the build curve.", nameof(request));
        }

        for (int i = 0; i < request.ZeroCurve.Count; i++)
        {
            if (request.ZeroCurve[i].Date.Date != _baseRequest.ZeroCurve[i].Date.Date)
            {
                throw new ArgumentException("The request zero curve must use the same pillar dates as the build curve.", nameof(request));
            }
        }
    }

    private static void ValidateFullPoint(double[] fullPoint)
    {
        ArgumentNullException.ThrowIfNull(fullPoint);
        if (fullPoint.Length != PublicInputDimensionCount)
        {
            throw new ArgumentException(
                $"Expected {PublicInputDimensionCount} coordinates: 60 curve bumps, coupon, and maturity.",
                nameof(fullPoint));
        }

        for (int i = 0; i < CurveBumpDimensionCount; i++)
        {
            if (fullPoint[i] < CurveBumpDomainLo || fullPoint[i] > CurveBumpDomainHi)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(fullPoint),
                    fullPoint[i],
                    $"Curve bump coordinate {i} is outside the supported [{CurveBumpDomainLo}, {CurveBumpDomainHi}] bp domain.");
            }
        }

        if (fullPoint[CouponDimension] < CouponDomainLo || fullPoint[CouponDimension] > CouponDomainHi)
        {
            throw new ArgumentOutOfRangeException(
                nameof(fullPoint),
                fullPoint[CouponDimension],
                $"Coupon is outside the supported [{CouponDomainLo}, {CouponDomainHi}] domain.");
        }

        if (fullPoint[MaturityDimension] < MaturityDomainLo || fullPoint[MaturityDimension] > MaturityDomainHi)
        {
            throw new ArgumentOutOfRangeException(
                nameof(fullPoint),
                fullPoint[MaturityDimension],
                $"Maturity is outside the supported [{MaturityDomainLo}, {MaturityDomainHi}] year domain.");
        }
    }

    private static void ValidateRiskBuffers(
        Span<double> curveBumpGradient,
        Span<double> rateCouponMixed)
    {
        if (curveBumpGradient.Length < CurveBumpDimensionCount)
        {
            throw new ArgumentException(
                $"Curve-gradient buffer must contain at least {CurveBumpDimensionCount} entries.",
                nameof(curveBumpGradient));
        }

        if (rateCouponMixed.Length < CurveBumpDimensionCount)
        {
            throw new ArgumentException(
                $"Rate-coupon mixed buffer must contain at least {CurveBumpDimensionCount} entries.",
                nameof(rateCouponMixed));
        }
    }

    private CashflowTemplate GetTemplate(DateTime maturityDate)
    {
        if (_scheduleCache.TryGetValue(maturityDate, out CashflowTemplate? template))
        {
            return template;
        }

        FixedRateBondResult result = _pricer.Price(_baseRequest with
        {
            Coupon = 1.0,
            MaturityDate = maturityDate,
        });

        CashflowComponent[] components = result.Cashflows
            .Where(cashflow => !cashflow.HasOccurred)
            .Select(cashflow => new CashflowComponent(
                cashflow.PaymentDate.Date,
                CouponMultiplier: cashflow.IsCoupon ? cashflow.Amount : 0.0,
                PrincipalAmount: cashflow.IsCoupon ? 0.0 : cashflow.Amount,
                GetDiscountKernel(cashflow.PaymentDate.Date)))
            .ToArray();

        template = new CashflowTemplate(components);
        _scheduleCache.Add(maturityDate, template);
        return template;
    }

    private DiscountKernel GetDiscountKernel(DateTime paymentDate)
    {
        if (_discountKernels.TryGetValue(paymentDate, out DiscountKernel? kernel))
        {
            return kernel;
        }

        kernel = DiscountKernel.Build(_baseRequest, paymentDate, DiscountKernelNodes);
        _discountKernels.Add(paymentDate, kernel);
        MaxKernelDimension = Math.Max(MaxKernelDimension, kernel.DimensionCount);
        return kernel;
    }

    private DateTime MaturityDateFromPoint(double[] fullPoint)
        => _baseRequest.ValuationDate.Date.AddDays(
            (int)Math.Round(365.25 * fullPoint[MaturityDimension]));

    private sealed record CashflowTemplate(IReadOnlyList<CashflowComponent> Components);

    private sealed record CashflowComponent(
        DateTime PaymentDate,
        double CouponMultiplier,
        double PrincipalAmount,
        DiscountKernel Kernel);

    private sealed class DiscountKernel
    {
        private readonly DiscountKernelSpec _spec;
        private readonly int[] _publicDimensions;
        private readonly double[] _nodes;
        private readonly double[] _weights;
        private readonly double[] _values;
        private readonly double _constantDiscount;

        private DiscountKernel(
            DiscountKernelSpec spec,
            int[] publicDimensions,
            double[] nodes,
            double[] weights,
            double[] values,
            double constantDiscount,
            int buildEvaluations,
            double buildSeconds)
        {
            _spec = spec;
            _publicDimensions = publicDimensions;
            _nodes = nodes;
            _weights = weights;
            _values = values;
            _constantDiscount = constantDiscount;
            BuildEvaluations = buildEvaluations;
            BuildSeconds = buildSeconds;
        }

        public int DimensionCount => _publicDimensions.Length;

        public int BuildEvaluations { get; }

        public double BuildSeconds { get; }

        public static DiscountKernel Build(
            FixedRateBondRequest baseRequest,
            DateTime paymentDate,
            int nNodes)
        {
            DiscountKernelSpec spec = DiscountKernelSpec.From(baseRequest, paymentDate);
            if (spec.PublicDimensions.Length == 0)
            {
                return new DiscountKernel(
                    spec,
                    spec.PublicDimensions,
                    nodes: [],
                    weights: [],
                    values: [],
                    constantDiscount: spec.Discount(Array.Empty<double>()),
                    buildEvaluations: 0,
                    buildSeconds: 0.0);
            }

            if (spec.PublicDimensions.Length > 2)
            {
                throw new InvalidOperationException("A linear zero-rate discount kernel should depend on at most two curve pillars.");
            }

            Stopwatch sw = Stopwatch.StartNew();
            double[] nodes = MakeChebyshevNodes(CurveBumpDomainLo, CurveBumpDomainHi, nNodes);
            double[] weights = ComputeBarycentricWeights(nodes);
            double[] values = BuildKernelValues(spec, nodes);
            sw.Stop();

            return new DiscountKernel(
                spec,
                spec.PublicDimensions,
                nodes,
                weights,
                values,
                constantDiscount: 0.0,
                buildEvaluations: values.Length,
                buildSeconds: sw.Elapsed.TotalSeconds);
        }

        public double Eval(double[] fullPoint)
        {
            if (_publicDimensions.Length == 0)
            {
                return _constantDiscount;
            }

            double x = fullPoint[_publicDimensions[0]];
            if (_publicDimensions.Length == 1)
            {
                return Interpolate1D(x, _nodes, _weights, _values);
            }

            double y = fullPoint[_publicDimensions[1]];
            return Interpolate2D(x, y, _nodes, _weights, _values);
        }

        public void EvalWithDerivatives(
            double[] fullPoint,
            Span<int> outputDimensions,
            Span<double> derivatives,
            out double discount,
            out int derivativeCount)
        {
            discount = Eval(fullPoint);
            derivativeCount = _spec.FillFirstDerivatives(discount, outputDimensions, derivatives);
        }

        private static double[] BuildKernelValues(DiscountKernelSpec spec, double[] nodes)
        {
            int n = nodes.Length;
            if (spec.PublicDimensions.Length == 1)
            {
                var values = new double[n];
                var point = new double[1];
                for (int i = 0; i < n; i++)
                {
                    point[0] = nodes[i];
                    values[i] = spec.Discount(point);
                }

                return values;
            }

            var result = new double[n * n];
            var twoDimPoint = new double[2];
            for (int i = 0; i < n; i++)
            {
                twoDimPoint[0] = nodes[i];
                for (int j = 0; j < n; j++)
                {
                    twoDimPoint[1] = nodes[j];
                    result[(i * n) + j] = spec.Discount(twoDimPoint);
                }
            }

            return result;
        }

        private static double Interpolate1D(
            double x,
            double[] nodes,
            double[] weights,
            double[] values)
        {
            Span<double> basis = stackalloc double[nodes.Length];
            FillBarycentricBasis(x, nodes, weights, basis);
            double result = 0.0;
            for (int i = 0; i < nodes.Length; i++)
            {
                result += basis[i] * values[i];
            }

            return result;
        }

        private static double Interpolate2D(
            double x,
            double y,
            double[] nodes,
            double[] weights,
            double[] values)
        {
            int n = nodes.Length;
            Span<double> xBasis = stackalloc double[n];
            Span<double> yBasis = stackalloc double[n];
            FillBarycentricBasis(x, nodes, weights, xBasis);
            FillBarycentricBasis(y, nodes, weights, yBasis);

            double result = 0.0;
            for (int i = 0; i < n; i++)
            {
                double xWeight = xBasis[i];
                int row = i * n;
                for (int j = 0; j < n; j++)
                {
                    result += xWeight * yBasis[j] * values[row + j];
                }
            }

            return result;
        }

        private static void FillBarycentricBasis(
            double x,
            double[] nodes,
            double[] weights,
            Span<double> basis)
        {
            double denominator = 0.0;
            for (int i = 0; i < nodes.Length; i++)
            {
                if (Math.Abs(x - nodes[i]) < 1e-14)
                {
                    basis.Clear();
                    basis[i] = 1.0;
                    return;
                }

                double value = weights[i] / (x - nodes[i]);
                basis[i] = value;
                denominator += value;
            }

            for (int i = 0; i < basis.Length; i++)
            {
                basis[i] /= denominator;
            }
        }

        private static double[] MakeChebyshevNodes(double lo, double hi, int n)
        {
            double[] nodes = new double[n];
            double mid = 0.5 * (lo + hi);
            double half = 0.5 * (hi - lo);
            for (int k = 0; k < n; k++)
            {
                nodes[k] = mid + half * Math.Cos(Math.PI * ((2 * k) + 1) / (2 * n));
            }

            Array.Sort(nodes);
            return nodes;
        }

        private static double[] ComputeBarycentricWeights(double[] nodes)
        {
            int n = nodes.Length;
            var weights = new double[n];
            Array.Fill(weights, 1.0);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        weights[i] /= nodes[i] - nodes[j];
                    }
                }
            }

            return weights;
        }
    }

    private sealed class DiscountKernelSpec
    {
        private readonly double _paymentTime;
        private readonly double _lowerTime;
        private readonly double _upperTime;
        private readonly double _lowerBaseRate;
        private readonly double _upperBaseRate;
        private readonly int _lowerLocalIndex;
        private readonly int _upperLocalIndex;
        private readonly double[] _publicRateWeights;

        private DiscountKernelSpec(
            int[] publicDimensions,
            double[] publicRateWeights,
            double paymentTime,
            double lowerTime,
            double upperTime,
            double lowerBaseRate,
            double upperBaseRate,
            int lowerLocalIndex,
            int upperLocalIndex)
        {
            PublicDimensions = publicDimensions;
            _publicRateWeights = publicRateWeights;
            _paymentTime = paymentTime;
            _lowerTime = lowerTime;
            _upperTime = upperTime;
            _lowerBaseRate = lowerBaseRate;
            _upperBaseRate = upperBaseRate;
            _lowerLocalIndex = lowerLocalIndex;
            _upperLocalIndex = upperLocalIndex;
        }

        public int[] PublicDimensions { get; }

        public static DiscountKernelSpec From(FixedRateBondRequest baseRequest, DateTime paymentDate)
        {
            IReadOnlyList<ZeroRatePillar> curve = baseRequest.ZeroCurve;
            DateTime valuationDate = baseRequest.ValuationDate.Date;
            double paymentTime = Actual365(valuationDate, paymentDate);

            int upper = 0;
            while (upper < curve.Count && curve[upper].Date.Date < paymentDate.Date)
            {
                upper++;
            }

            if (upper >= curve.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(paymentDate), "Payment date is outside the zero-curve domain.");
            }

            int lower = upper;
            if (curve[upper].Date.Date != paymentDate.Date)
            {
                lower = Math.Max(0, upper - 1);
            }

            int[] curveIndices = lower == upper ? [lower] : [lower, upper];
            int[] publicDimensions = curveIndices
                .Where(curveIndex => curveIndex > 0)
                .Select(curveIndex => curveIndex - 1)
                .Distinct()
                .ToArray();
            double[] publicRateWeights = BuildPublicRateWeights(
                publicDimensions,
                lower,
                upper,
                Actual365(valuationDate, curve[lower].Date),
                Actual365(valuationDate, curve[upper].Date),
                paymentTime);

            return new DiscountKernelSpec(
                publicDimensions,
                publicRateWeights,
                paymentTime,
                Actual365(valuationDate, curve[lower].Date),
                Actual365(valuationDate, curve[upper].Date),
                curve[lower].ZeroRate,
                curve[upper].ZeroRate,
                LocalIndex(publicDimensions, lower),
                LocalIndex(publicDimensions, upper));
        }

        public double Discount(double[] localBumps)
        {
            double lowerRate = BumpedRate(_lowerBaseRate, _lowerLocalIndex, localBumps);
            double upperRate = BumpedRate(_upperBaseRate, _upperLocalIndex, localBumps);
            double zeroRate = Math.Abs(_upperTime - _lowerTime) < 1e-14
                ? upperRate
                : lowerRate + (upperRate - lowerRate) * ((_paymentTime - _lowerTime) / (_upperTime - _lowerTime));

            return Math.Exp(-zeroRate * _paymentTime);
        }

        public int FillFirstDerivatives(
            double discount,
            Span<int> outputDimensions,
            Span<double> derivatives)
        {
            for (int i = 0; i < PublicDimensions.Length; i++)
            {
                outputDimensions[i] = PublicDimensions[i];
                derivatives[i] = -_paymentTime * _publicRateWeights[i] * 1e-4 * discount;
            }

            return PublicDimensions.Length;
        }

        private static double[] BuildPublicRateWeights(
            int[] publicDimensions,
            int lowerCurveIndex,
            int upperCurveIndex,
            double lowerTime,
            double upperTime,
            double paymentTime)
        {
            var weights = new double[publicDimensions.Length];
            if (lowerCurveIndex == upperCurveIndex || Math.Abs(upperTime - lowerTime) < 1e-14)
            {
                AddPublicWeight(weights, publicDimensions, upperCurveIndex, 1.0);
                return weights;
            }

            double upperWeight = (paymentTime - lowerTime) / (upperTime - lowerTime);
            AddPublicWeight(weights, publicDimensions, lowerCurveIndex, 1.0 - upperWeight);
            AddPublicWeight(weights, publicDimensions, upperCurveIndex, upperWeight);
            return weights;
        }

        private static void AddPublicWeight(
            double[] weights,
            int[] publicDimensions,
            int curveIndex,
            double weight)
        {
            if (curveIndex == 0)
            {
                return;
            }

            int publicDimension = curveIndex - 1;
            for (int i = 0; i < publicDimensions.Length; i++)
            {
                if (publicDimensions[i] == publicDimension)
                {
                    weights[i] += weight;
                    return;
                }
            }
        }

        private static int LocalIndex(int[] publicDimensions, int curveIndex)
        {
            if (curveIndex == 0)
            {
                return -1;
            }

            int publicDimension = curveIndex - 1;
            for (int i = 0; i < publicDimensions.Length; i++)
            {
                if (publicDimensions[i] == publicDimension)
                {
                    return i;
                }
            }

            return -1;
        }

        private static double BumpedRate(double baseRate, int localIndex, double[] localBumps)
            => localIndex < 0 ? baseRate : baseRate + localBumps[localIndex] * 1e-4;

        private static double Actual365(DateTime valuationDate, DateTime paymentDate)
            => (paymentDate.Date - valuationDate.Date).TotalDays / 365.0;
    }
}
