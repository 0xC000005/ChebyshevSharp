using FixedRateBondSurrogate;
using QLNet;
using QDate = QLNet.Date;

namespace CallableBondSurrogate;

public sealed class ReferenceSemanticsCallableTreePricer
{
    private const double TimeTolerance = 1e-10;
    private const double ExerciseBoundarySmoothing = 0.05;

    private readonly CallableBondFullDimensionalWrapper _wrapper;
    private readonly int? _treeTimeStepsOverride;
    private readonly Dictionary<ProductKey, ProductData> _cache = [];
    private readonly object _cacheLock = new();

    public ReferenceSemanticsCallableTreePricer(
        CallableBondFullDimensionalWrapper wrapper,
        int? treeTimeStepsOverride = null)
    {
        _wrapper = wrapper;
        _treeTimeStepsOverride = treeTimeStepsOverride;
    }

    public double Price(double[] fullPoint)
    {
        CallableBondRequest request = ToRequest(fullPoint);
        ProductData product = GetProductData(request);
        double[] phi = CalibratePhi(request, product);

        int lastIndex = product.Grid.Times.Length - 1;
        double[] values = Enumerable.Repeat(product.Redemption, product.Tree.Size(lastIndex)).ToArray();
        ApplyAdjustments(values, product.Grid.Times[lastIndex], product.Inputs, request);

        for (int step = lastIndex - 1; step >= 0; step--)
        {
            double[] previous = new double[product.Tree.Size(step)];
            double dt = product.Grid.Dt[step];
            for (int j = 0; j < previous.Length; j++)
            {
                double continuation = 0.0;
                for (int branch = 0; branch < 3; branch++)
                {
                    continuation += product.Tree.Probability(step, j, branch)
                        * values[product.Tree.Descendant(step, j, branch)];
                }

                double shortRate = product.Tree.Underlying(step, j) + phi[step];
                previous[j] = continuation * Math.Exp(-shortRate * dt);
            }

            values = previous;
            ApplyAdjustments(values, product.Grid.Times[step], product.Inputs, request);
        }

        return values[0] / request.Notional * 100.0;
    }

    public double[] FullDv01Vector(double[] fullPoint)
    {
        CallableBondRequest request = ToRequest(fullPoint);
        ProductData product = GetProductData(request);
        PhiDerivatives calibrated = CalibratePhiDerivatives(request, product);

        int lastIndex = product.Grid.Times.Length - 1;
        double[] values = Enumerable.Repeat(product.Redemption, product.Tree.Size(lastIndex)).ToArray();
        double[][] derivatives = ZeroDerivatives(values.Length);
        ApplyAdjustments(values, derivatives, product.Grid.Times[lastIndex], product.Inputs, request);

        for (int step = lastIndex - 1; step >= 0; step--)
        {
            double[] previous = new double[product.Tree.Size(step)];
            double[][] previousDerivatives = ZeroDerivatives(previous.Length);
            double dt = product.Grid.Dt[step];
            for (int j = 0; j < previous.Length; j++)
            {
                double continuation = 0.0;
                var continuationDerivative = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
                for (int branch = 0; branch < 3; branch++)
                {
                    int descendant = product.Tree.Descendant(step, j, branch);
                    double probability = product.Tree.Probability(step, j, branch);
                    continuation += probability * values[descendant];
                    for (int p = 0; p < continuationDerivative.Length; p++)
                    {
                        continuationDerivative[p] += probability * derivatives[descendant][p];
                    }
                }

                double shortRate = product.Tree.Underlying(step, j) + calibrated.Phi[step];
                double discount = Math.Exp(-shortRate * dt);
                previous[j] = continuation * discount;
                for (int p = 0; p < continuationDerivative.Length; p++)
                {
                    previousDerivatives[j][p] =
                        (continuationDerivative[p] * discount)
                        - (continuation * discount * dt * calibrated.DPhiByBump[step][p]);
                }
            }

            values = previous;
            derivatives = previousDerivatives;
            ApplyAdjustments(values, derivatives, product.Grid.Times[step], product.Inputs, request);
        }

        double scale = 100.0 / request.Notional;
        double[] result = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
        for (int p = 0; p < result.Length; p++)
        {
            result[p] = derivatives[0][p] * scale;
        }

        return result;
    }

    public double[] HybridFullDv01Vector(double[] fullPoint, int correctionCount = 16)
    {
        double[] vector = FullDv01Vector(fullPoint);
        int[] correctionIndices = vector
            .Select((value, index) => (Magnitude: Math.Abs(value), Index: index))
            .OrderByDescending(item => item.Magnitude)
            .Take(Math.Clamp(correctionCount, 0, CallableBondFullDimensionalWrapper.CurveBumpCount))
            .Select(item => item.Index)
            .ToArray();

        Parallel.ForEach(correctionIndices, index =>
        {
            double[] down = Shift(fullPoint, index, -1.0);
            double[] up = Shift(fullPoint, index, 1.0);
            vector[index] = (Price(up) - Price(down)) / 2.0;
        });

        return vector;
    }

    private static double[] Shift(double[] point, int dimension, double amount)
    {
        double[] shifted = (double[])point.Clone();
        shifted[dimension] += amount;
        return shifted;
    }

    private ProductData GetProductData(CallableBondRequest request)
    {
        var key = new ProductKey(
            request.ValuationDate.Date,
            request.EffectiveDate.Date,
            request.MaturityDate.Date,
            request.FirstCallDate.Date,
            request.Coupon,
            request.Notional,
            request.CallPrice,
            request.HullWhiteMeanReversion,
            request.HullWhiteSigma,
            request.TreeTimeSteps,
            request.SettlementDays);

        lock (_cacheLock)
        {
            if (_cache.TryGetValue(key, out ProductData? cached))
            {
                return cached;
            }
        }

        CallableBond.Arguments arguments = BuildArguments(request);
        TreeInputs inputs = BuildTreeInputs(request, arguments);
        TimeGridData grid = TimeGridData.Build(inputs.MandatoryTimes, request.TreeTimeSteps);
        TrinomialTreeData tree = TrinomialTreeData.Build(grid, request.HullWhiteMeanReversion, request.HullWhiteSigma);
        CurveInterpolation[] curveInterpolations = CurveInterpolation.Build(request, grid.Times);
        var product = new ProductData(arguments.redemption, inputs, grid, tree, curveInterpolations);

        lock (_cacheLock)
        {
            _cache[key] = product;
        }

        return product;
    }

    private CallableBondRequest ToRequest(double[] fullPoint)
    {
        CallableBondRequest request = _wrapper.ToRequest(fullPoint);
        return _treeTimeStepsOverride is null
            ? request
            : request with { TreeTimeSteps = _treeTimeStepsOverride.Value };
    }

    private static CallableBond.Arguments BuildArguments(CallableBondRequest request)
    {
        QDate valuationDate = ToQlDate(request.ValuationDate);
        QDate effectiveDate = ToQlDate(request.EffectiveDate);
        QDate maturityDate = ToQlDate(request.MaturityDate);

        Settings.setEvaluationDate(valuationDate);

        Calendar calendar = new UnitedStates(UnitedStates.Market.GovernmentBond);
        DayCounter couponDayCounter = new Thirty360(Thirty360.Thirty360Convention.USA);

        var schedule = new Schedule(
            effectiveDate,
            maturityDate,
            new Period(Frequency.Semiannual),
            calendar,
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.ModifiedFollowing,
            DateGeneration.Rule.Backward,
            false);

        CallabilitySchedule callability = BuildCallabilitySchedule(request, calendar);
        var callableBond = new CallableFixedRateBond(
            request.SettlementDays,
            request.Notional,
            schedule,
            [request.Coupon],
            couponDayCounter,
            BusinessDayConvention.ModifiedFollowing,
            redemption: 100.0,
            issueDate: effectiveDate,
            putCallSchedule: callability);

        var arguments = new CallableBond.Arguments();
        callableBond.setupArguments(arguments);
        arguments.validate();
        return arguments;
    }

    private static CallabilitySchedule BuildCallabilitySchedule(
        CallableBondRequest request,
        Calendar calendar)
    {
        var schedule = new CallabilitySchedule();
        var price = new Bond.Price(request.CallPrice, Bond.Price.Type.Clean);

        DateTime callDate = request.FirstCallDate.Date;
        while (callDate < request.MaturityDate.Date)
        {
            QDate adjusted = calendar.adjust(ToQlDate(callDate), BusinessDayConvention.ModifiedFollowing);
            if (adjusted < ToQlDate(request.MaturityDate))
            {
                schedule.Add(new Callability(price, Callability.Type.Call, adjusted));
            }

            callDate = callDate.AddMonths(6);
        }

        return schedule;
    }

    private static TreeInputs BuildTreeInputs(
        CallableBondRequest request,
        CallableBond.Arguments arguments)
    {
        var couponEvents = new List<CouponEvent>();
        var callEvents = new List<CallEvent>();
        var mandatoryTimes = new List<double>();
        var couponAdjustments = Enumerable.Repeat(CouponAdjustment.Post, arguments.couponDates.Count).ToArray();
        double[] couponTimes = new double[arguments.couponDates.Count];
        DateTime[] couponDates = new DateTime[arguments.couponDates.Count];

        double redemptionTime = YearFraction(request.ValuationDate, arguments.redemptionDate.ToDateTime());
        if (redemptionTime >= 0.0)
        {
            mandatoryTimes.Add(redemptionTime);
        }

        for (int i = 0; i < arguments.couponDates.Count; i++)
        {
            DateTime date = arguments.couponDates[i].ToDateTime().Date;
            double time = YearFraction(request.ValuationDate, date);
            couponDates[i] = date;
            couponTimes[i] = time;
            if (time >= 0.0)
            {
                mandatoryTimes.Add(time);
            }
        }

        for (int i = 0; i < arguments.callabilityDates.Count; i++)
        {
            DateTime callDate = arguments.callabilityDates[i].ToDateTime().Date;
            double callabilityTime = YearFraction(request.ValuationDate, callDate);
            double callPrice = arguments.callabilityPrices[i] * arguments.faceAmount / 100.0;
            DateTime? couponDateForPriceAdjustment = null;

            for (int j = 0; j < couponTimes.Length; j++)
            {
                if (WithinNextWeek(callabilityTime, couponTimes[j]) && callDate < couponDates[j])
                {
                    callabilityTime = couponTimes[j];
                    couponAdjustments[j] = CouponAdjustment.Pre;
                    couponDateForPriceAdjustment = couponDates[j];
                    break;
                }
            }

            if (callabilityTime >= 0.0)
            {
                mandatoryTimes.Add(callabilityTime);
            }

            Callability callability = arguments.putCallSchedule[i];
            callEvents.Add(new CallEvent(
                callabilityTime,
                callPrice,
                callability.type(),
                callDate,
                couponDateForPriceAdjustment));
        }

        for (int i = 0; i < couponTimes.Length; i++)
        {
            couponEvents.Add(new CouponEvent(couponTimes[i], arguments.couponAmounts[i], couponAdjustments[i]));
        }

        return new TreeInputs(
            mandatoryTimes,
            couponEvents,
            callEvents);
    }

    private static double[] CalibratePhi(
        CallableBondRequest request,
        ProductData product)
    {
        TimeGridData grid = product.Grid;
        TrinomialTreeData tree = product.Tree;
        var phi = new double[grid.Times.Length - 1];
        var statePrices = new double[] { 1.0 };

        for (int step = 0; step < phi.Length; step++)
        {
            double dt = grid.Dt[step];
            double value = 0.0;
            for (int j = 0; j < statePrices.Length; j++)
            {
                value += statePrices[j] * Math.Exp(-tree.Underlying(step, j) * dt);
            }

            double discountBond = DiscountFactor(request, product.CurveInterpolations[step + 1]);
            phi[step] = Math.Log(value / discountBond) / dt;

            var nextStatePrices = new double[tree.Size(step + 1)];
            for (int j = 0; j < statePrices.Length; j++)
            {
                double shortRate = tree.Underlying(step, j) + phi[step];
                double discount = Math.Exp(-shortRate * dt);
                for (int branch = 0; branch < 3; branch++)
                {
                    nextStatePrices[tree.Descendant(step, j, branch)] +=
                        statePrices[j] * discount * tree.Probability(step, j, branch);
                }
            }

            statePrices = nextStatePrices;
        }

        return phi;
    }

    private static PhiDerivatives CalibratePhiDerivatives(
        CallableBondRequest request,
        ProductData product)
    {
        TimeGridData grid = product.Grid;
        TrinomialTreeData tree = product.Tree;
        var phi = new double[grid.Times.Length - 1];
        double[][] dPhiByBump = ZeroDerivatives(phi.Length);
        var statePrices = new double[] { 1.0 };
        double[][] stateDerivatives = ZeroDerivatives(1);

        for (int step = 0; step < phi.Length; step++)
        {
            double dt = grid.Dt[step];
            double value = 0.0;
            var dValue = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
            for (int j = 0; j < statePrices.Length; j++)
            {
                double stateDiscount = Math.Exp(-tree.Underlying(step, j) * dt);
                value += statePrices[j] * stateDiscount;
                for (int p = 0; p < dValue.Length; p++)
                {
                    dValue[p] += stateDerivatives[j][p] * stateDiscount;
                }
            }

            double discountBond = DiscountFactor(request, product.CurveInterpolations[step + 1]);
            phi[step] = Math.Log(value / discountBond) / dt;
            for (int p = 0; p < dValue.Length; p++)
            {
                double dDiscountBond = DiscountFactorDerivativeByBump(
                    request,
                    product.CurveInterpolations[step + 1],
                    p);
                dPhiByBump[step][p] = ((dValue[p] / value) - (dDiscountBond / discountBond)) / dt;
            }

            var nextStatePrices = new double[tree.Size(step + 1)];
            double[][] nextStateDerivatives = ZeroDerivatives(nextStatePrices.Length);
            for (int j = 0; j < statePrices.Length; j++)
            {
                double shortRate = tree.Underlying(step, j) + phi[step];
                double discount = Math.Exp(-shortRate * dt);
                for (int branch = 0; branch < 3; branch++)
                {
                    int descendant = tree.Descendant(step, j, branch);
                    double probability = tree.Probability(step, j, branch);
                    nextStatePrices[descendant] += statePrices[j] * discount * probability;
                    for (int p = 0; p < dValue.Length; p++)
                    {
                        nextStateDerivatives[descendant][p] +=
                            ((stateDerivatives[j][p] * discount)
                            - (statePrices[j] * discount * dt * dPhiByBump[step][p]))
                            * probability;
                    }
                }
            }

            statePrices = nextStatePrices;
            stateDerivatives = nextStateDerivatives;
        }

        return new PhiDerivatives(phi, dPhiByBump);
    }

    private static void ApplyAdjustments(
        double[] values,
        double time,
        TreeInputs inputs,
        CallableBondRequest request)
    {
        foreach (CouponEvent coupon in inputs.Coupons)
        {
            if (coupon.Adjustment == CouponAdjustment.Pre && Close(coupon.Time, time))
            {
                AddCoupon(values, coupon.Amount);
            }
        }

        foreach (CallEvent call in inputs.Calls)
        {
            if (Close(call.Time, time))
            {
                ApplyCallability(values, call.Price(request), call.Type);
            }
        }

        foreach (CouponEvent coupon in inputs.Coupons)
        {
            if (coupon.Adjustment == CouponAdjustment.Post && Close(coupon.Time, time))
            {
                AddCoupon(values, coupon.Amount);
            }
        }
    }

    private static void ApplyAdjustments(
        double[] values,
        double[][] derivatives,
        double time,
        TreeInputs inputs,
        CallableBondRequest request)
    {
        foreach (CouponEvent coupon in inputs.Coupons)
        {
            if (coupon.Adjustment == CouponAdjustment.Pre && Close(coupon.Time, time))
            {
                AddCoupon(values, coupon.Amount);
            }
        }

        foreach (CallEvent call in inputs.Calls)
        {
            if (Close(call.Time, time))
            {
                ApplyCallability(values, derivatives, call, request);
            }
        }

        foreach (CouponEvent coupon in inputs.Coupons)
        {
            if (coupon.Adjustment == CouponAdjustment.Post && Close(coupon.Time, time))
            {
                AddCoupon(values, coupon.Amount);
            }
        }
    }

    private static void AddCoupon(double[] values, double amount)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += amount;
        }
    }

    private static void ApplyCallability(
        double[] values,
        double price,
        Callability.Type type)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = type switch
            {
                Callability.Type.Call => Math.Min(price, values[i]),
                Callability.Type.Put => Math.Max(price, values[i]),
                _ => throw new NotSupportedException("Only call and put callability types are supported."),
            };
        }
    }

    private static void ApplyCallability(
        double[] values,
        double[][] derivatives,
        CallEvent call,
        CallableBondRequest request)
    {
        double price = call.Price(request);
        double[] priceDerivatives = call.PriceDerivativesByBump(request);
        for (int i = 0; i < values.Length; i++)
        {
            switch (call.Type)
            {
                case Callability.Type.Call:
                    BlendDerivatives(
                        derivatives[i],
                        priceDerivatives,
                        SmoothCallDerivativeWeight(values[i], price));
                    if (price < values[i])
                    {
                        values[i] = price;
                    }
                    break;
                case Callability.Type.Put:
                    BlendDerivatives(
                        derivatives[i],
                        priceDerivatives,
                        SmoothPutDerivativeWeight(values[i], price));
                    if (price > values[i])
                    {
                        values[i] = price;
                    }
                    break;
                default:
                    throw new NotSupportedException("Only call and put callability types are supported.");
            }
        }
    }

    private static double SmoothCallDerivativeWeight(double continuation, double callPrice)
    {
        double diff = continuation - callPrice;
        return 0.5 * (1.0 - diff / Math.Sqrt((diff * diff) + (ExerciseBoundarySmoothing * ExerciseBoundarySmoothing)));
    }

    private static double SmoothPutDerivativeWeight(double continuation, double putPrice)
    {
        double diff = continuation - putPrice;
        return 0.5 * (1.0 + diff / Math.Sqrt((diff * diff) + (ExerciseBoundarySmoothing * ExerciseBoundarySmoothing)));
    }

    private static void BlendDerivatives(
        double[] derivatives,
        IReadOnlyList<double> priceDerivatives,
        double continuationWeight)
    {
        for (int i = 0; i < derivatives.Length; i++)
        {
            derivatives[i] =
                (continuationWeight * derivatives[i])
                + ((1.0 - continuationWeight) * priceDerivatives[i]);
        }
    }

    private static double[][] ZeroDerivatives(int valueCount)
    {
        var derivatives = new double[valueCount][];
        for (int i = 0; i < derivatives.Length; i++)
        {
            derivatives[i] = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
        }

        return derivatives;
    }

    private static double DiscountFactor(CallableBondRequest request, DateTime date)
        => DiscountFactor(request, YearFraction(request.ValuationDate, date));

    private static double DiscountFactorDerivativeByBump(
        CallableBondRequest request,
        DateTime date,
        int bumpIndex)
        => DiscountFactorDerivativeByBump(
            request,
            CurveInterpolation.At(request, YearFraction(request.ValuationDate, date)),
            bumpIndex);

    private static double DiscountFactor(CallableBondRequest request, double time)
        => Math.Exp(-ZeroRateAtTime(request, time) * time);

    private static double DiscountFactor(
        CallableBondRequest request,
        CurveInterpolation interpolation)
        => Math.Exp(-ZeroRateAt(request, interpolation) * interpolation.Time);

    private static double DiscountFactorDerivativeByBump(
        CallableBondRequest request,
        CurveInterpolation interpolation,
        int bumpIndex)
    {
        double dz = ZeroRateDerivativeByBump(interpolation, bumpIndex);
        return -interpolation.Time * DiscountFactor(request, interpolation) * dz;
    }

    private static double ZeroRateAtTime(CallableBondRequest request, double targetTime)
    {
        IReadOnlyList<ZeroRatePillar> curve = request.ZeroCurve;
        if (targetTime <= 0.0)
        {
            return curve[0].ZeroRate;
        }

        for (int i = 1; i < curve.Count; i++)
        {
            double rightTime = YearFraction(request.ValuationDate, curve[i].Date);
            if (targetTime <= rightTime)
            {
                double leftTime = YearFraction(request.ValuationDate, curve[i - 1].Date);
                double width = Math.Max(rightTime - leftTime, 1e-12);
                double weight = (targetTime - leftTime) / width;
                return ((1.0 - weight) * curve[i - 1].ZeroRate) + (weight * curve[i].ZeroRate);
            }
        }

        return curve[^1].ZeroRate;
    }

    private static double ZeroRateAt(
        CallableBondRequest request,
        CurveInterpolation interpolation)
    {
        if (interpolation.LeftIndex == interpolation.RightIndex)
        {
            return request.ZeroCurve[interpolation.LeftIndex].ZeroRate;
        }

        double left = request.ZeroCurve[interpolation.LeftIndex].ZeroRate;
        double right = request.ZeroCurve[interpolation.RightIndex].ZeroRate;
        return ((1.0 - interpolation.RightWeight) * left) + (interpolation.RightWeight * right);
    }

    private static double ZeroRateDerivativeByBump(
        CurveInterpolation interpolation,
        int bumpIndex)
    {
        if (interpolation.Time <= 0.0)
        {
            return 0.0;
        }

        if (interpolation.LeftIndex == interpolation.RightIndex)
        {
            int onlyBumpIndex = interpolation.LeftIndex - 1;
            return onlyBumpIndex == bumpIndex ? 1e-4 : 0.0;
        }

        double derivative = 0.0;
        int leftBumpIndex = interpolation.LeftIndex - 1;
        int rightBumpIndex = interpolation.RightIndex - 1;
        if (leftBumpIndex == bumpIndex)
        {
            derivative += (1.0 - interpolation.RightWeight) * 1e-4;
        }

        if (rightBumpIndex == bumpIndex)
        {
            derivative += interpolation.RightWeight * 1e-4;
        }

        return derivative;
    }

    private static double YearFraction(DateTime valuationDate, DateTime date)
        => (date.Date - valuationDate.Date).TotalDays / 365.0;

    private static bool WithinNextWeek(double t1, double t2)
        => t1 <= t2 && t2 <= t1 + (1.0 / 52.0);

    private static bool Close(double left, double right)
        => Math.Abs(left - right) <= TimeTolerance * Math.Max(1.0, Math.Max(Math.Abs(left), Math.Abs(right)));

    private static QDate ToQlDate(DateTime date)
        => new(date.Day, date.Month, date.Year);

    private enum CouponAdjustment
    {
        Pre,
        Post,
    }

    private sealed record ProductKey(
        DateTime ValuationDate,
        DateTime EffectiveDate,
        DateTime MaturityDate,
        DateTime FirstCallDate,
        double Coupon,
        double Notional,
        double CallPrice,
        double MeanReversion,
        double Sigma,
        int TreeTimeSteps,
        int SettlementDays);

    private sealed record ProductData(
        double Redemption,
        TreeInputs Inputs,
        TimeGridData Grid,
        TrinomialTreeData Tree,
        CurveInterpolation[] CurveInterpolations);

    private sealed record PhiDerivatives(
        double[] Phi,
        double[][] DPhiByBump);

    private sealed record CurveInterpolation(
        double Time,
        int LeftIndex,
        int RightIndex,
        double RightWeight)
    {
        public static CurveInterpolation[] Build(
            CallableBondRequest request,
            IReadOnlyList<double> times)
        {
            var interpolations = new CurveInterpolation[times.Count];
            for (int i = 0; i < interpolations.Length; i++)
            {
                interpolations[i] = At(request, times[i]);
            }

            return interpolations;
        }

        public static CurveInterpolation At(
            CallableBondRequest request,
            double targetTime)
        {
            IReadOnlyList<ZeroRatePillar> curve = request.ZeroCurve;
            if (targetTime <= 0.0)
            {
                return new CurveInterpolation(targetTime, 0, 0, 0.0);
            }

            for (int i = 1; i < curve.Count; i++)
            {
                double rightTime = YearFraction(request.ValuationDate, curve[i].Date);
                if (targetTime <= rightTime)
                {
                    double leftTime = YearFraction(request.ValuationDate, curve[i - 1].Date);
                    double width = Math.Max(rightTime - leftTime, 1e-12);
                    double weight = (targetTime - leftTime) / width;
                    return new CurveInterpolation(targetTime, i - 1, i, weight);
                }
            }

            int last = curve.Count - 1;
            return new CurveInterpolation(targetTime, last, last, 0.0);
        }
    }

    private sealed record TreeInputs(
        IReadOnlyList<double> MandatoryTimes,
        IReadOnlyList<CouponEvent> Coupons,
        IReadOnlyList<CallEvent> Calls);

    private sealed record CouponEvent(
        double Time,
        double Amount,
        CouponAdjustment Adjustment);

    private sealed record CallEvent(
        double Time,
        double BasePrice,
        Callability.Type Type,
        DateTime CallDate,
        DateTime? CouponDateForPriceAdjustment)
    {
        public double Price(CallableBondRequest request)
        {
            if (CouponDateForPriceAdjustment is not { } couponDate)
            {
                return BasePrice;
            }

            return BasePrice * DiscountFactor(request, CallDate) / DiscountFactor(request, couponDate);
        }

        public double[] PriceDerivativesByBump(CallableBondRequest request)
        {
            var derivatives = new double[CallableBondFullDimensionalWrapper.CurveBumpCount];
            if (CouponDateForPriceAdjustment is not { } couponDate)
            {
                return derivatives;
            }

            double callDiscount = DiscountFactor(request, CallDate);
            double couponDiscount = DiscountFactor(request, couponDate);
            double adjustedPrice = BasePrice * callDiscount / couponDiscount;
            for (int p = 0; p < derivatives.Length; p++)
            {
                double dCallDiscount = DiscountFactorDerivativeByBump(request, CallDate, p);
                double dCouponDiscount = DiscountFactorDerivativeByBump(request, couponDate, p);
                derivatives[p] = adjustedPrice
                    * ((dCallDiscount / callDiscount) - (dCouponDiscount / couponDiscount));
            }

            return derivatives;
        }
    }

    private sealed class TimeGridData
    {
        private TimeGridData(double[] times, double[] dt)
        {
            Times = times;
            Dt = dt;
        }

        public double[] Times { get; }
        public double[] Dt { get; }

        public static TimeGridData Build(
            IReadOnlyList<double> mandatoryTimes,
            int steps)
        {
            if (mandatoryTimes.Count == 0)
            {
                throw new ArgumentException("At least one mandatory time is required.", nameof(mandatoryTimes));
            }

            List<double> mandatory = mandatoryTimes
                .Order()
                .Where(time => time >= 0.0)
                .ToList();
            for (int i = 0; i < mandatory.Count - 1; i++)
            {
                if (Close(mandatory[i], mandatory[i + 1]))
                {
                    mandatory.RemoveAt(i);
                    i--;
                }
            }

            double last = mandatory[^1];
            double dtMax = last / steps;
            var times = new List<double> { 0.0 };
            double periodBegin = 0.0;
            foreach (double periodEnd in mandatory)
            {
                if (!Close(periodEnd, 0.0))
                {
                    int nSteps = (int)(((periodEnd - periodBegin) / dtMax) + 0.5);
                    nSteps = nSteps != 0 ? nSteps : 1;
                    double dt = (periodEnd - periodBegin) / nSteps;
                    for (int n = 1; n <= nSteps; n++)
                    {
                        times.Add(periodBegin + n * dt);
                    }
                }

                periodBegin = periodEnd;
            }

            double[] timeArray = times.ToArray();
            var dtArray = new double[timeArray.Length - 1];
            for (int i = 0; i < dtArray.Length; i++)
            {
                dtArray[i] = timeArray[i + 1] - timeArray[i];
            }

            return new TimeGridData(timeArray, dtArray);
        }
    }

    private sealed class TrinomialTreeData
    {
        private readonly BranchingData[] _branchings;
        private readonly double[] _dx;

        private TrinomialTreeData(BranchingData[] branchings, double[] dx)
        {
            _branchings = branchings;
            _dx = dx;
        }

        public static TrinomialTreeData Build(
            TimeGridData grid,
            double meanReversion,
            double sigma)
        {
            int nTimeSteps = grid.Times.Length - 1;
            var branchings = new BranchingData[nTimeSteps];
            var dx = new double[nTimeSteps + 1];
            int jMin = 0;
            int jMax = 0;

            for (int step = 0; step < nTimeSteps; step++)
            {
                double dt = grid.Dt[step];
                double variance = sigma * sigma * (1.0 - Math.Exp(-2.0 * meanReversion * dt)) / (2.0 * meanReversion);
                double std = Math.Sqrt(variance);
                dx[step + 1] = std * Math.Sqrt(3.0);

                var branching = new BranchingBuilder();
                for (int j = jMin; j <= jMax; j++)
                {
                    double x = j * dx[step];
                    double expectation = x * Math.Exp(-meanReversion * dt);
                    int k = (int)Math.Floor((expectation / dx[step + 1]) + 0.5);
                    double e = expectation - (k * dx[step + 1]);
                    double e2 = e * e;
                    double e3 = e * Math.Sqrt(3.0);

                    double p1 = (1.0 + (e2 / variance) - (e3 / std)) / 6.0;
                    double p2 = (2.0 - (e2 / variance)) / 3.0;
                    double p3 = (1.0 + (e2 / variance) + (e3 / std)) / 6.0;

                    branching.Add(k, p1, p2, p3);
                }

                BranchingData data = branching.Build();
                branchings[step] = data;
                jMin = data.JMin;
                jMax = data.JMax;
            }

            return new TrinomialTreeData(branchings, dx);
        }

        public int Size(int step)
            => step == 0 ? 1 : _branchings[step - 1].Size;

        public double Underlying(int step, int index)
            => step == 0 ? 0.0 : (_branchings[step - 1].JMin + index) * _dx[step];

        public int Descendant(int step, int index, int branch)
            => _branchings[step].K[index] - _branchings[step].JMin - 1 + branch;

        public double Probability(int step, int index, int branch)
            => _branchings[step].Probabilities[branch][index];
    }

    private sealed class BranchingBuilder
    {
        private readonly List<int> _k = [];
        private readonly List<double>[] _probabilities = [[], [], []];
        private int _kMin = int.MaxValue;
        private int _kMax = int.MinValue;

        public void Add(int k, double p1, double p2, double p3)
        {
            _k.Add(k);
            _probabilities[0].Add(p1);
            _probabilities[1].Add(p2);
            _probabilities[2].Add(p3);
            _kMin = Math.Min(_kMin, k);
            _kMax = Math.Max(_kMax, k);
        }

        public BranchingData Build()
        {
            int jMin = _kMin - 1;
            int jMax = _kMax + 1;
            return new BranchingData(
                _k.ToArray(),
                _probabilities.Select(values => values.ToArray()).ToArray(),
                jMin,
                jMax);
        }
    }

    private sealed record BranchingData(
        int[] K,
        double[][] Probabilities,
        int JMin,
        int JMax)
    {
        public int Size => JMax - JMin + 1;
    }
}
