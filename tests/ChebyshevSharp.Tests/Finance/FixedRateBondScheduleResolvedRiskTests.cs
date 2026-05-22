using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondScheduleResolvedRiskTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();

    [Fact]
    public void Risk_snapshot_matches_finite_differences_of_schedule_resolved_model()
    {
        ScheduleResolvedCashflowChebyshevBondPricer model = BuildModel();
        double[] point = InteriorPoint();

        ScheduleResolvedRiskResult risk = model.EvalRisk(point);

        Assert.Equal(model.Eval(point), risk.DirtyPrice, precision: 12);
        Assert.Equal(ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount, risk.CurveBumpGradient.Length);
        Assert.Equal(ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount, risk.RateCouponMixed.Length);

        for (int dim = 0; dim < ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount; dim++)
        {
            double expected = FirstDerivative(model.Eval, point, dim, 1e-3);
            Assert.True(
                Math.Abs(risk.CurveBumpGradient[dim] - expected) < 1e-7,
                $"Curve bump gradient dim {dim}: expected {expected:E6}, actual {risk.CurveBumpGradient[dim]:E6}");

            double expectedMixed = MixedDerivative(
                model.Eval,
                point,
                dim,
                1e-3,
                ScheduleResolvedCashflowChebyshevBondPricer.CouponDimension,
                1e-4);
            Assert.True(
                Math.Abs(risk.RateCouponMixed[dim] - expectedMixed) < 1e-6,
                $"Rate-coupon mixed dim {dim}: expected {expectedMixed:E6}, actual {risk.RateCouponMixed[dim]:E6}");
        }

        double expectedCoupon = FirstDerivative(
            model.Eval,
            point,
            ScheduleResolvedCashflowChebyshevBondPricer.CouponDimension,
            1e-4);
        Assert.True(Math.Abs(risk.CouponDerivative - expectedCoupon) < 1e-7);
    }

    [Fact]
    public void Risk_snapshot_span_overload_reuses_and_clears_buffers()
    {
        ScheduleResolvedCashflowChebyshevBondPricer model = BuildModel();
        double[] point = InteriorPoint();
        var curveGradient = Enumerable.Repeat(123.0, ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount).ToArray();
        var rateCouponMixed = Enumerable.Repeat(456.0, ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount).ToArray();

        double dirtyPrice = model.EvalRisk(point, curveGradient, rateCouponMixed, out double couponDerivative);

        Assert.Equal(model.Eval(point), dirtyPrice, precision: 12);
        Assert.True(double.IsFinite(couponDerivative));

        int firstUnsupportedPostMaturityPillar = 21;
        for (int dim = firstUnsupportedPostMaturityPillar; dim < curveGradient.Length; dim++)
        {
            Assert.True(Math.Abs(curveGradient[dim]) < 1e-12, $"Expected post-maturity curve gradient dim {dim} to be cleared.");
            Assert.True(Math.Abs(rateCouponMixed[dim]) < 1e-12, $"Expected post-maturity mixed dim {dim} to be cleared.");
        }

        Assert.Throws<ArgumentException>(() =>
            model.EvalRisk(point, curveGradient.AsSpan(0, 59), rateCouponMixed, out _));
        Assert.Throws<ArgumentException>(() =>
            model.EvalRisk(point, curveGradient, rateCouponMixed.AsSpan(0, 59), out _));
    }

    private static ScheduleResolvedCashflowChebyshevBondPricer BuildModel()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        return new ScheduleResolvedCashflowChebyshevBondPricer(Pricer, baseRequest);
    }

    private static double[] InteriorPoint()
    {
        var point = new double[ScheduleResolvedCashflowChebyshevBondPricer.PublicInputDimensionCount];
        for (int i = 0; i < ScheduleResolvedCashflowChebyshevBondPricer.CurveBumpDimensionCount; i++)
        {
            point[i] = 65.0 * Math.Sin((i + 1) * Math.PI / 13.0);
        }

        point[ScheduleResolvedCashflowChebyshevBondPricer.CouponDimension] = 0.0575;
        point[ScheduleResolvedCashflowChebyshevBondPricer.MaturityDimension] = 10.25;
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

    private static double MixedDerivative(
        Func<double[], double> function,
        double[] point,
        int firstDimension,
        double firstStep,
        int secondDimension,
        double secondStep)
    {
        double[] plusPlus = Shift(Shift(point, firstDimension, firstStep), secondDimension, secondStep);
        double[] plusMinus = Shift(Shift(point, firstDimension, firstStep), secondDimension, -secondStep);
        double[] minusPlus = Shift(Shift(point, firstDimension, -firstStep), secondDimension, secondStep);
        double[] minusMinus = Shift(Shift(point, firstDimension, -firstStep), secondDimension, -secondStep);

        return (function(plusPlus) - function(plusMinus) - function(minusPlus) + function(minusMinus)) /
            (4.0 * firstStep * secondStep);
    }

    private static double[] Shift(double[] point, int dimension, double delta)
    {
        double[] copy = (double[])point.Clone();
        copy[dimension] += delta;
        return copy;
    }
}
