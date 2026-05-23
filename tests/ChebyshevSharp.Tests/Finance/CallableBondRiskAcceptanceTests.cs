using CallableBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class CallableBondRiskAcceptanceTests
{
    [Fact]
    public void Default_validation_bank_preserves_65d_public_contract()
    {
        IReadOnlyList<CallableRiskValidationPoint> bank = CallableRiskAcceptance.BuildDefaultValidationBank();

        Assert.Equal(7, bank.Count);
        Assert.All(bank, point =>
        {
            Assert.False(string.IsNullOrWhiteSpace(point.Name));
            Assert.Equal(CallableBondFullDimensionalWrapper.DimensionCount, point.Coordinates.Length);
            Assert.All(point.Coordinates, value => Assert.True(double.IsFinite(value)));
        });
    }

    [Fact]
    public void Risk_metrics_are_zero_when_model_matches_baseline()
    {
        IReadOnlyList<CallableRiskValidationPoint> bank =
        [
            new(
                "linear-base",
                CallableRiskAcceptance.BuildCurveFactorPoint(
                    level: 0.0,
                    slope: 0.0,
                    curvature: 0.0,
                    coupon: 0.06,
                    maturityYears: 20.0,
                    firstCallYears: 5.0,
                    callPrice: 100.0,
                    sigma: 0.01)),
        ];

        double LinearFunction(double[] point)
            => point.Take(CallableBondFullDimensionalWrapper.CurveBumpCount).Sum() * 0.001
                + 100.0 * point[CallableRiskAcceptance.CouponDimension]
                + 0.25 * point[CallableRiskAcceptance.CallPriceDimension]
                - 20.0 * point[CallableRiskAcceptance.SigmaDimension];

        CallableRiskAcceptanceMetrics metrics = CallableRiskAcceptance.Summarize(
            LinearFunction,
            LinearFunction,
            bank);

        Assert.All(metrics.ScalarMetrics, metric =>
        {
            Assert.Equal(0.0, metric.MeanAbsoluteError, precision: 12);
            Assert.Equal(0.0, metric.MaxAbsoluteError, precision: 12);
        });
        Assert.All(metrics.VectorMetrics, metric =>
        {
            Assert.Equal(0.0, metric.MeanMaxComponentAbsoluteError, precision: 12);
            Assert.Equal(0.0, metric.MaxComponentAbsoluteError, precision: 12);
            Assert.Equal(0.0, metric.MeanL1RelativeError, precision: 12);
        });
    }

    [Fact]
    public void Anchored_hdmr_reports_full_pillar_component_counts()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(new QlNetCallableBondReferencePricer());
        CallableAnchoredHdmrSurrogate surrogate = CallableAnchoredHdmrSurrogate.Build(
            wrapper,
            oneDimensionalNodes: 3,
            pairNodes: 3);

        Assert.Equal(CallableBondFullDimensionalWrapper.DimensionCount, surrogate.OneDimensionalComponentCount);
        Assert.True(surrogate.PairComponentCount > CallableBondFullDimensionalWrapper.CurveBumpCount);
        Assert.True(surrogate.BuildEvaluations > 0);
        Assert.True(surrogate.BuildSeconds >= 0.0);
        Assert.True(double.IsFinite(surrogate.Eval(wrapper.CreateBasePoint())));
    }
}
