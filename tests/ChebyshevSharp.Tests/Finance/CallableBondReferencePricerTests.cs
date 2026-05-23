using CallableBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class CallableBondReferencePricerTests
{
    private static readonly ICallableBondReferencePricer Pricer = new QlNetCallableBondReferencePricer();

    [Fact]
    public void Price_returns_finite_outputs_for_callable_fixed_rate_bond()
    {
        CallableBondResult result = Pricer.Price(CallableBondScenarios.StandardThirtyYear());

        Assert.True(double.IsFinite(result.DirtyPrice));
        Assert.True(double.IsFinite(result.CleanPrice));
        Assert.True(double.IsFinite(result.AccruedAmount));
        Assert.True(double.IsFinite(result.NetPresentValue));
        Assert.True(double.IsFinite(result.StraightDirtyPrice));
        Assert.True(double.IsFinite(result.EmbeddedCallValue));
        Assert.True(result.CallabilityCount > 0);
    }

    [Fact]
    public void Callable_price_is_no_greater_than_equivalent_straight_bond()
    {
        CallableBondResult result = Pricer.Price(CallableBondScenarios.StandardThirtyYear());

        Assert.True(result.DirtyPrice <= result.StraightDirtyPrice + 1e-10);
        Assert.Equal(result.StraightDirtyPrice - result.DirtyPrice, result.EmbeddedCallValue, precision: 9);
    }

    [Fact]
    public void Higher_curve_level_lowers_callable_bond_price()
    {
        CallableBondRequest baseRequest = CallableBondScenarios.StandardThirtyYear();
        CallableBondRequest bumpedRequest = BumpCurve(baseRequest, 0.01);

        CallableBondResult baseResult = Pricer.Price(baseRequest);
        CallableBondResult bumpedResult = Pricer.Price(bumpedRequest);

        Assert.True(bumpedResult.DirtyPrice < baseResult.DirtyPrice);
    }

    [Fact]
    public void Higher_call_price_increases_callable_bond_price()
    {
        CallableBondResult lowCallPrice = Pricer.Price(CallableBondScenarios.StandardThirtyYear(callPrice: 100.0));
        CallableBondResult highCallPrice = Pricer.Price(CallableBondScenarios.StandardThirtyYear(callPrice: 105.0));

        Assert.True(highCallPrice.DirtyPrice > lowCallPrice.DirtyPrice);
        Assert.True(highCallPrice.EmbeddedCallValue < lowCallPrice.EmbeddedCallValue);
    }

    [Fact]
    public void Higher_hull_white_volatility_lowers_callable_bond_price()
    {
        CallableBondResult lowVolatility = Pricer.Price(CallableBondScenarios.StandardThirtyYear(hullWhiteSigma: 0.005));
        CallableBondResult highVolatility = Pricer.Price(CallableBondScenarios.StandardThirtyYear(hullWhiteSigma: 0.020));

        Assert.True(highVolatility.DirtyPrice < lowVolatility.DirtyPrice);
        Assert.True(highVolatility.EmbeddedCallValue > lowVolatility.EmbeddedCallValue);
    }

    [Fact]
    public void Tree_step_convergence_is_stable_for_reference_case()
    {
        CallableBondRequest baseRequest = CallableBondScenarios.StandardThirtyYear();

        CallableBondResult coarse = Pricer.Price(baseRequest with { TreeTimeSteps = 60 });
        CallableBondResult fine = Pricer.Price(baseRequest with { TreeTimeSteps = 120 });

        Assert.InRange(Math.Abs(fine.DirtyPrice - coarse.DirtyPrice), 0.0, 0.25);
    }

    private static CallableBondRequest BumpCurve(CallableBondRequest request, double rateBump)
        => request with
        {
            ZeroCurve = request.ZeroCurve
                .Select(pillar => pillar with { ZeroRate = pillar.ZeroRate + rateBump })
                .ToArray(),
        };
}

public sealed class CallableBondFullDimensionalWrapperTests
{
    private static readonly ICallableBondReferencePricer Pricer = new QlNetCallableBondReferencePricer();

    [Fact]
    public void Public_contract_is_sixty_five_dimensional()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(Pricer);

        Assert.Equal(60, CallableBondFullDimensionalWrapper.CurveBumpCount);
        Assert.Equal(65, CallableBondFullDimensionalWrapper.DimensionCount);
        Assert.Equal(65, wrapper.DimensionLabels.Count);
        Assert.Equal("curveBump_0.5Y_bp", wrapper.DimensionLabels[0]);
        Assert.Equal("curveBump_30Y_bp", wrapper.DimensionLabels[59]);
        Assert.Equal("coupon", wrapper.DimensionLabels[60]);
        Assert.Equal("maturityYears", wrapper.DimensionLabels[61]);
        Assert.Equal("firstCallYears", wrapper.DimensionLabels[62]);
        Assert.Equal("callPrice", wrapper.DimensionLabels[63]);
        Assert.Equal("hullWhiteSigma", wrapper.DimensionLabels[64]);
    }

    [Fact]
    public void Base_point_matches_reference_pricer()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(Pricer);
        double[] point = wrapper.CreateBasePoint();

        double wrappedPrice = wrapper.Price(point);
        CallableBondResult reference = Pricer.Price(wrapper.ToRequest(point));

        Assert.Equal(reference.DirtyPrice, wrappedPrice, precision: 10);
    }

    [Fact]
    public void Curve_bump_coordinates_are_basis_point_bumps_to_dense_pillars()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(Pricer);
        double[] point = wrapper.CreateBasePoint();
        point[19] = 25.0;

        CallableBondRequest bumped = wrapper.ToRequest(point);
        CallableBondRequest unbumped = wrapper.ToRequest(wrapper.CreateBasePoint());

        Assert.Equal(CallableBondFullDimensionalWrapper.DimensionCount, point.Length);
        Assert.Equal(
            unbumped.ZeroCurve[20].ZeroRate + 25.0e-4,
            bumped.ZeroCurve[20].ZeroRate,
            precision: 12);
    }

    [Fact]
    public void Wrong_dimension_count_is_rejected()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(Pricer);

        Assert.Throws<ArgumentException>(() => wrapper.ToRequest(new double[64]));
        Assert.Throws<ArgumentException>(() => wrapper.Price(new double[66]));
    }

    [Fact]
    public void Post_maturity_pillar_bump_does_not_change_shorter_callable_price()
    {
        var wrapper = CallableBondFullDimensionalWrapper.CreateDefault(Pricer);
        double[] point = wrapper.CreateBasePoint();
        point[61] = 20.0;
        point[62] = 5.0;

        double basePrice = wrapper.Price(point);
        double[] bumped = (double[])point.Clone();
        bumped[59] = 100.0;

        Assert.Equal(basePrice, wrapper.Price(bumped), precision: 8);
    }
}
