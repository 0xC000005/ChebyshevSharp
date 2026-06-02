using System.Reflection;
using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

public sealed class AmericanOptionDynamicChebyshevTests
{
    private static readonly IAmericanOptionReferencePricer ReferencePricer = new QlNetAmericanOptionReferencePricer();

    [Fact]
    public void Dynamic_chebyshev_price_is_close_to_qlnet_reference_for_standard_put()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        AmericanOptionResult reference = ReferencePricer.Price(request);
        var settings = new DynamicChebyshevSettings(
            ExerciseSteps: 80,
            SpotNodeCount: 81,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevResult result = pricer.Price(request, settings);

        Assert.True(double.IsFinite(result.Price));
        Assert.InRange(Math.Abs(result.Price - reference.Price), 0.0, 0.15);
        Assert.InRange(result.Delta, -1.0, 0.0);
        Assert.True(result.Gamma > 0.0);
        Assert.True(result.BuildEvaluations > 0);
        Assert.True(result.BuildTimeSeconds >= 0.0);
        Assert.Equal(settings.ExerciseSteps, result.ExerciseSteps);
        Assert.Equal(settings.SpotNodeCount, result.SpotNodeCount);
        Assert.Equal(settings.QuadratureOrder, result.QuadratureOrder);
    }

    [Fact]
    public void Dynamic_chebyshev_is_reproducible_for_fixed_inputs()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new DynamicChebyshevSettings(
            ExerciseSteps: 40,
            SpotNodeCount: 61,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevResult first = pricer.Price(request, settings);
        DynamicChebyshevResult second = pricer.Price(request, settings);

        Assert.Equal(first.Price, second.Price, precision: 12);
        Assert.Equal(first.Delta, second.Delta, precision: 12);
        Assert.Equal(first.Gamma, second.Gamma, precision: 12);
    }

    [Fact]
    public void Dynamic_chebyshev_model_reuses_build_for_spot_grid_evaluation()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new DynamicChebyshevSettings(
            ExerciseSteps: 40,
            SpotNodeCount: 61,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevAmericanOptionModel model = pricer.Build(request, settings);
        DynamicChebyshevEvaluation atRequestSpot = model.Evaluate(request.Spot);
        DynamicChebyshevResult direct = pricer.Price(request, settings);
        DynamicChebyshevEvaluation lowSpot = model.Evaluate(95.0);
        DynamicChebyshevEvaluation highSpot = model.Evaluate(105.0);

        Assert.Equal(direct.Price, atRequestSpot.Price, precision: 12);
        Assert.Equal(direct.Delta, atRequestSpot.Delta, precision: 12);
        Assert.Equal(direct.Gamma, atRequestSpot.Gamma, precision: 12);
        Assert.True(highSpot.Price < lowSpot.Price);
    }

    [Fact]
    public void Dynamic_chebyshev_handles_high_dividend_call_exercise_region()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardCall(
            spot: 180.0,
            strike: 100.0,
            riskFreeRate: 0.01,
            dividendYield: 0.25,
            volatility: 0.20);
        var settings = new DynamicChebyshevSettings(
            ExerciseSteps: 30,
            SpotNodeCount: 51,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);

        DynamicChebyshevAmericanOptionModel model = new DynamicChebyshevAmericanOptionPricer().Build(request, settings);
        DynamicChebyshevEvaluation evaluation = model.Evaluate(220.0);

        Assert.True(double.IsFinite(evaluation.Price));
        Assert.True(evaluation.Price >= 120.0);
        Assert.InRange(evaluation.Delta, 0.0, 1.0);
    }

    [Fact]
    public void Dynamic_chebyshev_bellman_expectation_matches_black_scholes_first_moment()
    {
        MethodInfo method = typeof(DynamicChebyshevAmericanOptionPricer)
            .GetMethod("ContinuationValue", BindingFlags.Static | BindingFlags.NonPublic)!;
        double spot = 100.0;
        double riskFreeRate = 0.05;
        double dividendYield = 0.02;
        double volatility = 0.20;
        double dt = 1.0 / 80.0;
        double discount = Math.Exp(-riskFreeRate * dt);
        double drift = (riskFreeRate - dividendYield - 0.5 * volatility * volatility) * dt;
        double diffusion = volatility * Math.Sqrt(dt);

        double continuation = (double)method.Invoke(
            null,
            [spot, drift, diffusion, discount, (Func<double, double>)(nextSpot => nextSpot)])!;

        // For nextValue(S') = S', the Bellman expectation is
        // e^{-r dt} E[S'] = S e^{-q dt} under risk-neutral GBM.
        double expected = spot * Math.Exp(-dividendYield * dt);
        Assert.InRange(Math.Abs(continuation - expected), 0.0, 1e-12);
    }

    [Fact]
    public void Dynamic_chebyshev_rejects_invalid_inputs()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new DynamicChebyshevSettings();
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        Assert.Throws<ArgumentException>(() =>
            pricer.Price(request with { MaturityDate = request.ValuationDate }, settings));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, settings with { ExerciseSteps = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, settings with { SpotNodeCount = 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, settings with { SpotLower = 0.0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request with { Spot = 500.0 }, settings));
        Assert.Throws<NotSupportedException>(() =>
            pricer.Price(request, settings with { QuadratureOrder = 16 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request with { Right = (VanillaOptionRight)999 }, settings));
    }

    [Fact]
    public void Dynamic_chebyshev_model_rejects_out_of_domain_evaluation()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new DynamicChebyshevSettings(
            ExerciseSteps: 20,
            SpotNodeCount: 41,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        DynamicChebyshevAmericanOptionModel model = new DynamicChebyshevAmericanOptionPricer().Build(request, settings);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Evaluate(300.0));
    }

    [Fact]
    public void Dynamic_payoff_delta_rejects_unsupported_option_right_defensively()
    {
        MethodInfo method = typeof(DynamicChebyshevAmericanOptionPricer)
            .GetMethod("PayoffDelta", BindingFlags.Static | BindingFlags.NonPublic)!;
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut() with
        {
            Right = (VanillaOptionRight)999,
        };

        TargetInvocationException exception = Assert.Throws<TargetInvocationException>(() =>
            method.Invoke(null, [request, request.Spot]));

        Assert.IsType<ArgumentOutOfRangeException>(exception.InnerException);
    }
}
