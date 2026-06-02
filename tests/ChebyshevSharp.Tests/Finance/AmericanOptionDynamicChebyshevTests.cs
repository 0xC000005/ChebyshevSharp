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
}
