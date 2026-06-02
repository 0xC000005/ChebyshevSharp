using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

public sealed class AmericanOptionRegressionMonteCarloTests
{
    private static readonly IAmericanOptionReferencePricer ReferencePricer = new QlNetAmericanOptionReferencePricer();

    [Fact]
    public void Longstaff_schwartz_price_is_reproducible_for_fixed_seed()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new RegressionMonteCarloSettings(
            PathCount: 4_000,
            ExerciseSteps: 40,
            Seed: 12345);
        var pricer = new LongstaffSchwartzAmericanOptionPricer();

        RegressionMonteCarloResult first = pricer.Price(request, settings);
        RegressionMonteCarloResult second = pricer.Price(request, settings);

        Assert.Equal(first.Price, second.Price, precision: 12);
        Assert.Equal(first.StandardError, second.StandardError, precision: 12);
    }

    [Fact]
    public void Longstaff_schwartz_price_is_close_to_qlnet_reference_for_standard_put()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        AmericanOptionResult reference = ReferencePricer.Price(request);
        var settings = new RegressionMonteCarloSettings(
            PathCount: 12_000,
            ExerciseSteps: 50,
            Seed: 8675309);
        var pricer = new LongstaffSchwartzAmericanOptionPricer();

        RegressionMonteCarloResult result = pricer.Price(request, settings);

        Assert.True(double.IsFinite(result.Price));
        Assert.True(double.IsFinite(result.StandardError));
        Assert.InRange(Math.Abs(result.Price - reference.Price), 0.0, 0.75);
        Assert.True(result.ExercisedPathFraction > 0.0);
        Assert.True(result.ExercisedPathFraction < 1.0);
    }
}
