using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

public sealed class AmericanOptionLspiTests
{
    private static readonly IAmericanOptionReferencePricer ReferencePricer = new QlNetAmericanOptionReferencePricer();

    [Fact]
    public void Lspi_price_is_reproducible_for_fixed_seed()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new LspiSettings(
            PathCount: 4_000,
            ExerciseSteps: 40,
            Seed: 24680);
        var pricer = new LspiAmericanOptionPricer();

        LspiResult first = pricer.Price(request, settings);
        LspiResult second = pricer.Price(request, settings);

        Assert.Equal(first.Price, second.Price, precision: 12);
        Assert.Equal(first.BoundaryDecisionCount, second.BoundaryDecisionCount);
    }

    [Fact]
    public void Lspi_price_is_close_to_qlnet_reference_for_standard_put()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        AmericanOptionResult reference = ReferencePricer.Price(request);
        var settings = new LspiSettings(
            PathCount: 12_000,
            ExerciseSteps: 50,
            Seed: 13579);
        var pricer = new LspiAmericanOptionPricer();

        LspiResult result = pricer.Price(request, settings);

        Assert.True(double.IsFinite(result.Price));
        Assert.True(double.IsFinite(result.StandardError));
        Assert.InRange(Math.Abs(result.Price - reference.Price), 0.0, 0.90);
        Assert.True(result.BoundaryDecisionCount > 0);
        Assert.True(result.PolicyIterations > 0);
        Assert.Equal(7, result.FeatureCount);
    }
}
