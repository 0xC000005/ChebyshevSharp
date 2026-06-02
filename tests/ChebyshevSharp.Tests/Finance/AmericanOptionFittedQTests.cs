using System.Reflection;
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
        Assert.Equal(settings.PathCount, result.PathCount);
        Assert.Equal(settings.ExerciseSteps, result.ExerciseSteps);
    }

    [Fact]
    public void Lspi_handles_call_payoff_and_degenerate_linear_system()
    {
        var pricer = new LspiAmericanOptionPricer();
        var settings = new LspiSettings(
            PathCount: 1,
            ExerciseSteps: 3,
            Seed: 27182,
            MaxPolicyIterations: 2);
        AmericanOptionRequest call = AmericanOptionScenarios.StandardCall(
            spot: 110.0,
            strike: 100.0,
            volatility: 0.0);

        LspiResult result = pricer.Price(call, settings);

        Assert.True(double.IsFinite(result.Price));
        Assert.True(double.IsFinite(result.StandardError));
        Assert.Equal(settings.PathCount, result.PathCount);
        Assert.Equal(settings.ExerciseSteps, result.ExerciseSteps);
    }

    [Fact]
    public void Lspi_rejects_invalid_inputs()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new LspiAmericanOptionPricer();

        Assert.Throws<ArgumentException>(() =>
            pricer.Price(request with { MaturityDate = request.ValuationDate }, new LspiSettings()));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, new LspiSettings(PathCount: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, new LspiSettings(ExerciseSteps: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request, new LspiSettings(MaxPolicyIterations: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            pricer.Price(request with { Right = (VanillaOptionRight)999 }, new LspiSettings()));
    }

    [Fact]
    public void Lspi_solver_returns_zero_weights_for_singular_system_defensively()
    {
        MethodInfo method = typeof(LspiAmericanOptionPricer)
            .GetMethod("Solve", BindingFlags.Static | BindingFlags.NonPublic)!;

        var matrix = new double[2, 2];
        var rhs = new double[2];

        double[] weights = (double[])method.Invoke(null, [matrix, rhs])!;

        Assert.Equal([0.0, 0.0], weights);
    }
}
