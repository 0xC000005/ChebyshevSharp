using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

public sealed class AmericanOptionReferencePricerTests
{
    private static readonly IAmericanOptionReferencePricer Pricer = new QlNetAmericanOptionReferencePricer();

    [Fact]
    public void American_put_reference_price_is_finite_and_above_european_put()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();

        AmericanOptionResult result = Pricer.Price(request);

        Assert.True(double.IsFinite(result.Price));
        Assert.True(double.IsFinite(result.EuropeanPrice));
        Assert.True(result.Price >= result.EuropeanPrice - 1e-10);
        Assert.True(result.EarlyExercisePremium >= -1e-10);
    }

    [Fact]
    public void Non_dividend_american_call_matches_european_call()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardCall(dividendYield: 0.0);

        AmericanOptionResult result = Pricer.Price(request);

        Assert.InRange(Math.Abs(result.Price - result.EuropeanPrice), 0.0, 1e-3);
        Assert.InRange(Math.Abs(result.EarlyExercisePremium), 0.0, 1e-3);
    }

    [Fact]
    public void Finite_difference_and_binomial_prices_converge_for_standard_put()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();

        AmericanOptionResult finiteDifference = Pricer.Price(request with
        {
            Engine = AmericanOptionReferenceEngine.FiniteDifference,
            TimeSteps = 300,
            GridPoints = 300,
        });
        AmericanOptionResult binomial = Pricer.Price(request with
        {
            Engine = AmericanOptionReferenceEngine.CoxRossRubinstein,
            TimeSteps = 1200,
        });

        Assert.InRange(Math.Abs(finiteDifference.Price - binomial.Price), 0.0, 0.03);
    }

    [Fact]
    public void American_put_reference_greeks_have_expected_signs()
    {
        AmericanOptionResult result = Pricer.Price(AmericanOptionScenarios.StandardPut());

        Assert.True(double.IsFinite(result.Delta));
        Assert.True(double.IsFinite(result.Gamma));
        Assert.InRange(result.Delta, -1.0, 0.0);
        Assert.True(result.Gamma > 0.0);
    }

    [Fact]
    public void Put_price_moves_monotonically_with_spot_and_volatility()
    {
        AmericanOptionResult lowSpot = Pricer.Price(AmericanOptionScenarios.StandardPut(spot: 95.0));
        AmericanOptionResult highSpot = Pricer.Price(AmericanOptionScenarios.StandardPut(spot: 105.0));
        AmericanOptionResult lowVolatility = Pricer.Price(AmericanOptionScenarios.StandardPut(volatility: 0.15));
        AmericanOptionResult highVolatility = Pricer.Price(AmericanOptionScenarios.StandardPut(volatility: 0.30));

        Assert.True(highSpot.Price < lowSpot.Price);
        Assert.True(highVolatility.Price > lowVolatility.Price);
    }
}
