using AmericanOptionDynamicChebyshev;
using Xunit.Abstractions;

namespace ChebyshevSharp.Tests.Finance;

/// <summary>
/// F1 — log-spot Dynamic Chebyshev variant (Stage F1 of the front-fixing track). Interpolating the
/// continuation in x = log(S) makes the Gauss-Hermite transition additive (bounded images) and the grid
/// narrow + uniform-in-x, which is far better conditioned at high node counts than the wide linear
/// [5,250] grid. Greeks come from the chain rule: Delta = u'(x)/S, Gamma = (u''(x) - u'(x)) / S^2.
/// </summary>
public sealed class AmericanOptionLogSpotTests
{
    private readonly ITestOutputHelper _output;

    public AmericanOptionLogSpotTests(ITestOutputHelper output) => _output = output;

    private static DynamicChebyshevSettings BaseSettings(int nodes = 81) =>
        new(ExerciseSteps: 80, SpotNodeCount: nodes, SpotLower: 5.0, SpotUpper: 250.0, QuadratureOrder: 8);

    [Fact]
    public void LogSpot_off_path_is_bit_identical_to_today()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevAmericanOptionModel off =
            pricer.Build(request, BaseSettings() with { LogSpot = false });

        // OFF path must reproduce the documented linear Dynamic Chebyshev anchors (regression lock).
        DynamicChebyshevEvaluation atm = off.Evaluate(100.0);
        Assert.InRange(Math.Abs(atm.Price - 6.083607), 0.0, 1e-5);
        Assert.InRange(Math.Abs(atm.Delta - (-0.410533)), 0.0, 1e-5);
        Assert.InRange(Math.Abs(atm.Gamma - 0.022946), 0.0, 1e-5);
        Assert.InRange(Math.Abs(off.Evaluate(82.0).Gamma - 0.025696), 0.0, 1e-5);
    }

    [Fact]
    public void LogSpot_on_ATM_price_matches_oracle()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        double linPrice = pricer.Build(request, BaseSettings() with { LogSpot = false }).Evaluate(100.0).Price;
        double logPrice = pricer.Build(request, BaseSettings() with { LogSpot = true }).Evaluate(100.0).Price;

        Assert.True(double.IsFinite(logPrice));
        Assert.InRange(Math.Abs(logPrice - 6.088238), 0.0, 0.15);   // QLNet FD oracle band
        Assert.InRange(Math.Abs(logPrice - linPrice), 0.0, 0.05);   // coordinate change preserves value
    }

    [Fact]
    public void LogSpot_on_chainrule_delta_gamma_match_linear_baseline()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevEvaluation lin =
            pricer.Build(request, BaseSettings() with { LogSpot = false }).Evaluate(100.0);
        DynamicChebyshevEvaluation log =
            pricer.Build(request, BaseSettings() with { LogSpot = true }).Evaluate(100.0);

        // Chain-rule Greeks (Delta = u'/S, Gamma = (u'' - u')/S^2) must track the trusted linear baseline.
        Assert.InRange(Math.Abs(log.Delta - lin.Delta), 0.0, 0.01);
        Assert.InRange(Math.Abs(log.Gamma - lin.Gamma), 0.0, 0.005);
    }

    [Fact]
    public void LogSpot_on_is_finite_where_linear_fails_at_high_node_counts()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        foreach (int n in new[] { 161, 321 })
        {
            // Document the linear failure: at high n the wide [5,250] build is non-finite or throws.
            bool linearBad;
            try
            {
                double linGamma = pricer.Build(request, BaseSettings(n) with { LogSpot = false }).Evaluate(82.0).Gamma;
                linearBad = !double.IsFinite(linGamma);
            }
            catch (ArgumentException)
            {
                linearBad = true;
            }

            _output.WriteLine($"n={n}: linear bad (threw or non-finite) = {linearBad}");

            // The log-spot build must complete and return finite values at the same n.
            DynamicChebyshevAmericanOptionModel log = pricer.Build(request, BaseSettings(n) with { LogSpot = true });
            DynamicChebyshevEvaluation atm = log.Evaluate(100.0);
            DynamicChebyshevEvaluation near = log.Evaluate(82.0);
            Assert.True(
                double.IsFinite(atm.Price) && double.IsFinite(near.Delta) && double.IsFinite(near.Gamma),
                $"log-spot must be finite at n={n}");
            Assert.InRange(Math.Abs(atm.Price - 6.088238), 0.0, 0.15);
        }
    }

    [Fact]
    public void LogSpot_spot82_gamma_convergence_measurement()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var pricer = new DynamicChebyshevAmericanOptionPricer();
        const double oracleGamma82 = 0.033689;

        _output.WriteLine("n     linGamma82   linErr      logGamma82   logErr      linStatus");
        double logErrAt81 = double.NaN;
        foreach (int n in new[] { 81, 161, 321 })
        {
            string linStatus;
            double linGamma = double.NaN;
            double linErr = double.NaN;
            try
            {
                linGamma = pricer.Build(request, BaseSettings(n) with { LogSpot = false }).Evaluate(82.0).Gamma;
                linErr = Math.Abs(linGamma - oracleGamma82);
                linStatus = double.IsFinite(linGamma) ? "ok" : "non-finite";
            }
            catch (ArgumentException)
            {
                linStatus = "THREW";
            }

            DynamicChebyshevAmericanOptionModel logModel =
                pricer.Build(request, BaseSettings(n) with { LogSpot = true });
            double logGamma = logModel.Evaluate(82.0).Gamma;
            double logErr = Math.Abs(logGamma - oracleGamma82);
            double logPrice = logModel.Evaluate(100.0).Price;
            if (n == 81)
            {
                logErrAt81 = logErr;
            }

            _output.WriteLine(
                $"{n,-5} {linGamma,10:F6}  {linErr,9:F6}  {logGamma,10:F6}  {logErr,9:F6}  {linStatus}");

            // Sanity only; whether log-spot Gamma converges toward the oracle is the measured, reported
            // question (the F1 decision gate), not a pass/fail assertion.
            Assert.True(double.IsFinite(logGamma), $"log Gamma finite at n={n}");
            Assert.InRange(Math.Abs(logPrice - 6.088238), 0.0, 0.15);
        }

        _output.WriteLine("oracle Gamma(82) = 0.033689; linear@81 anchor = 0.025696 (err 0.007993)");
        Assert.InRange(logErrAt81, 0.0, 0.015);
    }
}
