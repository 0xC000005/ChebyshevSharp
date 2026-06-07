using System.Reflection;
using AmericanOptionDynamicChebyshev;
using Xunit.Abstractions;

namespace ChebyshevSharp.Tests.Finance;

/// <summary>
/// W2 — closed-form European terminal step. The Dynamic Chebyshev recursion can seed its
/// terminal backward step with the exact one-period European Black-Scholes value instead of
/// quadrature against the kinked payoff. These tests pin the closed-form European helper.
/// </summary>
public sealed class AmericanOptionTerminalStepTests
{
    private static readonly IAmericanOptionReferencePricer ReferencePricer =
        new QlNetAmericanOptionReferencePricer();

    private readonly ITestOutputHelper _output;

    public AmericanOptionTerminalStepTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Closed_form_terminal_step_accuracy_profile_against_the_oracle()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var baseSettings = new DynamicChebyshevSettings(80, 81, 5.0, 250.0, 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();
        DynamicChebyshevAmericanOptionModel quad = pricer.Build(request, baseSettings);
        DynamicChebyshevAmericanOptionModel closed =
            pricer.Build(request, baseSettings with { ClosedFormTerminalStep = true });

        double[] spots = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
        _output.WriteLine(
            "spot  oracleP   quadPerr  closedPerr | quadDerr closedDerr | quadGerr closedGerr");
        double quadPmax = 0.0, closedPmax = 0.0, quadGmax = 0.0, closedGmax = 0.0;
        foreach (double s in spots)
        {
            AmericanOptionResult o = ReferencePricer.Price(request with { Spot = s });
            DynamicChebyshevEvaluation q = quad.Evaluate(s);
            DynamicChebyshevEvaluation c = closed.Evaluate(s);
            double qpe = Math.Abs(q.Price - o.Price), cpe = Math.Abs(c.Price - o.Price);
            double qde = Math.Abs(q.Delta - o.Delta), cde = Math.Abs(c.Delta - o.Delta);
            double qge = Math.Abs(q.Gamma - o.Gamma), cge = Math.Abs(c.Gamma - o.Gamma);
            quadPmax = Math.Max(quadPmax, qpe);
            closedPmax = Math.Max(closedPmax, cpe);
            quadGmax = Math.Max(quadGmax, qge);
            closedGmax = Math.Max(closedGmax, cge);
            _output.WriteLine(
                $"{s,4:F0}  {o.Price,8:F5}  {qpe,8:F6}  {cpe,8:F6} | {qde,7:F5} {cde,7:F5} | {qge,7:F6} {cge,7:F6}");
        }

        _output.WriteLine(
            $"Max |price err| over grid: quadrature={quadPmax:F6}, closedForm={closedPmax:F6}");
        _output.WriteLine(
            $"Max |gamma err| over grid: quadrature={quadGmax:F6}, closedForm={closedGmax:F6}");

        // Sanity guard only; whether the closed-form terminal step REDUCES the errors is the
        // empirical question reported above.
        Assert.InRange(closedPmax, 0.0, 0.2);
    }

    [Fact]
    public void European_put_price_matches_the_black_scholes_analytic_value()
    {
        // Standard ATM put: S=100, K=100, T=1, r=5%, sigma=20%, q=0.
        // QLNet's AnalyticEuropeanEngine reports 5.573526 for this contract (the case study's
        // "European analytic" control), so the closed-form helper must match it.
        double price = InvokeEuropeanPutPrice(s: 100.0, k: 100.0, t: 1.0, r: 0.05, sigma: 0.20, q: 0.0);

        Assert.InRange(Math.Abs(price - 5.573526), 0.0, 1e-3);
    }

    [Fact]
    public void Closed_form_terminal_step_changes_the_build_and_stays_near_the_oracle()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var baseSettings = new DynamicChebyshevSettings(
            ExerciseSteps: 80,
            SpotNodeCount: 81,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevResult quadrature = pricer.Price(request, baseSettings);
        DynamicChebyshevResult closedForm = pricer.Price(
            request,
            baseSettings with { ClosedFormTerminalStep = true });

        // The flag must actually change the terminal-step computation,
        Assert.NotEqual(quadrature.Price, closedForm.Price);
        // and the closed-form variant must stay accurate against the QLNet FD oracle (6.088238).
        Assert.InRange(Math.Abs(closedForm.Price - 6.088238), 0.0, 0.15);
    }

    private static double InvokeEuropeanPutPrice(double s, double k, double t, double r, double sigma, double q)
    {
        MethodInfo method = typeof(DynamicChebyshevAmericanOptionPricer)
            .GetMethod("EuropeanPutPrice", BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(
                "EuropeanPutPrice(double, double, double, double, double, double) is not implemented yet.");
        return (double)method.Invoke(null, [s, k, t, r, sigma, q])!;
    }
}
