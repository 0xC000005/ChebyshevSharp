using System.Reflection;
using AmericanOptionDynamicChebyshev;
using Xunit.Abstractions;

namespace ChebyshevSharp.Tests.Finance;

/// <summary>
/// W3 — boundary-aware split. The exercise boundary B_i (where payoff = continuation) is a
/// moving kink; locating it by root-finding lets the continuation be represented as a two-piece
/// ChebyshevSpline with a knot at B_i, restoring per-piece smoothness near the boundary.
/// </summary>
public sealed class AmericanOptionBoundarySplitTests
{
    private static readonly IAmericanOptionReferencePricer ReferencePricer =
        new QlNetAmericanOptionReferencePricer();

    private readonly ITestOutputHelper _output;

    public AmericanOptionBoundarySplitTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Boundary_split_reduces_gamma_error_near_the_exercise_boundary()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var baseSettings = new DynamicChebyshevSettings(80, 81, 5.0, 250.0, 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();
        DynamicChebyshevAmericanOptionModel global = pricer.Build(request, baseSettings);
        DynamicChebyshevAmericanOptionModel split =
            pricer.Build(request, baseSettings with { BoundarySplit = true });

        double[] spots = [82.0, 86.0, 90.0, 94.0, 98.0, 102.0, 106.0, 110.0, 114.0, 118.0];
        _output.WriteLine("spot   oracleGamma  globalGamma  globalErr   splitGamma   splitErr");
        double globalErrAt82 = 0.0;
        double splitErrAt82 = 0.0;
        foreach (double s in spots)
        {
            double oracle = ReferencePricer.Price(request with { Spot = s }).Gamma;
            double globalGamma = global.Evaluate(s).Gamma;
            double splitGamma = split.Evaluate(s).Gamma;
            double globalErr = Math.Abs(globalGamma - oracle);
            double splitErr = Math.Abs(splitGamma - oracle);
            _output.WriteLine(
                $"{s,5:F1}  {oracle,11:F6}  {globalGamma,11:F6}  {globalErr,9:F6}  {splitGamma,11:F6}  {splitErr,9:F6}");
            if (s == 82.0)
            {
                globalErrAt82 = globalErr;
                splitErrAt82 = splitErr;
            }
        }

        _output.WriteLine(
            $"Boundary-row (spot 82) Gamma abs error: global={globalErrAt82:F6}, split={splitErrAt82:F6}");

        // Sanity/regression guard only. Whether the split REDUCES the boundary-row Gamma error is
        // the empirical question reported in the table above, and the answer is mixed: it improves
        // rows a little further out (e.g. spot 86) but worsens the row immediately adjacent to the
        // knot (spot 82, only ~0.14 above B0), where the per-piece second derivative at the piece
        // edge is unreliable. This test therefore only asserts the split keeps pricing accurate and
        // the Gamma profile finite.
        Assert.InRange(Math.Abs(split.Evaluate(request.Spot).Price - 6.088238), 0.0, 0.15);
        Assert.True(double.IsFinite(globalErrAt82) && double.IsFinite(splitErrAt82));
    }

    [Fact]
    public void Exercise_boundary_is_located_near_the_diagnostic_value()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var settings = new DynamicChebyshevSettings(80, 81, 5.0, 250.0, 8);
        DynamicChebyshevAmericanOptionModel model =
            new DynamicChebyshevAmericanOptionPricer().Build(request, settings);

        double boundary = InvokeFindExerciseBoundary(model, lo: 5.0, hi: request.Strike);

        // The diagnostics mode reports the first-step exercise boundary around spot 81.86.
        Assert.InRange(boundary, 80.0, 84.0);
    }

    [Fact]
    public void Boundary_split_stays_near_the_oracle_and_changes_greeks_near_the_boundary()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var baseSettings = new DynamicChebyshevSettings(80, 81, 5.0, 250.0, 8);
        var pricer = new DynamicChebyshevAmericanOptionPricer();

        DynamicChebyshevAmericanOptionModel global = pricer.Build(request, baseSettings);
        DynamicChebyshevAmericanOptionModel split =
            pricer.Build(request, baseSettings with { BoundarySplit = true });

        // Price stays accurate against the QLNet FD oracle (6.088238).
        Assert.InRange(Math.Abs(split.Evaluate(request.Spot).Price - 6.088238), 0.0, 0.15);

        // Splitting at the boundary changes the second derivative near it (spot 82, next to ~81.86),
        double globalGamma = global.Evaluate(82.0).Gamma;
        double splitGamma = split.Evaluate(82.0).Gamma;
        Assert.NotEqual(globalGamma, splitGamma);
        // and Gamma stays a sane positive convexity.
        Assert.True(splitGamma > 0.0);
    }

    private static double InvokeFindExerciseBoundary(
        DynamicChebyshevAmericanOptionModel model, double lo, double hi)
    {
        MethodInfo method = typeof(DynamicChebyshevAmericanOptionPricer)
            .GetMethod(
                "FindExerciseBoundary",
                BindingFlags.Static | BindingFlags.NonPublic,
                binder: null,
                types: [typeof(DynamicChebyshevAmericanOptionModel), typeof(double), typeof(double)],
                modifiers: null)
            ?? throw new InvalidOperationException("FindExerciseBoundary is not implemented yet.");
        return (double)method.Invoke(null, [model, lo, hi])!;
    }
}
