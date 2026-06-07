using System.Reflection;
using AmericanOptionDynamicChebyshev;

namespace ChebyshevSharp.Tests.Finance;

/// <summary>
/// The exercise-boundary finder: B (where payoff = continuation) located by Math.NET Brent
/// root-finding. The boundary itself is a useful diagnostic and the basis for any future
/// boundary-aware variant. (The earlier boundary-<i>split</i> experiment was retired as a documented
/// negative result — see the case study — once the log-spot variant superseded it.)
/// </summary>
public sealed class AmericanOptionBoundaryFinderTests
{
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
            ?? throw new InvalidOperationException("FindExerciseBoundary is not implemented.");
        return (double)method.Invoke(null, [model, lo, hi])!;
    }
}
