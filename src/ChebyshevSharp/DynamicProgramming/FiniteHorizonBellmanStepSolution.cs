namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Solved action-value branches for one finite-horizon step.
/// </summary>
public sealed class FiniteHorizonBellmanStepSolution
{
    internal FiniteHorizonBellmanStepSolution(
        int step,
        IReadOnlyList<FiniteHorizonActionValueApproximation> actionValues)
    {
        Step = step;
        ActionValues = actionValues;
    }

    /// <summary>Zero-based step index.</summary>
    public int Step { get; }

    /// <summary>Chebyshev action-value approximations at this step.</summary>
    public IReadOnlyList<FiniteHorizonActionValueApproximation> ActionValues { get; }

    /// <summary>Evaluate the value and selected action at this step.</summary>
    public BellmanEvaluation Evaluate(double state, double stateLower, double stateUpper)
    {
        if (state < stateLower || state > stateUpper)
            throw new ArgumentOutOfRangeException(nameof(state), "State must be inside the Bellman model domain.");

        FiniteHorizonActionValueApproximation? best = null;
        double bestValue = double.NegativeInfinity;

        foreach (FiniteHorizonActionValueApproximation actionValue in ActionValues)
        {
            double value = actionValue.Evaluate(state, derivativeOrder: 0);
            if (value > bestValue)
            {
                best = actionValue;
                bestValue = value;
            }
        }

        if (best is null)
            throw new InvalidOperationException("The Bellman step has no action-value branches.");

        return new BellmanEvaluation(
            bestValue,
            best.Action,
            best.Evaluate(state, derivativeOrder: 1),
            best.Evaluate(state, derivativeOrder: 2));
    }
}
