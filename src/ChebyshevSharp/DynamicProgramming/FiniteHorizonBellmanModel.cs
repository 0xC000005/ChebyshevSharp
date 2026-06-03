namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Solved finite-horizon Bellman model with reusable Chebyshev action-value
/// branches for online evaluation.
/// </summary>
public sealed class FiniteHorizonBellmanModel
{
    internal FiniteHorizonBellmanModel(
        double stateLower,
        double stateUpper,
        IReadOnlyList<FiniteHorizonBellmanStepSolution> stepSolutions,
        int buildEvaluations)
    {
        StateLower = stateLower;
        StateUpper = stateUpper;
        StepSolutions = stepSolutions;
        BuildEvaluations = buildEvaluations;
    }

    /// <summary>Lower state-domain bound.</summary>
    public double StateLower { get; }

    /// <summary>Upper state-domain bound.</summary>
    public double StateUpper { get; }

    /// <summary>Backward-induction step solutions in chronological order.</summary>
    public IReadOnlyList<FiniteHorizonBellmanStepSolution> StepSolutions { get; }

    /// <summary>Total number of function evaluations used to build action-value branches.</summary>
    public int BuildEvaluations { get; }

    /// <summary>Evaluate the solved value and selected action at the first step.</summary>
    public BellmanEvaluation Evaluate(double state)
    {
        if (StepSolutions.Count == 0)
            throw new InvalidOperationException("The Bellman model has no solved steps.");

        return StepSolutions[0].Evaluate(state, StateLower, StateUpper);
    }
}
