namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Chebyshev approximation of one action-value branch at one Bellman step.
/// </summary>
public sealed class FiniteHorizonActionValueApproximation
{
    internal FiniteHorizonActionValueApproximation(
        BellmanAction action,
        ChebyshevApproximation approximation)
    {
        Action = action;
        Approximation = approximation;
    }

    /// <summary>Action represented by this branch.</summary>
    public BellmanAction Action { get; }

    /// <summary>Underlying Chebyshev approximation.</summary>
    public ChebyshevApproximation Approximation { get; }

    /// <summary>Evaluate this action-value branch.</summary>
    public double Evaluate(double state, int derivativeOrder = 0)
        => Approximation.VectorizedEval([state], [derivativeOrder]);
}
