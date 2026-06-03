namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Time-step metadata passed to a Bellman action-value callback.
/// </summary>
/// <param name="Step">Zero-based step currently being solved.</param>
/// <param name="StepCount">Total number of finite-horizon steps.</param>
public sealed record BellmanStepContext(int Step, int StepCount)
{
    /// <summary>True for the final backward-induction step before terminal value.</summary>
    public bool IsLastStepBeforeTerminal => Step == StepCount - 1;
}
