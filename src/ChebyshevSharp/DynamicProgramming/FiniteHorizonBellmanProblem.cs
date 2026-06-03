namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Defines a one-dimensional continuous-state, finite-horizon Bellman problem
/// with a discrete action set.
/// </summary>
/// <remarks>
/// The action-value callback receives the already-solved next-step value
/// function. This supports backward induction, optimal stopping, and simple
/// grid-action control problems without requiring a continuous-action optimizer.
/// </remarks>
public sealed class FiniteHorizonBellmanProblem
{
    /// <summary>Create a finite-horizon Bellman problem.</summary>
    public FiniteHorizonBellmanProblem(
        int StepCount,
        double StateLower,
        double StateUpper,
        int StateNodeCount,
        IReadOnlyList<BellmanAction> Actions,
        Func<double, double> TerminalValue,
        Func<BellmanStepContext, double, BellmanAction, Func<double, double>, double> ActionValue,
        int MaxDerivativeOrder = 2)
    {
        ArgumentNullException.ThrowIfNull(Actions);
        ArgumentNullException.ThrowIfNull(TerminalValue);
        ArgumentNullException.ThrowIfNull(ActionValue);

        if (StepCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(StepCount), "Step count must be positive.");
        if (!double.IsFinite(StateLower) || !double.IsFinite(StateUpper) || StateUpper <= StateLower)
            throw new ArgumentOutOfRangeException(nameof(StateLower), "State domain must satisfy finite lower < upper.");
        if (StateNodeCount < 3)
            throw new ArgumentOutOfRangeException(nameof(StateNodeCount), "State node count must be at least 3.");
        if (Actions.Count == 0)
            throw new ArgumentException("At least one action is required.", nameof(Actions));
        if (MaxDerivativeOrder < 2)
            throw new ArgumentOutOfRangeException(nameof(MaxDerivativeOrder), "Maximum derivative order must be at least 2.");

        this.StepCount = StepCount;
        this.StateLower = StateLower;
        this.StateUpper = StateUpper;
        this.StateNodeCount = StateNodeCount;
        this.Actions = Actions.ToArray();
        this.TerminalValue = TerminalValue;
        this.ActionValue = ActionValue;
        this.MaxDerivativeOrder = MaxDerivativeOrder;
    }

    /// <summary>Total finite-horizon step count.</summary>
    public int StepCount { get; }

    /// <summary>Lower state-domain bound.</summary>
    public double StateLower { get; }

    /// <summary>Upper state-domain bound.</summary>
    public double StateUpper { get; }

    /// <summary>Number of Chebyshev state nodes.</summary>
    public int StateNodeCount { get; }

    /// <summary>Discrete action grid.</summary>
    public IReadOnlyList<BellmanAction> Actions { get; }

    /// <summary>Terminal value function.</summary>
    public Func<double, double> TerminalValue { get; }

    /// <summary>Action-value callback used during backward induction.</summary>
    public Func<BellmanStepContext, double, BellmanAction, Func<double, double>, double> ActionValue { get; }

    /// <summary>Maximum derivative order requested from each Chebyshev action-value branch. Must be at least 2.</summary>
    public int MaxDerivativeOrder { get; }
}
