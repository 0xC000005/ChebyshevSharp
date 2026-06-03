namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Builds finite-horizon Bellman solutions by fitting Chebyshev action-value
/// branches backward from a terminal value function.
/// </summary>
public sealed class FiniteHorizonBellmanSolver
{
    /// <summary>Solve a one-dimensional continuous-state finite-horizon Bellman problem.</summary>
    public FiniteHorizonBellmanModel Solve(FiniteHorizonBellmanProblem problem)
    {
        ArgumentNullException.ThrowIfNull(problem);

        var stepSolutions = new FiniteHorizonBellmanStepSolution[problem.StepCount];
        Func<double, double> nextValue = problem.TerminalValue;
        int buildEvaluations = 0;

        for (int step = problem.StepCount - 1; step >= 0; step--)
        {
            var context = new BellmanStepContext(step, problem.StepCount);
            Func<double, double> nextValueAtStep = nextValue;
            var actionApproximations = new List<FiniteHorizonActionValueApproximation>(problem.Actions.Count);

            foreach (BellmanAction action in problem.Actions)
            {
                ChebyshevApproximation approximation = BuildActionApproximation(
                    problem,
                    context,
                    action,
                    nextValueAtStep);
                buildEvaluations += approximation.NEvaluations;
                actionApproximations.Add(new FiniteHorizonActionValueApproximation(action, approximation));
            }

            var stepSolution = new FiniteHorizonBellmanStepSolution(step, actionApproximations);
            stepSolutions[step] = stepSolution;
            nextValue = state =>
            {
                double boundedState = Math.Clamp(state, problem.StateLower, problem.StateUpper);
                return stepSolution.Evaluate(boundedState, problem.StateLower, problem.StateUpper).Value;
            };
        }

        return new FiniteHorizonBellmanModel(
            problem.StateLower,
            problem.StateUpper,
            stepSolutions,
            buildEvaluations);
    }

    private static ChebyshevApproximation BuildActionApproximation(
        FiniteHorizonBellmanProblem problem,
        BellmanStepContext context,
        BellmanAction action,
        Func<double, double> nextValue)
    {
        double Function(double[] point, object? _)
            => problem.ActionValue(context, point[0], action, nextValue);

        var approximation = new ChebyshevApproximation(
            Function,
            numDimensions: 1,
            domain: new[] { new[] { problem.StateLower, problem.StateUpper } },
            nNodes: new[] { problem.StateNodeCount },
            maxDerivativeOrder: problem.MaxDerivativeOrder);
        approximation.Build(verbose: false);
        return approximation;
    }
}
