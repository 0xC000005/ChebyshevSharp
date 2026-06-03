using ChebyshevSharp.DynamicProgramming;

namespace ChebyshevSharp.Tests.DynamicProgramming;

public sealed class FiniteHorizonBellmanSolverTests
{
    [Fact]
    public void Solver_selects_better_discrete_action_for_simple_one_step_problem()
    {
        var problem = new FiniteHorizonBellmanProblem(
            StepCount: 1,
            StateLower: 0.0,
            StateUpper: 2.0,
            StateNodeCount: 17,
            Actions: new[]
            {
                new BellmanAction(0, "low"),
                new BellmanAction(1, "high"),
            },
            TerminalValue: state => state,
            ActionValue: (context, state, action, nextValue) =>
                action.Index == 0 ? state : 2.0 * state);

        FiniteHorizonBellmanModel model = new FiniteHorizonBellmanSolver().Solve(problem);
        BellmanEvaluation evaluation = model.Evaluate(1.25);

        Assert.Equal(1, evaluation.Action.Index);
        Assert.Equal("high", evaluation.Action.Name);
        Assert.Equal(2.5, evaluation.Value, precision: 10);
        Assert.Equal(2.0, evaluation.FirstDerivative, precision: 8);
        Assert.Equal(0.0, evaluation.SecondDerivative, precision: 7);
        Assert.Equal(2, model.StepSolutions[0].ActionValues.Count);
    }

    [Fact]
    public void Solver_reuses_value_function_for_dynamic_asset_allocation_grid()
    {
        BellmanAction[] allocations =
        [
            new(0, "cash", 0.0),
            new(1, "balanced", 0.5),
            new(2, "risky", 1.0),
        ];
        double[] riskyReturns = [-0.10, 0.18];
        double[] probabilities = [0.45, 0.55];
        const double riskFreeReturn = 0.02;

        var problem = new FiniteHorizonBellmanProblem(
            StepCount: 3,
            StateLower: 25.0,
            StateUpper: 250.0,
            StateNodeCount: 31,
            Actions: allocations,
            TerminalValue: Math.Sqrt,
            ActionValue: (context, wealth, action, nextValue) =>
            {
                double riskyWeight = action.Value;
                double expected = 0.0;
                for (int i = 0; i < riskyReturns.Length; i++)
                {
                    double grossReturn = (1.0 - riskyWeight) * (1.0 + riskFreeReturn)
                        + riskyWeight * (1.0 + riskyReturns[i]);
                    expected += probabilities[i] * nextValue(wealth * grossReturn);
                }

                return expected;
            });

        FiniteHorizonBellmanModel model = new FiniteHorizonBellmanSolver().Solve(problem);
        BellmanEvaluation lowWealth = model.Evaluate(80.0);
        BellmanEvaluation highWealth = model.Evaluate(160.0);

        Assert.True(highWealth.Value > lowWealth.Value);
        Assert.Contains(highWealth.Action.Name, new[] { "cash", "balanced", "risky" });
        Assert.True(double.IsFinite(highWealth.FirstDerivative));
        Assert.Equal(3 * allocations.Length * 31, model.BuildEvaluations);
    }

    [Fact]
    public void Solver_handles_stochastic_growth_with_savings_action_grid()
    {
        BellmanAction[] savingsRates =
        [
            new(0, "save-20", 0.20),
            new(1, "save-40", 0.40),
            new(2, "save-60", 0.60),
        ];
        double[] productivityShocks = [0.9, 1.1];
        const double beta = 0.95;
        const double alpha = 0.35;

        var problem = new FiniteHorizonBellmanProblem(
            StepCount: 4,
            StateLower: 0.5,
            StateUpper: 8.0,
            StateNodeCount: 33,
            Actions: savingsRates,
            TerminalValue: capital => Math.Log(1.0 + capital),
            ActionValue: (context, capital, action, nextValue) =>
            {
                double output = Math.Pow(capital, alpha);
                double savings = action.Value * output;
                double consumption = Math.Max(output - savings, 1e-12);
                double expectedNext = 0.0;
                foreach (double shock in productivityShocks)
                {
                    expectedNext += 0.5 * nextValue(Math.Max(0.05, shock * savings));
                }

                return Math.Log(consumption) + beta * expectedNext;
            });

        FiniteHorizonBellmanModel model = new FiniteHorizonBellmanSolver().Solve(problem);
        BellmanEvaluation smallCapital = model.Evaluate(1.0);
        BellmanEvaluation largeCapital = model.Evaluate(4.0);

        Assert.True(largeCapital.Value > smallCapital.Value);
        Assert.Contains(largeCapital.Action.Name, new[] { "save-20", "save-40", "save-60" });
        Assert.True(double.IsFinite(largeCapital.Value));
    }

    [Fact]
    public void Solver_rejects_invalid_problem_definitions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FiniteHorizonBellmanProblem(
                StepCount: 0,
                StateLower: 0.0,
                StateUpper: 1.0,
                StateNodeCount: 5,
                Actions: [new BellmanAction(0, "a")],
                TerminalValue: state => state,
                ActionValue: (_, state, _, _) => state));

        Assert.Throws<ArgumentException>(() =>
            new FiniteHorizonBellmanProblem(
                StepCount: 1,
                StateLower: 0.0,
                StateUpper: 1.0,
                StateNodeCount: 5,
                Actions: [],
                TerminalValue: state => state,
                ActionValue: (_, state, _, _) => state));

        Assert.Throws<ArgumentNullException>(() =>
            new FiniteHorizonBellmanProblem(
                StepCount: 1,
                StateLower: 0.0,
                StateUpper: 1.0,
                StateNodeCount: 5,
                Actions: null!,
                TerminalValue: state => state,
                ActionValue: (_, state, _, _) => state));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FiniteHorizonBellmanProblem(
                StepCount: 1,
                StateLower: 0.0,
                StateUpper: 1.0,
                StateNodeCount: 5,
                Actions: [new BellmanAction(0, "a")],
                TerminalValue: state => state,
                ActionValue: (_, state, _, _) => state,
                MaxDerivativeOrder: 1));
    }

    [Fact]
    public void Model_rejects_evaluation_outside_state_domain()
    {
        var problem = new FiniteHorizonBellmanProblem(
            StepCount: 1,
            StateLower: 0.0,
            StateUpper: 1.0,
            StateNodeCount: 9,
            Actions: [new BellmanAction(0, "identity")],
            TerminalValue: state => state,
            ActionValue: (_, state, _, _) => state);

        FiniteHorizonBellmanModel model = new FiniteHorizonBellmanSolver().Solve(problem);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Evaluate(1.1));
    }
}
