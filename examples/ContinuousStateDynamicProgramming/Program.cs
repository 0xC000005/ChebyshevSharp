using ChebyshevSharp.DynamicProgramming;

namespace ContinuousStateDynamicProgramming;

public static class Program
{
    public static void Main()
    {
        PrintDynamicAssetAllocation();
        Console.WriteLine();
        PrintStochasticGrowth();
    }

    private static void PrintDynamicAssetAllocation()
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

        Console.WriteLine("Continuous-state DP: dynamic asset allocation");
        Console.WriteLine("Wealth | value | action | dV/dWealth");
        foreach (double wealth in new[] { 50.0, 100.0, 150.0, 200.0 })
        {
            BellmanEvaluation evaluation = model.Evaluate(wealth);
            Console.WriteLine(
                $"{wealth,6:F1} | {evaluation.Value,8:F5} | {evaluation.Action.Name,-8} | {evaluation.FirstDerivative,10:F6}");
        }
        Console.WriteLine($"Build evaluations: {model.BuildEvaluations}");
    }

    private static void PrintStochasticGrowth()
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

        Console.WriteLine("Continuous-state DP: stochastic growth");
        Console.WriteLine("Capital | value | action | dV/dCapital");
        foreach (double capital in new[] { 1.0, 2.0, 4.0, 6.0 })
        {
            BellmanEvaluation evaluation = model.Evaluate(capital);
            Console.WriteLine(
                $"{capital,7:F1} | {evaluation.Value,8:F5} | {evaluation.Action.Name,-8} | {evaluation.FirstDerivative,11:F6}");
        }
        Console.WriteLine($"Build evaluations: {model.BuildEvaluations}");
    }
}
