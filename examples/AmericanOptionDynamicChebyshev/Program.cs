using System.Diagnostics;

namespace AmericanOptionDynamicChebyshev;

public static class Program
{
    public static void Main()
    {
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var referencePricer = new QlNetAmericanOptionReferencePricer();
        AmericanOptionResult reference = referencePricer.Price(request);

        var lsm = new LongstaffSchwartzAmericanOptionPricer().Price(
            request,
            new RegressionMonteCarloSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 8675309));
        var lspi = new LspiAmericanOptionPricer().Price(
            request,
            new LspiSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 13579));
        var dynamicSettings = new DynamicChebyshevSettings(
            ExerciseSteps: 80,
            SpotNodeCount: 81,
            SpotLower: 5.0,
            SpotUpper: 250.0,
            QuadratureOrder: 8);
        DynamicChebyshevAmericanOptionModel dynamicModel = new DynamicChebyshevAmericanOptionPricer().Build(
            request,
            dynamicSettings);
        DynamicChebyshevEvaluation dynamicChebyshev = dynamicModel.Evaluate(request.Spot);
        SpeedComparison speed = MeasureSpeed(request, referencePricer, dynamicModel);

        Console.WriteLine("American option Dynamic Chebyshev case study");
        Console.WriteLine($"QLNet FD reference:   {reference.Price:F6}");
        Console.WriteLine($"European analytic:   {reference.EuropeanPrice:F6}");
        Console.WriteLine($"LSM regression:      {lsm.Price:F6} (se {lsm.StandardError:F6})");
        Console.WriteLine($"Stanford-style LSPI: {lspi.Price:F6} (se {lspi.StandardError:F6})");
        Console.WriteLine($"Dynamic Chebyshev:   {dynamicChebyshev.Price:F6}");
        Console.WriteLine($"  Delta/Gamma:       {dynamicChebyshev.Delta:F6} / {dynamicChebyshev.Gamma:F6}");
        Console.WriteLine($"  Build evals/time:  {dynamicModel.BuildEvaluations} / {dynamicModel.BuildTimeSeconds:F3}s");
        Console.WriteLine($"  Online eval:       {speed.ChebyshevMicrosecondsPerEval:F3} us");
        Console.WriteLine($"  QLNet ref eval:    {speed.ReferenceMillisecondsPerEval:F3} ms");
        Console.WriteLine($"  Online speedup:    {speed.Speedup:F1}x");
        Console.WriteLine($"  Grid max abs err:  price {speed.MaxPriceAbsError:E3}, delta {speed.MaxDeltaAbsError:E3}, gamma {speed.MaxGammaAbsError:E3}");
    }

    private static SpeedComparison MeasureSpeed(
        AmericanOptionRequest request,
        QlNetAmericanOptionReferencePricer referencePricer,
        DynamicChebyshevAmericanOptionModel dynamicModel)
    {
        double[] spots = Enumerable.Range(0, 21)
            .Select(i => 80.0 + 2.0 * i)
            .ToArray();

        _ = dynamicModel.Evaluate(request.Spot);
        _ = referencePricer.Price(request);

        var referenceResults = new AmericanOptionResult[spots.Length];
        var referenceTimer = Stopwatch.StartNew();
        for (int i = 0; i < spots.Length; i++)
        {
            referenceResults[i] = referencePricer.Price(request with { Spot = spots[i] });
        }

        referenceTimer.Stop();

        int repetitions = 2_000;
        var chebyshevTimer = Stopwatch.StartNew();
        for (int repeat = 0; repeat < repetitions; repeat++)
        {
            foreach (double spot in spots)
            {
                _ = dynamicModel.Evaluate(spot);
            }
        }

        chebyshevTimer.Stop();

        double maxPriceError = 0.0;
        double maxDeltaError = 0.0;
        double maxGammaError = 0.0;
        for (int i = 0; i < spots.Length; i++)
        {
            DynamicChebyshevEvaluation approximation = dynamicModel.Evaluate(spots[i]);
            maxPriceError = Math.Max(maxPriceError, Math.Abs(approximation.Price - referenceResults[i].Price));
            maxDeltaError = Math.Max(maxDeltaError, Math.Abs(approximation.Delta - referenceResults[i].Delta));
            maxGammaError = Math.Max(maxGammaError, Math.Abs(approximation.Gamma - referenceResults[i].Gamma));
        }

        double referenceMillisecondsPerEval = referenceTimer.Elapsed.TotalMilliseconds / spots.Length;
        double chebyshevMicrosecondsPerEval = chebyshevTimer.Elapsed.TotalMilliseconds * 1_000.0 / (spots.Length * repetitions);
        double speedup = (referenceMillisecondsPerEval * 1_000.0) / chebyshevMicrosecondsPerEval;

        return new SpeedComparison(
            referenceMillisecondsPerEval,
            chebyshevMicrosecondsPerEval,
            speedup,
            maxPriceError,
            maxDeltaError,
            maxGammaError);
    }

    private sealed record SpeedComparison(
        double ReferenceMillisecondsPerEval,
        double ChebyshevMicrosecondsPerEval,
        double Speedup,
        double MaxPriceAbsError,
        double MaxDeltaAbsError,
        double MaxGammaAbsError);
}
