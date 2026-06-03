using System.Diagnostics;

namespace AmericanOptionDynamicChebyshev;

public static class Program
{
    public static void Main(string[]? args = null)
    {
        args ??= [];
        AmericanOptionRequest request = AmericanOptionScenarios.StandardPut();
        var referencePricer = new QlNetAmericanOptionReferencePricer();
        AmericanOptionResult reference = referencePricer.Price(request);

        var lsmTimer = Stopwatch.StartNew();
        var lsm = new LongstaffSchwartzAmericanOptionPricer().Price(
            request,
            new RegressionMonteCarloSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 8675309));
        lsmTimer.Stop();
        var lspiTimer = Stopwatch.StartNew();
        var lspi = new LspiAmericanOptionPricer().Price(
            request,
            new LspiSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 13579));
        lspiTimer.Stop();
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
        Console.WriteLine($"  LSM diagnostics:   time {lsmTimer.Elapsed.TotalMilliseconds:F3} ms, exercised {lsm.ExercisedPathFraction:P2}");
        Console.WriteLine($"Stanford-style LSPI: {lspi.Price:F6} (se {lspi.StandardError:F6})");
        Console.WriteLine($"  LSPI diagnostics:  time {lspiTimer.Elapsed.TotalMilliseconds:F3} ms, policy iters {lspi.PolicyIterations}, features {lspi.FeatureCount}, boundary paths {lspi.BoundaryDecisionCount}");
        Console.WriteLine($"Dynamic Chebyshev:   {dynamicChebyshev.Price:F6}");
        Console.WriteLine($"  Delta/Gamma:       {dynamicChebyshev.Delta:F6} / {dynamicChebyshev.Gamma:F6}");
        Console.WriteLine($"  Build evals/time:  {dynamicModel.BuildEvaluations} / {dynamicModel.BuildTimeSeconds:F3}s");
        Console.WriteLine($"  Online eval:       {speed.ChebyshevMicrosecondsPerEval:F3} us");
        Console.WriteLine($"  QLNet ref eval:    {speed.ReferenceMillisecondsPerEval:F3} ms");
        Console.WriteLine($"  Online speedup:    {speed.Speedup:F1}x");
        Console.WriteLine($"  Grid max abs err:  price {speed.MaxPriceAbsError:E3}, delta {speed.MaxDeltaAbsError:E3}, gamma {speed.MaxGammaAbsError:E3}");

        if (args.Contains("--case-assessment", StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine();
            PrintCaseAssessment(request, referencePricer, dynamicModel);
        }

        if (args.Contains("--thesis-benchmark", StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine();
            PrintThesisBenchmark();
        }

        if (args.Contains("--boundary-diagnostics", StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine();
            PrintBoundaryDiagnostics(request, dynamicModel, speed);
        }
    }

    private static void PrintThesisBenchmark()
    {
        Console.WriteLine("Kan thesis Table 2.1 comparable benchmark");
        Console.WriteLine("n | S0 | thesis LS | thesis se | ChebyshevSharp LS | abs diff | rel diff");
        foreach (ThesisMaxCallRow row in ThesisMaxCallBenchmark.Run())
        {
            Console.WriteLine(
                $"{row.AssetCount} | " +
                $"{row.InitialSpot:F0} | " +
                $"{row.ThesisLongstaffSchwartz:F3} | " +
                $"{row.ThesisStandardError:F3} | " +
                $"{row.ChebyshevSharpLongstaffSchwartz:F3} | " +
                $"{row.AbsoluteDifference:F3} | " +
                $"{row.RelativeDifference:P3}");
        }
    }

    private static void PrintCaseAssessment(
        AmericanOptionRequest request,
        QlNetAmericanOptionReferencePricer referencePricer,
        DynamicChebyshevAmericanOptionModel dynamicModel)
    {
        double[] spots = [82.0, 86.0, 90.0, 94.0, 98.0, 102.0, 106.0, 110.0, 114.0, 118.0];

        PrintDynamicChebyshevCases(request, referencePricer, dynamicModel, spots);
        Console.WriteLine();
        PrintLongstaffSchwartzCases(request, referencePricer, spots);
        Console.WriteLine();
        PrintLspiCases(request, referencePricer, spots);
    }

    private static void PrintBoundaryDiagnostics(
        AmericanOptionRequest request,
        DynamicChebyshevAmericanOptionModel dynamicModel,
        SpeedComparison speed)
    {
        double[] spots = [70.0, 75.0, 80.0, 82.0, 86.0, 90.0, 94.0, 98.0, 102.0, 106.0, 110.0];

        Console.WriteLine("Dynamic Chebyshev payoff/continuation boundary diagnostics");
        Console.WriteLine("Spot | payoff h(S) | continuation C(S) | value max(h,C) | decision | C(S)-h(S)");
        foreach (double spot in spots)
        {
            double payoff = dynamicModel.Payoff(spot);
            double continuation = dynamicModel.Continuation(spot);
            double value = Math.Max(payoff, continuation);
            string decision = continuation >= payoff ? "continue" : "exercise";
            Console.WriteLine(
                $"{spot:F1} | {payoff:F6} | {continuation:F6} | {value:F6} | {decision} | {continuation - payoff:F6}");
        }

        double? boundary = EstimateBoundary(dynamicModel, 50.0, 120.0, step: 0.25);
        Console.WriteLine();
        Console.WriteLine("Bellman-style implemented diagnostics");
        Console.WriteLine("Check | Result");
        Console.WriteLine($"Transition first moment residual | < 1e-12 in unit test");
        Console.WriteLine($"Max spot-grid price abs error vs QLNet FD | {speed.MaxPriceAbsError:E3}");
        Console.WriteLine($"Max spot-grid Delta abs error vs QLNet FD | {speed.MaxDeltaAbsError:E3}");
        Console.WriteLine($"Max spot-grid Gamma abs error vs QLNet FD | {speed.MaxGammaAbsError:E3}");
        Console.WriteLine($"Estimated first-step exercise boundary | {(boundary.HasValue ? boundary.Value.ToString("F2") : "not found")} spot");
        Console.WriteLine($"Domain | [{5.0:F1}, {250.0:F1}] spot");
    }

    private static double? EstimateBoundary(
        DynamicChebyshevAmericanOptionModel dynamicModel,
        double lower,
        double upper,
        double step)
    {
        double previousSpot = lower;
        double previousGap = dynamicModel.Continuation(previousSpot) - dynamicModel.Payoff(previousSpot);

        for (double spot = lower + step; spot <= upper + 1e-12; spot += step)
        {
            double gap = dynamicModel.Continuation(spot) - dynamicModel.Payoff(spot);
            if (Math.Sign(previousGap) != Math.Sign(gap))
            {
                double weight = Math.Abs(previousGap) / (Math.Abs(previousGap) + Math.Abs(gap));
                return previousSpot + weight * (spot - previousSpot);
            }

            previousSpot = spot;
            previousGap = gap;
        }

        return null;
    }

    private static void PrintDynamicChebyshevCases(
        AmericanOptionRequest request,
        QlNetAmericanOptionReferencePricer referencePricer,
        DynamicChebyshevAmericanOptionModel dynamicModel,
        double[] spots)
    {
        Console.WriteLine("Case-level QLNet vs Dynamic Chebyshev assessment");
        Console.WriteLine("Spot | QLNet price | Cheb price | price rel diff | QLNet delta | Cheb delta | delta rel diff | QLNet gamma | Cheb gamma | gamma rel diff");
        foreach (double spot in spots)
        {
            AmericanOptionResult reference = referencePricer.Price(request with { Spot = spot });
            DynamicChebyshevEvaluation approximation = dynamicModel.Evaluate(spot);
            Console.WriteLine(
                $"{spot:F1} | " +
                $"{reference.Price:F6} | {approximation.Price:F6} | {RelativeDiff(approximation.Price, reference.Price):P3} | " +
                $"{reference.Delta:F6} | {approximation.Delta:F6} | {RelativeDiff(approximation.Delta, reference.Delta):P3} | " +
                $"{reference.Gamma:F6} | {approximation.Gamma:F6} | {RelativeDiff(approximation.Gamma, reference.Gamma):P3}");
        }
    }

    private static void PrintLongstaffSchwartzCases(
        AmericanOptionRequest request,
        QlNetAmericanOptionReferencePricer referencePricer,
        double[] spots)
    {
        var pricer = new LongstaffSchwartzAmericanOptionPricer();
        var settings = new RegressionMonteCarloSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 8675309);
        Console.WriteLine("Case-level QLNet vs Longstaff-Schwartz assessment");
        Console.WriteLine("Spot | QLNet price | LSM price | price rel diff | QLNet delta | LSM delta | delta rel diff | QLNet gamma | LSM gamma | gamma rel diff");
        foreach (double spot in spots)
        {
            AmericanOptionResult reference = referencePricer.Price(request with { Spot = spot });
            BaselineGreeks approximation = EvaluateLongstaffSchwartzWithGreeks(pricer, request, settings, spot);
            Console.WriteLine(
                $"{spot:F1} | " +
                $"{reference.Price:F6} | {approximation.Price:F6} | {RelativeDiff(approximation.Price, reference.Price):P3} | " +
                $"{reference.Delta:F6} | {approximation.Delta:F6} | {RelativeDiff(approximation.Delta, reference.Delta):P3} | " +
                $"{reference.Gamma:F6} | {approximation.Gamma:F6} | {RelativeDiff(approximation.Gamma, reference.Gamma):P3}");
        }
    }

    private static void PrintLspiCases(
        AmericanOptionRequest request,
        QlNetAmericanOptionReferencePricer referencePricer,
        double[] spots)
    {
        var pricer = new LspiAmericanOptionPricer();
        var settings = new LspiSettings(PathCount: 12_000, ExerciseSteps: 50, Seed: 13579);
        Console.WriteLine("Case-level QLNet vs LSPI assessment");
        Console.WriteLine("Spot | QLNet price | LSPI price | price rel diff | QLNet delta | LSPI delta | delta rel diff | QLNet gamma | LSPI gamma | gamma rel diff");
        foreach (double spot in spots)
        {
            AmericanOptionResult reference = referencePricer.Price(request with { Spot = spot });
            BaselineGreeks approximation = EvaluateLspiWithGreeks(pricer, request, settings, spot);
            Console.WriteLine(
                $"{spot:F1} | " +
                $"{reference.Price:F6} | {approximation.Price:F6} | {RelativeDiff(approximation.Price, reference.Price):P3} | " +
                $"{reference.Delta:F6} | {approximation.Delta:F6} | {RelativeDiff(approximation.Delta, reference.Delta):P3} | " +
                $"{reference.Gamma:F6} | {approximation.Gamma:F6} | {RelativeDiff(approximation.Gamma, reference.Gamma):P3}");
        }
    }

    private static BaselineGreeks EvaluateLongstaffSchwartzWithGreeks(
        LongstaffSchwartzAmericanOptionPricer pricer,
        AmericanOptionRequest request,
        RegressionMonteCarloSettings settings,
        double spot)
    {
        double bump = request.SpotBump;
        double down = pricer.Price(request with { Spot = spot - bump }, settings).Price;
        double mid = pricer.Price(request with { Spot = spot }, settings).Price;
        double up = pricer.Price(request with { Spot = spot + bump }, settings).Price;
        return FromBumpedPrices(down, mid, up, bump);
    }

    private static BaselineGreeks EvaluateLspiWithGreeks(
        LspiAmericanOptionPricer pricer,
        AmericanOptionRequest request,
        LspiSettings settings,
        double spot)
    {
        double bump = request.SpotBump;
        double down = pricer.Price(request with { Spot = spot - bump }, settings).Price;
        double mid = pricer.Price(request with { Spot = spot }, settings).Price;
        double up = pricer.Price(request with { Spot = spot + bump }, settings).Price;
        return FromBumpedPrices(down, mid, up, bump);
    }

    private static BaselineGreeks FromBumpedPrices(
        double down,
        double mid,
        double up,
        double bump)
        => new(
            Price: mid,
            Delta: (up - down) / (2.0 * bump),
            Gamma: (up - 2.0 * mid + down) / (bump * bump));

    private static double RelativeDiff(double approximation, double reference)
        => Math.Abs(approximation - reference) / Math.Max(Math.Abs(reference), 1e-12);

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

    private sealed record BaselineGreeks(
        double Price,
        double Delta,
        double Gamma);
}
