using CallableBondSurrogate;

CallableBondExample.Run(Environment.GetCommandLineArgs().Skip(1).ToArray(), Console.Out);

public static class CallableBondExample
{
    public static void Run(TextWriter output)
        => Run([], output);

    public static void Run(string[] args, TextWriter output)
    {
        if (args is ["--naive-surrogate-discovery"])
        {
            RunNaiveSurrogateDiscovery(output);
            return;
        }

        RunPricingExample(output);
    }

    private static void RunPricingExample(TextWriter output)
    {
        var pricer = new QlNetCallableBondReferencePricer();
        CallableBondRequest request = CallableBondScenarios.StandardThirtyYear();
        CallableBondResult result = pricer.Price(request);

        output.WriteLine("Callable fixed-rate bond reference pricer");
        output.WriteLine();
        output.WriteLine($"Valuation date : {request.ValuationDate:yyyy-MM-dd}");
        output.WriteLine($"Maturity date  : {request.MaturityDate:yyyy-MM-dd}");
        output.WriteLine($"First call     : {request.FirstCallDate:yyyy-MM-dd}");
        output.WriteLine($"Coupon         : {request.Coupon:P2}");
        output.WriteLine($"Call price     : {request.CallPrice:F4}");
        output.WriteLine($"Hull-White a   : {request.HullWhiteMeanReversion:F4}");
        output.WriteLine($"Hull-White sig : {request.HullWhiteSigma:F4}");
        output.WriteLine($"Tree steps     : {request.TreeTimeSteps}");
        output.WriteLine();
        output.WriteLine($"Callable dirty : {result.DirtyPrice:F8}");
        output.WriteLine($"Straight dirty : {result.StraightDirtyPrice:F8}");
        output.WriteLine($"Call option    : {result.EmbeddedCallValue:F8}");
        output.WriteLine($"Call dates     : {result.CallabilityCount}");
    }

    private static void RunNaiveSurrogateDiscovery(TextWriter output)
    {
        var pricer = new QlNetCallableBondReferencePricer();
        CallableNaiveSurrogateDiscoveryReport report = CallableNaiveSurrogateDiscovery.RunDefault(pricer);

        output.WriteLine("Callable bond naive surrogate discovery");
        output.WriteLine();
        output.WriteLine($"Curve fixture    : {report.FixtureId}");
        output.WriteLine($"Curve date       : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"65D public wrapper: {string.Join(", ", report.DimensionLabels.Take(3))}, ..., {string.Join(", ", report.DimensionLabels.Skip(60))}");
        output.WriteLine($"Dense full tensor : {report.Feasibility.ThreeNodeDenseGridLabel} = {report.Feasibility.ThreeNodeDenseGridCount}");
        output.WriteLine(report.Feasibility.DenseTensorConclusion);
        output.WriteLine();

        foreach (CallableNaiveSurrogateModelSummary model in report.Models)
        {
            output.WriteLine($"{model.ModelName}");
            output.WriteLine($"  Input dims   : {model.InputDimensionCount}");
            output.WriteLine($"  Build evals  : {model.BuildEvaluations:N0}");
            output.WriteLine($"  Build seconds: {model.BuildSeconds:F3}");
            foreach (CallableNaiveSurrogateMetricSummary metric in model.Metrics)
            {
                output.WriteLine(
                    $"  {metric.Name,-24} max abs {metric.MaxAbsoluteError:E6} " +
                    $"max rel {metric.MaxRelativeError:P2} worst {metric.WorstPointName}");
            }

            output.WriteLine();
        }
    }
}
