using FixedRateBondSurrogate;

FixedRateBondExample.Run(Console.Out);

public static class FixedRateBondExample
{
    public static void Run(TextWriter output)
        => Run(Environment.GetCommandLineArgs().Skip(1).ToArray(), output);

    public static void Run(string[] args, TextWriter output)
    {
        if (args is ["--diagnostics"])
        {
            RunDiagnostics(output);
            return;
        }

        if (args is ["--surrogate-reproduction"])
        {
            RunSurrogateReproduction(output);
            return;
        }

        RunPricingExample(output);
    }

    private static void RunPricingExample(TextWriter output)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDefaultCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularTenYearFromFixture(fixture);
        IFixedRateBondReferencePricer pricer = new QlNetFixedRateBondReferencePricer();
        FixedRateBondResult result = pricer.Price(request);

        output.WriteLine("Fixed-rate bond reference pricer");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {fixture.FixtureId}");
        output.WriteLine($"Curve date    : {fixture.Source.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Curve source  : {fixture.Source.Institution}");
        output.WriteLine();
        output.WriteLine($"Valuation date : {request.ValuationDate:yyyy-MM-dd}");
        output.WriteLine($"Effective date : {request.EffectiveDate:yyyy-MM-dd}");
        output.WriteLine($"Maturity date  : {request.MaturityDate:yyyy-MM-dd}");
        output.WriteLine($"Coupon         : {request.Coupon:P2}");
        output.WriteLine($"Notional       : {request.Notional:N2}");
        output.WriteLine();
        output.WriteLine($"Dirty price    : {result.DirtyPrice:F8}");
        output.WriteLine($"Clean price    : {result.CleanPrice:F8}");
        output.WriteLine($"Accrued amount : {result.AccruedAmount:F8}");
        output.WriteLine($"NPV            : {result.NetPresentValue:F8}");
        output.WriteLine($"Cashflows      : {result.Cashflows.Count}");
    }

    private static void RunDiagnostics(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        SmoothnessDiagnosticReport report = SmoothnessDiagnostics.RunDefault(pricer);
        RateSensitivityPoint largestDv01 = report.RateSensitivities
            .OrderByDescending(point => Math.Abs(point.ZeroPillarDv01))
            .First();

        output.WriteLine("Fixed-rate bond smoothness diagnostics");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {report.FixtureId}");
        output.WriteLine($"Curve date    : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Coupon second-difference max : {report.MaxAbsCouponSecondDifference:E6}");
        output.WriteLine(
            $"Largest absolute zero-pillar DV01 : {largestDv01.PillarYears}Y = {largestDv01.ZeroPillarDv01:E6}");
        output.WriteLine();
        output.WriteLine("Coupon slice");
        foreach (CouponSlicePoint point in report.CouponSlice)
        {
            output.WriteLine(
                $"{point.Coupon,7:P2} dirty {point.DirtyPrice,12:F8} " +
                $"dPV/dc {point.CouponDerivative,12:F8} second {point.SecondDifference:E6}");
        }

        output.WriteLine();
        output.WriteLine("Zero-pillar DV01");
        foreach (RateSensitivityPoint point in report.RateSensitivities)
        {
            output.WriteLine(
                $"{point.PillarYears,3}Y dv01 {point.ZeroPillarDv01,13:E6} " +
                $"second {point.LocalSecondDifference:E6}");
        }

        output.WriteLine($"Rate bump slice points : {report.RateBumpSlice.Count}");

        output.WriteLine();
        output.WriteLine("Top maturity spike candidates");

        foreach (MaturitySlicePoint point in report.TopMaturitySpikeCandidates.Take(5))
        {
            output.WriteLine(
                $"{point.MaturityDate:yyyy-MM-dd} offset {point.OffsetDays,3}d " +
                $"cashflows {point.CashflowCount,2} spike {point.SecondDifference!.Value:E6}");
        }
    }

    private static void RunSurrogateReproduction(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        SurrogateExperimentReport report = FixedRateBondSurrogateExperiment.RunDefault(pricer);

        output.WriteLine("Fixed-rate bond surrogate reproduction");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {report.FixtureId}");
        output.WriteLine($"Curve date    : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Dimensions    : {string.Join(", ", report.Dimensions.Select(dimension => dimension.Name))}");
        output.WriteLine($"Validation points : {report.ValidationPoints.Count}");
        output.WriteLine();

        foreach (SurrogateModelSummary model in report.Models)
        {
            SurrogateMetricSummary pv = model.Metrics.Single(metric => metric.Name == "PV");
            SurrogateMetricSummary rateCoupon = model.Metrics.Single(metric => metric.Name == "rate-coupon mixed");

            output.WriteLine(
                $"{model.ModelName}: build evals {model.BuildEvaluations}, " +
                $"build seconds {model.BuildSeconds:F3}, " +
                $"PV rel max {pv.MaxRelativeError:P2}, " +
                $"rate-coupon mixed rel max {rateCoupon.MaxRelativeError:P2}");

            foreach (SurrogateMetricSummary metric in model.Metrics)
            {
                output.WriteLine(
                    $"  {metric.Name,-22} abs max {metric.MaxAbsoluteError,12:E6} " +
                    $"rel max {metric.MaxRelativeError,12:P2}");
            }
        }
    }
}
