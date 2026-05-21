using System.Globalization;
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

        if (args is ["--naive-surrogate-discovery"])
        {
            RunNaiveSurrogateDiscovery(output);
            return;
        }

        if (args is ["--naive-maturity-scan-csv"])
        {
            RunNaiveMaturityScanCsv(output);
            return;
        }

        if (args is ["--structured-alternatives"])
        {
            RunStructuredAlternatives(output);
            return;
        }

        if (args is ["--analytic-coupon-decomposition"])
        {
            RunAnalyticCouponDecomposition(output);
            return;
        }

        RunPricingExample(output);
    }

    private static void RunPricingExample(TextWriter output)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        IFixedRateBondReferencePricer pricer = new QlNetFixedRateBondReferencePricer();
        FixedRateBondResult result = pricer.Price(request);
        FixedRateBondConventionSummary conventions = QlNetFixedRateBondReferencePricer.SupportedConventions;

        output.WriteLine("Fixed-rate bond reference pricer");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {fixture.FixtureId}");
        output.WriteLine($"Curve date    : {fixture.Source.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Curve source  : {fixture.Source.Institution}");
        output.WriteLine($"Curve pillars : {request.ZeroCurve.Count}");
        output.WriteLine();
        output.WriteLine("Conventions");
        output.WriteLine($"Calendar       : {conventions.Calendar}");
        output.WriteLine($"Schedule       : {conventions.ScheduleFrequency}, {conventions.DateGeneration}");
        output.WriteLine($"Day counts     : coupon {conventions.CouponDayCount}, curve {conventions.CurveDayCount}");
        output.WriteLine($"Business days  : {conventions.BusinessDayConvention}");
        output.WriteLine($"Curve method   : {conventions.CurveInterpolation}, {conventions.CurveCompounding}");
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

    private static void RunNaiveSurrogateDiscovery(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        NaiveSurrogateDiscoveryReport report = NaiveSurrogateDiscovery.RunDefault(pricer);

        output.WriteLine("Fixed-rate bond naive surrogate discovery");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {report.FixtureId}");
        output.WriteLine($"Curve date    : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Dimensions    : {string.Join(", ", report.Dimensions.Select(dimension => dimension.Name))}");
        output.WriteLine($"Validation points : {report.ValidationPoints.Count}");
        output.WriteLine();
        output.WriteLine(
            $"Dense full tensor : {report.Feasibility.ThreeNodeDenseGridLabel} = " +
            $"{report.Feasibility.ThreeNodeDenseGridCount} nodes; " +
            $"{report.Feasibility.FiveNodeDenseGridLabel} = {report.Feasibility.FiveNodeDenseGridCount} nodes");
        output.WriteLine(report.Feasibility.DenseTensorConclusion);
        output.WriteLine();

        foreach (NaiveSurrogateModelSummary model in report.Models)
        {
            NaiveSurrogateMetricSummary pv = model.Metrics.Single(metric => metric.Name == "PV");
            NaiveSurrogateMetricSummary maturity = model.Metrics.Single(metric => metric.Name == "maturity sensitivity");
            NaiveSurrogateMetricSummary couponMaturity = model.Metrics.Single(metric => metric.Name == "coupon-maturity mixed");

            output.WriteLine(
                $"{model.ModelName}: build evals {model.BuildEvaluations}, " +
                $"build seconds {model.BuildSeconds:F3}, " +
                $"PV rel max {pv.MaxRelativeError:P2}, " +
                $"maturity sensitivity rel max {maturity.MaxRelativeError:P2}, " +
                $"coupon-maturity mixed rel max {couponMaturity.MaxRelativeError:P2}");

            foreach (NaiveSurrogateMetricSummary metric in model.Metrics)
            {
                output.WriteLine(
                    $"  {metric.Name,-24} abs max {metric.MaxAbsoluteError,12:E6} " +
                    $"rel max {metric.MaxRelativeError,12:P2} worst {metric.WorstPointName}");
            }
        }

        output.WriteLine();
        output.WriteLine("Structural sanity checks");
        foreach (NaiveSurrogateSanityCheck check in report.SanityChecks)
        {
            output.WriteLine(
                $"{check.Name}: baseline {check.BaselineValue:E6} " +
                $"tolerance {check.BaselineTolerance:E1}");
            foreach (NaiveSurrogateSanityModelValue modelValue in check.ModelValues)
            {
                output.WriteLine(
                    $"  {modelValue.ModelName,-11} value {modelValue.Value,12:E6} " +
                    $"abs error {modelValue.AbsoluteError,12:E6}");
            }
        }

        output.WriteLine();
        output.WriteLine("Top maturity spike candidates");
        foreach (NaiveMaturitySpikeCandidate point in report.TopMaturitySpikeCandidates.Take(5))
        {
            output.WriteLine(
                $"{point.MaturityDate:yyyy-MM-dd} offset {point.OffsetDays,3}d " +
                $"cashflows {point.CashflowCount,2} second {point.SecondDifference:E6} " +
                $"left {point.LeftSlopePerYear:E6} right {point.RightSlopePerYear:E6}");
        }
    }

    private static void RunNaiveMaturityScanCsv(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        IReadOnlyList<NaiveMaturityScanPoint> maturityScan = NaiveSurrogateDiscovery.RunMaturityScanDefault(pricer);

        output.WriteLine(
            "boundary_date,offset_days,maturity_date,cashflow_count,dirty_price,central_slope_per_year,second_difference");
        foreach (NaiveMaturityScanPoint point in maturityScan)
        {
            output.WriteLine(
                string.Join(
                    ",",
                    point.BoundaryDate.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    point.OffsetDays.ToString(CultureInfo.InvariantCulture),
                    point.MaturityDate.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    point.CashflowCount.ToString(CultureInfo.InvariantCulture),
                    FormatCsvDouble(point.DirtyPrice),
                    FormatCsvDouble(point.CentralSlopePerYear),
                    FormatCsvDouble(point.SecondDifference)));
        }
    }

    private static void RunStructuredAlternatives(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        StructuredAlternativesReport report = StructuredAlternativesBenchmark.RunDefault(pricer);

        output.WriteLine("Fixed-rate bond structured alternatives");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {report.FixtureId}");
        output.WriteLine($"Curve date    : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"full wrapper  : {report.WrapperContract}");
        output.WriteLine($"Research basis: {report.ResearchBasis}");
        output.WriteLine($"Clone validation points        : {report.CloneValidationPoints.Count}");
        output.WriteLine($"Factor-aligned validation points: {report.FactorAlignedValidationPoints.Count}");
        output.WriteLine();

        foreach (StructuredAlternativeModelSummary model in report.Models)
        {
            NaiveSurrogateMetricSummary pv = model.Metrics.Single(metric => metric.Name == "PV");
            NaiveSurrogateMetricSummary maturity = model.Metrics.Single(metric => metric.Name == "maturity sensitivity");
            NaiveSurrogateMetricSummary couponMaturity = model.Metrics.Single(metric => metric.Name == "coupon-maturity mixed");
            NaiveSurrogateMetricSummary factorPv = model.FactorAlignedMetrics.Single(metric => metric.Name == "PV");

            output.WriteLine(
                $"{model.ModelName}: build evals {model.BuildEvaluations}, " +
                $"build seconds {model.BuildSeconds:F3}, " +
                $"internal dims {model.InternalDimensionCount}, " +
                $"buckets {model.BucketCount}");
            output.WriteLine($"  Method        : {model.InternalMethod}");
            output.WriteLine(
                $"  Clone metrics : PV rel max {pv.MaxRelativeError:P2}, " +
                $"maturity rel max {maturity.MaxRelativeError:P2}, " +
                $"coupon-maturity rel max {couponMaturity.MaxRelativeError:P2}");
            output.WriteLine($"  Factor metrics: PV rel max {factorPv.MaxRelativeError:P2}");
            output.WriteLine($"  Interpretation: {model.Interpretation}");
        }
    }

    private static void RunAnalyticCouponDecomposition(TextWriter output)
    {
        var pricer = new QlNetFixedRateBondReferencePricer();
        AnalyticCouponDecompositionReport report = AnalyticCouponDecompositionBenchmark.RunDefault(pricer);

        output.WriteLine("Fixed-rate bond analytic coupon decomposition");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {report.FixtureId}");
        output.WriteLine($"Curve date    : {report.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"full wrapper  : {report.WrapperContract}");
        output.WriteLine($"Formula       : {report.Formula}");
        output.WriteLine(
            $"coupon-linearity max abs {report.Identity.MaxAbsoluteError:E6}, " +
            $"max rel {report.Identity.MaxRelativeError:P6}");
        output.WriteLine($"Identity points: {report.Identity.ValidationPointCount}");
        output.WriteLine();

        foreach (AnalyticCouponModelSummary model in report.Models)
        {
            NaiveSurrogateMetricSummary pv = model.Metrics.Single(metric => metric.Name == "PV");
            NaiveSurrogateMetricSummary coupon = model.Metrics.Single(metric => metric.Name == "coupon derivative");
            NaiveSurrogateMetricSummary maturity = model.Metrics.Single(metric => metric.Name == "maturity sensitivity");
            NaiveSurrogateMetricSummary couponMaturity = model.Metrics.Single(metric => metric.Name == "coupon-maturity mixed");
            NaiveSurrogateMetricSummary factorPv = model.FactorAlignedMetrics.Single(metric => metric.Name == "PV");

            output.WriteLine(
                $"{model.ModelName}: build evals {model.BuildEvaluations}, " +
                $"build seconds {model.BuildSeconds:F3}, " +
                $"internal dims {model.InternalDimensionCount}, " +
                $"buckets {model.BucketCount}");
            output.WriteLine($"  Method        : {model.InternalMethod}");
            output.WriteLine(
                $"  Clone metrics : PV rel max {pv.MaxRelativeError:P2}, " +
                $"coupon rel max {coupon.MaxRelativeError:P2}, " +
                $"maturity rel max {maturity.MaxRelativeError:P2}, " +
                $"coupon-maturity rel max {couponMaturity.MaxRelativeError:P2}");
            output.WriteLine($"  Factor metrics: PV rel max {factorPv.MaxRelativeError:P2}");
            output.WriteLine($"  Interpretation: {model.Interpretation}");
        }
    }

    private static string FormatCsvDouble(double? value)
        => value.HasValue ? value.Value.ToString("G17", CultureInfo.InvariantCulture) : string.Empty;
}
