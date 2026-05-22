using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondAccuracyRecipeSearchTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();
    private static readonly Lazy<AccuracyRecipeSearchReport> Report =
        new(() => AccuracyRecipeSearch.RunDefault(Pricer));

    [Fact]
    public void Phase12_report_preserves_full_wrapper_and_records_oracles()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.NotEmpty(report.CloneValidationPoints);
        Assert.NotEmpty(report.ProjectionOracle.Points);
        Assert.NotEmpty(report.DerivativeOracle.RateStepDiagnostics);
        Assert.NotEmpty(report.DerivativeOracle.MaturityStepDiagnostics);
        Assert.NotEmpty(report.ScheduleDispatch.Diagnostics);
        Assert.Equal(3, report.NotionalScaling.ValidationPointCount);
        Assert.Contains("projection", report.Decision, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Projection_oracle_separates_arbitrary_clone_error_from_factor_aligned_points()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.True(report.ProjectionOracle.MaxClonePvAbsoluteError > 1e-6);
        Assert.True(report.ProjectionOracle.MaxClonePvRelativeError > 1e-6);
        Assert.True(report.ProjectionOracle.MaxFactorAlignedPvAbsoluteError < report.ProjectionOracle.MaxClonePvAbsoluteError);
        Assert.True(report.ProjectionOracle.MaxFactorAlignedPvRelativeError < report.ProjectionOracle.MaxClonePvRelativeError);
    }

    [Fact]
    public void Projection_oracle_compares_richer_deterministic_curve_basis()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        AccuracyProjectionBasisSummary richer = Assert.Single(
            report.ProjectionOracle.AlternativeBases,
            basis => basis.Name == "Five-factor deterministic curve basis");

        Assert.Equal(5, richer.FactorCount);
        Assert.True(double.IsFinite(richer.MaxClonePvAbsoluteError));
        Assert.True(double.IsFinite(richer.MaxClonePvRelativeError));
        Assert.True(richer.MaxClonePvRelativeError > 1e-6);
    }

    [Fact]
    public void Derivative_oracle_records_step_sensitivity_and_post_maturity_support()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.All(report.DerivativeOracle.RateStepDiagnostics, diagnostic =>
        {
            Assert.True(double.IsFinite(diagnostic.Step));
            Assert.True(double.IsFinite(diagnostic.Value));
        });
        Assert.All(report.DerivativeOracle.MaturityStepDiagnostics, diagnostic =>
        {
            Assert.True(double.IsFinite(diagnostic.Step));
            Assert.True(double.IsFinite(diagnostic.Value));
        });
        Assert.True(Math.Abs(report.DerivativeOracle.PostMaturityUnsupportedPillarDv01) < 1e-10);
    }

    [Fact]
    public void Active_support_oracle_preserves_price_when_post_maturity_pillars_are_removed()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.NotEmpty(report.ActiveSupport.Points);
        Assert.True(report.ActiveSupport.MaxPvAbsoluteError < 1e-8);
        Assert.Contains(report.ActiveSupport.Points, point => point.ActiveCurveBumpDimensions < 60);
        Assert.Contains(report.ActiveSupport.Points, point => point.ActiveCurveBumpDimensions == 60);
    }

    [Fact]
    public void Active_pillar_candidate_keeps_full_wrapper_with_smaller_local_dimension()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        AccuracyRecipeModelSummary activeTt = Assert.Single(
            report.CandidateModels,
            model => model.ModelName == "10Y active-pillar TT");
        Assert.Contains(report.CandidateModels, model => model.ModelName == "10Y narrow active-pillar TT");
        Assert.Contains(report.CandidateModels, model => model.ModelName == "10Y fixed-trade curve-only TT");
        Assert.Contains(report.CandidateModels, model => model.ModelName == "Schedule-resolved cashflow Chebyshev kernels");

        Assert.Equal(62, activeTt.PublicInputDimensionCount);
        Assert.InRange(activeTt.InternalDimensionCount, 1, 61);
        Assert.True(activeTt.BuildEvaluations > 0);
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "PV");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "10Y DV01");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "coupon derivative");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "maturity sensitivity");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "maturity left sensitivity");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "maturity right sensitivity");
        Assert.Contains(activeTt.Metrics, metric => metric.Name == "coupon-maturity mixed");
        Assert.All(activeTt.Metrics, metric =>
        {
            Assert.True(double.IsFinite(metric.MaxAbsoluteError));
            Assert.True(double.IsFinite(metric.MaxRelativeError));
        });
    }

    [Fact]
    public void Schedule_resolved_cashflow_kernel_candidate_preserves_full_wrapper_and_matches_reference_risk()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        AccuracyRecipeModelSummary model = Assert.Single(
            report.CandidateModels,
            model => model.ModelName == "Schedule-resolved cashflow Chebyshev kernels");

        Assert.Equal(62, model.PublicInputDimensionCount);
        Assert.Equal(2, model.InternalDimensionCount);
        Assert.True(model.BuildEvaluations > 0);
        Assert.True(double.IsFinite(model.BaselineEvalMicroseconds) && model.BaselineEvalMicroseconds > 0.0);
        Assert.True(double.IsFinite(model.ModelEvalMicroseconds) && model.ModelEvalMicroseconds > 0.0);
        Assert.True(double.IsFinite(model.EvalSpeedup) && model.EvalSpeedup > 0.0);
        Assert.True(model.ValidationPointCount >= 80);

        AssertMetricBelow(model, "PV", maxAbsoluteError: 1e-6, maxRelativeError: 1e-8);
        AssertMetricBelow(model, "10Y DV01", maxAbsoluteError: 1e-7, maxRelativeError: 1e-5);
        AssertMetricBelow(model, "all-pillar DV01", maxAbsoluteError: 1e-7, maxRelativeError: 1e-5);
        AssertMetricBelow(model, "coupon derivative", maxAbsoluteError: 1e-5, maxRelativeError: 1e-7);
        AssertMetricBelow(model, "maturity sensitivity", maxAbsoluteError: 1e-4, maxRelativeError: 1e-4);
        AssertMetricBelow(model, "10Y rate-coupon mixed", maxAbsoluteError: 1e-4, maxRelativeError: 1e-3);
        AssertMetricAbsoluteBelow(model, "10Y rate-maturity mixed", maxAbsoluteError: 1e-6);
        AssertMetricBelow(model, "coupon-maturity mixed", maxAbsoluteError: 1e-4, maxRelativeError: 1e-4);
        AssertMetricAbsoluteBelow(model, "10Y-10.5Y rate-rate mixed", maxAbsoluteError: 1e-5);
    }

    [Fact]
    public void Schedule_resolved_cashflow_kernel_preserves_dirty_price_under_non_100_notional()
    {
        AccuracyRecipeSearchReport report = Report.Value;

        Assert.Equal(250.0, report.NotionalScaling.Notional);
        Assert.True(report.NotionalScaling.MaxDirtyPriceAbsoluteError < 1e-6);
        Assert.True(report.NotionalScaling.MaxDirtyPriceRelativeError < 1e-8);
    }

    [Fact]
    public void Schedule_resolved_cashflow_pricer_prices_full_request_like_reference_pricer()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        var surrogate = new ScheduleResolvedCashflowChebyshevBondPricer(Pricer, baseRequest);

        ZeroRatePillar[] curve = baseRequest.ZeroCurve.ToArray();
        for (int i = 1; i < curve.Length; i++)
        {
            double bumpBp = -70.0 + 140.0 * (i - 1) / 59.0;
            curve[i] = curve[i] with { ZeroRate = curve[i].ZeroRate + bumpBp * 1e-4 };
        }

        FixedRateBondRequest request = baseRequest with
        {
            Coupon = 0.0675,
            MaturityDate = baseRequest.ValuationDate.Date.AddDays((int)Math.Round(365.25 * 18.75)),
            Notional = 250.0,
            ZeroCurve = curve,
        };

        double expected = Pricer.Price(request).DirtyPrice;
        double actual = surrogate.PriceDirty(request);

        Assert.Equal(expected, actual, precision: 8);
    }

    [Fact]
    public void Schedule_resolved_cashflow_pricer_rejects_curve_bumps_outside_training_domain()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        var surrogate = new ScheduleResolvedCashflowChebyshevBondPricer(Pricer, baseRequest);

        ZeroRatePillar[] curve = baseRequest.ZeroCurve.ToArray();
        curve[20] = curve[20] with { ZeroRate = curve[20].ZeroRate + 0.0200 };

        FixedRateBondRequest request = baseRequest with
        {
            MaturityDate = baseRequest.ValuationDate.Date.AddDays((int)Math.Round(365.25 * 10.0)),
            ZeroCurve = curve,
        };

        Assert.Throws<ArgumentOutOfRangeException>(() => surrogate.PriceDirty(request));
    }

    [Fact]
    public void Schedule_resolved_cashflow_pricer_rejects_incompatible_requests_and_coordinates()
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDenseSemiannualCurveFixture();
        FixedRateBondRequest baseRequest = FixedRateBondMarketData.RegularThirtyYearFromDenseFixture(fixture);
        var surrogate = new ScheduleResolvedCashflowChebyshevBondPricer(Pricer, baseRequest);

        Assert.Throws<ArgumentException>(() =>
            new ScheduleResolvedCashflowChebyshevBondPricer(
                Pricer,
                baseRequest with { ZeroCurve = baseRequest.ZeroCurve.Take(2).ToArray() }));
        Assert.Throws<ArgumentException>(() => surrogate.Eval(new double[61]));

        double[] point = new double[ScheduleResolvedCashflowChebyshevBondPricer.PublicInputDimensionCount];
        point[60] = 0.13;
        point[61] = 10.0;
        Assert.Throws<ArgumentOutOfRangeException>(() => surrogate.Eval(point));

        point[60] = 0.045;
        point[61] = 31.0;
        Assert.Throws<ArgumentOutOfRangeException>(() => surrogate.Eval(point));

        Assert.Throws<ArgumentException>(() => surrogate.PriceDirty(baseRequest with
        {
            ValuationDate = baseRequest.ValuationDate.AddDays(1),
        }));
        Assert.Throws<ArgumentException>(() => surrogate.PriceDirty(baseRequest with
        {
            EffectiveDate = baseRequest.EffectiveDate.AddDays(1),
        }));
        Assert.Throws<ArgumentException>(() => surrogate.PriceDirty(baseRequest with
        {
            SettlementDays = 1,
        }));
        Assert.Throws<ArgumentException>(() => surrogate.PriceDirty(baseRequest with
        {
            ZeroCurve = baseRequest.ZeroCurve.SkipLast(1).ToArray(),
        }));

        ZeroRatePillar[] shiftedDates = baseRequest.ZeroCurve.ToArray();
        shiftedDates[10] = shiftedDates[10] with { Date = shiftedDates[10].Date.AddDays(1) };
        Assert.Throws<ArgumentException>(() => surrogate.PriceDirty(baseRequest with
        {
            ZeroCurve = shiftedDates,
        }));
    }

    [Fact]
    public void Accuracy_recipe_search_mode_writes_phase12_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--accuracy-recipe-search"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond accuracy recipe search", output);
        Assert.Contains("Projection oracle", output);
        Assert.Contains("Derivative oracle", output);
        Assert.Contains("Schedule dispatch", output);
        Assert.Contains("Notional scaling check", output);
        Assert.Contains("Next decision", output);
    }

    private static void AssertMetricBelow(
        AccuracyRecipeModelSummary model,
        string metricName,
        double maxAbsoluteError,
        double maxRelativeError)
    {
        AccuracyRecipeMetricSummary metric = model.Metrics.Single(metric => metric.Name == metricName);

        Assert.True(
            metric.MaxAbsoluteError <= maxAbsoluteError,
            $"{model.ModelName} {metricName} max abs {metric.MaxAbsoluteError:E6} > {maxAbsoluteError:E6}");
        Assert.True(
            metric.MaxRelativeError <= maxRelativeError,
            $"{model.ModelName} {metricName} max rel {metric.MaxRelativeError:E6} > {maxRelativeError:E6}");
    }

    private static void AssertMetricAbsoluteBelow(
        AccuracyRecipeModelSummary model,
        string metricName,
        double maxAbsoluteError)
    {
        AccuracyRecipeMetricSummary metric = model.Metrics.Single(metric => metric.Name == metricName);

        Assert.True(
            metric.MaxAbsoluteError <= maxAbsoluteError,
            $"{model.ModelName} {metricName} max abs {metric.MaxAbsoluteError:E6} > {maxAbsoluteError:E6}");
    }
}
