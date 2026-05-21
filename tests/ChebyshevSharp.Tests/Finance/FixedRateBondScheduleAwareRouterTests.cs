using FixedRateBondSurrogate;

namespace ChebyshevSharp.Tests.Finance;

public sealed class FixedRateBondScheduleAwareRouterTests
{
    private static readonly IFixedRateBondReferencePricer Pricer = new QlNetFixedRateBondReferencePricer();
    private static readonly Lazy<ScheduleAwareRouterReport> Report =
        new(() => ScheduleAwareRouterBenchmark.RunDefault(Pricer));

    [Fact]
    public void Phase10_report_preserves_full_public_wrapper()
    {
        ScheduleAwareRouterReport report = Report.Value;

        Assert.Equal("curve bumps[60], coupon, maturity -> dirty PV", report.WrapperContract);
        Assert.Equal(62, report.PublicInputDimensionCount);
        Assert.NotEmpty(report.Pieces);
        Assert.Contains(
            report.Models,
            model => model.ModelName == "Schedule-aware router decomposed factor tensor");
    }

    [Fact]
    public void Schedule_aware_router_mode_writes_phase10_summary()
    {
        using var writer = new StringWriter();

        FixedRateBondExample.Run(["--schedule-aware-router"], writer);

        string output = writer.ToString();
        Assert.Contains("Fixed-rate bond schedule-aware router", output);
        Assert.Contains("full wrapper", output);
        Assert.Contains("Pieces", output);
        Assert.Contains("One-sided maturity diagnostics", output);
        Assert.Contains("left abs error", output);
        Assert.DoesNotContain("Pending", output);
        Assert.Contains("Decision", output);
    }

    [Fact]
    public void Phase10_pieces_cover_domain_with_half_open_boundaries()
    {
        ScheduleAwareRouterReport report = Report.Value;

        Assert.Equal(2.0, report.Pieces[0].MaturityLo, precision: 12);
        Assert.Equal(30.0, report.Pieces[^1].MaturityHi, precision: 12);
        Assert.True(report.Pieces[^1].IncludesUpperBound);

        for (int i = 0; i < report.Pieces.Count; i++)
        {
            ScheduleAwareRouterPieceSummary piece = report.Pieces[i];
            Assert.Equal(i, piece.Index);
            Assert.True(piece.MaturityHi > piece.MaturityLo);

            if (i < report.Pieces.Count - 1)
            {
                ScheduleAwareRouterPieceSummary next = report.Pieces[i + 1];
                Assert.False(piece.IncludesUpperBound);
                Assert.Equal(piece.MaturityHi, next.MaturityLo, precision: 12);
            }
        }
    }

    [Fact]
    public void Phase10_router_routes_boundaries_to_right_piece_except_final_boundary()
    {
        ScheduleAwareRouterReport report = Report.Value;
        var router = new ScheduleAwarePiecewiseRouter(report.Pieces, (_, pieceIndex) => pieceIndex);

        Assert.Equal(0.0, router.Eval(FullPoint(2.0)));
        Assert.Equal(report.Pieces.Count - 1, router.Eval(FullPoint(30.0)));

        foreach (ScheduleAwareRouterPieceSummary piece in report.Pieces.Skip(1).Take(5))
        {
            Assert.Equal(piece.Index, router.Eval(FullPoint(piece.MaturityLo)));
        }
    }

    [Fact]
    public void Phase10_schedule_breakpoints_keep_phase9_candidate_provenance()
    {
        ScheduleAwareRouterReport report = Report.Value;
        HashSet<double> phase9Candidates = report.Phase9ScheduleCandidateYears
            .Select(years => Math.Round(years, 8))
            .ToHashSet();

        Assert.NotEmpty(report.ScheduleBreakpoints);
        Assert.All(
            report.ScheduleBreakpoints,
            breakpoint =>
            {
                Assert.InRange(breakpoint, 2.0, 30.0);
                Assert.Contains(Math.Round(breakpoint, 8), phase9Candidates);
            });
    }

    [Fact]
    public void Phase10_report_contains_finite_one_sided_maturity_diagnostics()
    {
        ScheduleAwareRouterReport report = Report.Value;

        Assert.True(report.OneSidedMaturityDiagnostics.Count >= 5);
        Assert.All(
            report.OneSidedMaturityDiagnostics,
            diagnostic =>
            {
                Assert.InRange(diagnostic.BreakpointYears, 2.0, 30.0);
                Assert.Contains(
                    Math.Round(diagnostic.BreakpointYears, 8),
                    report.ScheduleBreakpoints.Select(years => Math.Round(years, 8)));
                Assert.True(double.IsFinite(diagnostic.EpsilonYears));
                Assert.True(double.IsFinite(diagnostic.BaselineLeftSlopePerYear));
                Assert.True(double.IsFinite(diagnostic.BaselineRightSlopePerYear));
                Assert.True(double.IsFinite(diagnostic.RouterLeftSlopePerYear));
                Assert.True(double.IsFinite(diagnostic.RouterRightSlopePerYear));
                Assert.True(double.IsFinite(diagnostic.LeftSlopeAbsoluteError));
                Assert.True(double.IsFinite(diagnostic.RightSlopeAbsoluteError));
            });
    }

    [Fact]
    public void Phase10_model_bank_contains_phase9_controls_and_router_candidate()
    {
        ScheduleAwareRouterReport report = Report.Value;

        string[] expectedModels =
        [
            "Phase 9 global decomposed factor control",
            "Phase 9 uniform 0.5Y control",
            "Phase 9 schedule-aware special-point control",
            "Schedule-aware router decomposed factor tensor",
        ];

        foreach (string expectedModel in expectedModels)
        {
            AnalyticCouponModelSummary model = Assert.Single(report.Models, model => model.ModelName == expectedModel);
            Assert.Equal(62, model.PublicInputDimensionCount);
            Assert.Contains(model.Metrics, metric => metric.Name == "maturity sensitivity");
            Assert.Contains(model.Metrics, metric => metric.Name == "coupon-maturity mixed");
        }
    }

    [Fact]
    public void Phase10_router_rejects_invalid_piece_shapes_and_inputs()
    {
        List<ScheduleAwareRouterPieceSummary> validPieces =
        [
            Piece(index: 0, lo: 2.0, hi: 10.0, includesUpperBound: false),
            Piece(index: 1, lo: 10.0, hi: 30.0, includesUpperBound: true),
        ];
        var router = new ScheduleAwarePiecewiseRouter(validPieces, (_, pieceIndex) => pieceIndex);

        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter([], (_, _) => 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => router.Route(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => router.Route(1.9));
        Assert.Throws<ArgumentOutOfRangeException>(() => router.Route(30.1));
        Assert.Throws<ArgumentException>(() => router.Eval(new double[61]));
        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter(
            [Piece(index: 1, lo: 2.0, hi: 30.0, includesUpperBound: true)],
            (_, _) => 0.0));
        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter(
            [Piece(index: 0, lo: 2.0, hi: 2.0, includesUpperBound: true)],
            (_, _) => 0.0));
        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter(
            [
                Piece(index: 0, lo: 2.0, hi: 10.0, includesUpperBound: true),
                Piece(index: 1, lo: 10.0, hi: 30.0, includesUpperBound: true),
            ],
            (_, _) => 0.0));
        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter(
            [
                Piece(index: 0, lo: 2.0, hi: 10.0, includesUpperBound: false),
                Piece(index: 1, lo: 11.0, hi: 30.0, includesUpperBound: true),
            ],
            (_, _) => 0.0));
        Assert.Throws<ArgumentException>(() => new ScheduleAwarePiecewiseRouter(
            [Piece(index: 0, lo: 2.0, hi: 30.0, includesUpperBound: false)],
            (_, _) => 0.0));
    }

    private static double[] FullPoint(double maturityYears)
    {
        var point = new double[62];
        point[61] = maturityYears;
        return point;
    }

    private static ScheduleAwareRouterPieceSummary Piece(
        int index,
        double lo,
        double hi,
        bool includesUpperBound)
        => new(
            Index: index,
            MaturityLo: lo,
            MaturityHi: hi,
            IncludesUpperBound: includesUpperBound,
            Source: "test");
}
