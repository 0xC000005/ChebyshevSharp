using System;
using System.IO;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_error_threshold.py (PyChebyshev v0.11)
// Tests added incrementally across Phase 1 tasks.
public class ErrorThresholdTests
{
}

public class ConstructorValidation
{
    private static readonly Func<double[], object?, double> Sin2D = (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]);
    private static readonly double[][] UnitSq = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };

    [Fact]
    public void Test_explicit_n_unchanged()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, new[] { 11, 11 });
        Assert.Equal(new[] { 11, 11 }, cheb.NNodes);
        Assert.Null(cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_error_threshold_only()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(1e-6, cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_neither_n_nor_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: null));
    }

    [Fact]
    public void Test_none_without_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: new int?[] { null, 11 }, errorThreshold: null));
    }

    [Fact]
    public void Test_max_n_default()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(64, cheb.MaxN);
    }

    [Fact]
    public void Test_max_n_custom()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 128);
        Assert.Equal(128, cheb.MaxN);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(-1)]
    public void Test_max_n_below_minimum_raises(int badMaxN)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: badMaxN));
        Assert.Contains("maxN must be at least 3", ex.Message);
    }

    [Fact]
    public void Test_max_n_equal_to_minimum_accepted()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 3);
        Assert.Equal(3, cheb.MaxN);
    }
}

public class DoublingLoopTests
{
    private static readonly double[][] UnitSq = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };

    [Fact]
    public void Test_1d_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-8);
        cheb.Build(verbose: false);
        Assert.True(cheb.NNodes[0] <= 64);
        Assert.True(cheb.ErrorEstimate() <= 1e-8);
    }

    [Fact]
    public void Test_2d_auto_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_3d_auto_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_semi_variable_respects_fixed_dims()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null, 15, 15 }, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(15, cheb.NNodes[1]);
        Assert.Equal(15, cheb.NNodes[2]);
    }

    [Fact]
    public void Test_already_accurate_stops_immediately()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0] + x[1],  // linear; exact at N=3
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(3, cheb.NNodes[0]);
        Assert.Equal(3, cheb.NNodes[1]);
    }

    [Fact]
    public void Test_tight_threshold_eventual()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Exp(-x[0] * x[0]),
            1, new[] { new[] { -2.0, 2.0 } },
            nNodes: null, errorThreshold: 1e-12);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-12);
    }

    [Fact]
    public void Test_max_n_cap_emits_warning_and_remains_usable()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(20 * x[0]) + Math.Cos(17 * x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-12, maxN: 16);
        cheb.Build(verbose: false);

        Assert.NotNull(cheb.BuildWarning);
        Assert.Contains("maxN", cheb.BuildWarning);
        Assert.True(cheb.NNodes[0] <= 16);
        double v = cheb.VectorizedEval(new[] { 0.1 }, new[] { 0 });
        Assert.True(double.IsFinite(v));
    }

    [Fact]
    public void Test_no_warning_when_threshold_met()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Null(cheb.BuildWarning);
    }

    [Fact]
    public void Test_rebuild_with_tighter_threshold_rebuilds_auto_dims()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-4);
        cheb.Build(verbose: false);
        int nFirst = cheb.NNodes[0];
        Assert.True(cheb.ErrorEstimate() <= 1e-4);

        cheb.ErrorThreshold = 1e-10;
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-10);
        Assert.True(cheb.NNodes[0] >= nFirst);
    }
}

public class ErrorEstimatePerDimTests
{
    [Fact]
    public void Test_per_dim_returns_one_entry_per_dimension()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 });
        cheb.Build(verbose: false);
        double[] perDim = cheb.ErrorEstimatePerDim();
        Assert.Equal(2, perDim.Length);
        Assert.All(perDim, e => Assert.True(e >= 0.0));
    }

    [Fact]
    public void Test_per_dim_sum_equals_error_estimate()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 9, 11, 13 });
        cheb.Build(verbose: false);
        double[] perDim = cheb.ErrorEstimatePerDim();
        double total = perDim.Sum();
        Assert.Equal(cheb.ErrorEstimate(), total, precision: 14);
    }

    [Fact]
    public void Test_per_dim_throws_before_build()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0],
            1, new[] { new[] { -1.0, 1.0 } },
            new[] { 5 });
        Assert.Throws<InvalidOperationException>(() => cheb.ErrorEstimatePerDim());
    }
}

public class GetErrorThresholdTests
{
    [Fact]
    public void Test_returns_threshold_when_set()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(1e-6, cheb.GetErrorThreshold());
    }

    [Fact]
    public void Test_returns_null_when_not_set()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 });
        cheb.Build(verbose: false);
        Assert.Null(cheb.GetErrorThreshold());
    }
}

public class GetOptimalN1Tests
{
    [Fact]
    public void Test_returns_int_above_minimum()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(x[0]),
            (-1.0, 1.0),
            errorThreshold: 1e-8);
        Assert.True(n >= 3 && n <= 64);
    }

    [Fact]
    public void Test_smooth_low_freq_small_n()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => x[0],
            (-1.0, 1.0),
            errorThreshold: 1e-10);
        Assert.Equal(3, n);
    }

    [Fact]
    public void Test_high_freq_larger_n()
    {
        int nLow = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[0]),
            (-1.0, 1.0), errorThreshold: 1e-8);
        int nHigh = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(10 * x[0]) + Math.Cos(10 * x[0]),
            (-1.0, 1.0), errorThreshold: 1e-8);
        Assert.True(nHigh > nLow);
    }

    [Fact]
    public void Test_respects_max_n()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(50 * x[0]) + Math.Cos(43 * x[0]),
            (-1.0, 1.0),
            errorThreshold: 1e-14,
            maxN: 8);
        Assert.Equal(8, n);
    }
}

public class SplineErrorThresholdTests
{
    private static readonly Func<double[], object?, double> Sin2D = (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]);

    [Fact]
    public void Test_1d_with_knot()
    {
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null },
            knots: new[] { new[] { 0.0 } },
            errorThreshold: 1e-6);
        spl.Build(verbose: false);
        foreach (var piece in spl.Pieces)
        {
            Assert.NotNull(piece);
            Assert.True(piece!.ErrorEstimate() <= 1e-6);
        }
    }

    [Fact]
    public void Test_2d_no_knots_matches_flat()
    {
        var spl = new ChebyshevSpline(
            Sin2D,
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null, null },
            knots: new[] { Array.Empty<double>(), Array.Empty<double>() },
            errorThreshold: 1e-6);
        spl.Build(verbose: false);
        Assert.Single(spl.Pieces);
        Assert.True(spl.Pieces[0]!.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_explicit_n_still_works()
    {
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: new int?[] { 15 },
            knots: new[] { new[] { 0.0 } });
        spl.Build(verbose: false);
        foreach (var piece in spl.Pieces)
            Assert.Equal(new[] { 15 }, piece!.NNodes);
    }

    [Fact]
    public void Test_spline_neither_n_nor_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: (int?[]?)null, knots: null));
    }

    [Fact]
    public void Test_spline_max_n_below_minimum_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: null, knots: null, errorThreshold: 1e-6, maxN: 2));
    }

    [Fact]
    public void Test_spline_knots_default_to_empty_per_dim()
    {
        var spl = new ChebyshevSpline(
            Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, knots: null, errorThreshold: 1e-6);
        Assert.Equal(2, spl.Knots.Length);
        Assert.Empty(spl.Knots[0]);
        Assert.Empty(spl.Knots[1]);
        spl.Build(verbose: false);
        Assert.Single(spl.Pieces);
    }
}

public class JsonMigrationTests
{
    [Fact]
    public void Test_save_load_roundtrip_auto_n()
    {
        string path = Path.GetTempFileName();
        try
        {
            var cheb = new ChebyshevApproximation(
                (x, _) => Math.Sin(x[0]),
                1, new[] { new[] { -1.0, 1.0 } },
                nNodes: null, errorThreshold: 1e-6);
            cheb.Build(verbose: false);
            cheb.Save(path);

            var loaded = ChebyshevApproximation.Load(path);
            Assert.Equal(cheb.NNodes, loaded.NNodes);
            Assert.Equal(cheb.OriginalNNodes.Length, loaded.OriginalNNodes.Length);
            Assert.Equal(cheb.OriginalNNodes[0], loaded.OriginalNNodes[0]);
            Assert.Equal(cheb.ErrorThreshold, loaded.ErrorThreshold);
            Assert.Equal(cheb.MaxN, loaded.MaxN);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_load_pre_v05_file_backfills_original_n_nodes()
    {
        string path = Path.GetTempFileName();
        try
        {
            // Save a fixed-N file the new way, then strip the new fields by re-serializing without them.
            var cheb = new ChebyshevApproximation(
                (x, _) => x[0] + x[1],
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5, 5 });
            cheb.Build(verbose: false);
            cheb.Save(path);

            string json = File.ReadAllText(path);
            // Remove the v0.5.0 fields to mimic an older file. (Keys must match property names.)
            string oldJson = System.Text.RegularExpressions.Regex.Replace(
                json, @",\s*""(OriginalNNodes|ErrorThreshold|MaxN)""\s*:\s*[^,}]+", "");
            File.WriteAllText(path, oldJson);

            var loaded = ChebyshevApproximation.Load(path);
            // OriginalNNodes backfilled from NNodes (fully-resolved fixed-N intent)
            Assert.Equal(2, loaded.OriginalNNodes.Length);
            Assert.Equal(5, loaded.OriginalNNodes[0]);
            Assert.Equal(5, loaded.OriginalNNodes[1]);
            Assert.Null(loaded.ErrorThreshold);
            Assert.Equal(64, loaded.MaxN);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
