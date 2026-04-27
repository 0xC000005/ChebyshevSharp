using System;
using System.IO;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_special_points.py (PyChebyshev v0.12)
// Tests added incrementally across Phase 1 tasks.
public class SpecialPointsTests
{
}

public class WithSpecialPointsTests
{
    private static readonly Func<double[], object?, double> Abs1D = (x, _) => Math.Abs(x[0]);

    [Fact]
    public void Test_factory_returns_spline_with_kink_as_knot()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        Assert.Equal(new[] { new[] { 0.0 } }, spl.Knots);
        Assert.Equal(2, spl.Pieces.Length);
    }

    [Fact]
    public void Test_abs_kink_reaches_machine_precision()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        for (double x = -0.95; x <= 0.95; x += 0.05)
        {
            if (Math.Abs(x) < 1e-8) continue;
            double v = spl.Eval(new[] { x }, new[] { 0 });
            TestFixtures.AssertClose(Math.Abs(x), v, atol: 1e-13);
        }
    }

    [Fact]
    public void Test_unsorted_points_raise()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.5, -0.5 } },
                nNodesNested: new[] { new[] { 11, 11, 11 } }));
        Assert.Contains("must be sorted", ex.Message);
    }

    [Fact]
    public void Test_point_on_boundary_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 1.0 } },
                nNodesNested: new[] { new[] { 11, 11 } }));
    }

    [Fact]
    public void Test_outer_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]),
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.0 } },  // missing dim 1
                nNodesNested: new[] { new[] { 11, 11 }, new[] { 13 } }));
        Assert.Contains("must have 2 entries", ex.Message);
    }

    [Fact]
    public void Test_factory_with_error_threshold()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            errorThreshold: 1e-10);
        spl.Build(verbose: false);
        TestFixtures.AssertClose(0.5, spl.Eval(new[] { 0.5 }, new[] { 0 }), atol: 1e-10);
    }
}

public class NestedNNodesTests
{
    [Fact]
    public void Test_nested_n_nodes_per_piece()
    {
        // 1D abs(x) with knot at 0; left piece uses 11 nodes, right piece 13.
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 11, 13 } },
            knots: new[] { new[] { 0.0 } });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 13 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_2d_per_sub_interval()
    {
        // 2D: dim 0 has knot at 0.2 (2 pieces with 7,9 nodes), dim 1 has no knot (1 piece, 11 nodes)
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + x[1] * x[1] * x[1] * x[1],
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } },
            knots: new[] { new[] { 0.2 }, Array.Empty<double>() });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 7, 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 9, 11 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_outer_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]),
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11, 11 } },  // only 1 entry, should be 2
                knots: new[] { new[] { 0.0 }, Array.Empty<double>() }));
        Assert.Contains("must have 2 entries", ex.Message);
    }

    [Fact]
    public void Test_nested_inner_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]),
                1, new[] { new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11 } },  // 1 entry, should be 2 (since knots has 1 knot)
                knots: new[] { new[] { 0.0 } }));
        Assert.Contains("must have 2 entries", ex.Message);
    }
}

public class CrossFeatureTests
{
    private static readonly Func<double[], object?, double> Abs1D = (x, _) => Math.Abs(x[0]);

    [Fact]
    public void Test_save_load_roundtrip()
    {
        string path = Path.GetTempFileName();
        try
        {
            var spl = ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.0 } },
                nNodesNested: new[] { new[] { 11, 11 } });
            spl.Build(verbose: false);
            spl.Save(path);

            var loaded = ChebyshevSpline.Load(path);
            foreach (double x in new[] { -0.5, 0.2, 0.8 })
                TestFixtures.AssertClose(spl.Eval(new[] { x }, new[] { 0 }),
                                         loaded.Eval(new[] { x }, new[] { 0 }), atol: 1e-14);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_algebra_with_sibling()
    {
        var a = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        a.Build(verbose: false);
        var b = ChebyshevSpline.WithSpecialPoints(
            (x, _) => x[0] * x[0], 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        b.Build(verbose: false);

        var c = a + b;
        foreach (double x in new[] { -0.5, 0.3, 0.7 })
        {
            double expected = Math.Abs(x) + x * x;
            TestFixtures.AssertClose(expected, c.Eval(new[] { x }, new[] { 0 }), atol: 1e-12);
        }
    }

    [Fact]
    public void Test_integrate()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        // Integral of |x| over [-1, 1] is 1.0 exactly.
        var result = (double)spl.Integrate();
        TestFixtures.AssertClose(1.0, result, atol: 1e-12);
    }
}

public class SplineSerializationMetadataTests
{
    private static readonly Func<double[], object?, double> Abs1D = (x, _) => Math.Abs(x[0]);

    [Fact]
    public void Test_save_load_preserves_error_threshold()
    {
        string path = Path.GetTempFileName();
        try
        {
            var spl = new ChebyshevSpline(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                nNodes: new int?[] { null },
                knots: new[] { new[] { 0.0 } },
                errorThreshold: 1e-6,
                maxN: 32);
            spl.Build(verbose: false);
            spl.Save(path);

            var loaded = ChebyshevSpline.Load(path);
            Assert.Equal(1e-6, loaded.ErrorThreshold);
            Assert.Equal(32, loaded.MaxN);
            Assert.Single(loaded.OriginalNNodes);
            Assert.Null(loaded.OriginalNNodes[0]);
            Assert.Equal(spl.GetErrorThreshold(), loaded.GetErrorThreshold());
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_save_load_preserves_nested_n_nodes()
    {
        string path = Path.GetTempFileName();
        try
        {
            var spl = ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.0 } },
                nNodesNested: new[] { new[] { 11, 13 } });
            spl.Build(verbose: false);
            spl.Save(path);

            var loaded = ChebyshevSpline.Load(path);
            Assert.NotNull(loaded.GetType().GetProperty("NestedNNodes",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance));
            // Eval still matches exactly (the pieces themselves carry their own resolved NNodes).
            foreach (double x in new[] { -0.5, 0.3 })
                TestFixtures.AssertClose(spl.Eval(new[] { x }, new[] { 0 }),
                                         loaded.Eval(new[] { x }, new[] { 0 }), atol: 1e-14);
            // Per-piece NNodes survive (asymmetric 11 vs 13).
            Assert.Equal(11, loaded.Pieces[0]!.NNodes[0]);
            Assert.Equal(13, loaded.Pieces[1]!.NNodes[0]);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_load_pre_v05_spline_file_uses_defaults()
    {
        // Simulate a pre-v0.5.0 spline file by saving then stripping the new fields.
        string path = Path.GetTempFileName();
        try
        {
            var spl = new ChebyshevSpline(
                (x, _) => x[0] + x[1],
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5, 5 },
                new[] { Array.Empty<double>(), Array.Empty<double>() });
            spl.Build(verbose: false);
            spl.Save(path);

            string json = File.ReadAllText(path);
            string oldJson = System.Text.RegularExpressions.Regex.Replace(
                json, @",\s*""(OriginalNNodes|ErrorThreshold|MaxN|NestedNNodes)""\s*:\s*[^,}]+", "");
            File.WriteAllText(path, oldJson);

            var loaded = ChebyshevSpline.Load(path);
            // Defaults: empty OriginalNNodes, null ErrorThreshold, MaxN=64.
            Assert.Empty(loaded.OriginalNNodes);
            Assert.Null(loaded.ErrorThreshold);
            Assert.Equal(64, loaded.MaxN);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
