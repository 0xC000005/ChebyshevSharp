using System;
using System.IO;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_binary_format.py (v0.14.0).
// Phase 3 Task 1 — header validation.
public class BinaryFormatTests
{
}

public class HeaderTests
{
    private static byte[] BuildValidHeader(int classTag)
    {
        // Magic "PCB\0" + major=1 + minor=0 + classTag (uint16 LE) + reserved=0
        return new byte[] {
            0x50, 0x43, 0x42, 0x00,                          // magic
            0x01, 0x00,                                       // major=1, minor=0
            (byte)(classTag & 0xFF), (byte)((classTag >> 8) & 0xFF), // class_tag LE
            0x00, 0x00, 0x00, 0x00                            // reserved
        };
    }

    [Fact]
    public void Test_write_header_produces_documented_bytes()
    {
        // Per docs/user-guide/binary-format.md: 12-byte header for class_tag=1
        // is exactly: 50 43 42 00 01 00 01 00 00 00 00 00
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms))
            PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
        byte[] expected = BuildValidHeader(PcbFormat.ClassTagApproximation);
        Assert.Equal(expected, ms.ToArray());
    }

    [Fact]
    public void Test_read_header_returns_class_tag_for_approximation()
    {
        var bytes = BuildValidHeader(PcbFormat.ClassTagApproximation);
        using var ms = new MemoryStream(bytes);
        using var r = new BinaryReader(ms);
        var header = PcbFormat.ReadHeader(r);
        Assert.Equal(1, header.Major);
        Assert.Equal(0, header.Minor);
        Assert.Equal(PcbFormat.ClassTagApproximation, header.ClassTag);
    }

    [Fact]
    public void Test_read_header_rejects_bad_magic()
    {
        byte[] bad = (byte[])BuildValidHeader(1).Clone();
        bad[0] = 0x58; // 'X' instead of 'P'
        using var ms = new MemoryStream(bad);
        using var r = new BinaryReader(ms);
        var ex = Assert.Throws<InvalidDataException>(() => PcbFormat.ReadHeader(r));
        Assert.Contains("magic", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_read_header_rejects_unsupported_major_version()
    {
        byte[] bad = (byte[])BuildValidHeader(1).Clone();
        bad[4] = 0x99; // major=153
        using var ms = new MemoryStream(bad);
        using var r = new BinaryReader(ms);
        var ex = Assert.Throws<InvalidDataException>(() => PcbFormat.ReadHeader(r));
        Assert.Contains("major", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_read_header_rejects_truncated_input()
    {
        byte[] truncated = new byte[6]; // less than 12-byte header
        truncated[0] = 0x50; truncated[1] = 0x43; truncated[2] = 0x42; truncated[3] = 0x00;
        using var ms = new MemoryStream(truncated);
        using var r = new BinaryReader(ms);
        Assert.Throws<EndOfStreamException>(() => PcbFormat.ReadHeader(r));
    }
}

public class ApproxBodyTests
{
    private static (double[][] domain, int[] nNodes, double[] tensor) BuildXPlusY()
    {
        // f(x,y) = x + y on [-1,1]^2 with n=[3,3], the binary-format.md worked example.
        // Type I Chebyshev nodes for n=3 in ascending order: -sqrt(3)/2, 0, sqrt(3)/2.
        double s = Math.Sqrt(3.0) / 2.0;
        double[] nodes = { -s, 0.0, s };
        double[] tensor = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                tensor[i * 3 + j] = nodes[i] + nodes[j]; // C-order
        return (
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 3, 3 },
            tensor);
    }

    [Fact]
    public void Test_write_approx_body_byte_count_matches_spec()
    {
        // Per binary-format.md: header(12) + d(4) + lo(16) + hi(16) + n(8) + t(72) = 128 bytes
        var (domain, nNodes, tensor) = BuildXPlusY();
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms))
        {
            PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
            PcbFormat.WriteApproximationBody(w, domain, nNodes, tensor);
        }
        Assert.Equal(128, ms.ToArray().Length);
    }

    [Fact]
    public void Test_round_trip_approx_body_2d()
    {
        var (domain, nNodes, tensor) = BuildXPlusY();
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
            PcbFormat.WriteApproximationBody(w, domain, nNodes, tensor);
        }
        ms.Position = 0;
        using (var r = new BinaryReader(ms))
        {
            PcbFormat.ReadHeader(r);
            var (rdDomain, rdNNodes, rdTensor) = PcbFormat.ReadApproximationBody(r);
            Assert.Equal(domain.Length, rdDomain.Length);
            for (int d = 0; d < domain.Length; d++)
            {
                Assert.Equal(domain[d][0], rdDomain[d][0]);
                Assert.Equal(domain[d][1], rdDomain[d][1]);
            }
            Assert.Equal(nNodes, rdNNodes);
            Assert.Equal(tensor, rdTensor);
        }
    }

    [Fact]
    public void Test_read_approx_body_rejects_zero_dimensions()
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
            w.Write((uint)0); // d = 0 — invalid
        }
        ms.Position = 0;
        using var r = new BinaryReader(ms);
        PcbFormat.ReadHeader(r);
        var ex = Assert.Throws<InvalidDataException>(() => PcbFormat.ReadApproximationBody(r));
        Assert.Contains("num_dimensions", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_read_approx_body_rejects_inverted_domain()
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
            w.Write((uint)1);    // d=1
            w.Write(2.0);        // lo
            w.Write(1.0);        // hi (lo > hi — invalid)
            w.Write((uint)3);    // n_nodes[0]
            w.Write(0.0); w.Write(0.0); w.Write(0.0); // tensor (3 doubles)
        }
        ms.Position = 0;
        using var r = new BinaryReader(ms);
        PcbFormat.ReadHeader(r);
        var ex = Assert.Throws<InvalidDataException>(() => PcbFormat.ReadApproximationBody(r));
        Assert.Contains("domain", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

public class ApproxRoundTripTests
{
    private static string TempPcb() => Path.Combine(
        Path.GetTempPath(),
        $"cheb_test_{Guid.NewGuid():N}.pcb");

    [Fact]
    public void Test_round_trip_3d_sin_evaluates_within_tolerance()
    {
        var cheb = new ChebyshevApproximation(
            (p, _) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2] * p[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 });
        cheb.Build(verbose: false);

        string path = TempPcb();
        try
        {
            cheb.Save(path, format: "binary");
            var loaded = ChebyshevApproximation.Load(path);

            Assert.Equal(cheb.NumDimensions, loaded.NumDimensions);
            double[] testPt = { 0.3, -0.2, 0.5 };
            double expected = cheb.Eval(testPt, new[] { 0, 0, 0 });
            double actual = loaded.Eval(testPt, new[] { 0, 0, 0 });
            Assert.Equal(expected, actual, precision: 12);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_round_trip_n_eq_1_dim()
    {
        // Edge case: a dimension with n=1 (constant in that dim).
        var cheb = new ChebyshevApproximation(
            (p, _) => p[0] * p[0],
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
            new[] { 5, 1 });
        cheb.Build(verbose: false);

        string path = TempPcb();
        try
        {
            cheb.Save(path, format: "binary");
            var loaded = ChebyshevApproximation.Load(path);
            Assert.Equal(new[] { 5, 1 }, loaded.NNodes);
            double expected = cheb.Eval(new[] { 0.4, 0.5 }, new[] { 0, 0 });
            double actual = loaded.Eval(new[] { 0.4, 0.5 }, new[] { 0, 0 });
            Assert.Equal(expected, actual, precision: 12);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_round_trip_5d_evaluates_within_tolerance()
    {
        var cheb = new ChebyshevApproximation(
            (p, _) => p[0] + 2 * p[1] - p[2] + p[3] * p[3] + Math.Sin(p[4]),
            5,
            new[]
            {
                new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
                new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }
            },
            new[] { 4, 4, 4, 4, 6 });
        cheb.Build(verbose: false);

        string path = TempPcb();
        try
        {
            cheb.Save(path, format: "binary");
            var loaded = ChebyshevApproximation.Load(path);
            double[] pt = { 0.1, -0.2, 0.3, -0.4, 0.5 };
            double expected = cheb.Eval(pt, new[] { 0, 0, 0, 0, 0 });
            double actual = loaded.Eval(pt, new[] { 0, 0, 0, 0, 0 });
            Assert.Equal(expected, actual, precision: 10);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class ApproxCrossFeatureBinaryTests
{
    private static string TempPcb() => Path.Combine(
        Path.GetTempPath(), $"cheb_test_{Guid.NewGuid():N}.pcb");

    [Fact]
    public void Test_round_trip_after_algebra_plus()
    {
        var f = new ChebyshevApproximation((p, _) => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        var g = new ChebyshevApproximation((p, _) => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        f.Build(verbose: false);
        g.Build(verbose: false);
        var sum = f + g;

        string path = TempPcb();
        try
        {
            sum.Save(path, format: "binary");
            var loaded = ChebyshevApproximation.Load(path);
            double[] pt = { 0.3, 0.4 };
            double expected = sum.Eval(pt, new[] { 0, 0 });
            double actual = loaded.Eval(pt, new[] { 0, 0 });
            Assert.Equal(expected, actual, precision: 12);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_round_trip_after_slice_and_extrude()
    {
        var cheb = new ChebyshevApproximation(
            (p, _) => Math.Sin(p[0]) + p[1] * p[1] + p[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 } },
            new[] { 6, 6, 6 });
        cheb.Build(verbose: false);

        // Slice dim=2 at value=1.0, then extrude back along the original axis.
        var sliced = cheb.Slice((2, 1.0));
        var extruded = sliced.Extrude((2, new[] { 0.0, 2.0 }, 4));

        string path = TempPcb();
        try
        {
            extruded.Save(path, format: "binary");
            var loaded = ChebyshevApproximation.Load(path);
            Assert.Equal(extruded.NumDimensions, loaded.NumDimensions);
            double[] pt = { 0.2, 0.3, 1.5 };
            double expected = extruded.Eval(pt, new[] { 0, 0, 0 });
            double actual = loaded.Eval(pt, new[] { 0, 0, 0 });
            Assert.Equal(expected, actual, precision: 10);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class SaveLoadApiTests
{
    private static string TempFile(string ext) => Path.Combine(
        Path.GetTempPath(), $"cheb_test_{Guid.NewGuid():N}{ext}");

    private static ChebyshevApproximation Built1D()
    {
        var cheb = new ChebyshevApproximation(
            (p, _) => p[0] * p[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        cheb.Build(verbose: false);
        return cheb;
    }

    [Fact]
    public void Test_save_format_binary_writes_magic_header()
    {
        var cheb = Built1D();
        string path = TempFile(".pcb");
        try
        {
            cheb.Save(path, format: "binary");
            byte[] head = new byte[4];
            using (var fs = File.OpenRead(path)) fs.Read(head, 0, 4);
            Assert.Equal(new byte[] { 0x50, 0x43, 0x42, 0x00 }, head);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_save_format_json_unchanged_default()
    {
        var cheb = Built1D();
        string path = TempFile(".json");
        try
        {
            cheb.Save(path); // no format arg → JSON
            string text = File.ReadAllText(path);
            Assert.StartsWith("{", text);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_load_auto_detects_format_via_magic()
    {
        var cheb = Built1D();
        string binPath = TempFile(".pcb");
        string jsonPath = TempFile(".json");
        try
        {
            cheb.Save(binPath, format: "binary");
            cheb.Save(jsonPath, format: "json");

            var fromBin = ChebyshevApproximation.Load(binPath);
            var fromJson = ChebyshevApproximation.Load(jsonPath);

            double pt = 0.3;
            double a = fromBin.Eval(new[] { pt }, new[] { 0 });
            double b = fromJson.Eval(new[] { pt }, new[] { 0 });
            Assert.Equal(a, b, precision: 12);
        }
        finally
        {
            if (File.Exists(binPath)) File.Delete(binPath);
            if (File.Exists(jsonPath)) File.Delete(jsonPath);
        }
    }

    [Fact]
    public void Test_save_unknown_format_throws()
    {
        var cheb = Built1D();
        string path = TempFile(".bin");
        try
        {
            var ex = Assert.Throws<ArgumentException>(() =>
                cheb.Save(path, format: "msgpack"));
            Assert.Contains("format", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class PeekFormatVersionTests
{
    private static string TempFile(string ext) => Path.Combine(
        Path.GetTempPath(), $"cheb_test_{Guid.NewGuid():N}{ext}");

    [Fact]
    public void Test_peek_returns_1_for_valid_binary_file()
    {
        var cheb = new ChebyshevApproximation((p, _) => p[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 });
        cheb.Build(verbose: false);
        string path = TempFile(".pcb");
        try
        {
            cheb.Save(path, format: "binary");
            int version = ChebyshevApproximation.PeekFormatVersion(path);
            Assert.Equal(1, version);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_peek_throws_for_json_file()
    {
        var cheb = new ChebyshevApproximation((p, _) => p[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 });
        cheb.Build(verbose: false);
        string path = TempFile(".json");
        try
        {
            cheb.Save(path, format: "json");
            var ex = Assert.Throws<InvalidDataException>(
                () => ChebyshevApproximation.PeekFormatVersion(path));
            Assert.Contains("magic", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_peek_throws_for_missing_file()
    {
        string path = TempFile(".pcb"); // not created
        Assert.Throws<FileNotFoundException>(
            () => ChebyshevApproximation.PeekFormatVersion(path));
    }
}

public class CorruptionRejectionTests
{
    private static string TempPcb() => Path.Combine(
        Path.GetTempPath(), $"cheb_test_{Guid.NewGuid():N}.pcb");

    [Fact]
    public void Test_load_rejects_unknown_class_tag()
    {
        // Write a header with class_tag=99 (unknown), no body.
        string path = TempPcb();
        try
        {
            using (var fs = File.Create(path))
            using (var w = new BinaryWriter(fs))
            {
                w.Write(new byte[] { 0x50, 0x43, 0x42, 0x00 }); // magic
                w.Write((byte)1); w.Write((byte)0);              // major/minor
                w.Write((ushort)99);                              // unknown class_tag
                w.Write((uint)0);                                 // reserved
            }
            // Approximation.Load should reject (mismatched class_tag).
            var ex = Assert.Throws<InvalidDataException>(
                () => ChebyshevApproximation.Load(path));
            Assert.Contains("class_tag", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_load_rejects_truncated_body()
    {
        // Save a valid binary, then truncate the file to drop the last few bytes.
        var cheb = new ChebyshevApproximation((p, _) => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        cheb.Build(verbose: false);

        string path = TempPcb();
        try
        {
            cheb.Save(path, format: "binary");
            long size = new FileInfo(path).Length;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
                fs.SetLength(size - 8); // drop last 8 bytes (1 f64)
            // Should fail on EOF reading tensor_values.
            Assert.Throws<EndOfStreamException>(() => ChebyshevApproximation.Load(path));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_load_rejects_unsorted_knots()
    {
        // Hand-craft a class_tag=2 file with non-ascending knots in dim 0.
        string path = TempPcb();
        try
        {
            using (var fs = File.Create(path))
            using (var w = new BinaryWriter(fs))
            {
                w.Write(new byte[] { 0x50, 0x43, 0x42, 0x00 }); // magic
                w.Write((byte)1); w.Write((byte)0);              // version
                w.Write((ushort)PcbFormat.ClassTagSpline);       // class_tag=2
                w.Write((uint)0);                                 // reserved
                w.Write((uint)1);                                 // d=1
                w.Write(-1.0); w.Write(1.0);                      // domain
                w.Write((uint)3);                                 // n_nodes[0]=3
                w.Write((uint)2);                                 // num_knots[0]=2
                w.Write(0.5); w.Write(-0.5);                      // knots: NOT ascending
                w.Write((uint)3);                                 // num_pieces=3
                for (int p = 0; p < 3; p++)
                    for (int j = 0; j < 3; j++) w.Write(0.0);
            }
            var ex = Assert.Throws<InvalidDataException>(
                () => ChebyshevSpline.Load(path));
            Assert.Contains("ascending", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class SplineBinaryTests
{
    private static string TempPcb() => Path.Combine(
        Path.GetTempPath(), $"cheb_test_{Guid.NewGuid():N}.pcb");

    [Fact]
    public void Test_round_trip_1d_abs_with_kink()
    {
        // Spec's worked example: |x| on [-1,1], n=[3], knots=[[0.0]].
        var spline = new ChebyshevSpline(
            (p, _) => Math.Abs(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
            knots: new[] { new[] { 0.0 } });
        spline.Build(verbose: false);

        string path = TempPcb();
        try
        {
            spline.Save(path, format: "binary");
            var loaded = ChebyshevSpline.Load(path);
            foreach (double x in new[] { -0.7, -0.1, 0.1, 0.6 })
            {
                double expected = spline.Eval(new[] { x }, new[] { 0 });
                double actual = loaded.Eval(new[] { x }, new[] { 0 });
                Assert.Equal(expected, actual, precision: 12);
            }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_round_trip_2d_multi_knot()
    {
        var spline = new ChebyshevSpline(
            (p, _) => Math.Abs(p[0]) + p[1] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            knots: new[] { new[] { 0.0 }, new[] { -0.5, 0.5 } });
        spline.Build(verbose: false);

        string path = TempPcb();
        try
        {
            spline.Save(path, format: "binary");
            var loaded = ChebyshevSpline.Load(path);
            double[] pt = { 0.3, 0.2 };
            double expected = spline.Eval(pt, new[] { 0, 0 });
            double actual = loaded.Eval(pt, new[] { 0, 0 });
            Assert.Equal(expected, actual, precision: 10);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_save_binary_throws_for_nested_nNodes()
    {
        // Phase 1's nested-nNodes form via specialPoints + per-piece counts.
        var spline = new ChebyshevSpline(
            (p, _) => Math.Abs(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 3, 5 } }, // [[3, 5]] — nested form
            knots: new[] { new[] { 0.0 } });
        spline.Build(verbose: false);

        string path = TempPcb();
        try
        {
            var ex = Assert.Throws<NotSupportedException>(() =>
                spline.Save(path, format: "binary"));
            Assert.Contains("nested", ex.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("json", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_save_binary_unbuilt_throws()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0], 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
            knots: new[] { Array.Empty<double>() });
        // Skip Build() — saving must throw.
        string path = TempPcb();
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                spline.Save(path, format: "binary"));
            Assert.Contains("Build", ex.Message);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_load_routes_to_spline_for_class_tag_2()
    {
        var spline = new ChebyshevSpline(
            (p, _) => Math.Abs(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
            knots: new[] { new[] { 0.0 } });
        spline.Build(verbose: false);

        string path = TempPcb();
        try
        {
            spline.Save(path, format: "binary");
            // Loading via ChebyshevApproximation.Load should reject (wrong class_tag).
            var ex = Assert.Throws<InvalidDataException>(() =>
                ChebyshevApproximation.Load(path));
            Assert.Contains("class_tag", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}

public class PcbFixtureTests
{
    [Fact]
    public void Test_fixture_approx_2d_simple_loads_and_evaluates()
    {
        string path = ChebyshevSharp.Tests.Helpers.PcbFixtures.Path("approx_2d_simple.pcb");
        Assert.True(File.Exists(path), $"missing fixture: {path}");
        var cheb = ChebyshevApproximation.Load(path);
        Assert.Equal(2, cheb.NumDimensions);
        Assert.Equal(new[] { 3, 3 }, cheb.NNodes);
        // f(x,y) = x + y, so f(0.3, 0.4) = 0.7 (Chebyshev with n=3 hits this exactly).
        double v = cheb.Eval(new[] { 0.3, 0.4 }, new[] { 0, 0 });
        Assert.Equal(0.7, v, precision: 12);
    }

    [Fact]
    public void Test_fixture_approx_5d_bs_loads_and_round_trips()
    {
        string path = ChebyshevSharp.Tests.Helpers.PcbFixtures.Path("approx_5d_bs.pcb");
        Assert.True(File.Exists(path), $"missing fixture: {path}");
        var cheb = ChebyshevApproximation.Load(path);
        Assert.Equal(5, cheb.NumDimensions);
        // Re-save and verify bytes match.
        string roundtrip = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), $"rt_{Guid.NewGuid():N}.pcb");
        try
        {
            cheb.Save(roundtrip, format: "binary");
            byte[] original = File.ReadAllBytes(path);
            byte[] re = File.ReadAllBytes(roundtrip);
            Assert.Equal(original, re);
        }
        finally { if (File.Exists(roundtrip)) File.Delete(roundtrip); }
    }

    [Fact]
    public void Test_fixture_spline_1d_kink_loads_and_evaluates()
    {
        string path = ChebyshevSharp.Tests.Helpers.PcbFixtures.Path("spline_1d_kink.pcb");
        Assert.True(File.Exists(path), $"missing fixture: {path}");
        var spline = ChebyshevSpline.Load(path);
        Assert.Equal(1, spline.NumDimensions);
        // abs(x) on [-1,1] with knot at 0 — recovers to machine precision because
        // each piece is a 3-node Chebyshev fit to a line segment.
        Assert.Equal(0.5, spline.Eval(new[] { 0.5 }, new[] { 0 }), precision: 12);
        Assert.Equal(0.7, spline.Eval(new[] { -0.7 }, new[] { 0 }), precision: 12);
        Assert.Equal(0.0, spline.Eval(new[] { 0.0 }, new[] { 0 }), precision: 12);
    }
}
