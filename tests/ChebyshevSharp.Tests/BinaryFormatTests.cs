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
