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
