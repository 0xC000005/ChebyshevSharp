using System;
using System.Diagnostics;
using System.IO;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Portable .pcb binary serialization format (v1).
///
/// Internal-only. Public access is via <see cref="ChebyshevApproximation.Save"/>
/// and <see cref="ChebyshevSpline.Save"/> with <c>format="binary"</c>.
///
/// Format spec: <c>ref/PyChebyshev/docs/user-guide/binary-format.md</c>.
/// All multi-byte fields are little-endian. Numeric arrays are raw f64
/// blobs in C-order (row-major).
/// </summary>
internal static class PcbFormat
{
    public const int ClassTagApproximation = 1;
    public const int ClassTagSpline = 2;
    public const int MajorVersion = 1;
    public const int MinorVersion = 0;
    public const int HeaderSize = 12;

    public static readonly byte[] Magic = { 0x50, 0x43, 0x42, 0x00 }; // "PCB\0"

    static PcbFormat()
    {
        // .pcb format is fixed little-endian. .NET supports big-endian platforms
        // in theory but no realistic deploy target. Fail fast if violated.
        Debug.Assert(BitConverter.IsLittleEndian,
            "PcbFormat requires a little-endian runtime");
    }

    public readonly record struct PcbHeader(int Major, int Minor, int ClassTag);

    public static void WriteHeader(BinaryWriter w, int classTag)
    {
        w.Write(Magic);
        w.Write((byte)MajorVersion);
        w.Write((byte)MinorVersion);
        w.Write((ushort)classTag);
        w.Write((uint)0); // reserved
    }

    public static PcbHeader ReadHeader(BinaryReader r)
    {
        // Read the full 12-byte header at once — mirrors Python's f.read(_HEADER_SIZE)
        // check so truncated files raise EndOfStreamException before any parsing.
        byte[] raw = r.ReadBytes(HeaderSize);
        if (raw.Length != HeaderSize)
            throw new EndOfStreamException(
                $"unexpected EOF reading header (wanted {HeaderSize} bytes, got {raw.Length})");

        if (raw[0] != Magic[0] || raw[1] != Magic[1] ||
            raw[2] != Magic[2] || raw[3] != Magic[3])
            throw new InvalidDataException(
                "not a PyChebyshev binary file (bad magic)");

        byte major = raw[4];
        byte minor = raw[5];
        if (major != MajorVersion)
            throw new InvalidDataException(
                $"unsupported .pcb major version {major} " +
                $"(this build reads major {MajorVersion})");

        ushort classTag = (ushort)(raw[6] | (raw[7] << 8));
        uint reserved = (uint)(raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24));
        if (reserved != 0)
            throw new InvalidDataException(
                "reserved header bytes nonzero — file may be corrupt");

        return new PcbHeader(major, minor, classTag);
    }

    /// <summary>
    /// Returns true if the file exists and starts with the .pcb magic.
    /// Returns false for nonexistent files (caller's File.OpenRead will throw).
    /// </summary>
    public static bool IsBinary(string path)
    {
        if (!File.Exists(path)) return false;
        using var fs = File.OpenRead(path);
        Span<byte> head = stackalloc byte[4];
        int read = fs.Read(head);
        if (read < 4) return false;
        return head[0] == Magic[0] && head[1] == Magic[1] &&
               head[2] == Magic[2] && head[3] == Magic[3];
    }
}
