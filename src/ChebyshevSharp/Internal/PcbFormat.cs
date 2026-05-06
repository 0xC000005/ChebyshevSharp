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

    /// <summary>
    /// Convert a uint32 read from a binary file to int, throwing
    /// <see cref="InvalidDataException"/> (not <see cref="OverflowException"/>)
    /// if the value exceeds int.MaxValue. Keeps the read-side error surface
    /// uniformly InvalidDataException for malformed/hostile input.
    /// </summary>
    private static int ToCheckedInt(uint x, string fieldName)
    {
        if (x > int.MaxValue)
            throw new InvalidDataException(
                $"{fieldName}={x} exceeds int.MaxValue ({int.MaxValue})");
        return (int)x;
    }

    /// <summary>
    /// Multiply two ints, converting <see cref="OverflowException"/> to
    /// <see cref="InvalidDataException"/> for spec-consistent error surface.
    /// </summary>
    private static int CheckedMul(int a, int b, string description)
    {
        try { return checked(a * b); }
        catch (OverflowException)
        {
            throw new InvalidDataException(
                $"{description} overflows int (a={a}, b={b})");
        }
    }

    private static double ReadFiniteDouble(BinaryReader r, string fieldName)
    {
        double value = r.ReadDouble();
        if (!double.IsFinite(value))
            throw new InvalidDataException(
                $"{fieldName} must be finite, got {value}");
        return value;
    }

    private static void ValidateDomainBounds(double lo, double hi, int dim)
    {
        if (lo >= hi)
            throw new InvalidDataException(
                $"domain[{dim}]: lo ({lo}) must be < hi ({hi})");
    }

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
    /// Read the major version byte from a .pcb file without deserializing the body.
    /// Mirrors Python <c>peek_format_version</c> (PyChebyshev v0.16).
    /// </summary>
    /// <param name="path">Path to a .pcb file.</param>
    /// <returns>The major format version byte (currently 1).</returns>
    /// <exception cref="FileNotFoundException">Thrown if the path does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown if the file is not a .pcb file
    /// (magic mismatch) or is shorter than the 12-byte header.</exception>
    public static int PeekFormatVersion(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"file not found: {path}", path);
        using var fs = File.OpenRead(path);
        Span<byte> head = stackalloc byte[HeaderSize];
        int read = fs.Read(head);
        if (read < HeaderSize)
            throw new InvalidDataException(
                $"file '{path}' is shorter than the {HeaderSize}-byte .pcb header");
        if (head[0] != Magic[0] || head[1] != Magic[1] ||
            head[2] != Magic[2] || head[3] != Magic[3])
            throw new InvalidDataException(
                $"file '{path}' is not a .pcb file (magic mismatch)");
        return head[4]; // major version byte
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

    /// <summary>
    /// Writes the body of a class_tag=1 (Approximation) record.
    /// Mirrors Python <c>_binary.write_approx</c> lines 208-236.
    ///
    /// Layout (all little-endian):
    ///   uint32  d           — number of dimensions
    ///   f64[d]  lo          — domain lower bounds
    ///   f64[d]  hi          — domain upper bounds
    ///   uint32[d] n_nodes   — node count per dimension
    ///   f64[prod(n)] tensor — tensor values in C-order (row-major)
    /// </summary>
    public static void WriteApproximationBody(
        BinaryWriter w, double[][] domain, int[] nNodes, double[] tensorValues)
    {
        int d = domain.Length;
        if (d != nNodes.Length)
            throw new ArgumentException(
                $"domain.Length ({d}) != nNodes.Length ({nNodes.Length})");

        w.Write((uint)d);
        for (int i = 0; i < d; i++) w.Write(domain[i][0]);
        for (int i = 0; i < d; i++) w.Write(domain[i][1]);
        for (int i = 0; i < d; i++) w.Write((uint)nNodes[i]);

        int total = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(nNodes, nameof(WriteApproximationBody)),
            nameof(WriteApproximationBody),
            nNodes);
        if (tensorValues.Length != total)
            throw new ArgumentException(
                $"tensorValues.Length={tensorValues.Length} does not match prod(nNodes)={total}");

        for (int i = 0; i < total; i++) w.Write(tensorValues[i]);
    }

    /// <summary>
    /// Reads the body of a class_tag=1 (Approximation) record.
    /// Mirrors Python <c>_binary.read_approx</c> lines 239-283.
    /// Must be called after <see cref="ReadHeader"/> has consumed the 12-byte header.
    /// </summary>
    /// <returns>
    /// A tuple of (domain, nNodes, tensorValues) where domain[i] = {lo, hi}.
    /// </returns>
    public static (double[][] domain, int[] nNodes, double[] tensorValues) ReadApproximationBody(
        BinaryReader r)
    {
        uint d32 = r.ReadUInt32();
        if (d32 < 1)
            throw new InvalidDataException($"num_dimensions must be >= 1, got {d32}");
        int d = ToCheckedInt(d32, "num_dimensions");

        double[] lo = new double[d];
        for (int i = 0; i < d; i++) lo[i] = ReadFiniteDouble(r, $"lo[{i}]");
        double[] hi = new double[d];
        for (int i = 0; i < d; i++) hi[i] = ReadFiniteDouble(r, $"hi[{i}]");

        var domain = new double[d][];
        for (int i = 0; i < d; i++)
        {
            ValidateDomainBounds(lo[i], hi[i], i);
            domain[i] = new[] { lo[i], hi[i] };
        }

        int[] nNodes = new int[d];
        int total = 1;
        for (int i = 0; i < d; i++)
        {
            uint n32 = r.ReadUInt32();
            if (n32 < 1)
                throw new InvalidDataException($"n_nodes[{i}] must be >= 1, got {n32}");
            nNodes[i] = ToCheckedInt(n32, $"n_nodes[{i}]");
            total = CheckedMul(total, nNodes[i], "prod(n_nodes)");
        }

        double[] tensor = new double[total];
        for (int i = 0; i < total; i++)
            tensor[i] = ReadFiniteDouble(r, $"tensor_values[{i}]");
        return (domain, nNodes, tensor);
    }

    /// <summary>
    /// Writes the body of a class_tag=2 (Spline) record.
    /// Mirrors Python <c>_binary.write_spline</c> lines 289-365.
    ///
    /// Layout (all little-endian):
    ///   uint32    d               — number of dimensions
    ///   f64[d]    lo              — domain lower bounds
    ///   f64[d]    hi              — domain upper bounds
    ///   uint32[d] n_nodes         — node count per dimension (shared across pieces)
    ///   uint32[d] num_knots       — knot count per dimension
    ///   f64[sum(num_knots)] knots — all knots concatenated in dim order
    ///   uint32    num_pieces      — total number of pieces = prod(num_knots[i]+1)
    ///   f64[prod(n_nodes)] * num_pieces — tensor values for each piece (C-order)
    /// </summary>
    public static void WriteSplineBody(
        BinaryWriter w,
        double[][] domain, int[] nNodes,
        double[][] knotsPerDim,
        double[][] pieceTensorValues)
    {
        int d = domain.Length;
        if (d != nNodes.Length || d != knotsPerDim.Length)
            throw new ArgumentException(
                $"dimension mismatch: domain.Length={d}, nNodes.Length={nNodes.Length}, " +
                $"knotsPerDim.Length={knotsPerDim.Length}");

        w.Write((uint)d);
        for (int i = 0; i < d; i++) w.Write(domain[i][0]);
        for (int i = 0; i < d; i++) w.Write(domain[i][1]);
        for (int i = 0; i < d; i++) w.Write((uint)nNodes[i]);
        for (int i = 0; i < d; i++) w.Write((uint)knotsPerDim[i].Length);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < knotsPerDim[i].Length; j++)
                w.Write(knotsPerDim[i][j]);

        int[] pieceShape = knotsPerDim.Select(k => k.Length + 1).ToArray();
        int expectedPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(pieceShape, nameof(WriteSplineBody)),
            nameof(WriteSplineBody),
            pieceShape);
        if (pieceTensorValues.Length != expectedPieces)
            throw new ArgumentException(
                $"pieceTensorValues.Length={pieceTensorValues.Length} does not match " +
                $"prod(num_knots[i]+1)={expectedPieces}");

        w.Write((uint)expectedPieces);
        int perPieceFloats = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(nNodes, nameof(WriteSplineBody)),
            nameof(WriteSplineBody),
            nNodes);
        foreach (var piece in pieceTensorValues)
        {
            if (piece.Length != perPieceFloats)
                throw new ArgumentException(
                    $"piece tensor length {piece.Length} != prod(nNodes)={perPieceFloats}");
            for (int j = 0; j < piece.Length; j++) w.Write(piece[j]);
        }
    }

    /// <summary>
    /// Reads the body of a class_tag=2 (Spline) record.
    /// Mirrors Python <c>_binary.read_spline</c> lines 368-421.
    /// Must be called after <see cref="ReadHeader"/> has consumed the 12-byte header.
    /// </summary>
    /// <returns>
    /// A tuple of (domain, nNodes, knotsPerDim, pieceTensors).
    /// </returns>
    public static (double[][] domain, int[] nNodes, double[][] knotsPerDim, double[][] pieceTensors)
        ReadSplineBody(BinaryReader r)
    {
        uint d32 = r.ReadUInt32();
        if (d32 < 1)
            throw new InvalidDataException($"num_dimensions must be >= 1, got {d32}");
        int d = ToCheckedInt(d32, "num_dimensions");

        double[] lo = new double[d];
        for (int i = 0; i < d; i++) lo[i] = ReadFiniteDouble(r, $"lo[{i}]");
        double[] hi = new double[d];
        for (int i = 0; i < d; i++) hi[i] = ReadFiniteDouble(r, $"hi[{i}]");
        var domain = new double[d][];
        for (int i = 0; i < d; i++)
        {
            ValidateDomainBounds(lo[i], hi[i], i);
            domain[i] = new[] { lo[i], hi[i] };
        }

        int[] nNodes = new int[d];
        int perPieceFloats = 1;
        for (int i = 0; i < d; i++)
        {
            uint n32 = r.ReadUInt32();
            if (n32 < 1)
                throw new InvalidDataException($"n_nodes[{i}] must be >= 1, got {n32}");
            nNodes[i] = ToCheckedInt(n32, $"n_nodes[{i}]");
            perPieceFloats = CheckedMul(perPieceFloats, nNodes[i], "prod(n_nodes)");
        }

        int[] numKnots = new int[d];
        int expectedPieces = 1;
        for (int i = 0; i < d; i++)
        {
            uint k32 = r.ReadUInt32();
            numKnots[i] = ToCheckedInt(k32, $"num_knots[{i}]");
            expectedPieces = CheckedMul(expectedPieces, numKnots[i] + 1, "expected pieces");
        }

        var knots = new double[d][];
        for (int i = 0; i < d; i++)
        {
            knots[i] = new double[numKnots[i]];
            for (int j = 0; j < numKnots[i]; j++)
            {
                double knot = ReadFiniteDouble(r, $"knots[{i}][{j}]");
                if (!(lo[i] < knot && knot < hi[i]))
                    throw new InvalidDataException(
                        $"knots[{i}][{j}]={knot} must be strictly inside " +
                        $"domain[{i}] [{lo[i]}, {hi[i]}]");
                knots[i][j] = knot;
            }
            for (int j = 1; j < numKnots[i]; j++)
            {
                if (knots[i][j - 1] >= knots[i][j])
                    throw new InvalidDataException(
                        $"knots in dim {i} not strictly ascending");
            }
        }

        uint pieceCount = r.ReadUInt32();
        if (pieceCount != expectedPieces)
            throw new InvalidDataException(
                $"num_pieces={pieceCount} does not match prod(num_knots+1)={expectedPieces}");

        var pieces = new double[expectedPieces][];
        for (int p = 0; p < expectedPieces; p++)
        {
            pieces[p] = new double[perPieceFloats];
            for (int j = 0; j < perPieceFloats; j++)
                pieces[p][j] = ReadFiniteDouble(r, $"piece_tensors[{p}][{j}]");
        }

        return (domain, nNodes, knots, pieces);
    }
}
