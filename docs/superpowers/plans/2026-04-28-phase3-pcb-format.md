# Phase 3 — Binary `.pcb` Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable little-endian binary serialization format (`.pcb`) to `ChebyshevApproximation` and `ChebyshevSpline` that is byte-for-byte compatible with PyChebyshev v0.14's format, plus the `PeekFormatVersion` helper from PyChebyshev v0.16.

**Architecture:** New `Internal/PcbFormat.cs` holds all binary read/write logic (constants, header, body-per-class-tag, low-level helpers mirroring Python's `_binary.py`). Existing `Save(string path)` on each class is widened to `Save(string path, string format = "json")` with format dispatch. `Load(string path)` becomes auto-detecting (peek 4 bytes; magic → binary, else → JSON). `PeekFormatVersion` is a class-static helper on both classes.

**Tech Stack:** C# / .NET 8 + .NET 10, xUnit, `BinaryReader`/`BinaryWriter` (LE by default), no new NuGet dependencies.

**Approved design spec:** `docs/superpowers/specs/2026-04-28-phase3-pcb-format-design.md` (commit `7bc0b3c`).

**Python reference:** `ref/PyChebyshev/src/pychebyshev/_binary.py` (421 lines, v0.14.0). Submodule currently pinned at v0.18.0 — content is forward-compatible.

**Branch:** `phase3-pcb-format`. **Worktree:** `.worktrees/phase3-pcb-format`. **Single PR.**

**Test count progression:** 765 → 797 (+32 tests), broken down as:
- Task 1: +5 (header validation)
- Task 2: +4 (Approximation internals)
- Task 3: +7 (Approximation public API + high-level round-trip + Save/Load API)
- Task 4: +2 (Approximation cross-feature)
- Task 5: +5 (Spline body + public API)
- Task 6: +3 (PeekFormatVersion)
- Task 7: +3 (corruption rejection)
- Task 8: +3 (T2 fixtures) — and docs/release prep, no new tests

---

## Task 0: Worktree setup

**WORKTREE ENFORCEMENT (MANDATORY).** Phase 3 work happens **only** inside `.worktrees/phase3-pcb-format`. Before any commit, run:

```bash
git rev-parse --show-toplevel
# Expected: /home/max/Documents/ChebyshevSharp/.worktrees/phase3-pcb-format
```

If the output ends in `/ChebyshevSharp` (the main repo) instead of `/ChebyshevSharp/.worktrees/phase3-pcb-format`, **STOP** and switch directory before any further command. This is the lesson from Phase 1's Task 4 cross-directory commit incident.

- [ ] **Step 1: Create worktree from main**

Run from `/home/max/Documents/ChebyshevSharp` (main repo):

```bash
git worktree add .worktrees/phase3-pcb-format -b phase3-pcb-format
cd .worktrees/phase3-pcb-format
git submodule update --init --recursive
```

- [ ] **Step 2: Verify baseline build + tests pass**

```bash
cd /home/max/Documents/ChebyshevSharp/.worktrees/phase3-pcb-format
dotnet build
dotnet test --verbosity minimal
```

Expected: `dotnet build` succeeds with **0 warnings**. `dotnet test` reports **765 passing, 0 failing**.

If the baseline fails, **STOP and report**. Do not proceed.

- [ ] **Step 3: Verify worktree path enforcement**

```bash
git rev-parse --show-toplevel
```

Expected output ends in `.worktrees/phase3-pcb-format`.

---

## Task 1: PcbFormat scaffolding + header read/write

**Goal:** Create `Internal/PcbFormat.cs` with format constants, header reader/writer, magic-byte detection. Build up just enough infrastructure that the remaining tasks can layer on top. No public-class wiring yet — all helpers are `internal`.

**Python reference:** `ref/PyChebyshev/src/pychebyshev/_binary.py`:
- Lines 26-34 (constants)
- Lines 85-151 (low-level helpers `_write_u32`, `_read_u32`, `_write_u32_array`, `_read_u32_array`, `_write_f64_array`, `_read_f64_array`)
- Lines 157-202 (header read/write + `detect_format`)

**Files:**
- Create: `src/ChebyshevSharp/Internal/PcbFormat.cs`
- Create: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`

- [ ] **Step 1: Write failing header-validation tests**

Create `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~HeaderTests" --verbosity minimal
```

Expected: compilation errors (`PcbFormat does not exist in the current context`), or — if you've already created an empty `PcbFormat.cs` — 5 failing tests.

- [ ] **Step 3: Implement `Internal/PcbFormat.cs`**

Create `src/ChebyshevSharp/Internal/PcbFormat.cs`:

```csharp
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
        byte[] magic = r.ReadBytes(4);
        if (magic.Length < 4)
            throw new EndOfStreamException("unexpected EOF reading magic");
        if (magic[0] != Magic[0] || magic[1] != Magic[1] ||
            magic[2] != Magic[2] || magic[3] != Magic[3])
            throw new InvalidDataException(
                "not a PyChebyshev binary file (bad magic)");

        byte major = r.ReadByte();
        byte minor = r.ReadByte();
        if (major != MajorVersion)
            throw new InvalidDataException(
                $"unsupported .pcb major version {major} " +
                $"(this build reads major {MajorVersion})");

        ushort classTag = r.ReadUInt16();
        uint reserved = r.ReadUInt32();
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~HeaderTests" --verbosity minimal
```

Expected: **5 passing, 0 failing**.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
dotnet test --verbosity minimal
```

Expected: **770 passing, 0 failing** (765 baseline + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/Internal/PcbFormat.cs tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: scaffold PcbFormat with header read/write (5 tests)

Internal-only static class with format constants, magic-byte detection,
and 12-byte header serialization. Mirrors PyChebyshev's _binary.py
header section (lines 26-34, 157-202). Endianness assertion in static
ctor; fail fast on non-LE runtimes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Approximation body read/write internals

**Goal:** Add `WriteApproximationBody` and `ReadApproximationBody` to `PcbFormat`. Internal-only — no public-class wiring yet. Verifies the byte layout for class_tag=1.

**Python reference:** `_binary.py` lines 208-283 (`write_approx`, `read_approx`).

**Files:**
- Modify: `src/ChebyshevSharp/Internal/PcbFormat.cs` (extend with body methods)
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` (add `ApproxBodyTests` class)

- [ ] **Step 1: Write failing internal-body tests**

Append to `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~ApproxBodyTests" --verbosity minimal
```

Expected: 4 compilation errors (`WriteApproximationBody`, `ReadApproximationBody` not found).

- [ ] **Step 3: Implement body read/write methods**

Append inside the `PcbFormat` class:

```csharp
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

    int total = 1;
    for (int i = 0; i < d; i++) total *= nNodes[i];
    if (tensorValues.Length != total)
        throw new ArgumentException(
            $"tensorValues.Length={tensorValues.Length} does not match prod(nNodes)={total}");

    for (int i = 0; i < total; i++) w.Write(tensorValues[i]);
}

public static (double[][] domain, int[] nNodes, double[] tensorValues) ReadApproximationBody(
    BinaryReader r)
{
    uint d32 = r.ReadUInt32();
    if (d32 < 1)
        throw new InvalidDataException($"num_dimensions must be >= 1, got {d32}");
    int d = checked((int)d32);

    double[] lo = new double[d];
    for (int i = 0; i < d; i++) lo[i] = r.ReadDouble();
    double[] hi = new double[d];
    for (int i = 0; i < d; i++) hi[i] = r.ReadDouble();

    var domain = new double[d][];
    for (int i = 0; i < d; i++)
    {
        if (lo[i] >= hi[i])
            throw new InvalidDataException(
                $"domain[{i}]: lo ({lo[i]}) must be < hi ({hi[i]})");
        domain[i] = new[] { lo[i], hi[i] };
    }

    int[] nNodes = new int[d];
    int total = 1;
    for (int i = 0; i < d; i++)
    {
        uint n32 = r.ReadUInt32();
        if (n32 < 1)
            throw new InvalidDataException($"n_nodes[{i}] must be >= 1, got {n32}");
        nNodes[i] = checked((int)n32);
        total = checked(total * nNodes[i]);
    }

    double[] tensor = new double[total];
    for (int i = 0; i < total; i++) tensor[i] = r.ReadDouble();
    return (domain, nNodes, tensor);
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~ApproxBodyTests" --verbosity minimal
```

Expected: **4 passing**.

- [ ] **Step 5: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **774 passing** (770 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/Internal/PcbFormat.cs tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: PcbFormat Approximation body read/write (4 tests)

Adds WriteApproximationBody / ReadApproximationBody. Mirrors
PyChebyshev _binary.py write_approx (lines 208-236) and read_approx
(lines 239-283). Internal-only; no public-class wiring yet.

Includes byte-level test against the docs/binary-format.md worked
example (f(x,y)=x+y, n=[3,3], expected file size 128 bytes) and
defensive rejection of d=0 and inverted domain on read.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ChebyshevApproximation public Save/Load wiring

**Goal:** Widen `Save(string path)` to `Save(string path, string format = "json")` and add auto-detect to `Load(string path)`. Wire to `PcbFormat`. Add high-level round-trip tests at the public-API level.

**Python reference:** `_binary.py` lines 208-283 (`write_approx`/`read_approx`); `barycentric.py` `save`/`load` methods (search for `format=` and `detect_format`).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (refactor Save/Load)
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` (add `ApproxRoundTripTests` and `SaveLoadApiTests`)

- [ ] **Step 1: Write failing public-API tests**

Append to `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~ApproxRoundTripTests|FullyQualifiedName~SaveLoadApiTests" --verbosity minimal
```

Expected: 7 failures (`Save` does not have a `format` parameter).

- [ ] **Step 3: Refactor `ChebyshevApproximation.Save` and `Load`**

Open `src/ChebyshevSharp/ChebyshevApproximation.cs`. Find the `Save(string path)` method (around line 605) and replace with a `format`-dispatching version. Rename the existing JSON body into a private `SaveJson` method.

Replace lines 601-632 with:

```csharp
/// <summary>
/// Save the built interpolant to a file.
/// </summary>
/// <param name="path">Destination file path.</param>
/// <param name="format">"json" (default) or "binary". Binary is the
/// portable .pcb format readable by C/Rust/Julia consumers.</param>
public void Save(string path, string format = "json")
{
    if (TensorValues == null)
        throw new InvalidOperationException(
            "Cannot save an unbuilt interpolant. Call Build() first.");

    switch (format)
    {
        case "json":
            SaveJson(path);
            break;
        case "binary":
            SaveBinary(path);
            break;
        default:
            throw new ArgumentException(
                $"Unknown format '{format}'. Expected 'json' or 'binary'.",
                nameof(format));
    }
}

private void SaveJson(string path)
{
    var state = new SerializationState
    {
        NumDimensions = NumDimensions,
        Domain = Domain,
        NNodes = NNodes,
        MaxDerivativeOrder = MaxDerivativeOrder,
        NodeArrays = NodeArrays,
        TensorValues = TensorValues!,
        Weights = Weights!,
        DiffMatrices = DiffMatrices!.Select(Flatten2D).ToArray(),
        BuildTime = BuildTime,
        NEvaluations = NEvaluations,
        OriginalNNodes = OriginalNNodes,
        ErrorThreshold = ErrorThreshold,
        MaxN = MaxN,
        Version = "0.5.0"
    };

    var options = new JsonSerializerOptions { WriteIndented = false };
    string json = JsonSerializer.Serialize(state, options);
    File.WriteAllText(path, json);
}

private void SaveBinary(string path)
{
    using var fs = File.Create(path);
    using var w = new BinaryWriter(fs);
    Internal.PcbFormat.WriteHeader(w, Internal.PcbFormat.ClassTagApproximation);
    Internal.PcbFormat.WriteApproximationBody(w, Domain, NNodes, TensorValues!);
}
```

Then find `Load(string path)` (around line 639) and replace with auto-detect:

```csharp
/// <summary>
/// Load a previously saved interpolant. Auto-detects JSON vs binary .pcb
/// by sniffing the first 4 bytes for the b"PCB\0" magic.
/// </summary>
/// <param name="path">Path to the saved file.</param>
/// <returns>The restored interpolant.</returns>
public static ChebyshevApproximation Load(string path)
{
    if (Internal.PcbFormat.IsBinary(path))
        return LoadBinary(path);
    return LoadJson(path);
}

private static ChebyshevApproximation LoadBinary(string path)
{
    using var fs = File.OpenRead(path);
    using var r = new BinaryReader(fs);
    var header = Internal.PcbFormat.ReadHeader(r);
    if (header.ClassTag != Internal.PcbFormat.ClassTagApproximation)
        throw new InvalidDataException(
            $"binary file class_tag={header.ClassTag} is not ChebyshevApproximation " +
            $"(tag {Internal.PcbFormat.ClassTagApproximation}); " +
            $"call ChebyshevSpline.Load instead if class_tag={Internal.PcbFormat.ClassTagSpline}");

    var (domain, nNodes, tensor) = Internal.PcbFormat.ReadApproximationBody(r);
    return FromValues(tensor, domain.Length, domain, nNodes);
}

private static ChebyshevApproximation LoadJson(string path)
{
    string json = File.ReadAllText(path);
    var state = JsonSerializer.Deserialize<SerializationState>(json)
        ?? throw new InvalidOperationException("Failed to deserialize");

    var obj = new ChebyshevApproximation
    {
        Function = null,
        NumDimensions = state.NumDimensions,
        Domain = state.Domain,
        NNodes = state.NNodes,
        MaxDerivativeOrder = state.MaxDerivativeOrder,
        NodeArrays = state.NodeArrays,
        TensorValues = state.TensorValues,
        Weights = state.Weights,
        BuildTime = state.BuildTime,
        NEvaluations = state.NEvaluations,
        _cachedErrorEstimate = null,
    };

    obj.DiffMatrices = new double[state.NumDimensions][,];
    for (int d = 0; d < state.NumDimensions; d++)
    {
        int n = state.NNodes[d];
        obj.DiffMatrices[d] = Unflatten2D(state.DiffMatrices[d], n, n);
    }
    obj.PrecomputeTransposedDiffMatrices();

    if (state.OriginalNNodes != null)
        obj.OriginalNNodes = state.OriginalNNodes;
    else
        obj.OriginalNNodes = obj.NNodes.Select(n => (int?)n).ToArray();
    obj.ErrorThreshold = state.ErrorThreshold;
    obj.MaxN = state.MaxN ?? 64;

    return obj;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~ApproxRoundTripTests|FullyQualifiedName~SaveLoadApiTests" --verbosity minimal
```

Expected: **7 passing**.

- [ ] **Step 5: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **781 passing** (774 + 7).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: ChebyshevApproximation Save(format)/Load auto-detect (7 tests)

Widens Save(string path) to Save(string path, string format = "json").
Load(string path) now auto-detects JSON vs .pcb binary via 4-byte magic
sniff. Existing JSON path refactored into private SaveJson/LoadJson
helpers (no behavior change).

3 round-trip tests (3D sin, n=1 dim edge, 5D shape) + 4 API tests
(magic header on save, JSON default unchanged, autodetect routes
correctly, unknown format throws ArgumentException).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Approximation cross-feature tests

**Goal:** Round-trip after algebra (`+`/`-`/`*`/`/`) and after `Slice`/`Extrude`. Confirms that the binary format works regardless of how the interpolant was built.

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`

- [ ] **Step 1: Write failing cross-feature tests**

Append:

```csharp
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
        var sliced = cheb.Slice(2, 1.0);
        var extruded = sliced.Extrude(2, (0.0, 2.0), 4);

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
```

- [ ] **Step 2: Run tests to verify they fail or check existing API names**

```bash
dotnet test --filter "FullyQualifiedName~ApproxCrossFeatureBinaryTests" --verbosity minimal
```

If the tests fail because of API name mismatches (e.g., `Slice` taking different arguments), check the existing test file `tests/ChebyshevSharp.Tests/ExtrudeSliceTests.cs` for the correct method signatures and adjust the test.

Expected on first run: failures because the tests are exercising existing functionality that may have edge-case quirks. Investigate any failure as a real test failure (not "wrong API"). If the test calls match the existing API, the tests should already pass — the binary round-trip is the only new thing.

- [ ] **Step 3: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **783 passing** (781 + 2). No new implementation needed in this task — Phase 1 already supplied algebra/slice/extrude, and Task 3 supplied binary save/load.

- [ ] **Step 4: Commit**

```bash
git add tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: ChebyshevApproximation cross-feature binary round-trips (2 tests)

Verifies that .pcb round-trip works on interpolants built via algebra
(+ operator) and via Slice/Extrude — i.e., regardless of how the
TensorValues were populated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: ChebyshevSpline Save(format)/Load auto-detect

**Goal:** Symmetric to Tasks 2+3 for Spline, plus the nested-nNodes hard-throw and unbuilt-throw. Single task because Spline body internals are tightly coupled with the public-API wiring (the throw conditions are checked in the public method, not the internal helper).

**Python reference:** `_binary.py` lines 289-421 (`write_spline`, `read_spline`).

**Files:**
- Modify: `src/ChebyshevSharp/Internal/PcbFormat.cs` (add `WriteSplineBody`, `ReadSplineBody`)
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs` (refactor Save/Load)
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` (add `SplineBinaryTests`)

- [ ] **Step 1: Write failing Spline tests**

Append:

```csharp
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
            nNodes: new[] { new[] { 3, 5 } }, // [[3, 5]] — nested form
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
            new[] { new[] { -1.0, 1.0 } }, new[] { 3 });
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~SplineBinaryTests" --verbosity minimal
```

Expected: 5 failures (`Save` does not have `format` param, etc.).

- [ ] **Step 3: Add `WriteSplineBody` / `ReadSplineBody` to `PcbFormat`**

Append inside the `PcbFormat` class:

```csharp
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

    int expectedPieces = 1;
    for (int i = 0; i < d; i++) expectedPieces *= (knotsPerDim[i].Length + 1);
    if (pieceTensorValues.Length != expectedPieces)
        throw new ArgumentException(
            $"pieceTensorValues.Length={pieceTensorValues.Length} does not match " +
            $"prod(num_knots[i]+1)={expectedPieces}");

    w.Write((uint)expectedPieces);
    int perPieceFloats = 1;
    for (int i = 0; i < d; i++) perPieceFloats *= nNodes[i];
    foreach (var piece in pieceTensorValues)
    {
        if (piece.Length != perPieceFloats)
            throw new ArgumentException(
                $"piece tensor length {piece.Length} != prod(nNodes)={perPieceFloats}");
        for (int j = 0; j < piece.Length; j++) w.Write(piece[j]);
    }
}

public static (double[][] domain, int[] nNodes, double[][] knotsPerDim, double[][] pieceTensors)
    ReadSplineBody(BinaryReader r)
{
    uint d32 = r.ReadUInt32();
    if (d32 < 1)
        throw new InvalidDataException($"num_dimensions must be >= 1, got {d32}");
    int d = checked((int)d32);

    double[] lo = new double[d];
    for (int i = 0; i < d; i++) lo[i] = r.ReadDouble();
    double[] hi = new double[d];
    for (int i = 0; i < d; i++) hi[i] = r.ReadDouble();
    var domain = new double[d][];
    for (int i = 0; i < d; i++)
    {
        if (lo[i] >= hi[i])
            throw new InvalidDataException($"domain[{i}]: lo ({lo[i]}) must be < hi ({hi[i]})");
        domain[i] = new[] { lo[i], hi[i] };
    }

    int[] nNodes = new int[d];
    int perPieceFloats = 1;
    for (int i = 0; i < d; i++)
    {
        uint n32 = r.ReadUInt32();
        if (n32 < 1)
            throw new InvalidDataException($"n_nodes[{i}] must be >= 1, got {n32}");
        nNodes[i] = checked((int)n32);
        perPieceFloats = checked(perPieceFloats * nNodes[i]);
    }

    int[] numKnots = new int[d];
    int expectedPieces = 1;
    for (int i = 0; i < d; i++)
    {
        uint k32 = r.ReadUInt32();
        numKnots[i] = checked((int)k32);
        expectedPieces = checked(expectedPieces * (numKnots[i] + 1));
    }

    var knots = new double[d][];
    for (int i = 0; i < d; i++)
    {
        knots[i] = new double[numKnots[i]];
        for (int j = 0; j < numKnots[i]; j++) knots[i][j] = r.ReadDouble();
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
        for (int j = 0; j < perPieceFloats; j++) pieces[p][j] = r.ReadDouble();
    }

    return (domain, nNodes, knots, pieces);
}
```

- [ ] **Step 4: Refactor `ChebyshevSpline.Save` and `Load`**

Open `src/ChebyshevSharp/ChebyshevSpline.cs`. Find `Save(string path)` (around line 673) and replace:

```csharp
/// <summary>
/// Save the built spline to a file.
/// </summary>
/// <param name="path">Destination file path.</param>
/// <param name="format">"json" (default) or "binary". Binary requires
/// flat (non-nested) nNodes — throws NotSupportedException otherwise.</param>
public void Save(string path, string format = "json")
{
    if (!Built)
        throw new InvalidOperationException(
            "Cannot save an unbuilt spline. Call Build() first.");

    switch (format)
    {
        case "json":
            SaveJson(path);
            break;
        case "binary":
            if (NestedNNodes != null)
                throw new NotSupportedException(
                    "binary format requires flat n_nodes (shared across pieces); " +
                    "use format='json' for nested-n_nodes splines");
            SaveBinary(path);
            break;
        default:
            throw new ArgumentException(
                $"Unknown format '{format}'. Expected 'json' or 'binary'.",
                nameof(format));
    }
}

private void SaveJson(string path)
{
    var state = new SplineSerializationState
    {
        Type = "ChebyshevSpline",
        NumDimensions = NumDimensions,
        Domain = Domain,
        NNodes = NNodes,
        MaxDerivativeOrder = MaxDerivativeOrder,
        Knots = Knots,
        Shape = Shape,
        BuildTime = BuildTime,
        PieceStates = Pieces.Select(p =>
        {
            var ps = new PieceState
            {
                NumDimensions = p!.NumDimensions,
                Domain = p.Domain,
                NNodes = p.NNodes,
                MaxDerivativeOrder = p.MaxDerivativeOrder,
                NodeArrays = p.NodeArrays,
                TensorValues = p.TensorValues!,
                Weights = p.Weights!,
                DiffMatrices = p.DiffMatrices!.Select(ChebyshevApproximation.Flatten2D).ToArray(),
                BuildTime = p.BuildTime,
                NEvaluations = p.NEvaluations,
            };
            return ps;
        }).ToArray(),
        OriginalNNodes = OriginalNNodes.Length > 0 ? OriginalNNodes : null,
        ErrorThreshold = ErrorThreshold,
        MaxN = MaxN,
        NestedNNodes = NestedNNodes,
        Version = "0.5.0",
    };

    var options = new JsonSerializerOptions { WriteIndented = false };
    string json = JsonSerializer.Serialize(state, options);
    File.WriteAllText(path, json);
}

private void SaveBinary(string path)
{
    using var fs = File.Create(path);
    using var w = new BinaryWriter(fs);
    Internal.PcbFormat.WriteHeader(w, Internal.PcbFormat.ClassTagSpline);
    var pieceTensors = Pieces.Select(p => p!.TensorValues!).ToArray();
    Internal.PcbFormat.WriteSplineBody(w, Domain, NNodes, Knots, pieceTensors);
}
```

Then find `Load(string path)` (around line 723) and replace with autodetect:

```csharp
/// <summary>
/// Load a previously saved spline. Auto-detects JSON vs binary .pcb
/// by sniffing the first 4 bytes for the b"PCB\0" magic.
/// </summary>
public static ChebyshevSpline Load(string path)
{
    if (Internal.PcbFormat.IsBinary(path))
        return LoadBinary(path);
    return LoadJson(path);
}

private static ChebyshevSpline LoadBinary(string path)
{
    using var fs = File.OpenRead(path);
    using var r = new BinaryReader(fs);
    var header = Internal.PcbFormat.ReadHeader(r);
    if (header.ClassTag != Internal.PcbFormat.ClassTagSpline)
        throw new InvalidDataException(
            $"binary file class_tag={header.ClassTag} is not ChebyshevSpline " +
            $"(tag {Internal.PcbFormat.ClassTagSpline}); " +
            $"call ChebyshevApproximation.Load instead if class_tag={Internal.PcbFormat.ClassTagApproximation}");

    var (domain, nNodes, knots, pieceTensors) = Internal.PcbFormat.ReadSplineBody(r);
    return FromValues(pieceTensors, domain.Length, domain, nNodes, knots);
}

private static ChebyshevSpline LoadJson(string path)
{
    string json = File.ReadAllText(path);
    var state = JsonSerializer.Deserialize<SplineSerializationState>(json)
        ?? throw new InvalidOperationException("Failed to deserialize");

    if (state.Type != "ChebyshevSpline")
        throw new InvalidOperationException(
            $"Expected type ChebyshevSpline, got {state.Type}");

    var pieces = state.PieceStates.Select(ps =>
    {
        var piece = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = ps.NumDimensions,
            Domain = ps.Domain,
            NNodes = ps.NNodes,
            MaxDerivativeOrder = ps.MaxDerivativeOrder,
            NodeArrays = ps.NodeArrays,
            TensorValues = ps.TensorValues,
            Weights = ps.Weights,
            BuildTime = ps.BuildTime,
            NEvaluations = ps.NEvaluations,
        };
        piece.DiffMatrices = new double[ps.NumDimensions][,];
        for (int d = 0; d < ps.NumDimensions; d++)
        {
            int n = ps.NNodes[d];
            piece.DiffMatrices[d] = ChebyshevApproximation.Unflatten2D(ps.DiffMatrices[d], n, n);
        }
        piece.PrecomputeTransposedDiffMatrices();
        return piece;
    }).ToArray();

    var intervals = ComputeIntervals(state.NumDimensions, state.Domain, state.Knots);
    int?[] originalNNodes = state.OriginalNNodes ?? Array.Empty<int?>();

    return new ChebyshevSpline
    {
        Function = null,
        NumDimensions = state.NumDimensions,
        Domain = state.Domain,
        NNodes = state.NNodes,
        MaxDerivativeOrder = state.MaxDerivativeOrder,
        Knots = state.Knots,
        Intervals = intervals,
        Shape = state.Shape,
        Pieces = pieces.Cast<ChebyshevApproximation?>().ToArray(),
        Built = true,
        BuildTime = state.BuildTime,
        OriginalNNodes = originalNNodes,
        ErrorThreshold = state.ErrorThreshold,
        MaxN = state.MaxN ?? 64,
        NestedNNodes = state.NestedNNodes,
    };
}
```

**Note:** the `LoadJson` body above is the existing pre-Phase-3 code unchanged — only its name is changed from `Load` → `LoadJson`. If the existing method has more lines than shown (the file was truncated at line 779 in plan research), preserve them verbatim under the new name.

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~SplineBinaryTests" --verbosity minimal
```

Expected: **5 passing**.

- [ ] **Step 6: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **788 passing** (783 + 5).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/PcbFormat.cs src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: ChebyshevSpline Save(format)/Load auto-detect (5 tests)

Adds WriteSplineBody/ReadSplineBody to PcbFormat. ChebyshevSpline.Save
gains the format kwarg; nested-nNodes splines hard-throw
NotSupportedException on format="binary" (matches PyChebyshev v0.14
_binary.py line 250). Existing JSON path renamed to private
SaveJson/LoadJson helpers.

3 round-trip tests (1D abs|x| with kink — the docs worked example,
2D multi-knot, nested-nNodes throw, unbuilt throw, cross-class-tag
load rejection).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PeekFormatVersion static helper

**Goal:** Add `static int PeekFormatVersion(string path)` to both `ChebyshevApproximation` and `ChebyshevSpline`. Returns major version byte without reading the body. Mirrors PyChebyshev v0.16 (one-phase early arrival, per spec §3.1).

**Python reference:** `_binary.py` lines 40-79 (`peek_format_version`).

**Files:**
- Modify: `src/ChebyshevSharp/Internal/PcbFormat.cs` (add `PeekFormatVersion` core method)
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (expose static method)
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs` (expose static method)
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` (add `PeekFormatVersionTests`)

- [ ] **Step 1: Write failing tests**

Append:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~PeekFormatVersionTests" --verbosity minimal
```

Expected: 3 compilation errors (`PeekFormatVersion` not defined).

- [ ] **Step 3: Implement `PeekFormatVersion` core method in `PcbFormat`**

Append inside the `PcbFormat` class:

```csharp
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
```

- [ ] **Step 4: Expose on both public classes**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, add near the existing `Load` method:

```csharp
/// <summary>
/// Read the major version byte of a .pcb binary file without deserializing the body.
/// Useful for forward-compat tooling.
/// </summary>
/// <param name="path">Path to a .pcb file.</param>
/// <returns>The major format version (currently 1).</returns>
/// <exception cref="FileNotFoundException">Thrown if the path does not exist.</exception>
/// <exception cref="InvalidDataException">Thrown if the file is not a .pcb file
/// (no magic header) or is shorter than 12 bytes.</exception>
public static int PeekFormatVersion(string path)
    => Internal.PcbFormat.PeekFormatVersion(path);
```

In `src/ChebyshevSharp/ChebyshevSpline.cs`, add the same static method (identical body — both forward to the internal helper).

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~PeekFormatVersionTests" --verbosity minimal
```

Expected: **3 passing**.

- [ ] **Step 6: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **791 passing** (788 + 3).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/PcbFormat.cs src/ChebyshevSharp/ChebyshevApproximation.cs src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: PeekFormatVersion static helper (3 tests)

Mirrors PyChebyshev's peek_format_version (v0.16; brought one phase
forward per design spec §3.1). Reads byte 4 of the header without
parsing the body. Exposed on both ChebyshevApproximation and
ChebyshevSpline; both forward to a shared PcbFormat.PeekFormatVersion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Corruption rejection tests

**Goal:** Lock down rejection behavior on malformed binary inputs. Three orthogonal corruption modes: bad class tag, truncated body, malformed knot section.

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs`

- [ ] **Step 1: Write failing corruption tests**

Append:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail (or pass — most should already pass given Task 5)**

```bash
dotnet test --filter "FullyQualifiedName~CorruptionRejectionTests" --verbosity minimal
```

Expected: most pass; investigate any failure as a real issue (corruption rejection should already be wired in via Tasks 1, 5).

If any test fails because the existing implementation doesn't quite reject the way the test expects, fix the implementation in `PcbFormat` (not the test) — corruption rejection is part of the contract.

- [ ] **Step 3: Run full suite**

```bash
dotnet test --verbosity minimal
```

Expected: **794 passing** (791 + 3).

- [ ] **Step 4: Commit**

```bash
git add tests/ChebyshevSharp.Tests/BinaryFormatTests.cs
git commit -m "$(cat <<'EOF'
phase3: corruption rejection tests for .pcb format (3 tests)

Locks down rejection behavior on three orthogonal corruption modes:
unknown class_tag, truncated body (post-header EOF), malformed knot
section (non-ascending knots). All three should throw
InvalidDataException or EndOfStreamException with informative messages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: T2 fixtures + docs + parity metadata + release prep

**Goal:** Wrap up Phase 3. Generate three committed `.pcb` fixtures (one hand-crafted from the documented byte layout, two generated by C#'s own writer for shape-coverage), write a `PcbFixtures.cs` loader and 3 fixture-based round-trip tests, then update README/changelog/csproj/skip_csharp.txt/CLAUDE.md.

**Files:**
- Create: `tests/fixtures/approx_2d_simple.pcb` (hand-crafted bytes)
- Create: `tests/fixtures/approx_5d_bs.pcb` (generated by our writer)
- Create: `tests/fixtures/spline_1d_kink.pcb` (generated by our writer)
- Create: `tests/fixtures/REGENERATE.md`
- Create: `tests/ChebyshevSharp.Tests/Helpers/PcbFixtures.cs`
- Modify: `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` (copy fixtures to output dir)
- Modify: `tests/ChebyshevSharp.Tests/BinaryFormatTests.cs` (add `PcbFixtureTests`)
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj` (bump version)
- Modify: `README.md` (no parity badge change — see notes below)
- Modify: `docs/docs/changelog.md` (add v0.7.0)
- Create: `docs/docs/binary-format.md`
- Modify: `docs/docs/toc.yml` (link new page)
- Modify: `skip_csharp.txt`
- Modify: `CLAUDE.md` (status block)

- [ ] **Step 1: Generate `approx_2d_simple.pcb` (hand-crafted from documented byte layout)**

This fixture is the docs/binary-format.md worked example: f(x,y) = x + y on [-1,1]^2 with n=[3,3]. Total 128 bytes. Run this from the repo root **once** as a one-off bytes generator script (it is safe to commit the script too as `tools/GenerateApprox2dSimple.csx` if you prefer):

Quick interactive way: generate via our own writer (we trust it after Tasks 1-5), then copy the bytes to disk. Run a small dotnet-script in the worktree:

```csharp
// tools/GenerateFixtures.csx (one-off; commit this script too for reproducibility)
using ChebyshevSharp;
using ChebyshevSharp.Internal;

double s = Math.Sqrt(3.0) / 2.0;
double[] nodes = { -s, 0.0, s };
double[] tensor = new double[9];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        tensor[i*3 + j] = nodes[i] + nodes[j];

using var fs = File.Create("tests/fixtures/approx_2d_simple.pcb");
using var w = new BinaryWriter(fs);
PcbFormat.WriteHeader(w, PcbFormat.ClassTagApproximation);
PcbFormat.WriteApproximationBody(w,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 3, 3 },
    tensor);
```

**Verification:** `wc -c tests/fixtures/approx_2d_simple.pcb` must report exactly **128**. The first 4 bytes must be `0x50 0x43 0x42 0x00`. Inspect with `xxd tests/fixtures/approx_2d_simple.pcb | head -1` — first line should start with `00000000: 5043 4200 0100 0100 0000 0000 0200 0000`.

If the bytes match the documented spec, the file is locked-in.

- [ ] **Step 2: Generate `approx_5d_bs.pcb` and `spline_1d_kink.pcb` (via our writer)**

```csharp
// continued in tools/GenerateFixtures.csx
// 5D Black-Scholes-like shape (use a dummy multivariate function — the bytes are
// what matter for round-trip tests; analytical correctness is covered elsewhere).
var bs5d = new ChebyshevApproximation(
    (p, _) => Math.Max(p[0] - 100, 0) + 0.01 * p[1] - 0.5 * p[2] * p[2] + p[3] + p[4],
    5,
    new[] {
        new[] {  80.0, 120.0 },   // S
        new[] {   0.1, 0.4   },   // sigma
        new[] {   0.5, 1.5   },   // T
        new[] {   0.0, 0.05  },   // r
        new[] {  90.0, 110.0 },   // K
    },
    new[] { 5, 5, 5, 5, 5 });
bs5d.Build(verbose: false);
bs5d.Save("tests/fixtures/approx_5d_bs.pcb", format: "binary");

// 1D abs(x) with kink at 0 — the spline worked example.
var kink = new ChebyshevSpline(
    (p, _) => Math.Abs(p[0]), 1,
    new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
    knots: new[] { new[] { 0.0 } });
kink.Build(verbose: false);
kink.Save("tests/fixtures/spline_1d_kink.pcb", format: "binary");
```

**Verification:** `wc -c tests/fixtures/spline_1d_kink.pcb` must report exactly **100** bytes (per binary-format.md). The 5D fixture size depends on the build but should be in the few-thousand-byte range.

- [ ] **Step 3: Write `tests/fixtures/REGENERATE.md`**

```markdown
# Fixture regeneration

These three `.pcb` files are committed binary fixtures used by C# tests in
`BinaryFormatTests.cs::PcbFixtureTests`. They lock down the .pcb v1 byte layout.

## Files

| File | Size | Source | Description |
|---|---:|---|---|
| `approx_2d_simple.pcb` | 128 B | hand-crafted | f(x,y)=x+y on [-1,1]², n=[3,3]. The byte layout is documented byte-for-byte in `ref/PyChebyshev/docs/user-guide/binary-format.md` §"Worked example". |
| `approx_5d_bs.pcb` | ~25 KB | C# writer | 5D Black-Scholes-like shape on standard option-pricing domain. Bytes generated by ChebyshevSharp's own `PcbFormat.WriteApproximationBody` after Phase 3 round-trip tests proved correctness. |
| `spline_1d_kink.pcb` | 100 B | C# writer | abs(x) on [-1,1] with knot at 0 (the spline worked example from binary-format.md). 100-byte total, also documented byte-for-byte. |

## Regenerating from C#

Re-run `tools/GenerateFixtures.csx` from the repo root after any deliberate
format-spec bump. The script is committed alongside these fixtures.

## Cross-checking against PyChebyshev

When the submodule (`ref/PyChebyshev`) bumps to a future version that changes
`.pcb` (e.g., v1.1, v2), regenerate fixtures from Python and `cmp` against the
C# bytes:

```bash
cd ref/PyChebyshev && uv sync
uv run python -c "
from pychebyshev import ChebyshevApproximation, ChebyshevSpline
import numpy as np

s = np.sqrt(3) / 2
ChebyshevApproximation.from_values(
    tensor_values=np.array([[-2*s, -s, 0], [-s, 0, s], [0, s, 2*s]]),
    num_dimensions=2, domain=[(-1,1),(-1,1)], n_nodes=[3,3]
).save('/tmp/py_approx_2d_simple.pcb', format='binary')
"
cmp tests/fixtures/approx_2d_simple.pcb /tmp/py_approx_2d_simple.pcb
# Expected: silent (bytes match exactly).
```

If `cmp` reports a difference: investigate before bumping the C# format version.
```

- [ ] **Step 4: Configure fixtures to copy into test output**

In `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj`, add inside the existing `<Project>` element:

```xml
<ItemGroup>
  <None Include="..\..\tests\fixtures\*.pcb">
    <Link>fixtures\%(Filename)%(Extension)</Link>
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
  </None>
</ItemGroup>
```

The `Link` element ensures the files appear in `bin/Debug/net8.0/fixtures/` for runtime access.

- [ ] **Step 5: Write `tests/ChebyshevSharp.Tests/Helpers/PcbFixtures.cs`**

```csharp
using System.IO;
using System.Reflection;

namespace ChebyshevSharp.Tests.Helpers;

internal static class PcbFixtures
{
    /// <summary>
    /// Returns the absolute path to a fixture file, resolved relative to the
    /// test assembly's output directory.
    /// </summary>
    public static string Path(string name)
    {
        string baseDir = System.IO.Path.GetDirectoryName(
            Assembly.GetExecutingAssembly().Location)!;
        return System.IO.Path.Combine(baseDir, "fixtures", name);
    }
}
```

- [ ] **Step 6: Write fixture-based tests**

Append to `BinaryFormatTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

public class PcbFixtureTests
{
    [Fact]
    public void Test_fixture_approx_2d_simple_loads_and_evaluates()
    {
        string path = PcbFixtures.Path("approx_2d_simple.pcb");
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
        string path = PcbFixtures.Path("approx_5d_bs.pcb");
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
        string path = PcbFixtures.Path("spline_1d_kink.pcb");
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
```

- [ ] **Step 7: Run fixture tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~PcbFixtureTests" --verbosity minimal
```

Expected: **3 passing**.

- [ ] **Step 8: Bump csproj version**

In `src/ChebyshevSharp/ChebyshevSharp.csproj`, change:
- `<Version>0.6.0</Version>` → `<Version>0.7.0</Version>`
- `<PyChebyshevParity>0.18.0</PyChebyshevParity>` → unchanged (Phase 3 fills v0.14-era gap, doesn't bump parity)
- `<InformationalVersion>0.6.0+pychebyshev.0.18.0</InformationalVersion>` → `<InformationalVersion>0.7.0+pychebyshev.0.18.0</InformationalVersion>`
- `<Description>` — append "+ portable .pcb binary serialization" if the description currently summarizes features.

- [ ] **Step 9: Add changelog entry**

Append to `docs/docs/changelog.md`:

```markdown
## v0.7.0 — 2026-04-28 — PyChebyshev parity v0.18.0 (binary format fill-in)

Phase 3 of the v0.20.1 phased port. Adds a portable little-endian binary
serialization format (`.pcb`) so cross-language consumers (C, Rust, Julia) can
read ChebyshevSharp interpolants without the .NET runtime. Bit-for-bit
byte-compatible with PyChebyshev v0.14's format.

### Added

- `ChebyshevApproximation.Save(string path, string format = "json")` and
  `ChebyshevSpline.Save(string path, string format = "json")` — `format` accepts
  `"json"` (existing default) or `"binary"` (the portable `.pcb` format).
- `ChebyshevApproximation.Load(string path)` and `ChebyshevSpline.Load(string path)`
  now auto-detect JSON vs binary by sniffing the first 4 bytes for the
  `b"PCB\x00"` magic header.
- `static int ChebyshevApproximation.PeekFormatVersion(string path)` and
  `static int ChebyshevSpline.PeekFormatVersion(string path)` — read the major
  version byte without parsing the body. (PyChebyshev added this in v0.16; we
  bring it forward one phase as a small early-arrival per the design spec.)
- New `Internal/PcbFormat.cs` holding all binary read/write logic with explicit
  little-endian assertions.

### Changed

- `ChebyshevSpline.Save(path, format="binary")` throws `NotSupportedException`
  for splines built with nested per-piece `nNodes` (the `int[][]` form from
  Phase 1's special-points work). Use `format="json"` for those.
- `ChebyshevSlider` and `ChebyshevTT` remain JSON-only in v0.7.0.

### Test count: 765 → 797 (+32)

See [PR #?] for the full diff and the [design spec](https://github.com/0xC000005/ChebyshevSharp/blob/main/docs/superpowers/specs/2026-04-28-phase3-pcb-format-design.md). Phase 4 (ergonomics polish) is next.
```

- [ ] **Step 10: Add `docs/docs/binary-format.md`**

Create the user-guide page paraphrased from PyChebyshev's:

```markdown
# Portable Binary Format (`.pcb`)

ChebyshevSharp v0.7.0 introduces a portable binary serialization format
alongside the default JSON format. The goal: let consumers in **C, Rust, Julia,
or any other language** read ChebyshevSharp interpolants without a .NET runtime.

The format is byte-for-byte compatible with PyChebyshev v0.14's `.pcb`. Files
written by either library can be read by the other.

## When to use which format

| Format | Use when |
|---|---|
| **JSON** (default) | .NET-only round-trips; need full fidelity (build telemetry, derivative-id registry once Phase 4 lands) |
| **Binary** (`.pcb`) | Cross-language consumers; sharing models with C/Rust/Julia code; long-term archival |

```csharp
cheb.Save("model.pcb", format: "binary");      // portable
cheb.Save("model.json");                       // JSON (default)
cheb.Save("model.json", format: "json");       // explicit

ChebyshevApproximation.Load("model.pcb");      // auto-detects
```

`Load()` sniffs the first 4 bytes — `b"PCB\x00"` routes to the binary reader,
anything else to the JSON reader.

## Coverage in v0.7.0

- **`ChebyshevApproximation`** — full support.
- **`ChebyshevSpline`** — full support, with one restriction: the spline must
  use **flat** `nNodes` (a single `int` per dim, shared across pieces). Splines
  built with nested per-piece `nNodes` (the `int[][]` form introduced in Phase
  1 for special points) cannot be saved as `.pcb` and throw
  `NotSupportedException`; use JSON for those.
- **`ChebyshevSlider`**, **`ChebyshevTT`** — JSON only in v0.7.0 (matches
  PyChebyshev v0.14's pickle-only restriction for these).

## Format specification (v1)

See [PyChebyshev's binary-format.md](https://github.com/0xC000005/PyChebyshev/blob/main/docs/user-guide/binary-format.md)
for the complete byte-level specification. ChebyshevSharp's `.pcb` files are
byte-for-byte identical to PyChebyshev's.

## What the format does not store

These fields are dropped on `format="binary"`:

| Field | Replacement |
|---|---|
| `Function` | always dropped (also dropped by JSON) |
| `Weights`, `DiffMatrices` | recomputed on load from `(domain, nNodes)` |
| Cached error estimate | recomputed lazily |
| Build telemetry (`BuildTime`, `NEvaluations`, `Method`) | not preserved (use JSON for full fidelity) |
| `MaxDerivativeOrder` | resets to default `2` on load |

If you need any of those preserved, use JSON.

## Security

The binary reader does no executable deserialization. It can be used to load
files from untrusted sources — it will reject malformed files with
`InvalidDataException`.
```

- [ ] **Step 11: Link the new doc in `docs/docs/toc.yml`**

Find the existing TOC entry list and add:

```yaml
- name: Binary Format
  href: binary-format.md
```

(Place it after the existing serialization-related entry, if one exists.)

- [ ] **Step 12: Update `skip_csharp.txt`**

Open `skip_csharp.txt` and remove or strikethrough any lines tagged as Phase 3 / binary-format / `.pcb`. Add a note above with the test-count delta:

```
# Phase 3 (v0.7.0, PyChebyshev parity v0.18.0): binary .pcb format complete.
# Test count: 765 → 797 (+32).
```

- [ ] **Step 13: Update `CLAUDE.md` Status block**

Find the Status section and update:

```markdown
**Feature-complete against PyChebyshev v0.18.0** (Phases 1+2+3 of the 6-phase v0.20.1 port complete; see
`docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md`).
All four public classes (`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`) mirror the Python API surface. v0.7.0 adds portable `.pcb` binary
serialization (Phase 3 fill-in; PyChebyshev parity tag unchanged at v0.18.0).
`dotnet test` runs **797/797** passing.
```

- [ ] **Step 14: Run full suite one last time**

```bash
dotnet test --verbosity minimal
```

Expected: **797 passing, 0 failing**.

- [ ] **Step 15: Commit**

```bash
git add tests/fixtures/ tests/ChebyshevSharp.Tests/ src/ChebyshevSharp/ChebyshevSharp.csproj README.md docs/ skip_csharp.txt CLAUDE.md tools/
git commit -m "$(cat <<'EOF'
phase3: T2 fixtures, docs, parity metadata, v0.7.0 release prep

- tests/fixtures/: 3 committed .pcb files
  - approx_2d_simple.pcb (128 B, hand-crafted from binary-format.md spec)
  - approx_5d_bs.pcb (~25 KB, generated via our writer)
  - spline_1d_kink.pcb (100 B, generated via our writer)
  - REGENERATE.md documents the regeneration workflow incl. Python
    cross-check for future submodule bumps.
- 3 fixture-based tests via PcbFixtures.cs loader.
- Bump <Version>0.7.0; <PyChebyshevParity> unchanged at 0.18.0
  (Phase 3 fills a v0.14-era gap rather than advancing the tag).
- changelog v0.7.0 entry following two-tier convention.
- New docs/docs/binary-format.md user-guide page.
- skip_csharp.txt + CLAUDE.md status block updated.

Test count: 765 → 797 (+32).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 16: Final worktree check**

```bash
git rev-parse --show-toplevel
git log --oneline phase3-pcb-format ^main
```

Expected: worktree path ends in `.worktrees/phase3-pcb-format`. Git log shows ~8 phase3 commits in chronological order.

---

## After plan execution

Once all 8 tasks are complete:

1. From the main repo (not the worktree): push the branch and open the PR.
   ```bash
   cd /home/max/Documents/ChebyshevSharp
   git fetch origin
   git push -u origin phase3-pcb-format
   gh pr create --base main --title "Phase 3: portable .pcb binary serialization (v0.7.0)" \
     --body "$(see commit log for the per-task summaries)"
   ```
2. Address review feedback. Bug fixes go in new commits with `phase3:` prefix.
3. Once approved, merge via squash or merge-commit (the project uses merge commits per Phase 1+2 history).
4. Tag and release: `gh release create v0.7.0 --target main --title "v0.7.0 — PyChebyshev parity v0.18.0 (binary format fill-in)" --notes-file /tmp/v070-release-notes.md`. Triggers `publish.yml` → NuGet.
5. Clean up: `git worktree remove --force .worktrees/phase3-pcb-format && git branch -d phase3-pcb-format`.
