// One-off fixture generator script reference.
//
// NOTE: Run via the project in tools/GenerateFixtures/ instead:
//   dotnet run --project tools/GenerateFixtures
//
// This .csx file is a reference copy of the same logic for environments where
// dotnet-script is available. Re-run after any deliberate format-spec bump.
//
// IMPORTANT: The 2D-simple fixture is hand-crafted from the binary-format.md
// worked example (128 bytes documented byte-for-byte) and serves as the
// cross-language anchor.
//
// Reference: docs/docs/binary-format.md

#r "src/ChebyshevSharp/bin/Debug/net10.0/ChebyshevSharp.dll"

using System;
using System.IO;
using ChebyshevSharp;

string fixtureDir = "tests/fixtures";
Directory.CreateDirectory(fixtureDir);

// 1) approx_2d_simple.pcb — f(x,y)=x+y on [-1,1]², n=[3,3]
//    The tensor values at Chebyshev Type-I nodes xi = cos((2i-1)*pi/6) are:
//    tensor[i,j] = nodes[i] + nodes[j] where nodes = [-sqrt(3)/2, 0, sqrt(3)/2]
//    Total file size: 128 bytes (header 12 + body 116).
{
    double s = Math.Sqrt(3.0) / 2.0;
    double[] nodes = { -s, 0.0, s };
    double[] tensor = new double[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            tensor[i * 3 + j] = nodes[i] + nodes[j];

    var cheb = ChebyshevApproximation.FromValues(
        tensor, 2,
        new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
        new[] { 3, 3 });

    string path = Path.Combine(fixtureDir, "approx_2d_simple.pcb");
    cheb.Save(path, format: "binary");
    Console.WriteLine($"Wrote {path} ({new FileInfo(path).Length} bytes, expected 128)");
}

// 2) approx_5d_bs.pcb — a 5D Black-Scholes-ish shape on standard option-pricing domains.
{
    var bs5d = new ChebyshevApproximation(
        (p, _) => Math.Max(p[0] - 100, 0) + 0.01 * p[1] - 0.5 * p[2] * p[2] + p[3] + p[4],
        5,
        new[] {
            new[] {  80.0, 120.0 },
            new[] {   0.1, 0.4   },
            new[] {   0.5, 1.5   },
            new[] {   0.0, 0.05  },
            new[] {  90.0, 110.0 },
        },
        new[] { 5, 5, 5, 5, 5 });
    bs5d.Build(verbose: false);
    string path = Path.Combine(fixtureDir, "approx_5d_bs.pcb");
    bs5d.Save(path, format: "binary");
    Console.WriteLine($"Wrote {path} ({new FileInfo(path).Length} bytes)");
}

// 3) spline_1d_kink.pcb — abs(x) on [-1,1] with knot at 0 (100 bytes)
{
    var kink = new ChebyshevSpline(
        (p, _) => Math.Abs(p[0]), 1,
        new[] { new[] { -1.0, 1.0 } }, new[] { 3 },
        knots: new[] { new[] { 0.0 } });
    kink.Build(verbose: false);
    string path = Path.Combine(fixtureDir, "spline_1d_kink.pcb");
    kink.Save(path, format: "binary");
    Console.WriteLine($"Wrote {path} ({new FileInfo(path).Length} bytes, expected 100)");
}

Console.WriteLine("Done.");
