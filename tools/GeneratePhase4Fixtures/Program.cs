using System;
using System.IO;
using ChebyshevSharp;

class Program
{
    static void Main()
    {
        // Output relative to the working directory when running from the solution root
        string outDir = Path.Combine("tests", "fixtures", "json-pre-v080");
        Directory.CreateDirectory(outDir);

        // --- ChebyshevApproximation ---
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        approx.Save(Path.Combine(outDir, "approx.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "approx.json")}");

        // --- ChebyshevSpline ---
        var spline = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { Array.Empty<double>(), Array.Empty<double>() });
        spline.Build(verbose: false);
        spline.Save(Path.Combine(outDir, "spline.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "spline.json")}");

        // --- ChebyshevSlider ---
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        slider.Save(Path.Combine(outDir, "slider.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "slider.json")}");

        // --- ChebyshevTT ---
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        tt.Build(verbose: false, seed: 42);
        tt.Save(Path.Combine(outDir, "tt.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "tt.json")}");
    }
}
