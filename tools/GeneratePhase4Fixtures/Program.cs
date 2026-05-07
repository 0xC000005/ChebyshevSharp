using System;
using System.IO;
using ChebyshevSharp;

class Program
{
    static void Main()
    {
        // These committed files are old-format migration fixtures. The current
        // library can only produce current-format JSON, so write candidates to
        // a temp directory and preserve canonical fixtures when they differ.
        string fixtureDir = Path.Combine("tests", "fixtures", "json-pre-v080");
        string candidateDir = Path.Combine(
            Path.GetTempPath(),
            $"chebsharp-json-pre-v080.{Guid.NewGuid():N}");
        Directory.CreateDirectory(candidateDir);

        try
        {
            // --- ChebyshevApproximation ---
            var approx = new ChebyshevApproximation(
                (p, _) => p[0] + p[1],
                numDimensions: 2,
                domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: new[] { 5, 5 });
            approx.Build(verbose: false);
            approx.Save(Path.Combine(candidateDir, "approx.json"));
            PreserveCanonicalFixture(candidateDir, fixtureDir, "approx.json");

            // --- ChebyshevSpline ---
            var spline = new ChebyshevSpline(
                (p, _) => p[0] + p[1],
                numDimensions: 2,
                domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: new[] { 5, 5 },
                knots: new[] { Array.Empty<double>(), Array.Empty<double>() });
            spline.Build(verbose: false);
            spline.Save(Path.Combine(candidateDir, "spline.json"));
            PreserveCanonicalFixture(candidateDir, fixtureDir, "spline.json");

            // --- ChebyshevSlider ---
            var slider = new ChebyshevSlider(
                (p, _) => p[0] + p[1] + p[2],
                numDimensions: 3,
                domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: new[] { 5, 5, 5 },
                partition: new[] { new[] { 0 }, new[] { 1, 2 } },
                pivotPoint: new[] { 0.0, 0.0, 0.0 });
            slider.Build(verbose: false);
            slider.Save(Path.Combine(candidateDir, "slider.json"));
            PreserveCanonicalFixture(candidateDir, fixtureDir, "slider.json");

            // --- ChebyshevTT ---
            var tt = new ChebyshevTT(
                p => p[0] + p[1] + p[2],
                numDimensions: 3,
                domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: new[] { 5, 5, 5 });
            tt.Build(verbose: false, seed: 42);
            tt.Save(Path.Combine(candidateDir, "tt.json"));
            PreserveCanonicalFixture(candidateDir, fixtureDir, "tt.json");
        }
        finally
        {
            if (Directory.Exists(candidateDir))
                Directory.Delete(candidateDir, recursive: true);
        }
    }

    private static void PreserveCanonicalFixture(string candidateDir, string fixtureDir, string name)
    {
        string candidatePath = Path.Combine(candidateDir, name);
        string fixturePath = Path.Combine(fixtureDir, name);

        if (!File.Exists(fixturePath))
        {
            throw new FileNotFoundException(
                "Cannot regenerate this fixture with the current library because " +
                "it must remain pre-v0.8 JSON.",
                fixturePath);
        }

        if (FilesEqual(fixturePath, candidatePath))
        {
            File.Copy(candidatePath, fixturePath, overwrite: true);
            Console.WriteLine($"Wrote {fixturePath}");
            return;
        }

        Console.WriteLine(
            $"Preserved {fixturePath}; current-format candidate differs from " +
            "the committed pre-v0.8 migration fixture.");
    }

    private static bool FilesEqual(string left, string right)
    {
        byte[] leftBytes = File.ReadAllBytes(left);
        byte[] rightBytes = File.ReadAllBytes(right);
        return leftBytes.SequenceEqual(rightBytes);
    }
}
