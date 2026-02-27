using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// Accuracy tests
// ======================================================================

public class TensorTrainAccuracyTests
{
    [Fact]
    public void Test_1D_Sin()
    {
        // 1D sin(x) — TT is trivially rank 1
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]), 1,
            new[] { new[] { -1.0, 1.0 } },
            new[] { 11 },
            maxRank: 3);
        tt.Build(verbose: false, method: "svd");

        foreach (double x in new[] { -0.9, -0.3, 0.0, 0.5, 0.99 })
            Assert.True(Math.Abs(tt.Eval(new[] { x }) - Math.Sin(x)) < 1e-8,
                $"Error at x={x}: {Math.Abs(tt.Eval(new[] { x }) - Math.Sin(x)):E2}");
    }

    [Fact]
    public void Test_3D_Sin_SVD()
    {
        var tt = TestFixtures.TtSin3DSvd;
        double[][] pts = {
            new[] { 0.5, 0.3, 0.1 },
            new[] { -0.7, 0.0, 0.8 },
            new[] { 0.0, 0.0, 0.0 },
            new[] { 0.99, -0.99, 0.5 },
        };
        foreach (var pt in pts)
        {
            double exact = Math.Sin(pt[0]) + Math.Sin(pt[1]) + Math.Sin(pt[2]);
            double approx = tt.Eval(pt);
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"SVD error {Math.Abs(approx - exact):E2} at [{string.Join(", ", pt)}]");
        }
    }

    [Fact]
    public void Test_3D_Sin_Cross()
    {
        var tt = TestFixtures.TtSin3D;
        double[][] pts = {
            new[] { 0.5, 0.3, 0.1 },
            new[] { -0.7, 0.0, 0.8 },
            new[] { 0.99, -0.99, 0.5 },
        };
        foreach (var pt in pts)
        {
            double exact = Math.Sin(pt[0]) + Math.Sin(pt[1]) + Math.Sin(pt[2]);
            double approx = tt.Eval(pt);
            Assert.True(Math.Abs(approx - exact) < 1e-6,
                $"Cross error {Math.Abs(approx - exact):E2} at [{string.Join(", ", pt)}]");
        }
    }

    [Fact]
    public void Test_3D_Polynomial()
    {
        // x0^2*x1 + x2
        var tt = new ChebyshevTT(
            x => x[0] * x[0] * x[1] + x[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 7, 7, 7 },
            maxRank: 5);
        tt.Build(verbose: false, method: "svd");

        double[][] pts = {
            new[] { 0.5, 0.3, 0.1 },
            new[] { -0.8, 0.7, -0.5 },
            new[] { 0.0, 1.0, -1.0 },
        };
        foreach (var pt in pts)
        {
            double exact = pt[0] * pt[0] * pt[1] + pt[2];
            double approx = tt.Eval(pt);
            Assert.True(Math.Abs(approx - exact) < 1e-8,
                $"Poly error {Math.Abs(approx - exact):E2} at [{string.Join(", ", pt)}]");
        }
    }

    [Fact]
    public void Test_5D_BS_Price()
    {
        var tt = TestFixtures.TtBs5D;
        var rng = new Random(99);
        double maxRel = 0.0;

        for (int i = 0; i < 30; i++)
        {
            double[] pt = new double[5];
            for (int d = 0; d < 5; d++)
            {
                double lo = TestFixtures.TT_5D_BS_DOMAIN[d][0];
                double hi = TestFixtures.TT_5D_BS_DOMAIN[d][1];
                pt[d] = lo + rng.NextDouble() * (hi - lo);
            }
            double exact = TestFixtures.Bs5DFunc(pt);
            double approx = tt.Eval(pt);
            if (Math.Abs(exact) > 0.1)
            {
                double rel = Math.Abs(approx - exact) / Math.Abs(exact);
                if (rel > maxRel) maxRel = rel;
            }
        }
        Assert.True(maxRel < 0.01, $"Max relative error {maxRel:E4} exceeds 1%");
    }
}

// ======================================================================
// Batch evaluation
// ======================================================================

public class TensorTrainBatchTests
{
    [Fact]
    public void Test_Batch_Matches_Loop()
    {
        var tt = TestFixtures.TtSin3D;
        var rng = new Random(77);
        int N = 50;
        double[,] pts = new double[N, 3];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 3; j++)
                pts[i, j] = -1.0 + 2.0 * rng.NextDouble();

        double[] batch = tt.EvalBatch(pts);
        for (int i = 0; i < N; i++)
        {
            double[] pt = { pts[i, 0], pts[i, 1], pts[i, 2] };
            double loop = tt.Eval(pt);
            Assert.True(Math.Abs(batch[i] - loop) < 1e-12,
                $"Batch vs loop mismatch at i={i}: {Math.Abs(batch[i] - loop):E2}");
        }
    }

    [Fact]
    public void Test_Batch_5D()
    {
        var tt = TestFixtures.TtBs5D;
        var rng = new Random(88);
        int N = 20;
        double[,] pts = new double[N, 5];
        for (int i = 0; i < N; i++)
            for (int d = 0; d < 5; d++)
            {
                double lo = TestFixtures.TT_5D_BS_DOMAIN[d][0];
                double hi = TestFixtures.TT_5D_BS_DOMAIN[d][1];
                pts[i, d] = lo + rng.NextDouble() * (hi - lo);
            }

        double[] batch = tt.EvalBatch(pts);
        for (int i = 0; i < N; i++)
        {
            double[] pt = new double[5];
            for (int d = 0; d < 5; d++) pt[d] = pts[i, d];
            double loop = tt.Eval(pt);
            Assert.True(Math.Abs(batch[i] - loop) < 1e-12,
                $"Batch vs loop mismatch at i={i}: {Math.Abs(batch[i] - loop):E2}");
        }
    }
}

// ======================================================================
// Derivative tests
// ======================================================================

public class TensorTrainDerivativeTests
{
    [Fact]
    public void Test_FD_Delta()
    {
        var tt = TestFixtures.TtBs5D;
        double[] pt = { 100.0, 100.0, 0.5, 0.25, 0.05 };
        var results = tt.EvalMulti(pt, new[] {
            new[] { 0, 0, 0, 0, 0 },
            new[] { 1, 0, 0, 0, 0 },
        });
        double analytical = BlackScholes.BsCallDelta(S: 100, K: 100, T: 0.5, r: 0.05, sigma: 0.25, q: 0.02);
        double relErr = Math.Abs(results[1] - analytical) / Math.Abs(analytical);
        Assert.True(relErr < 0.05, $"Delta rel error {relErr:E2}");
    }

    [Fact]
    public void Test_FD_Gamma()
    {
        var tt = TestFixtures.TtBs5D;
        double[] pt = { 100.0, 100.0, 0.5, 0.25, 0.05 };
        var results = tt.EvalMulti(pt, new[] {
            new[] { 2, 0, 0, 0, 0 },
        });
        double analytical = BlackScholes.BsCallGamma(S: 100, K: 100, T: 0.5, r: 0.05, sigma: 0.25, q: 0.02);
        double relErr = Math.Abs(results[0] - analytical) / Math.Abs(analytical);
        Assert.True(relErr < 0.10, $"Gamma rel error {relErr:E2}");
    }

    [Fact]
    public void Test_FD_3D_Sin_Deriv()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        var results = tt.EvalMulti(pt, new[] {
            new[] { 0, 0, 0 },
            new[] { 1, 0, 0 },
        });
        double analytical = Math.Cos(0.5);
        double relErr = Math.Abs(results[1] - analytical) / Math.Abs(analytical);
        Assert.True(relErr < 0.01, $"dsin/dx0 rel error {relErr:E2}");
    }
}

// ======================================================================
// Rank & structure
// ======================================================================

public class TensorTrainStructureTests
{
    [Fact]
    public void Test_Separable_Rank()
    {
        var tt = TestFixtures.TtSin3DSvd;
        int[] ranks = tt.TtRanks;
        Assert.Equal(1, ranks[0]);
        Assert.Equal(1, ranks[^1]);
        for (int i = 1; i < ranks.Length - 1; i++)
            Assert.True(ranks[i] <= 2, $"Interior rank {ranks[i]} > 2 for separable function");
    }

    [Fact]
    public void Test_Higher_Rank_Improves()
    {
        static double f(double[] x) => Math.Sin(x[0] * x[1]) + Math.Cos(x[2]);

        var errors = new Dictionary<int, double>();
        foreach (int mr in new[] { 3, 8 })
        {
            var tt = new ChebyshevTT(f, 3,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 11, 11, 11 },
                maxRank: mr);
            tt.Build(verbose: false, method: "svd");

            var rng = new Random(42);
            double maxErr = 0;
            for (int i = 0; i < 20; i++)
            {
                double[] pt = { -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble() };
                double exact = f(pt);
                double approx = tt.Eval(pt);
                double err = Math.Abs(approx - exact);
                if (err > maxErr) maxErr = err;
            }
            errors[mr] = maxErr;
        }
        Assert.True(errors[8] <= errors[3] + 1e-14,
            $"rank 8 error {errors[8]:E2} > rank 3 error {errors[3]:E2}");
    }

    [Fact]
    public void Test_Compression_Ratio_5D()
    {
        Assert.True(TestFixtures.TtBs5D.CompressionRatio > 1.0);
    }

    [Fact]
    public void Test_TT_Ranks_Property()
    {
        int[] ranks = TestFixtures.TtSin3D.TtRanks;
        Assert.Equal(1, ranks[0]);
        Assert.Equal(1, ranks[^1]);
        Assert.Equal(4, ranks.Length); // d + 1
    }
}

// ======================================================================
// Infrastructure
// ======================================================================

public class TensorTrainInfrastructureTests
{
    [Fact]
    public void Test_Not_Built_Raises()
    {
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]), 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 });

        Assert.Throws<InvalidOperationException>(() => tt.Eval(new[] { 0.0, 0.0, 0.0 }));
        Assert.Throws<InvalidOperationException>(() => tt.EvalBatch(new double[1, 3]));
        Assert.Throws<InvalidOperationException>(() => tt.EvalMulti(new[] { 0.0, 0.0, 0.0 }, new[] { new[] { 0, 0, 0 } }));
        Assert.Throws<InvalidOperationException>(() => _ = tt.TtRanks);
        Assert.Throws<InvalidOperationException>(() => tt.ErrorEstimate());
        Assert.Throws<InvalidOperationException>(() => tt.Save("/tmp/test.json"));
    }

    [Fact]
    public void Test_Invalid_Method()
    {
        var tt = new ChebyshevTT(
            x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]), 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 });
        var ex = Assert.Throws<ArgumentException>(() => tt.Build(method: "bad"));
        Assert.Contains("method", ex.Message);
    }

    [Fact]
    public void Test_Domain_Validation()
    {
        Assert.Throws<ArgumentException>(() => new ChebyshevTT(
            x => x[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 }));
    }

    [Fact]
    public void Test_Nodes_Validation()
    {
        Assert.Throws<ArgumentException>(() => new ChebyshevTT(
            x => x[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 }));
    }

    [Fact]
    public void Test_Serialization_Roundtrip()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        double expected = tt.Eval(pt);

        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var loaded = ChebyshevTT.Load(path);
            Assert.True(Math.Abs(loaded.Eval(pt) - expected) < 1e-14);
            Assert.Equal(tt.TtRanks, loaded.TtRanks);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_Error_Estimate_Positive()
    {
        double err = TestFixtures.TtSin3D.ErrorEstimate();
        Assert.True(err >= 0.0);
    }

    [Fact]
    public void Test_ToString()
    {
        string s = TestFixtures.TtSin3D.ToString();
        Assert.Contains("ChebyshevTT", s);
        Assert.Contains("TT ranks", s);
        Assert.Contains("Compression", s);
    }

    [Fact]
    public void Test_Total_Build_Evals()
    {
        Assert.True(TestFixtures.TtSin3D.TotalBuildEvals > 0);
    }
}

// ======================================================================
// Cross vs SVD consistency
// ======================================================================

public class TensorTrainCrossVsSvdTests
{
    [Fact]
    public void Test_3D_Sin_Cross_Vs_SVD()
    {
        var ttCross = TestFixtures.TtSin3D;
        var ttSvd = TestFixtures.TtSin3DSvd;
        var rng = new Random(55);

        for (int i = 0; i < 20; i++)
        {
            double[] pt = { -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble(), -1 + 2 * rng.NextDouble() };
            double vCross = ttCross.Eval(pt);
            double vSvd = ttSvd.Eval(pt);
            Assert.True(Math.Abs(vCross - vSvd) < 1e-6,
                $"Cross-SVD diff {Math.Abs(vCross - vSvd):E2} at [{string.Join(", ", pt)}]");
        }
    }
}

// ======================================================================
// Additional coverage tests
// ======================================================================

public class TensorTrainCoverageTests
{
    [Fact]
    public void Test_Verbose_Cross_Build()
    {
        var original = Console.Out;
        try
        {
            var sw = new StringWriter();
            Console.SetOut(sw);

            var tt = new ChebyshevTT(
                x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]), 3,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 7, 7, 7 },
                maxRank: 3);
            tt.Build(verbose: true, method: "cross");

            string output = sw.ToString();
            Assert.Contains("Building", output);
            Assert.Contains("TT-Cross", output);
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void Test_Verbose_SVD_Build()
    {
        var original = Console.Out;
        try
        {
            var sw = new StringWriter();
            Console.SetOut(sw);

            var tt = new ChebyshevTT(
                x => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]), 3,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5, 5, 5 },
                maxRank: 3);
            tt.Build(verbose: true, method: "svd");

            string output = sw.ToString();
            Assert.Contains("Building", output);
            Assert.True(output.Contains("TT-SVD") || output.Contains("full tensor"));
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void Test_Cross_Derivative_Mixed_Partial()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        var results = tt.EvalMulti(pt, new[] { new[] { 1, 1, 0 } });
        // For f = sin(x0) + sin(x1) + sin(x2), d^2f/dx0dx1 = 0
        Assert.True(Math.Abs(results[0]) < 0.01, $"Mixed partial = {results[0]:E4}");
    }

    [Fact]
    public void Test_FD_Derivative_Near_Boundary()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { -0.999, 0.3, 0.1 };
        var results = tt.EvalMulti(pt, new[] { new[] { 1, 0, 0 } });
        double analytical = Math.Cos(-0.999);
        double relErr = Math.Abs(results[0] - analytical) / Math.Abs(analytical);
        Assert.True(relErr < 0.05, $"Boundary FD rel error {relErr:E2}");
    }

    [Fact]
    public void Test_FD_Derivative_Near_Right_Boundary()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.3, 0.1, 0.999 };
        var results = tt.EvalMulti(pt, new[] { new[] { 0, 0, 1 } });
        double analytical = Math.Cos(0.999);
        double relErr = Math.Abs(results[0] - analytical) / Math.Abs(analytical);
        Assert.True(relErr < 0.05, $"Boundary FD rel error {relErr:E2}");
    }

    [Fact]
    public void Test_Derivative_Order_3_Raises()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        var ex = Assert.Throws<ArgumentException>(() =>
            tt.EvalMulti(pt, new[] { new[] { 3, 0, 0 } }));
        Assert.Contains("not supported", ex.Message);
    }

    [Fact]
    public void Test_Load_Wrong_Type_Raises()
    {
        string path = Path.GetTempFileName();
        try
        {
            // Write a non-TT JSON file (e.g. a ChebyshevApproximation save)
            File.WriteAllText(path, "{\"NotAChebyshevTT\": true}");
            Assert.Throws<InvalidOperationException>(() => ChebyshevTT.LoadStrict(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_Version_Mismatch_Warning()
    {
        var tt = TestFixtures.TtSin3D;
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            // Manipulate the JSON to change the version
            string json = File.ReadAllText(path);
            json = json.Replace(
                "\"Version\":",
                "\"Version\":\"0.0.0-fake\",\"OrigVersion\":");
            // Actually, let's just replace the version value
            // The saved version will be the assembly version; replace it with "0.0.0-fake"
            var doc = System.Text.Json.JsonDocument.Parse(json);
            using var ms = new MemoryStream();
            using var writer = new System.Text.Json.Utf8JsonWriter(ms);
            writer.WriteStartObject();
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "Version")
                    writer.WriteString("Version", "0.0.0-fake");
                else
                    prop.WriteTo(writer);
            }
            writer.WriteEndObject();
            writer.Flush();
            string modifiedJson = System.Text.Encoding.UTF8.GetString(ms.ToArray());
            File.WriteAllText(path, modifiedJson);

            var loaded = ChebyshevTT.Load(path);
            Assert.NotNull(loaded.LoadWarning);
            Assert.Contains("0.0.0-fake", loaded.LoadWarning);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Test_Str_Unbuilt()
    {
        var tt = new ChebyshevTT(
            x => x[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 },
            maxRank: 3);
        string s = tt.ToString();
        Assert.Contains("not built", s);
        Assert.Contains("Domain", s);
    }

    [Fact]
    public void Test_High_Dim_Str_Truncation()
    {
        var tt = new ChebyshevTT(
            x => x.Sum(), 7,
            Enumerable.Range(0, 7).Select(_ => new[] { -1.0, 1.0 }).ToArray(),
            Enumerable.Repeat(3, 7).ToArray(),
            maxRank: 2);
        string s = tt.ToString();
        Assert.Contains("...]", s);
        Assert.Contains("...", s);
    }

    [Fact]
    public void Test_Higher_Order_Cross_Derivative()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        // d^3f/dx0^2 dx1 for f = sin(x0)+sin(x1)+sin(x2) = 0
        var results = tt.EvalMulti(pt, new[] { new[] { 2, 1, 0 } });
        Assert.True(Math.Abs(results[0]) < 0.1, $"Higher-order cross deriv = {results[0]:E4}");
    }

    [Fact]
    public void Test_Triple_Cross_Derivative()
    {
        var tt = TestFixtures.TtSin3D;
        double[] pt = { 0.5, 0.3, 0.1 };
        // d^3f/dx0 dx1 dx2 for separable f = 0
        var results = tt.EvalMulti(pt, new[] { new[] { 1, 1, 1 } });
        Assert.True(Math.Abs(results[0]) < 0.1, $"Triple cross deriv = {results[0]:E4}");
    }
}
