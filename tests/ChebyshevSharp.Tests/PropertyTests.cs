using FsCheck.Xunit;

namespace ChebyshevSharp.Tests;

[Properties(MaxTest = 8, Replay = "12345,67891", QuietOnSuccess = true)]
public class PropertyTests
{
    private static readonly double[][] SquareDomain =
    [
        [-1.0, 1.0],
        [-0.5, 1.5]
    ];

    private static readonly double[][] CubeDomain =
    [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0]
    ];

    [Property]
    public void Spline_EvalBatch_Equals_Looping_Eval(int seed)
    {
        var spline = new ChebyshevSpline(
            (p, _) => Math.Sin(p[0]) + p[1] * p[1],
            numDimensions: 2,
            domain: SquareDomain,
            nNodes: [6, 6],
            knots: [[0.0], [0.5]]);
        spline.Build(verbose: false);

        double[][] points = SamplePoints(seed, count: 5, SquareDomain);
        double[] batch = spline.EvalBatch(points, [0, 0]);

        for (int i = 0; i < points.Length; i++)
            AssertClose(spline.Eval(points[i], [0, 0]), batch[i], 1e-10);
    }

    [Property]
    public void Save_Then_Load_Preserves_Approximation_Evaluation(int seed)
    {
        var approx = BuildApproximation();
        double[][] points = SamplePoints(seed, count: 4, SquareDomain);
        string path = Path.Combine(Path.GetTempPath(), $"chebsharp-property-{Guid.NewGuid():N}.json");

        try
        {
            approx.Save(path);
            var loaded = ChebyshevApproximation.Load(path);

            foreach (double[] point in points)
                AssertClose(approx.Eval(point), loaded.Eval(point), 1e-12);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Property]
    public void Extrude_Then_Slice_Recovers_Original_Approximation(int seed)
    {
        var approx = BuildApproximation();
        var extruded = approx.Extrude((2, new[] { -2.0, 2.0 }, 5));
        double sliceValue = Sample(seed, -2.0, 2.0);
        var sliced = extruded.Slice((2, sliceValue));

        foreach (double[] point in SamplePoints(seed + 17, count: 4, SquareDomain))
            AssertClose(approx.Eval(point), sliced.Eval(point), 1e-10);
    }

    [Property]
    public void Algebra_Identities_Hold_For_Compatible_Approximations(int seed)
    {
        var approx = BuildApproximation();
        var zero = ChebyshevApproximation.FromValues(
            new double[6 * 6],
            numDimensions: 2,
            domain: SquareDomain,
            nNodes: [6, 6]);

        var plusZero = approx + zero;
        var minusSelf = approx - approx;
        var timesOne = approx * 1.0;
        var dividedByOne = approx / 1.0;

        foreach (double[] point in SamplePoints(seed, count: 4, SquareDomain))
        {
            double expected = approx.Eval(point);
            AssertClose(expected, plusZero.Eval(point), 1e-10);
            AssertClose(0.0, minusSelf.Eval(point), 1e-10);
            AssertClose(expected, timesOne.Eval(point), 1e-10);
            AssertClose(expected, dividedByOne.Eval(point), 1e-10);
        }
    }

    [Property]
    public void Tt_Reorder_Preserves_User_Frame_Evaluation(int seed)
    {
        var tt = new ChebyshevTT(
            p => p[0] + 0.5 * p[1] * p[1] - p[2],
            numDimensions: 3,
            domain: CubeDomain,
            nNodes: [5, 5, 5],
            maxRank: 8,
            tolerance: 1e-12,
            maxSweeps: 5);
        tt.Build(verbose: false, seed: NormalizeSeed(seed), method: "svd");

        var reordered = tt.Reorder([2, 0, 1], maxRank: 12, tolerance: 1e-12);

        foreach (double[] point in SamplePoints(seed + 31, count: 4, CubeDomain))
            AssertClose(tt.Eval(point), reordered.Eval(point), 1e-8);
    }

    private static ChebyshevApproximation BuildApproximation()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => Math.Cos(p[0]) + 0.25 * p[0] * p[1] + p[1],
            numDimensions: 2,
            domain: SquareDomain,
            nNodes: [6, 6]);
        approx.Build(verbose: false);
        return approx;
    }

    private static double[][] SamplePoints(int seed, int count, double[][] domain)
    {
        var random = new Random(NormalizeSeed(seed));
        var points = new double[count][];
        for (int i = 0; i < count; i++)
        {
            points[i] = new double[domain.Length];
            for (int d = 0; d < domain.Length; d++)
                points[i][d] = Sample(random, domain[d][0], domain[d][1]);
        }
        return points;
    }

    private static double Sample(int seed, double lo, double hi)
        => Sample(new Random(NormalizeSeed(seed)), lo, hi);

    private static double Sample(Random random, double lo, double hi)
        => lo + (hi - lo) * random.NextDouble();

    private static int NormalizeSeed(int seed)
        => seed == int.MinValue ? 0 : Math.Abs(seed);

    private static void AssertClose(double expected, double actual, double tolerance)
    {
        Assert.True(
            Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:E16}, got {actual:E16}, tolerance {tolerance:E1}");
    }
}
