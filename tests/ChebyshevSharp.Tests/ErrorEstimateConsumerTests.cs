using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

public class TestErrorEstimateConsumers
{
    [Fact]
    public void ErrorEstimatePerDim_MatchesDirectLastCoefficientMaxima()
    {
        int[] shape = { 4, 5, 3 };
        double[][] domain =
        {
            new[] { -1.0, 1.0 },
            new[] { -2.0, 3.0 },
            new[] { 0.25, 1.25 }
        };

        var values = new double[shape.Aggregate(1, (acc, n) => acc * n)];
        for (int flat = 0; flat < values.Length; flat++)
        {
            int[] idx = UnravelIndex(flat, shape);
            values[flat] = Math.Sin(0.19 * (flat + 1))
                + 0.07 * idx[0] * idx[1]
                - 0.13 * idx[2] * idx[2]
                + 0.03 * idx[0] * idx[2];
        }

        var approx = ChebyshevApproximation.FromValues(values, 3, domain, shape);
        double[] actual = approx.ErrorEstimatePerDim();
        double[] expected = DirectErrorEstimatePerDim(values, shape);

        for (int d = 0; d < shape.Length; d++)
            TestFixtures.AssertClose(expected[d], actual[d], rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void AutoN_RefinesDimensionWithLargestEstimatedTail()
    {
        static double F(double[] x, object? _) => 0.0001 * x[0] + Math.Sin(7.0 * x[1]);

        var approx = new ChebyshevApproximation(
            F,
            2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null,
            errorThreshold: 1e-6,
            maxN: 48);

        approx.Build(verbose: false);

        Assert.Equal(3, approx.NNodes[0]);
        Assert.True(approx.NNodes[1] > approx.NNodes[0],
            $"Expected auto-N to refine the high-frequency second dimension, got [{string.Join(", ", approx.NNodes)}]");
        Assert.True(approx.ErrorEstimate() <= 1e-6 || approx.NNodes[1] == 48);
        Assert.True(approx.NEvaluations > approx.NNodes.Aggregate(1, (acc, n) => acc * n),
            "Auto-N evaluation count should include validation and earlier grid work, not only the final tensor.");

        double maxActualError = 0.0;
        for (int i = 0; i <= 20; i++)
        {
            double x = -1.0 + 2.0 * i / 20.0;
            for (int j = 0; j <= 20; j++)
            {
                double y = -1.0 + 2.0 * j / 20.0;
                double[] point = { x, y };
                double diff = Math.Abs(F(point, null) - approx.VectorizedEval(point, new[] { 0, 0 }));
                maxActualError = Math.Max(maxActualError, diff);
            }
        }

        Assert.True(maxActualError < 1e-4,
            $"Auto-N accepted grid [{string.Join(", ", approx.NNodes)}] with empirical error {maxActualError:E2}");
    }

    private static double[] DirectErrorEstimatePerDim(double[] values, int[] shape)
    {
        var result = new double[shape.Length];
        for (int dim = 0; dim < shape.Length; dim++)
        {
            int[] otherShape = shape.Where((_, d) => d != dim).ToArray();
            int otherTotal = otherShape.Length == 0 ? 1 : otherShape.Aggregate(1, (acc, n) => acc * n);
            double max = 0.0;

            for (int otherFlat = 0; otherFlat < otherTotal; otherFlat++)
            {
                int[] otherIdx = UnravelIndex(otherFlat, otherShape);
                double coeff = DirectLastCoefficientForSlice(values, shape, dim, otherIdx);
                max = Math.Max(max, Math.Abs(coeff));
            }

            result[dim] = max;
        }

        return result;
    }

    private static double DirectLastCoefficientForSlice(double[] values, int[] shape, int dim, int[] otherIdx)
    {
        int n = shape[dim];
        int degree = n - 1;
        double sum = 0.0;

        for (int k = 0; k < n; k++)
        {
            var idx = new int[shape.Length];
            int otherDim = 0;
            for (int d = 0; d < shape.Length; d++)
            {
                if (d == dim)
                    idx[d] = k;
                else
                    idx[d] = otherIdx[otherDim++];
            }

            int descendingIndex = n - 1 - k;
            double factor = (2.0 / n) *
                Math.Cos(Math.PI * degree * (2 * descendingIndex + 1) / (2.0 * n));
            if (degree == 0)
                factor *= 0.5;

            sum += values[FlatIndex(idx, shape)] * factor;
        }

        return sum;
    }

    private static int[] UnravelIndex(int flat, int[] shape)
    {
        var idx = new int[shape.Length];
        int rem = flat;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        return idx;
    }

    private static int FlatIndex(int[] idx, int[] shape)
    {
        int flat = 0;
        for (int d = 0; d < shape.Length; d++)
            flat = flat * shape[d] + idx[d];
        return flat;
    }
}
