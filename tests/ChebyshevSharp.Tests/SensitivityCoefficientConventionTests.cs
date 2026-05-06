using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

public class TestSensitivityCoefficientConventions
{
    [Fact]
    public void ChebyshevCoefficientsND_MatchesDirectAxisWiseDctIIConvention()
    {
        CheckCoefficientShape(new[] { 2, 3, 4 });
        CheckCoefficientShape(new[] { 1, 5 });
        CheckCoefficientShape(new[] { 3, 1, 2 });
        CheckCoefficientShape(new[] { 2, 33 });
    }

    [Fact]
    public void ComputeSobolFromCoeffs_UsesChebyshevOrthogonalityWeights()
    {
        int[] shape = { 3, 4 };
        var coeffs = new double[shape[0] * shape[1]];
        coeffs[FlatIndex(new[] { 0, 0 }, shape)] = 10.0;
        coeffs[FlatIndex(new[] { 1, 0 }, shape)] = 2.0;
        coeffs[FlatIndex(new[] { 0, 1 }, shape)] = -3.0;
        coeffs[FlatIndex(new[] { 1, 1 }, shape)] = 4.0;
        coeffs[FlatIndex(new[] { 2, 0 }, shape)] = 0.5;

        var result = Sensitivity.ComputeSobolFromCoeffs(coeffs, shape);

        double pi = Math.PI;
        double e10 = 2.0 * pi * pi;
        double e01 = 4.5 * pi * pi;
        double e11 = 4.0 * pi * pi;
        double e20 = 0.125 * pi * pi;
        double variance = e10 + e01 + e11 + e20;

        TestFixtures.AssertClose(variance, result.Variance, rtol: 1e-12, atol: 1e-12);
        TestFixtures.AssertClose((e10 + e20) / variance, result.FirstOrder[0], rtol: 1e-12, atol: 1e-12);
        TestFixtures.AssertClose(e01 / variance, result.FirstOrder[1], rtol: 1e-12, atol: 1e-12);
        TestFixtures.AssertClose((e10 + e11 + e20) / variance, result.TotalOrder[0], rtol: 1e-12, atol: 1e-12);
        TestFixtures.AssertClose((e01 + e11) / variance, result.TotalOrder[1], rtol: 1e-12, atol: 1e-12);
    }

    private static void CheckCoefficientShape(int[] shape)
    {
        int total = shape.Aggregate(1, (acc, n) => acc * n);
        var values = new double[total];
        for (int flat = 0; flat < total; flat++)
        {
            int[] idx = UnravelIndex(flat, shape);
            values[flat] = 0.25 + Math.Sin(0.37 * (flat + 1));
            for (int d = 0; d < idx.Length; d++)
                values[flat] += (d + 1) * 0.11 * (idx[d] + 1) * (idx[d] + 2);
        }

        double[] actual = Sensitivity.ChebyshevCoefficientsND(values, shape);
        double[] expected = DirectTypeICoefficients(values, shape);

        for (int i = 0; i < expected.Length; i++)
            TestFixtures.AssertClose(expected[i], actual[i], rtol: 1e-12, atol: 1e-12);
    }

    private static double[] DirectTypeICoefficients(double[] values, int[] shape)
    {
        int total = shape.Aggregate(1, (acc, n) => acc * n);
        var coeffs = new double[total];

        for (int alphaFlat = 0; alphaFlat < total; alphaFlat++)
        {
            int[] alpha = UnravelIndex(alphaFlat, shape);
            double sum = 0.0;

            for (int valueFlat = 0; valueFlat < total; valueFlat++)
            {
                int[] idx = UnravelIndex(valueFlat, shape);
                double factor = 1.0;
                for (int d = 0; d < shape.Length; d++)
                {
                    int n = shape[d];
                    int descendingIndex = n - 1 - idx[d];
                    double axisFactor = (2.0 / n) *
                        Math.Cos(Math.PI * alpha[d] * (2 * descendingIndex + 1) / (2.0 * n));
                    if (alpha[d] == 0)
                        axisFactor *= 0.5;
                    factor *= axisFactor;
                }

                sum += values[valueFlat] * factor;
            }

            coeffs[alphaFlat] = sum;
        }

        return coeffs;
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
