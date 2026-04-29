namespace ChebyshevSharp.Internal;

/// <summary>
/// Sobol sensitivity indices computed from Chebyshev spectral coefficients.
/// Mirrors PyChebyshev <c>_sensitivity.py</c> (v0.20.0).
/// </summary>
internal static class Sensitivity
{
    /// <summary>Chebyshev T_n inner product norm² under weight 1/√(1-x²) on [-1,1].</summary>
    private static double ChebyshevNormSquared(int n) => n == 0 ? Math.PI : Math.PI / 2.0;

    /// <summary>Multi-D inner product norm² = ∏ per-dim norms.</summary>
    private static double MultiIndexNormSquared(int[] alpha)
    {
        double r = 1.0;
        for (int i = 0; i < alpha.Length; i++) r *= ChebyshevNormSquared(alpha[i]);
        return r;
    }

    /// <summary>Convert flat row-major index → multi-index (one int per dimension).</summary>
    private static int[] UnravelIndex(long flat, int[] shape)
    {
        int n = shape.Length;
        var idx = new int[n];
        long rem = flat;
        for (int d = n - 1; d >= 0; d--)
        {
            idx[d] = (int)(rem % shape[d]);
            rem /= shape[d];
        }
        return idx;
    }

    /// <summary>
    /// Apply <see cref="BarycentricKernel.ChebyshevCoefficients1D"/> along every axis,
    /// matching PyChebyshev's <c>_compute_chebyshev_coefficients</c> convention
    /// (DCT-II per axis with c_0 halving, applied dim-by-dim).
    /// </summary>
    /// <param name="tensorValues">Row-major tensor of values at Type-I Chebyshev nodes.</param>
    /// <param name="shape">Per-dim node counts.</param>
    /// <returns>Row-major tensor of Chebyshev coefficients, same shape.</returns>
    internal static double[] ChebyshevCoefficientsND(double[] tensorValues, int[] shape)
    {
        int nDim = shape.Length;
        var coeffs = (double[])tensorValues.Clone();

        // Apply 1D DCT-II axis-by-axis.
        // For each axis d, compute the leading "outer" size (product of dims to the
        // left of d) and the trailing "inner" size (product of dims to the right).
        // Iterate over (outer, inner) coordinates, extract a 1D slice along d,
        // apply BarycentricKernel.ChebyshevCoefficients1D, write back.
        for (int d = 0; d < nDim; d++)
        {
            int n = shape[d];
            int outer = 1;
            for (int k = 0; k < d; k++) outer *= shape[k];
            int inner = 1;
            for (int k = d + 1; k < nDim; k++) inner *= shape[k];

            var slice = new double[n];
            for (int o = 0; o < outer; o++)
            {
                for (int j = 0; j < inner; j++)
                {
                    // Extract slice: coeffs[o, :, j] in (outer, n, inner) layout.
                    for (int i = 0; i < n; i++)
                        slice[i] = coeffs[(o * n + i) * inner + j];

                    var c = BarycentricKernel.ChebyshevCoefficients1D(slice);
                    // c[0] is already halved by ChebyshevCoefficients1D (matches PyChebyshev convention).

                    for (int i = 0; i < n; i++)
                        coeffs[(o * n + i) * inner + j] = c[i];
                }
            }
        }

        return coeffs;
    }

    /// <summary>
    /// Compute first- and total-order Sobol sensitivity indices from a Chebyshev
    /// coefficient tensor. Throws on NaN/Inf in input. Returns zero-filled
    /// <see cref="SobolResult"/> for constant functions (Variance == 0).
    /// </summary>
    internal static SobolResult ComputeSobolFromCoeffs(double[] coeffs, int[] shape)
    {
        for (int i = 0; i < coeffs.Length; i++)
            if (!double.IsFinite(coeffs[i]))
                throw new ArgumentException(
                    "coefficients contain NaN or Inf; SobolIndices() requires finite spectral coefficients");

        int nDim = shape.Length;
        var firstOrder = new double[nDim];
        var totalOrder = new double[nDim];
        double variance = 0;

        for (long flat = 0; flat < coeffs.Length; flat++)
        {
            var alpha = UnravelIndex(flat, shape);
            int nonzeroCount = 0;
            int firstNonzeroDim = -1;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) { nonzeroCount++; if (firstNonzeroDim == -1) firstNonzeroDim = d; }
            if (nonzeroCount == 0) continue;  // skip α = 0 (mean term).

            double c = coeffs[flat];
            if (c == 0) continue;
            double energy = c * c * MultiIndexNormSquared(alpha);
            variance += energy;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) totalOrder[d] += energy;
            if (nonzeroCount == 1) firstOrder[firstNonzeroDim] += energy;
        }

        // Constant / near-constant function: variance is either exactly 0 (Python parity)
        // or floating-point DCT-II noise (~1e-29 in C#). Returning zero indices on the
        // noise-floor path keeps the contract sane. Variance is reported as-is so callers
        // can choose their own constancy threshold.
        if (variance == 0 || variance < 1e-20)
            return new SobolResult(new double[nDim], new double[nDim], variance);
        for (int d = 0; d < nDim; d++)
        {
            firstOrder[d] /= variance;
            totalOrder[d] /= variance;
        }
        return new SobolResult(firstOrder, totalOrder, variance);
    }
}
