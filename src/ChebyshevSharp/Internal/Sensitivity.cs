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
    /// Compute first-order and total-order Sobol sensitivity indices directly from
    /// TT coefficient cores. O(d · n · r²) per dimension — no dense materialization.
    /// Mirrors PyChebyshev <c>_sensitivity.py:_compute_sobol_from_tt_cores</c>.
    /// </summary>
    /// <param name="cores">
    /// TT coefficient cores in storage order. Each core has shape
    /// (r_left, n_k, r_right). cores[0] has r_left=1; cores[d-1] has r_right=1.
    /// </param>
    /// <returns>
    /// <see cref="SobolResult"/> with arrays keyed by storage-frame indices (0..d-1).
    /// The caller is responsible for translating storage-frame → user-frame if
    /// a non-identity dim-order is in effect.
    /// </returns>
    internal static SobolResult ComputeSobolFromTtCores(TensorTrainKernel.TtCore[] cores)
    {
        int d = cores.Length;
        double pi = Math.PI;

        // Per-dim Chebyshev inner-product weights:
        // w[k][0] = π,  w[k][p ≥ 1] = π/2
        var wFull = new double[d][];
        for (int k = 0; k < d; k++)
        {
            int nk = cores[k].NNodes;
            wFull[k] = new double[nk];
            for (int p = 0; p < nk; p++) wFull[k][p] = pi / 2.0;
            wFull[k][0] = pi;
        }

        // ---- total_weighted_squared
        // Iterative left-to-right contraction:
        //   M_{k+1}[a,b] = Σ_{i,j,p} M_k[i,j] * (A_k[i,p,a] * w_k[p]) * A_k[j,p,b]
        // M_0 = [[1.0]] (shape 1×1 → shape r_{k} × r_{k} after step k)
        int r0 = cores[0].RLeft;  // Should be 1
        var M = new double[r0, r0];
        M[0, 0] = 1.0;

        for (int k = 0; k < d; k++)
        {
            var A = cores[k];
            int rL = A.RLeft, nk = A.NNodes, rR = A.RRight;
            var Mnew = new double[rR, rR];
            for (int a = 0; a < rR; a++)
            for (int b = 0; b < rR; b++)
            {
                double acc = 0.0;
                for (int i = 0; i < rL; i++)
                for (int j = 0; j < rL; j++)
                {
                    double mij = M[i, j];
                    if (mij == 0.0) continue;
                    for (int p = 0; p < nk; p++)
                        acc += mij * A[i, p, a] * wFull[k][p] * A[j, p, b];
                }
                Mnew[a, b] = acc;
            }
            M = Mnew;
        }
        double totalWeightedSquared = M[0, 0];

        // ---- constant term c_0 (all-zero multi-index)
        // v = product of cores[k][:, 0, :] chained left-to-right from [1.0]
        double[] v = new double[cores[0].RLeft];
        v[0] = 1.0;
        for (int k = 0; k < d; k++)
        {
            var A = cores[k];
            int rL = A.RLeft, rR = A.RRight;
            var vnew = new double[rR];
            for (int j = 0; j < rR; j++)
            {
                double acc = 0.0;
                for (int i = 0; i < rL; i++) acc += v[i] * A[i, 0, j];
                vnew[j] = acc;
            }
            v = vnew;
        }
        double c0 = v[0];  // scalar since rRight of last core == 1
        double constantWeightedSquared = c0 * c0 * Math.Pow(pi, d);

        double variance = totalWeightedSquared - constantWeightedSquared;

        if (variance <= 0.0 || variance < 1e-20)
        {
            // Constant or near-constant function.
            // Clamp variance to 0 for clean reporting (TT-Cross floating-point noise
            // can leave a tiny positive residual even for truly constant f).
            // Absolute threshold 1e-20 matches the dense path (ComputeSobolFromCoeffs line 348)
            // and is well above TT-Cross noise floor (~1e-29) but well below legitimate small
            // signals such as f(x,y,z) = 1 + 1e-6*x (variance ~1.55e-11).
            return new SobolResult(new double[d], new double[d], 0.0);
        }

        // ---- Precompute left partial matrices L[0..d] and right partial matrices R[0..d]
        // L[0] = [[1]], L[k+1] = einsum("ij,ipa,jpb->ab", L[k], Aw_k, A_k)
        // R[d] = [[1]], R[k]   = einsum("ab,ipa,jpb->ij", R[k+1], Aw_k, A_k)
        var L = new double[d + 1][,];
        L[0] = new double[1, 1];
        L[0][0, 0] = 1.0;
        for (int k = 0; k < d; k++)
        {
            var A = cores[k];
            int rL = A.RLeft, nk = A.NNodes, rR = A.RRight;
            var Lnew = new double[rR, rR];
            for (int a = 0; a < rR; a++)
            for (int b = 0; b < rR; b++)
            {
                double acc = 0.0;
                for (int i = 0; i < rL; i++)
                for (int j = 0; j < rL; j++)
                {
                    double lij = L[k][i, j];
                    if (lij == 0.0) continue;
                    for (int p = 0; p < nk; p++)
                        acc += lij * A[i, p, a] * wFull[k][p] * A[j, p, b];
                }
                Lnew[a, b] = acc;
            }
            L[k + 1] = Lnew;
        }

        var R = new double[d + 1][,];
        R[d] = new double[1, 1];
        R[d][0, 0] = 1.0;
        for (int k = d - 1; k >= 0; k--)
        {
            var A = cores[k];
            int rL = A.RLeft, nk = A.NNodes, rR = A.RRight;
            var Rnew = new double[rL, rL];
            for (int i = 0; i < rL; i++)
            for (int j = 0; j < rL; j++)
            {
                double acc = 0.0;
                for (int a = 0; a < rR; a++)
                for (int b = 0; b < rR; b++)
                {
                    double rab = R[k + 1][a, b];
                    if (rab == 0.0) continue;
                    for (int p = 0; p < nk; p++)
                        acc += rab * A[i, p, a] * wFull[k][p] * A[j, p, b];
                }
                Rnew[i, j] = acc;
            }
            R[k] = Rnew;
        }

        var firstOrder = new double[d];
        var totalOrder = new double[d];

        for (int j = 0; j < d; j++)
        {
            // ---- first-order energy[j]:
            // alpha_j >= 1 AND alpha_k = 0 for all k != j
            // left[r_j] = chain cores[k][:,0,:] for k < j (left @ core)
            var left = new double[cores[j].RLeft];
            left[0] = 1.0;
            for (int k = 0; k < j; k++)
            {
                var A = cores[k];
                int rL = A.RLeft, rR = A.RRight;
                var lnew = new double[rR];
                for (int b = 0; b < rR; b++)
                {
                    double acc = 0.0;
                    for (int i = 0; i < rL; i++) acc += left[i] * A[i, 0, b];
                    lnew[b] = acc;
                }
                left = lnew;
            }

            // right[r_{j+1}] = chain cores[k][:,0,:] for k > j (core @ right), right-to-left
            var right = new double[cores[j].RRight];
            right[0] = 1.0;
            for (int k = d - 1; k > j; k--)
            {
                var A = cores[k];
                int rL = A.RLeft, rR = A.RRight;
                var rnew = new double[rL];
                for (int i = 0; i < rL; i++)
                {
                    double acc = 0.0;
                    for (int b = 0; b < rR; b++) acc += A[i, 0, b] * right[b];
                    rnew[i] = acc;
                }
                right = rnew;
            }

            var Gj = cores[j];
            int rLj = Gj.RLeft, njNodes = Gj.NNodes, rRj = Gj.RRight;
            double sumSquared = 0.0;
            for (int m = 1; m < njNodes; m++)  // skip m=0 (constant slice)
            {
                // coef_m = left @ G_j[:, m, :] @ right  (scalar)
                double coefM = 0.0;
                for (int i = 0; i < rLj; i++)
                {
                    double lGi = 0.0;
                    for (int a = 0; a < rRj; a++) lGi += Gj[i, m, a] * right[a];
                    coefM += left[i] * lGi;
                }
                sumSquared += coefM * coefM;
            }
            // weight: (π/2) * π^(d-1)
            double weightFirst = (pi / 2.0) * Math.Pow(pi, d - 1);
            firstOrder[j] = sumSquared * weightFirst / variance;

            // ---- total-order energy[j]:
            // = total_weighted_squared - sum_{alpha_j = 0} weighted
            // sum_alpha_j_zero = π * einsum("ij,ia,jb,ab->", L[j], c_j0, c_j0, R[j+1])
            // where c_j0 = cores[j][:, 0, :]  (shape rLj × rRj)
            int rLjj = cores[j].RLeft, rRjj = cores[j].RRight;
            double sumAlphaJZero = 0.0;
            for (int i = 0; i < rLjj; i++)
            for (int jj = 0; jj < rLjj; jj++)
            {
                double lij = L[j][i, jj];
                if (lij == 0.0) continue;
                for (int a = 0; a < rRjj; a++)
                for (int b = 0; b < rRjj; b++)
                {
                    sumAlphaJZero += lij * cores[j][i, 0, a] * cores[j][jj, 0, b] * R[j + 1][a, b];
                }
            }
            sumAlphaJZero *= pi;
            totalOrder[j] = (totalWeightedSquared - sumAlphaJZero) / variance;
        }

        return new SobolResult(firstOrder, totalOrder, variance);
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
