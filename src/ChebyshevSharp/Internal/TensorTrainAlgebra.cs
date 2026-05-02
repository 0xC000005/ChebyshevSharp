namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train algebra (addition, scalar mul, negation,
/// TT-SVD rounding, inner product). Operates on flat <see cref="TensorTrainKernel.TtCore"/>
/// arrays. Members are added incrementally across Phase 2 Tasks 4 (InnerProduct),
/// 9 (scalar algebra), and 10 (binary algebra + rounding).
/// </summary>
internal static class TensorTrainAlgebra
{
    /// <summary>
    /// Frobenius inner product of two TTs' Chebyshev coefficient tensors.
    /// Computes Σ_{i_1,…,i_d} C_a[i] * C_b[i] in O(d * n * r_a^2 * r_b^2).
    /// Mirrors Python's <c>ChebyshevTT.inner_product</c> (tensor_train.py:1431).
    /// </summary>
    /// <param name="coresA">Coefficient cores of the first TT.</param>
    /// <param name="coresB">Coefficient cores of the second TT (matching shape per dim).</param>
    /// <returns>Frobenius inner product.</returns>
    internal static double InnerProductCores(
        TensorTrainKernel.TtCore[] coresA,
        TensorTrainKernel.TtCore[] coresB)
    {
        int d = coresA.Length;
        // M starts as 1x1 identity: shape (rA_0, rB_0) = (1, 1).
        int rAcur = 1, rBcur = 1;
        double[] M = { 1.0 };

        for (int k = 0; k < d; k++)
        {
            var A = coresA[k];   // (rA_left, n, rA_right)
            var B = coresB[k];   // (rB_left, n, rB_right)
            int n = A.NNodes;
            int rAr = A.RRight, rBr = B.RRight;

            // newM[a, b] = sum_{i, j, p} M[i, j] * A[i, p, a] * B[j, p, b]
            int newMLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { rAr, rBr }, nameof(InnerProductCores)),
                nameof(InnerProductCores),
                new[] { rAr, rBr });
            var newM = new double[newMLength];
            for (int a = 0; a < rAr; a++)
                for (int b = 0; b < rBr; b++)
                {
                    double s = 0;
                    for (int i = 0; i < rAcur; i++)
                        for (int j = 0; j < rBcur; j++)
                        {
                            double mij = M[i * rBcur + j];
                            if (mij == 0) continue;
                            for (int p = 0; p < n; p++)
                                s += mij * A[i, p, a] * B[j, p, b];
                        }
                    newM[a * rBr + b] = s;
                }

            M = newM;
            rAcur = rAr;
            rBcur = rBr;
        }

        // M is (1, 1) at the end since rA_d = rB_d = 1.
        return M[0];
    }

    /// <summary>
    /// Return a deep-copy of <paramref name="cores"/> with core 0's data scaled by
    /// <paramref name="scalar"/>. The represented function is multiplied by <paramref name="scalar"/>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] ScalarMulCores(
        TensorTrainKernel.TtCore[] cores, double scalar)
    {
        int d = cores.Length;
        var result = new TensorTrainKernel.TtCore[d];
        // Deep-copy all cores; scale core 0.
        for (int k = 0; k < d; k++) result[k] = cores[k].Copy();
        for (int i = 0; i < result[0].Data.Length; i++)
            result[0].Data[i] *= scalar;
        return result;
    }

    /// <summary>In-place variant of <see cref="ScalarMulCores"/>.</summary>
    internal static void ScalarMulCoresInPlace(
        TensorTrainKernel.TtCore[] cores, double scalar)
    {
        for (int i = 0; i < cores[0].Data.Length; i++)
            cores[0].Data[i] *= scalar;
    }

    /// <summary>Return a deep-copy of <paramref name="cores"/> negated.</summary>
    internal static TensorTrainKernel.TtCore[] NegateCores(TensorTrainKernel.TtCore[] cores)
        => ScalarMulCores(cores, -1.0);

    /// <summary>In-place negation.</summary>
    internal static void NegateCoresInPlace(TensorTrainKernel.TtCore[] cores)
        => ScalarMulCoresInPlace(cores, -1.0);

    /// <summary>
    /// Block-diagonal stacking of TT cores → exact TT representation of the sum.
    /// Mirror of Python's <c>_tt_add_cores</c> (_algebra.py:63).
    /// </summary>
    /// <remarks>
    /// The d==1 path requires identical (RLeft, NNodes, RRight) on both inputs
    /// because there is no rank to absorb a mismatch into — the lone core is the
    /// whole representation. The multi-dim path block-diagonal-stacks regardless
    /// of input ranks: per-core nNodes still must match (validated below), but
    /// rank differences are absorbed into the new combined ranks (rA + rB) at
    /// each interior bond. Output ranks may exceed maxRank; callers typically
    /// follow with <see cref="RoundCores"/>.
    /// </remarks>
    internal static TensorTrainKernel.TtCore[] AddCores(
        TensorTrainKernel.TtCore[] coresA,
        TensorTrainKernel.TtCore[] coresB)
    {
        int d = coresA.Length;
        if (d != coresB.Length)
            throw new ArgumentException("AddCores: cores must have same length");

        // d == 1 special case: elementwise coefficient sum (only correct rep keeping rank-1 endpoints).
        if (d == 1)
        {
            var a0 = coresA[0]; var b0 = coresB[0];
            if (a0.RLeft != b0.RLeft || a0.NNodes != b0.NNodes || a0.RRight != b0.RRight)
                throw new ArgumentException(
                    $"AddCores: 1D core shape mismatch ({a0.RLeft},{a0.NNodes},{a0.RRight}) vs ({b0.RLeft},{b0.NNodes},{b0.RRight})");
            var sum = new TensorTrainKernel.TtCore(a0.RLeft, a0.NNodes, a0.RRight);
            for (int i = 0; i < a0.Data.Length; i++)
                sum.Data[i] = a0.Data[i] + b0.Data[i];
            return new[] { sum };
        }

        var result = new TensorTrainKernel.TtCore[d];
        for (int k = 0; k < d; k++)
        {
            var a = coresA[k]; var b = coresB[k];
            int n = a.NNodes;
            if (b.NNodes != n)
                throw new ArgumentException($"AddCores: core {k} nNodes mismatch: {n} vs {b.NNodes}");

            if (k == 0)
            {
                // Concat along right rank: shape (1, n, ra_r + rb_r)
                int rR = a.RRight + b.RRight;
                var newCore = new TensorTrainKernel.TtCore(1, n, rR);
                for (int j = 0; j < n; j++)
                {
                    for (int kk = 0; kk < a.RRight; kk++) newCore[0, j, kk] = a[0, j, kk];
                    for (int kk = 0; kk < b.RRight; kk++) newCore[0, j, a.RRight + kk] = b[0, j, kk];
                }
                result[k] = newCore;
            }
            else if (k == d - 1)
            {
                // Concat along left rank: shape (ra_l + rb_l, n, 1)
                int rL = a.RLeft + b.RLeft;
                var newCore = new TensorTrainKernel.TtCore(rL, n, 1);
                for (int j = 0; j < n; j++)
                {
                    for (int ii = 0; ii < a.RLeft; ii++) newCore[ii, j, 0] = a[ii, j, 0];
                    for (int ii = 0; ii < b.RLeft; ii++) newCore[a.RLeft + ii, j, 0] = b[ii, j, 0];
                }
                result[k] = newCore;
            }
            else
            {
                // Block diagonal: shape (ra_l + rb_l, n, ra_r + rb_r)
                int rL = a.RLeft + b.RLeft;
                int rR = a.RRight + b.RRight;
                var newCore = new TensorTrainKernel.TtCore(rL, n, rR);
                for (int j = 0; j < n; j++)
                {
                    for (int ii = 0; ii < a.RLeft; ii++)
                        for (int kk = 0; kk < a.RRight; kk++)
                            newCore[ii, j, kk] = a[ii, j, kk];
                    for (int ii = 0; ii < b.RLeft; ii++)
                        for (int kk = 0; kk < b.RRight; kk++)
                            newCore[a.RLeft + ii, j, a.RRight + kk] = b[ii, j, kk];
                }
                result[k] = newCore;
            }
        }
        return result;
    }

    /// <summary>
    /// Round TT to lower rank via TT-SVD recompression. Right-to-left QR sweep
    /// (right-canonicalize cores d-1..1) followed by left-to-right SVD truncation
    /// (cores 0..d-2). Truncation keeps min(maxRank, num_above_relative_tol)
    /// singular values. Mirror of Python's <c>_tt_round_cores</c> (_algebra.py:118).
    /// </summary>
    internal static TensorTrainKernel.TtCore[] RoundCores(
        TensorTrainKernel.TtCore[] cores, int maxRank, double tolerance = 1e-12)
    {
        int d = cores.Length;
        var result = new TensorTrainKernel.TtCore[d];
        for (int k = 0; k < d; k++) result[k] = cores[k].Copy();
        if (d == 1) return result;

        // Right-to-left QR sweep: right-canonicalize cores k = d-1, ..., 1.
        for (int k = d - 1; k > 0; k--)
        {
            int rL = result[k].RLeft, n = result[k].NNodes, rR = result[k].RRight;
            // Reshape (rL, n*rR), QR of transpose
            int rows = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { n, rR }, nameof(RoundCores)),
                nameof(RoundCores),
                new[] { n, rR });
            var Mt = new double[rows, rL];
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        Mt[j * rR + p, i] = result[k][i, j, p];
            var Mtm = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(Mt);
            var qr = Mtm.QR(MathNet.Numerics.LinearAlgebra.Factorization.QRMethod.Thin);
            int newRL = qr.Q.ColumnCount;
            var newCk = new TensorTrainKernel.TtCore(newRL, n, rR);
            for (int a = 0; a < newRL; a++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        newCk[a, j, p] = qr.Q[j * rR + p, a];
            // Push R^T into the previous core's right rank.
            var prev = result[k - 1];
            int rLp = prev.RLeft, nP = prev.NNodes;
            var newPrev = new TensorTrainKernel.TtCore(rLp, nP, newRL);
            for (int i = 0; i < rLp; i++)
                for (int j = 0; j < nP; j++)
                    for (int r = 0; r < newRL; r++)
                    {
                        double s = 0;
                        for (int sIdx = 0; sIdx < rL; sIdx++)
                            s += prev[i, j, sIdx] * qr.R[r, sIdx];
                        newPrev[i, j, r] = s;
                    }
            result[k] = newCk;
            result[k - 1] = newPrev;
        }

        // Left-to-right SVD truncation: cores k = 0, ..., d-2.
        for (int k = 0; k < d - 1; k++)
        {
            int rL = result[k].RLeft, n = result[k].NNodes, rR = result[k].RRight;
            var Mat = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(rL * n, rR, 0.0);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        Mat.At(i * n + j, p, result[k][i, j, p]);

            var svd = Mat.Svd(computeVectors: true);
            var U = svd.U; var S = svd.S; var Vt = svd.VT;
            int sLen = S.Count;
            int keep = Math.Min(maxRank, sLen);
            double sMax = sLen > 0 ? S[0] : 0.0;
            if (sMax > 0 && tolerance > 0)
            {
                int eff = 0;
                for (int i = 0; i < sLen; i++) if (S[i] > sMax * tolerance) eff++;
                keep = Math.Max(1, Math.Min(keep, eff));
            }
            else
            {
                keep = Math.Max(1, keep);
            }

            var newCk = new TensorTrainKernel.TtCore(rL, n, keep);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int r = 0; r < keep; r++)
                        newCk[i, j, r] = U[i * n + j, r];
            result[k] = newCk;

            // Push S @ Vt into next core's left rank.
            var nextC = result[k + 1];
            int n2 = nextC.NNodes, rR2 = nextC.RRight;
            var newNext = new TensorTrainKernel.TtCore(keep, n2, rR2);
            for (int r = 0; r < keep; r++)
                for (int j = 0; j < n2; j++)
                    for (int p = 0; p < rR2; p++)
                    {
                        double sAcc = 0;
                        for (int sIdx = 0; sIdx < rR; sIdx++)
                            sAcc += S[r] * Vt[r, sIdx] * nextC[sIdx, j, p];
                        newNext[r, j, p] = sAcc;
                    }
            result[k + 1] = newNext;
        }
        return result;
    }

    /// <summary>
    /// Swap adjacent storage axes <paramref name="i"/> and <c>i+1</c>
    /// of a TT in coefficient space via SVD truncation. Mirrors PyChebyshev
    /// <c>_algebra.py:177</c>.
    /// </summary>
    /// <param name="cores">Coefficient cores. Not mutated; returns a fresh list.</param>
    /// <param name="i">Position of the leftmost core in the swap pair (0 ≤ i &lt; cores.Length - 1).</param>
    /// <param name="maxRank">Maximum rank for the SVD truncation between the swapped cores.</param>
    /// <param name="tolerance">Relative singular-value cutoff (s_max × tolerance). Default 1e-12.</param>
    /// <returns>New cores list with axes <c>i</c> and <c>i+1</c> swapped.</returns>
    internal static TensorTrainKernel.TtCore[] TtSwapAdjacent(
        TensorTrainKernel.TtCore[] cores, int i, int maxRank, double tolerance = 1e-12)
    {
        if (i < 0 || i >= cores.Length - 1)
            throw new ArgumentOutOfRangeException(nameof(i),
                $"i={i} out of range [0, {cores.Length - 1})");

        var newCores = new TensorTrainKernel.TtCore[cores.Length];
        for (int k = 0; k < cores.Length; k++) newCores[k] = cores[k].Copy();

        var A = newCores[i];        // (rL, nA, rM)
        var B = newCores[i + 1];    // (rM, nB, rR)
        int rL = A.RLeft, nA = A.NNodes, rM = A.RRight;
        int rM2 = B.RLeft, nB = B.NNodes, rR = B.RRight;
        if (rM != rM2)
            throw new ArgumentException($"core shape mismatch at {i}: A.RRight={rM}, B.RLeft={rM2}");

        // Form joint M[rL, nA, nB, rR] = Σ_rM A · B
        // M is stored row-major: M[l, a, b, r] at index ((l * nA + a) * nB + b) * rR + r
        int jointLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { rL, nA, nB, rR }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { rL, nA, nB, rR });
        var M = new double[jointLength];
        for (int l = 0; l < rL; l++)
            for (int a = 0; a < nA; a++)
                for (int b = 0; b < nB; b++)
                    for (int r = 0; r < rR; r++)
                    {
                        double acc = 0;
                        for (int m = 0; m < rM; m++)
                            acc += A[l, a, m] * B[m, b, r];
                        M[((l * nA + a) * nB + b) * rR + r] = acc;
                    }

        // Transpose middle axes: Mt[l, b, a, r] = M[l, a, b, r]
        int transposedLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { rL, nB, nA, rR }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { rL, nB, nA, rR });
        var Mt = new double[transposedLength];
        for (int l = 0; l < rL; l++)
            for (int a = 0; a < nA; a++)
                for (int b = 0; b < nB; b++)
                    for (int r = 0; r < rR; r++)
                        Mt[((l * nB + b) * nA + a) * rR + r] =
                            M[((l * nA + a) * nB + b) * rR + r];

        // Reshape to matrix: rows = (rL × nB), cols = (nA × rR)
        int rows = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { rL, nB }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { rL, nB });
        int cols = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { nA, rR }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { nA, rR });
        var matData = new double[rows, cols];
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
                matData[row, col] = Mt[row * cols + col];

        var matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matData);
        var svd = matrix.Svd(true);
        var U = svd.U;
        var S = svd.S;
        var Vh = svd.VT;

        double sMax = S.Count > 0 ? S[0] : 0.0;
        int keep = Math.Min(maxRank, S.Count);
        if (sMax > 0 && tolerance > 0)
        {
            double cutoff = sMax * tolerance;
            int keepByTol = 0;
            for (int k = 0; k < S.Count; k++) if (S[k] > cutoff) keepByTol++;
            keep = Math.Max(1, Math.Min(keep, keepByTol));
        }
        else
        {
            keep = Math.Max(1, keep);
        }

        // Repack: A' = U * S, shape (rL, nB, keep); B' = Vh, shape (keep, nA, rR).
        int aNewLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { rL, nB, keep }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { rL, nB, keep });
        var aNewData = new double[aNewLength];
        for (int row = 0; row < rows; row++)
            for (int k = 0; k < keep; k++)
                aNewData[row * keep + k] = U[row, k] * S[k];

        int bNewLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { keep, nA, rR }, nameof(TtSwapAdjacent)),
            nameof(TtSwapAdjacent),
            new[] { keep, nA, rR });
        var bNewData = new double[bNewLength];
        for (int k = 0; k < keep; k++)
            for (int col = 0; col < cols; col++)
                bNewData[k * cols + col] = Vh[k, col];

        newCores[i] = new TensorTrainKernel.TtCore(rL, nB, keep, aNewData);
        newCores[i + 1] = new TensorTrainKernel.TtCore(keep, nA, rR, bNewData);
        return newCores;
    }
}
