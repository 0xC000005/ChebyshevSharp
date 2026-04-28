namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train extrusion / slicing / materialization
/// (Extrude, Slice, ToDense, FromValuesTtSvd).
/// </summary>
internal static class TensorTrainExtrude
{
    /// <summary>
    /// TT-SVD decomposition of a precomputed dense tensor. Mirror of Python's
    /// <c>_tt_svd_from_tensor</c> (tensor_train.py:636).
    /// Returns value cores (function values at Chebyshev nodes along axis 1).
    /// </summary>
    /// <param name="tensor">Flat row-major dense tensor of shape <paramref name="nNodes"/>.</param>
    /// <param name="nNodes">Per-dimension grid sizes.</param>
    /// <param name="maxRank">Cap on TT rank.</param>
    /// <param name="tol">Singular value truncation tolerance relative to sigma_max.</param>
    internal static TensorTrainKernel.TtCore[] FromValuesTtSvd(
        double[] tensor, int[] nNodes, int maxRank, double tol)
    {
        int d = nNodes.Length;
        var cores = new TensorTrainKernel.TtCore[d];
        // Working tensor reshapes after each step. Track current "matrix" as flat row-major.
        double[] C = (double[])tensor.Clone();
        int rPrev = 1;

        for (int k = 0; k < d - 1; k++)
        {
            int rows = rPrev * nNodes[k];
            int cols = C.Length / rows;
            // Build MathNet matrix from C (flat row-major)
            var Cm = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(rows, cols, 0.0);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Cm.At(i, j, C[i * cols + j]);

            var svd = Cm.Svd(computeVectors: true);
            var U = svd.U;     // (rows, min)
            var S = svd.S;     // (min)
            var Vt = svd.VT;   // (min, cols)

            int sLen = S.Count;
            int rank = Math.Min(maxRank, sLen);
            double sMax = sLen > 0 ? S[0] : 0.0;
            if (sMax > 0)
            {
                int eff = 0;
                for (int i = 0; i < sLen; i++) if (S[i] > tol * sMax) eff++;
                rank = Math.Max(1, Math.Min(rank, eff));
            }
            else
            {
                rank = Math.Max(1, rank);
            }

            // Pack new core_k from U[:, :rank] reshaped (rPrev, n_k, rank).
            var core = new TensorTrainKernel.TtCore(rPrev, nNodes[k], rank);
            for (int i = 0; i < rPrev; i++)
                for (int p = 0; p < nNodes[k]; p++)
                    for (int r = 0; r < rank; r++)
                        core[i, p, r] = U[i * nNodes[k] + p, r];
            cores[k] = core;

            // C = diag(S[:rank]) @ Vt[:rank, :]
            var newC = new double[rank * cols];
            for (int r = 0; r < rank; r++)
                for (int j = 0; j < cols; j++)
                    newC[r * cols + j] = S[r] * Vt[r, j];
            C = newC;
            rPrev = rank;
        }

        // Last core: shape (rPrev, n_{d-1}, 1)
        var lastCore = new TensorTrainKernel.TtCore(rPrev, nNodes[d - 1], 1);
        for (int i = 0; i < rPrev; i++)
            for (int p = 0; p < nNodes[d - 1]; p++)
                lastCore[i, p, 0] = C[i * nNodes[d - 1] + p];
        cores[d - 1] = lastCore;
        return cores;
    }
}
