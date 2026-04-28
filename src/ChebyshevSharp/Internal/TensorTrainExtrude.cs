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

    /// <summary>
    /// Materialize the TT chain into a flat row-major dense tensor of length Π nNodes.
    /// Converts coefficient cores to value cores first, then chains contractions.
    /// Mirror of Python's <c>ChebyshevTT.to_dense</c> (tensor_train.py:1637).
    /// </summary>
    internal static double[] ToDenseEinsumChain(TensorTrainKernel.TtCore[] coeffCores, int[] nNodes)
    {
        // Convert all coefficient cores to value cores.
        var valueCores = new TensorTrainKernel.TtCore[coeffCores.Length];
        for (int i = 0; i < coeffCores.Length; i++)
            valueCores[i] = TensorTrainKernel.CoeffCoreToValueCore(coeffCores[i]);
        return TensorTrainKernel.ReconstructDense(valueCores, nNodes);
    }

    /// <summary>
    /// Insert a constant rank-preserving core at position <paramref name="dim"/>
    /// into a TT. The new core encodes the constant function 1 in DCT-II
    /// coefficient space (only c_0 = 1.0 is set; the core is rank-preserving:
    /// new_core[i, 0, i] = 1.0 for all i).
    /// Mirror of Python's <c>_extrude_tt_core</c>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] ExtrudeCores(
        TensorTrainKernel.TtCore[] coeffCores, int dim, int nNew)
    {
        int d = coeffCores.Length;
        if (dim < 0 || dim > d)
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"dim={dim} out of range [0, {d}]");
        if (nNew < 2)
            throw new ArgumentException($"newN must be >= 2, got {nNew}", nameof(nNew));

        // Determine rank at insertion boundary.
        int rAt;
        if (dim == 0) rAt = 1;
        else if (dim == d) rAt = 1;
        else rAt = coeffCores[dim - 1].RRight;

        var newCore = new TensorTrainKernel.TtCore(rAt, nNew, rAt);
        for (int i = 0; i < rAt; i++)
            newCore[i, 0, i] = 1.0;

        var result = new TensorTrainKernel.TtCore[d + 1];
        for (int k = 0; k < dim; k++) result[k] = coeffCores[k];
        result[dim] = newCore;
        for (int k = dim; k < d; k++) result[k + 1] = coeffCores[k];
        return result;
    }

    /// <summary>
    /// Contract a TT coefficient core along <paramref name="dim"/> at <paramref name="value"/>.
    /// Converts the target core to value space, evaluates the barycentric interpolant at
    /// <paramref name="value"/> to produce a matrix M of shape (rL, rR), then absorbs M
    /// into the right neighbor (or left neighbor for the rightmost core).
    /// Mirror of Python's <c>_slice_tt_core</c>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] SliceCores(
        TensorTrainKernel.TtCore[] coeffCores, int dim, double value, double[] nodes)
    {
        var coeffCore = coeffCores[dim];
        var valueCore = TensorTrainKernel.CoeffCoreToValueCore(coeffCore);
        int rL = valueCore.RLeft, n = valueCore.NNodes, rR = valueCore.RRight;

        // Find nearest node and check fast-path.
        int exactIdx = 0;
        double minAbs = double.PositiveInfinity;
        double[] diff = new double[n];
        for (int j = 0; j < n; j++)
        {
            diff[j] = value - nodes[j];
            double abs = Math.Abs(diff[j]);
            if (abs < minAbs) { minAbs = abs; exactIdx = j; }
        }

        double[] M = new double[rL * rR];
        if (minAbs < 1e-14)
        {
            // Fast path: value coincides with a node — just take a slice.
            for (int i = 0; i < rL; i++)
                for (int k = 0; k < rR; k++)
                    M[i * rR + k] = valueCore[i, exactIdx, k];
        }
        else
        {
            // Compute barycentric weights for nodes.
            double[] baryW = BarycentricKernel.ComputeBarycentricWeights(nodes);
            double[] wOverDiff = new double[n];
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                wOverDiff[j] = baryW[j] / diff[j];
                sum += wOverDiff[j];
            }
            for (int j = 0; j < n; j++) wOverDiff[j] /= sum;

            for (int i = 0; i < rL; i++)
                for (int k = 0; k < rR; k++)
                {
                    double s = 0;
                    for (int j = 0; j < n; j++) s += wOverDiff[j] * valueCore[i, j, k];
                    M[i * rR + k] = s;
                }
        }

        int d = coeffCores.Length;
        var result = new TensorTrainKernel.TtCore[d - 1];

        if (dim < d - 1)
        {
            // Absorb M into right neighbor: newNeighbor[l, j, s] = sum_r M[l, r] * neighbor[r, j, s]
            var neighbor = coeffCores[dim + 1];
            int n2 = neighbor.NNodes, rR2 = neighbor.RRight;
            var newNeighbor = new TensorTrainKernel.TtCore(rL, n2, rR2);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n2; j++)
                    for (int k = 0; k < rR2; k++)
                    {
                        double s = 0;
                        for (int r = 0; r < rR; r++)
                            s += M[i * rR + r] * neighbor[r, j, k];
                        newNeighbor[i, j, k] = s;
                    }
            for (int k = 0; k < dim; k++) result[k] = coeffCores[k];
            result[dim] = newNeighbor;
            for (int k = dim + 2; k < d; k++) result[k - 1] = coeffCores[k];
        }
        else
        {
            // Rightmost core: absorb M into left neighbor.
            var neighbor = coeffCores[dim - 1];
            int rLp = neighbor.RLeft, np = neighbor.NNodes;
            var newNeighbor = new TensorTrainKernel.TtCore(rLp, np, rR);
            for (int i = 0; i < rLp; i++)
                for (int j = 0; j < np; j++)
                    for (int k = 0; k < rR; k++)
                    {
                        double s = 0;
                        for (int r = 0; r < rL; r++)
                            s += neighbor[i, j, r] * M[r * rR + k];
                        newNeighbor[i, j, k] = s;
                    }
            for (int k = 0; k < dim - 1; k++) result[k] = coeffCores[k];
            result[dim - 1] = newNeighbor;
        }
        return result;
    }
}
