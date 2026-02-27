using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train decomposition algorithms.
/// Contains maxvol, TT-Cross, and TT-SVD implementations.
/// </summary>
internal static class TensorTrainKernel
{
    // ------------------------------------------------------------------
    // TtCore — lightweight value type for a single TT core
    // ------------------------------------------------------------------

    /// <summary>
    /// A single TT core of shape (rLeft, nNodes, rRight).
    /// Stored as flat row-major double[] for BLAS-friendly access.
    /// </summary>
    internal readonly struct TtCore
    {
        public readonly int RLeft;
        public readonly int NNodes;
        public readonly int RRight;
        public readonly double[] Data;

        public TtCore(int rLeft, int nNodes, int rRight)
        {
            RLeft = rLeft;
            NNodes = nNodes;
            RRight = rRight;
            Data = new double[rLeft * nNodes * rRight];
        }

        public TtCore(int rLeft, int nNodes, int rRight, double[] data)
        {
            RLeft = rLeft;
            NNodes = nNodes;
            RRight = rRight;
            Data = data;
        }

        /// <summary>Row-major indexer: data[i * nNodes * rRight + j * rRight + k].</summary>
        public double this[int i, int j, int k]
        {
            get => Data[i * NNodes * RRight + j * RRight + k];
            set => Data[i * NNodes * RRight + j * RRight + k] = value;
        }

        /// <summary>Total number of elements.</summary>
        public int Size => RLeft * NNodes * RRight;

        /// <summary>Create a deep copy.</summary>
        public TtCore Copy()
        {
            var copy = new double[Data.Length];
            Array.Copy(Data, copy, Data.Length);
            return new TtCore(RLeft, NNodes, RRight, copy);
        }
    }

    // ------------------------------------------------------------------
    // Maxvol — find rows with approximately maximal volume submatrix
    // ------------------------------------------------------------------

    /// <summary>
    /// Find r rows of a tall (m x r) matrix whose submatrix has approximately maximal |det|.
    /// Phase 1: column-pivoted QR on A^T for initialization.
    /// Phase 2: iterative row-swapping refinement.
    /// </summary>
    internal static int[] Maxvol(double[,] A, double tol = 1.05, int maxIters = 100)
    {
        int m = A.GetLength(0);
        int r = A.GetLength(1);

        if (m <= r)
        {
            var trivial = new int[m];
            for (int i = 0; i < m; i++) trivial[i] = i;
            return trivial;
        }

        // Phase 1: Column-pivoted QR on A^T to get initial r row indices.
        // A^T is (r, m). Column pivoting selects the most linearly independent columns of A^T = rows of A.
        int[] idx = ColumnPivotedQRIndices(A, r);

        // Phase 2: Iterative refinement.
        // B = A * inv(A[idx]) — coefficient matrix of shape (m, r).
        // Solve A[idx]^T * B^T = A^T  →  B^T = inv(A[idx]^T) * A^T  →  B = A * inv(A[idx])
        var aIdx = new double[r, r];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < r; j++)
                aIdx[i, j] = A[idx[i], j];

        var matAIdx = DenseMatrix.OfArray(aIdx);
        var matA = DenseMatrix.OfArray(A);

        Matrix<double> matB;
        try
        {
            // B = A * inv(A[idx])  ↔  B * A[idx] = A  ↔  A[idx]^T * B^T = A^T
            matB = matAIdx.Transpose().Solve(matA.Transpose()).Transpose();
        }
        catch
        {
            return idx;
        }

        double[,] B = matB.ToArray();

        for (int iter = 0; iter < maxIters; iter++)
        {
            // Find argmax|B[i,j]|
            int bestI = 0, bestJ = 0;
            double bestVal = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < r; j++)
                {
                    double v = Math.Abs(B[i, j]);
                    if (v > bestVal) { bestVal = v; bestI = i; bestJ = j; }
                }

            if (bestVal <= tol)
                break;

            // Swap: replace idx[bestJ] with row bestI
            idx[bestJ] = bestI;

            // Rank-1 update of B
            double bij = B[bestI, bestJ];
            double[] colJ = new double[m];
            double[] rowI = new double[r];
            for (int i = 0; i < m; i++) colJ[i] = B[i, bestJ];
            for (int j = 0; j < r; j++) rowI[j] = B[bestI, j];

            for (int i = 0; i < m; i++)
                for (int j = 0; j < r; j++)
                    B[i, j] -= colJ[i] * rowI[j] / bij;

            for (int i = 0; i < m; i++)
                B[i, bestJ] = colJ[i] / bij;
        }

        return idx;
    }

    /// <summary>
    /// Column-pivoted QR on A^T to select the r most linearly independent rows of A.
    /// Uses MathNet's GramSchmid QR with manual pivot tracking.
    /// </summary>
    private static int[] ColumnPivotedQRIndices(double[,] A, int r)
    {
        int m = A.GetLength(0);
        int cols = A.GetLength(1); // = r

        // We need column-pivoted QR of A^T (shape r x m).
        // Column pivoting on A^T selects the m columns (= rows of A) with largest norms.
        // We implement Householder QR with column pivoting manually.
        double[,] At = new double[cols, m]; // A^T
        for (int i = 0; i < m; i++)
            for (int j = 0; j < cols; j++)
                At[j, i] = A[i, j];

        int rows = cols; // rows of A^T = r
        int ncols = m;   // cols of A^T = m
        int minDim = Math.Min(rows, ncols);

        // Work on a copy
        double[,] R = (double[,])At.Clone();
        int[] piv = new int[ncols];
        for (int i = 0; i < ncols; i++) piv[i] = i;

        // Column norms squared
        double[] colNormSq = new double[ncols];
        for (int j = 0; j < ncols; j++)
        {
            double s = 0;
            for (int i = 0; i < rows; i++)
                s += R[i, j] * R[i, j];
            colNormSq[j] = s;
        }

        for (int k = 0; k < minDim; k++)
        {
            // Find pivot: column with max remaining norm
            int pivCol = k;
            double maxNorm = colNormSq[k];
            for (int j = k + 1; j < ncols; j++)
            {
                if (colNormSq[j] > maxNorm)
                {
                    maxNorm = colNormSq[j];
                    pivCol = j;
                }
            }

            // Swap columns k and pivCol
            if (pivCol != k)
            {
                (piv[k], piv[pivCol]) = (piv[pivCol], piv[k]);
                (colNormSq[k], colNormSq[pivCol]) = (colNormSq[pivCol], colNormSq[k]);
                for (int i = 0; i < rows; i++)
                    (R[i, k], R[i, pivCol]) = (R[i, pivCol], R[i, k]);
            }

            // Householder reflection for column k
            double norm = 0;
            for (int i = k; i < rows; i++)
                norm += R[i, k] * R[i, k];
            norm = Math.Sqrt(norm);

            if (norm < 1e-300) continue;

            double sign = R[k, k] >= 0 ? 1.0 : -1.0;
            double alpha = sign * norm;
            double[] v = new double[rows - k];
            v[0] = R[k, k] + alpha;
            for (int i = 1; i < v.Length; i++)
                v[i] = R[k + i, k];

            double vNorm = 0;
            for (int i = 0; i < v.Length; i++)
                vNorm += v[i] * v[i];

            if (vNorm < 1e-300) continue;
            double tau = 2.0 / vNorm;

            // Apply Householder to remaining columns
            for (int j = k; j < ncols; j++)
            {
                double dot = 0;
                for (int i = 0; i < v.Length; i++)
                    dot += v[i] * R[k + i, j];
                for (int i = 0; i < v.Length; i++)
                    R[k + i, j] -= tau * dot * v[i];
            }

            // Update column norms for remaining columns
            for (int j = k + 1; j < ncols; j++)
            {
                double s = 0;
                for (int i = k + 1; i < rows; i++)
                    s += R[i, j] * R[i, j];
                colNormSq[j] = s;
            }
        }

        // First r pivot indices = most linearly independent rows of A
        int[] result = new int[r];
        Array.Copy(piv, result, r);
        return result;
    }

    // ------------------------------------------------------------------
    // TT-Cross — alternating cross approximation with maxvol pivoting
    // ------------------------------------------------------------------

    /// <summary>
    /// Build TT value cores via alternating TT-Cross with maxvol pivoting.
    /// Returns value cores (not coefficient cores) and the number of unique function evaluations.
    /// </summary>
    internal static (TtCore[] Cores, int TotalEvals) TtCross(
        Func<double[], double> func,
        double[][] grids,
        int maxRank,
        double tol,
        int maxSweeps,
        bool verbose,
        int? seed)
    {
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        int d = grids.Length;
        int[] n = new int[d];
        for (int i = 0; i < d; i++) n[i] = grids[i].Length;

        // Eval cache: key = mixed-radix encoding of grid indices
        var cache = new Dictionary<long, double>();

        double EvalFunc(int[] gridIndices)
        {
            long key = 0;
            for (int dim = 0; dim < d; dim++)
                key = key * n[dim] + gridIndices[dim];

            if (!cache.TryGetValue(key, out double val))
            {
                double[] point = new double[d];
                for (int dim = 0; dim < d; dim++)
                    point[dim] = grids[dim][gridIndices[dim]];
                val = func(point);
                cache[key] = val;
            }
            return val;
        }

        double EvalTt(TtCore[] cores, int[] gridIndices)
        {
            // Chain of matrix multiplications: result = Π_d core[:, gridIndices[d], :]
            double[] v = new double[1]; v[0] = 1.0;
            for (int dim = 0; dim < d; dim++)
            {
                int rr = cores[dim].RRight;
                double[] vNew = new double[rr];
                int j = gridIndices[dim];
                for (int k = 0; k < rr; k++)
                {
                    double sum = 0;
                    for (int ii = 0; ii < v.Length; ii++)
                        sum += v[ii] * cores[dim][ii, j, k];
                    vNew[k] = sum;
                }
                v = vNew;
            }
            return v[0];
        }

        // Per-mode rank caps
        int[] rankCaps = new int[d + 1];
        rankCaps[0] = 1; rankCaps[d] = 1;
        for (int k = 1; k < d; k++)
        {
            long leftSize = 1;
            for (int i = 0; i < k; i++) leftSize *= n[i];
            long rightSize = 1;
            for (int i = k; i < d; i++) rightSize *= n[i];
            rankCaps[k] = (int)Math.Min(maxRank, Math.Min(leftSize, rightSize));
        }

        // Initialize ranks
        int[] r = new int[d + 1];
        r[0] = 1; r[d] = 1;
        for (int k = 1; k < d; k++)
            r[k] = Math.Min(rankCaps[k], Math.Min(n[k - 1], n[k]));

        // J_right[k]: right multi-indices for dimension k, shape (r[k+1], d-k-1)
        int[][,] J_right = new int[d][,];
        for (int k = 0; k < d - 1; k++)
        {
            int rk = r[k + 1];
            int nRight = d - k - 1;
            if (nRight == 0)
            {
                J_right[k] = new int[1, 0];
            }
            else
            {
                J_right[k] = new int[rk, nRight];
                for (int row = 0; row < rk; row++)
                    for (int col = 0; col < nRight; col++)
                        J_right[k][row, col] = rng.Next(n[k + 1 + col]);
            }
        }
        J_right[d - 1] = new int[1, 0];

        // J_left[k]: left multi-indices for dimension k, shape (r[k], k)
        int[][,] J_left = new int[d][,];
        J_left[0] = new int[1, 0];

        // Best-cores tracking
        double bestError = double.PositiveInfinity;
        TtCore[]? bestCores = null;
        int staleChecks = 0;
        int nTest = Math.Min(20, Math.Max(5, d));

        double CheckError(TtCore[] coresList)
        {
            int[][] pts = new int[nTest][];
            for (int t = 0; t < nTest; t++)
            {
                pts[t] = new int[d];
                for (int dim = 0; dim < d; dim++)
                    pts[t][dim] = rng.Next(n[dim]);
            }

            double[] ttV = new double[nTest];
            double[] exV = new double[nTest];
            for (int t = 0; t < nTest; t++)
            {
                ttV[t] = EvalTt(coresList, pts[t]);
                exV[t] = EvalFunc(pts[t]);
            }

            double refNorm = 0;
            for (int t = 0; t < nTest; t++) refNorm += exV[t] * exV[t];
            refNorm = Math.Sqrt(refNorm);

            double errNorm = 0;
            for (int t = 0; t < nTest; t++)
            {
                double diff = ttV[t] - exV[t];
                errNorm += diff * diff;
            }
            errNorm = Math.Sqrt(errNorm);

            return refNorm > 0 ? errNorm / refNorm : errNorm;
        }

        // Sweep loop
        TtCore[] cores = new TtCore[d];

        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            // ============================================================
            // Left-to-right half-sweep (k = 0, ..., d-2)
            // ============================================================
            for (int k = 0; k < d - 1; k++)
            {
                var left = J_left[k];
                var right = J_right[k];
                int rl = left.GetLength(0);
                int rr = right.GetLength(0);
                int nk = n[k];
                int cap = rankCaps[k + 1];

                // Build cross matrix C of shape (rl * nk, rr)
                double[,] C = new double[rl * nk, rr];
                int[] idxBuf = new int[d];
                for (int a = 0; a < rl; a++)
                    for (int i = 0; i < nk; i++)
                        for (int b = 0; b < rr; b++)
                        {
                            // Compose index: left[a] + [i] + right[b]
                            for (int dd = 0; dd < k; dd++) idxBuf[dd] = left[a, dd];
                            idxBuf[k] = i;
                            for (int dd = 0; dd < d - k - 1; dd++) idxBuf[k + 1 + dd] = right[b, dd];
                            C[a * nk + i, b] = EvalFunc(idxBuf);
                        }

                // SVD-based adaptive rank
                var matC = DenseMatrix.OfArray(C);
                var svd = matC.Svd(true);
                var S = svd.S;
                int effective = 1;
                if (S[0] > 0)
                {
                    double threshold = 1e-12 * S[0];
                    effective = 0;
                    for (int i = 0; i < S.Count; i++)
                        if (S[i] > threshold) effective++;
                }
                int rank = Math.Max(1, Math.Min(cap, Math.Min(effective, svd.U.ColumnCount)));
                var U = new double[rl * nk, rank];
                for (int ii = 0; ii < rl * nk; ii++)
                    for (int jj = 0; jj < rank; jj++)
                        U[ii, jj] = svd.U[ii, jj];

                // Maxvol pivot selection
                int[] pivots;
                if (rl * nk > rank)
                    pivots = Maxvol(U);
                else
                {
                    pivots = new int[rl * nk];
                    for (int i = 0; i < pivots.Length; i++) pivots[i] = i;
                }
                if (pivots.Length > rank)
                {
                    var trimmed = new int[rank];
                    Array.Copy(pivots, trimmed, rank);
                    pivots = trimmed;
                }

                // Form TT core: C_hat = U * inv(U[pivots])
                var uPiv = new double[rank, rank];
                for (int ii = 0; ii < rank; ii++)
                    for (int jj = 0; jj < rank; jj++)
                        uPiv[ii, jj] = U[pivots[ii], jj];

                double[,] cHat;
                try
                {
                    var matU = DenseMatrix.OfArray(U);
                    var matUPiv = DenseMatrix.OfArray(uPiv);
                    cHat = (matU * matUPiv.Inverse()).ToArray();
                }
                catch
                {
                    cHat = U;
                }

                // Reshape to 3D core (rl, nk, rank)
                var core = new TtCore(rl, nk, rank);
                for (int a = 0; a < rl; a++)
                    for (int i = 0; i < nk; i++)
                        for (int rIdx = 0; rIdx < rank; rIdx++)
                            core[a, i, rIdx] = cHat[a * nk + i, rIdx];
                cores[k] = core;

                // Update left index set for k+1
                var newLeft = new int[rank, k + 1];
                for (int pIdx = 0; pIdx < rank; pIdx++)
                {
                    int prow = pivots[pIdx];
                    int a = Math.DivRem(prow, nk, out int ik);
                    a = Math.Min(a, rl - 1);
                    for (int dd = 0; dd < k; dd++)
                        newLeft[pIdx, dd] = J_left[k][a, dd];
                    newLeft[pIdx, k] = ik;
                }
                J_left[k + 1] = newLeft;
                r[k + 1] = rank;
            }

            // Last core (from L→R)
            {
                int k = d - 1;
                var left = J_left[k];
                int rl = left.GetLength(0);
                int nk = n[k];
                var lastCore = new TtCore(rl, nk, 1);
                int[] idxBuf = new int[d];
                for (int a = 0; a < rl; a++)
                    for (int i = 0; i < nk; i++)
                    {
                        for (int dd = 0; dd < k; dd++) idxBuf[dd] = left[a, dd];
                        idxBuf[k] = i;
                        lastCore[a, i, 0] = EvalFunc(idxBuf);
                    }
                cores[d - 1] = lastCore;
            }

            // Half-sweep convergence check
            double relErrorLR = CheckError(cores);

            if (verbose)
            {
                var ranksStr = "1";
                for (int i = 0; i < d; i++) ranksStr += $", {cores[i].RRight}";
                Console.WriteLine($"    Sweep {sweep + 1} L->R: rel error = {relErrorLR:E2}, " +
                                  $"unique evals = {cache.Count:N0}, ranks = [{ranksStr}]");
            }

            if (relErrorLR < bestError * 0.9)
            {
                bestError = relErrorLR;
                bestCores = new TtCore[d];
                for (int i = 0; i < d; i++) bestCores[i] = cores[i].Copy();
                staleChecks = 0;
            }
            else
            {
                staleChecks++;
            }

            if (relErrorLR < tol)
            {
                if (verbose) Console.WriteLine($"    Converged after {sweep + 1} sweeps (L->R)");
                cores = bestCores!;
                break;
            }

            if (staleChecks >= 3 && bestError < 1e-3)
            {
                if (verbose) Console.WriteLine($"    No improvement in {staleChecks} checks (best = {bestError:E2}) — stopping");
                cores = bestCores!;
                break;
            }

            // ============================================================
            // Right-to-left half-sweep (k = d-1, ..., 1)
            // ============================================================
            for (int k = d - 1; k >= 1; k--)
            {
                var left = J_left[k];
                var right = J_right[k];
                int rl = left.GetLength(0);
                int rr = right.GetLength(0);
                int nk = n[k];
                int cap = rankCaps[k];

                // Build cross matrix C of shape (rl, nk * rr)
                double[,] C = new double[rl, nk * rr];
                int[] idxBuf = new int[d];
                for (int a = 0; a < rl; a++)
                    for (int i = 0; i < nk; i++)
                        for (int b = 0; b < rr; b++)
                        {
                            for (int dd = 0; dd < k; dd++) idxBuf[dd] = left[a, dd];
                            idxBuf[k] = i;
                            for (int dd = 0; dd < d - k - 1; dd++) idxBuf[k + 1 + dd] = right[b, dd];
                            C[a, i * rr + b] = EvalFunc(idxBuf);
                        }

                // Transpose: Ct = C^T of shape (nk * rr, rl)
                double[,] Ct = new double[nk * rr, rl];
                for (int ii = 0; ii < rl; ii++)
                    for (int jj = 0; jj < nk * rr; jj++)
                        Ct[jj, ii] = C[ii, jj];

                var matCt = DenseMatrix.OfArray(Ct);
                var svd = matCt.Svd(true);
                var S = svd.S;
                int effective = 1;
                if (S[0] > 0)
                {
                    double threshold = 1e-12 * S[0];
                    effective = 0;
                    for (int i = 0; i < S.Count; i++)
                        if (S[i] > threshold) effective++;
                }
                int rank = Math.Max(1, Math.Min(cap, Math.Min(effective, svd.U.ColumnCount)));
                var U = new double[nk * rr, rank];
                for (int ii = 0; ii < nk * rr; ii++)
                    for (int jj = 0; jj < rank; jj++)
                        U[ii, jj] = svd.U[ii, jj];

                int[] pivots;
                if (nk * rr > rank)
                    pivots = Maxvol(U);
                else
                {
                    pivots = new int[nk * rr];
                    for (int i = 0; i < pivots.Length; i++) pivots[i] = i;
                }
                if (pivots.Length > rank)
                {
                    var trimmed = new int[rank];
                    Array.Copy(pivots, trimmed, rank);
                    pivots = trimmed;
                }

                // Cross interpolation on transposed: C_hat^T = U * inv(U[pivots]), then transpose
                var uPiv = new double[rank, rank];
                for (int ii = 0; ii < rank; ii++)
                    for (int jj = 0; jj < rank; jj++)
                        uPiv[ii, jj] = U[pivots[ii], jj];

                double[,] cHatT;
                try
                {
                    var matU = DenseMatrix.OfArray(U);
                    var matUPiv = DenseMatrix.OfArray(uPiv);
                    cHatT = (matU * matUPiv.Inverse()).ToArray();
                }
                catch
                {
                    cHatT = U;
                }

                // cores[k] = cHatT^T reshaped to (rank, nk, rr)
                var core = new TtCore(rank, nk, rr);
                for (int row = 0; row < nk * rr; row++)
                    for (int col = 0; col < rank; col++)
                    {
                        // cHatT[row, col] → transpose → cHat[col, row]
                        // reshape (rank, nk, rr): [col, row/rr, row%rr]
                        int nodeIdx = row / rr;
                        int rightIdx = row % rr;
                        core[col, nodeIdx, rightIdx] = cHatT[row, col];
                    }
                cores[k] = core;

                // Update right index set for k-1
                int nRightNew = d - k;
                var newRight = new int[rank, nRightNew];
                for (int pIdx = 0; pIdx < rank; pIdx++)
                {
                    int prow = pivots[pIdx];
                    int rrMax = Math.Max(rr, 1);
                    int ik = Math.DivRem(prow, rrMax, out int bIdx);
                    ik = Math.Min(ik, nk - 1);
                    bIdx = Math.Min(bIdx, rrMax - 1);

                    newRight[pIdx, 0] = ik;
                    for (int dd = 0; dd < right.GetLength(1); dd++)
                        newRight[pIdx, 1 + dd] = right[bIdx, dd];
                }
                J_right[k - 1] = newRight;
                r[k] = rank;
            }

            // First core (from R→L)
            {
                var right = J_right[0];
                int rr = right.GetLength(0);
                int nk = n[0];
                var firstCore = new TtCore(1, nk, rr);
                int[] idxBuf = new int[d];
                for (int i = 0; i < nk; i++)
                    for (int b = 0; b < rr; b++)
                    {
                        idxBuf[0] = i;
                        for (int dd = 0; dd < right.GetLength(1); dd++)
                            idxBuf[1 + dd] = right[b, dd];
                        firstCore[0, i, b] = EvalFunc(idxBuf);
                    }
                cores[0] = firstCore;
            }

            // Full convergence check after R→L
            double relError = CheckError(cores);

            if (verbose)
            {
                Console.WriteLine($"    Sweep {sweep + 1} R->L: rel error = {relError:E2}, " +
                                  $"unique evals = {cache.Count:N0}");
            }

            if (relError < bestError * 0.9)
            {
                bestError = relError;
                bestCores = new TtCore[d];
                for (int i = 0; i < d; i++) bestCores[i] = cores[i].Copy();
                staleChecks = 0;
            }
            else
            {
                staleChecks++;
            }

            if (relError < tol)
            {
                if (verbose) Console.WriteLine($"    Converged after {sweep + 1} sweeps");
                cores = bestCores!;
                break;
            }

            if (staleChecks >= 3 && bestError < 1e-3)
            {
                if (verbose)
                    Console.WriteLine($"    No improvement in {staleChecks} checks (best = {bestError:E2}) — stopping");
                cores = bestCores!;
                break;
            }

            // If max_sweeps exhausted on the last iteration, use best cores
            if (sweep == maxSweeps - 1 && bestCores != null)
                cores = bestCores;
        }

        return (cores, cache.Count);
    }

    // ------------------------------------------------------------------
    // TT-SVD — full tensor evaluation then sequential SVD decomposition
    // ------------------------------------------------------------------

    /// <summary>
    /// Build TT value cores via SVD of the full tensor.
    /// Evaluates f at all grid points, then decomposes via sequential truncated SVD.
    /// Only feasible for moderate dimensions (d ≤ 6).
    /// </summary>
    internal static (TtCore[] Cores, int TotalEvals) TtSvd(
        Func<double[], double> func,
        double[][] grids,
        int maxRank,
        double tol,
        bool verbose)
    {
        int d = grids.Length;
        int[] n = new int[d];
        for (int i = 0; i < d; i++) n[i] = grids[i].Length;

        long fullSize = 1;
        for (int i = 0; i < d; i++) fullSize *= n[i];

        if (verbose)
            Console.WriteLine($"  Building full tensor ({fullSize:N0} evaluations)...");

        // Build full tensor as flat array (row-major, dimensions ordered 0..d-1)
        double[] T = new double[fullSize];
        int[] idx = new int[d];
        double[] point = new double[d];
        int totalEvals = 0;

        for (long flat = 0; flat < fullSize; flat++)
        {
            // Compute multi-index from flat index
            long rem = flat;
            for (int dim = d - 1; dim >= 0; dim--)
            {
                idx[dim] = (int)(rem % n[dim]);
                rem /= n[dim];
            }

            for (int dim = 0; dim < d; dim++)
                point[dim] = grids[dim][idx[dim]];

            T[flat] = func(point);
            totalEvals++;
        }

        // Sequential SVD decomposition
        var cores = new TtCore[d];
        double[] C = T; // will be reshaped progressively
        int rPrev = 1;
        int remainingSize = (int)fullSize;

        for (int k = 0; k < d - 1; k++)
        {
            int rows = rPrev * n[k];
            int cols = remainingSize / (rPrev * n[k]);
            // Clamp to avoid degenerate case
            if (cols < 1) cols = 1;

            // Reshape C to (rows, cols) — already in row-major order
            var matC = DenseMatrix.Build.Dense(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matC[i, j] = C[i * cols + j];

            var svd = matC.Svd(true);
            var S = svd.S;

            int rank = Math.Min(maxRank, S.Count);
            if (S[0] > 0)
            {
                double threshold = tol * S[0];
                int eff = 0;
                for (int i = 0; i < S.Count; i++)
                    if (S[i] > threshold) eff++;
                rank = Math.Max(1, Math.Min(rank, eff));
            }

            // Core = U[:, :rank] reshaped to (rPrev, n[k], rank)
            var core = new TtCore(rPrev, n[k], rank);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < rank; j++)
                    core.Data[i * rank + j] = svd.U[i, j];
            cores[k] = core;

            // C = diag(S[:rank]) @ Vt[:rank, :]
            int newRows = rank;
            int newCols = cols;
            C = new double[newRows * newCols];
            for (int i = 0; i < newRows; i++)
            {
                double si = S[i];
                for (int j = 0; j < newCols; j++)
                    C[i * newCols + j] = si * svd.VT[i, j];
            }

            remainingSize = newRows * newCols;
            rPrev = rank;
        }

        // Last core
        {
            int nk = n[d - 1];
            var lastCore = new TtCore(rPrev, nk, 1);
            for (int i = 0; i < rPrev; i++)
                for (int j = 0; j < nk; j++)
                    lastCore[i, j, 0] = C[i * nk + j];
            cores[d - 1] = lastCore;
        }

        if (verbose)
        {
            var ranksStr = "1";
            for (int i = 0; i < d; i++) ranksStr += $", {cores[i].RRight}";
            Console.WriteLine($"  TT-SVD ranks: [{ranksStr}]");
        }

        return (cores, totalEvals);
    }

    // ------------------------------------------------------------------
    // Value-to-coefficient core conversion
    // ------------------------------------------------------------------

    /// <summary>
    /// Convert value cores to Chebyshev coefficient cores via DCT-II along the node axis.
    /// For each core of shape (rLeft, nNodes, rRight):
    ///   1. Reverse along node axis
    ///   2. Apply DCT-II and normalize by nNodes
    ///   3. Halve the zeroth coefficient
    /// </summary>
    internal static TtCore[] ValueToCoeffCores(TtCore[] valueCores)
    {
        var coeffCores = new TtCore[valueCores.Length];
        for (int k = 0; k < valueCores.Length; k++)
        {
            var vc = valueCores[k];
            int rL = vc.RLeft, nk = vc.NNodes, rR = vc.RRight;
            var cc = new TtCore(rL, nk, rR);

            // For each (i, k) fiber along the node axis, apply ChebyshevCoefficients1D
            double[] fiber = new double[nk];
            for (int i = 0; i < rL; i++)
            {
                for (int j = 0; j < rR; j++)
                {
                    // Extract fiber along node axis
                    for (int ni = 0; ni < nk; ni++)
                        fiber[ni] = vc[i, ni, j];

                    // Apply ChebyshevCoefficients1D (handles reverse + DCT-II + normalize + halve c0)
                    double[] coeffs = BarycentricKernel.ChebyshevCoefficients1D(fiber);

                    // Store back
                    for (int ni = 0; ni < nk; ni++)
                        cc[i, ni, j] = coeffs[ni];
                }
            }

            coeffCores[k] = cc;
        }
        return coeffCores;
    }
}
