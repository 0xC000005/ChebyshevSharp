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
            var newM = new double[rAr * rBr];
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
}
