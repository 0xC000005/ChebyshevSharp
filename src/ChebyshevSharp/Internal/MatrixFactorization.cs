using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace ChebyshevSharp.Internal;

internal static class MatrixFactorization
{
    internal static (Matrix<double> Q, Matrix<double> R) ReducedQr(DenseMatrix matrix)
    {
        if (matrix.RowCount >= matrix.ColumnCount)
        {
            var qr = matrix.QR(QRMethod.Thin);
            return (qr.Q, qr.R);
        }

        // MathNet's dense QR rejects wide matrices. NumPy's reduced QR accepts
        // them and returns Q(m x m), R(m x n). Use the equivalent thin SVD
        // factorization Q = U, R = S * V^T for that shape.
        var svd = matrix.Svd(computeVectors: true);
        int rankDim = svd.S.Count;
        var q = DenseMatrix.Create(matrix.RowCount, rankDim, 0.0);
        for (int i = 0; i < matrix.RowCount; i++)
            for (int j = 0; j < rankDim; j++)
                q.At(i, j, svd.U[i, j]);

        var r = DenseMatrix.Create(rankDim, matrix.ColumnCount, 0.0);
        for (int i = 0; i < rankDim; i++)
            for (int j = 0; j < matrix.ColumnCount; j++)
                r.At(i, j, svd.S[i] * svd.VT[i, j]);

        return (q, r);
    }
}
