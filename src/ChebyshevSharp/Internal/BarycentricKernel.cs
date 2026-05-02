using BlasSharp;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Free functions for barycentric interpolation: weight computation,
/// differentiation matrices, and evaluation kernels.
/// </summary>
internal static class BarycentricKernel
{
    private static readonly IBlasOperations Blas = new BlasSharp.OpenBlas.OpenBlasOperations();
    /// <summary>
    /// Compute barycentric weights for given interpolation nodes.
    /// w_i = 1 / prod_{j!=i} (x_i - x_j)
    /// </summary>
    internal static double[] ComputeBarycentricWeights(double[] nodes)
    {
        int n = nodes.Length;
        double[] weights = new double[n];
        for (int i = 0; i < n; i++)
            weights[i] = 1.0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                    weights[i] /= (nodes[i] - nodes[j]);
            }
        }
        return weights;
    }

    /// <summary>
    /// Compute spectral differentiation matrix for barycentric interpolation.
    /// Based on Berrut and Trefethen (2004), Section 9.3.
    /// </summary>
    internal static double[,] ComputeDifferentiationMatrix(double[] nodes, double[] weights)
    {
        int n = nodes.Length;
        double[,] c = new double[n, n];

        // c[i,j] = w[j] / ((x[i] - x[j]) * w[i]) for i != j
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    c[i, j] = weights[j] / ((nodes[i] - nodes[j]) * weights[i]);
                }
            }
        }

        // Diagonal: c[i,i] = -sum of row
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (j != i) sum += c[i, j];
            }
            c[i, i] = -sum;
        }

        return c;
    }

    /// <summary>
    /// Evaluate barycentric interpolation at a single point.
    /// If x coincides with a node (within 1e-14), returns the function value at that node.
    /// </summary>
    internal static double BarycentricInterpolate(double x, double[] nodes, double[] values, double[] weights, bool skipCheck = false)
    {
        if (!skipCheck)
        {
            for (int i = 0; i < nodes.Length; i++)
            {
                if (Math.Abs(nodes[i] - x) < 1e-14)
                    return values[i];
            }
        }
        return BarycentricInterpolateCore(x, nodes, values, weights);
    }

    /// <summary>
    /// Core barycentric interpolation formula without node coincidence check.
    /// sum(w_i/(x-x_i) * f_i) / sum(w_i/(x-x_i))
    /// </summary>
    internal static double BarycentricInterpolateCore(double x, double[] nodes, double[] values, double[] weights)
    {
        double sumNumerator = 0.0;
        double sumDenominator = 0.0;
        for (int i = 0; i < nodes.Length; i++)
        {
            double wi = weights[i] / (x - nodes[i]);
            sumNumerator += wi * values[i];
            sumDenominator += wi;
        }
        return sumNumerator / sumDenominator;
    }

    /// <summary>
    /// Compute analytical derivative using the spectral differentiation matrix.
    /// Supports order 1 and 2 (higher orders via repeated application).
    /// </summary>
    internal static double BarycentricDerivativeAnalytical(
        double x, double[] nodes, double[] values, double[] weights,
        double[,] diffMatrix, int order = 1)
    {
        if (order < 1)
            throw new ArgumentException($"Derivative order {order} not supported (use >= 1)");

        int n = values.Length;
        double[] current = values;

        for (int o = 0; o < order; o++)
        {
            double[] derivValues = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    sum += diffMatrix[i, j] * current[j];
                derivValues[i] = sum;
            }
            current = derivValues;
        }

        return BarycentricInterpolate(x, nodes, current, weights);
    }

    /// <summary>
    /// Generate Chebyshev Type I nodes on [lo, hi] with n points.
    /// Uses cos((2i-1)*pi/(2n)) for i=1..n, mapped to [lo, hi], sorted ascending.
    /// </summary>
    internal static double[] MakeNodesForDim(double lo, double hi, int n)
    {
        // NumPy chebpts1(n) uses Type-I roots cos(pi*(2k+1)/(2n)).
        // Generate the roots and sort ascending to match PyChebyshev parity.
        double[] nodesStd = new double[n];
        for (int k = 0; k < n; k++)
        {
            nodesStd[k] = Math.Cos(Math.PI * (2 * k + 1) / (2 * n));
        }
        Array.Sort(nodesStd);

        // Map from [-1,1] to [lo, hi]
        double mid = 0.5 * (lo + hi);
        double half = 0.5 * (hi - lo);
        double[] nodes = new double[n];
        for (int i = 0; i < n; i++)
            nodes[i] = mid + half * nodesStd[i];

        return nodes;
    }

    /// <summary>
    /// Multiply flat data[leadSize x lastDim] by vector rhs[lastDim], producing result[leadSize].
    /// Uses OpenBLAS GEMV via BlasSharp.
    /// </summary>
    internal static unsafe double[] MatmulLastAxis(double[] data, int leadSize, int lastDim, double[] rhs)
    {
        double[] result = new double[leadSize];
        fixed (double* pA = data, pX = rhs, pY = result)
        {
            Blas.Dgemv((uint)CBLAS_ORDER.CblasRowMajor, (uint)CBLAS_TRANSPOSE.CblasNoTrans,
                leadSize, lastDim, 1.0, pA, lastDim, pX, 1, 0.0, pY, 1);
        }
        return result;
    }

    /// <summary>
    /// Multiply flat data[leadSize x lastDim] by pre-flattened matrix rhsFlat[lastDim * rhsCols] (row-major).
    /// Uses OpenBLAS GEMM via BlasSharp.
    /// </summary>
    internal static unsafe double[] MatmulLastAxisMatrixFlat(double[] data, int leadSize, int lastDim, double[] rhsFlat, int rhsCols)
    {
        int resultLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { leadSize, rhsCols }, nameof(MatmulLastAxisMatrixFlat)),
            nameof(MatmulLastAxisMatrixFlat),
            new[] { leadSize, rhsCols });
        double[] result = new double[resultLength];
        fixed (double* pA = data, pB = rhsFlat, pC = result)
        {
            Blas.Dgemm((uint)CBLAS_ORDER.CblasRowMajor, (uint)CBLAS_TRANSPOSE.CblasNoTrans, (uint)CBLAS_TRANSPOSE.CblasNoTrans,
                leadSize, rhsCols, lastDim, 1.0, pA, lastDim, pB, rhsCols, 0.0, pC, rhsCols);
        }
        return result;
    }

    /// <summary>
    /// Apply a differentiation matrix along axis <paramref name="axis"/> of a flat N-D tensor.
    /// Equivalent to Python's <c>np.moveaxis(data, axis, -1) @ D_T; np.moveaxis(..., -1, axis)</c>.
    /// The result has the same total size as <paramref name="data"/>.
    /// </summary>
    /// <param name="data">Flat N-D tensor in row-major order, total length = product of shape.</param>
    /// <param name="shape">Shape of the tensor.</param>
    /// <param name="axis">Axis along which to apply the differentiation matrix.</param>
    /// <param name="diffMatrix">Differentiation matrix D, shape [n_axis, n_axis].</param>
    /// <returns>New flat tensor with diff matrix applied along the specified axis.</returns>
    internal static double[] MatmulAlongAxis(double[] data, int[] shape, int axis, double[,] diffMatrix)
    {
        int ndim = shape.Length;
        int n_axis = shape[axis];

        // Compute trailSize = product of dimensions after axis
        int[] trailingShape = shape.Skip(axis + 1).ToArray();
        int trailSize = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(trailingShape, nameof(MatmulAlongAxis)),
            nameof(MatmulAlongAxis),
            trailingShape);

        // Compute leadSize = product of dimensions before axis
        int[] leadingShape = shape.Take(axis).ToArray();
        int leadSize = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(leadingShape, nameof(MatmulAlongAxis)),
            nameof(MatmulAlongAxis),
            leadingShape);

        double[] result = new double[data.Length];

        // For each (lead, trail) pair, apply the diff matrix to the vector along axis.
        // data[lead * n_axis * trailSize + k * trailSize + trail]
        // result[lead * n_axis * trailSize + k' * trailSize + trail]
        //   = sum_k D[k', k] * data[...]
        for (int lead = 0; lead < leadSize; lead++)
        {
            int leadOff = lead * n_axis * trailSize;
            for (int trail = 0; trail < trailSize; trail++)
            {
                for (int kPrime = 0; kPrime < n_axis; kPrime++)
                {
                    double sum = 0.0;
                    int srcBase = leadOff + trail;
                    for (int k = 0; k < n_axis; k++)
                        sum += diffMatrix[kPrime, k] * data[srcBase + k * trailSize];
                    result[leadOff + kPrime * trailSize + trail] = sum;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Extract a 1-D slice from a flat N-D tensor along a given dimension at a given index.
    /// For indexing: takes all elements where dimension d has a specific index value.
    /// </summary>
    internal static double[] TakeAlongAxis(double[] data, int[] shape, int axis, int index)
    {
        int ndim = shape.Length;
        int[] newShape = new int[ndim - 1];
        for (int i = 0, j = 0; i < ndim; i++)
        {
            if (i != axis) newShape[j++] = shape[i];
        }

        int totalNew = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(newShape, nameof(TakeAlongAxis)),
            nameof(TakeAlongAxis),
            newShape);

        double[] result = new double[totalNew];

        // Compute strides for the original tensor
        int[] strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            int[] trailingShape = shape.Skip(i + 1).ToArray();
            strides[i] = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(trailingShape, nameof(TakeAlongAxis)),
                nameof(TakeAlongAxis),
                trailingShape);
        }

        // For each element in the result, compute the corresponding index in the source
        int[] newStrides = new int[newShape.Length];
        if (newShape.Length > 0)
        {
            newStrides[newShape.Length - 1] = 1;
            for (int i = newShape.Length - 2; i >= 0; i--)
            {
                int[] trailingShape = newShape.Skip(i + 1).ToArray();
                newStrides[i] = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(trailingShape, nameof(TakeAlongAxis)),
                    nameof(TakeAlongAxis),
                    trailingShape);
            }
        }

        for (int flatIdx = 0; flatIdx < totalNew; flatIdx++)
        {
            // Decompose flatIdx into multi-index in newShape
            int remaining = flatIdx;
            int srcIdx = 0;
            int newDim = 0;
            for (int d = 0; d < ndim; d++)
            {
                if (d == axis)
                {
                    srcIdx += index * strides[d];
                }
                else
                {
                    int coord = remaining / newStrides[newDim];
                    remaining %= newStrides[newDim];
                    srcIdx += coord * strides[d];
                    newDim++;
                }
            }
            result[flatIdx] = data[srcIdx];
        }

        return result;
    }

    /// <summary>
    /// Contract a flat N-D tensor along the given axis with a weight vector via dot product.
    /// Equivalent to np.tensordot(tensor, weights, axes=([axis], [0])).
    /// </summary>
    internal static double[] TensordotVector(double[] data, int[] shape, int axis, double[] weights)
    {
        int ndim = shape.Length;
        int axisLen = shape[axis];

        int[] newShape = new int[ndim - 1];
        for (int i = 0, j = 0; i < ndim; i++)
        {
            if (i != axis) newShape[j++] = shape[i];
        }

        int totalNew = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(newShape, nameof(TensordotVector)),
            nameof(TensordotVector),
            newShape);

        double[] result = new double[totalNew];

        // Compute strides for the original tensor
        int[] strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
        {
            int[] trailingShape = shape.Skip(i + 1).ToArray();
            strides[i] = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(trailingShape, nameof(TensordotVector)),
                nameof(TensordotVector),
                trailingShape);
        }

        // Compute strides for the result
        int[] newStrides = new int[newShape.Length];
        if (newShape.Length > 0)
        {
            newStrides[newShape.Length - 1] = 1;
            for (int i = newShape.Length - 2; i >= 0; i--)
            {
                int[] trailingShape = newShape.Skip(i + 1).ToArray();
                newStrides[i] = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(trailingShape, nameof(TensordotVector)),
                    nameof(TensordotVector),
                    trailingShape);
            }
        }

        int axisStride = strides[axis];

        for (int flatIdx = 0; flatIdx < totalNew; flatIdx++)
        {
            // Decompose flatIdx into multi-index in newShape
            int remaining = flatIdx;
            int baseIdx = 0;
            int newDim = 0;
            for (int d = 0; d < ndim; d++)
            {
                if (d == axis)
                    continue;
                int coord = remaining / newStrides[newDim];
                remaining %= newStrides[newDim];
                baseIdx += coord * strides[d];
                newDim++;
            }

            double sum = 0.0;
            for (int k = 0; k < axisLen; k++)
                sum += data[baseIdx + k * axisStride] * weights[k];
            result[flatIdx] = sum;
        }

        return result;
    }

    /// <summary>
    /// Compute Chebyshev expansion coefficients from values at Type I nodes via DCT-II.
    /// Uses O(n log n) FFT for n > 32, O(n^2) direct for small n.
    /// </summary>
    internal static double[] ChebyshevCoefficients1D(double[] values)
    {
        int n = values.Length;
        // Reverse to decreasing-node order for DCT-II convention
        double[] reversed = new double[n];
        for (int i = 0; i < n; i++)
            reversed[i] = values[n - 1 - i];

        double[] coeffs;
        if (n > 32)
            coeffs = DctIIviaFFT(reversed);
        else
            coeffs = DctIINaive(reversed);

        return coeffs;
    }

    /// <summary>
    /// O(n^2) direct DCT-II computation.
    /// </summary>
    private static double[] DctIINaive(double[] reversed)
    {
        int n = reversed.Length;
        double[] coeffs = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += reversed[j] * Math.Cos(Math.PI * k * (2 * j + 1) / (2.0 * n));
            coeffs[k] = sum * 2.0 / n;
        }
        coeffs[0] /= 2.0;
        return coeffs;
    }

    /// <summary>
    /// O(n log n) DCT-II via FFT using the standard half-sample-shift trick.
    /// DCT-II[k] = 2 * Re(W^{k/2} * FFT(reordered)[k]) where W = exp(-2*pi*i/(4n)).
    /// </summary>
    private static double[] DctIIviaFFT(double[] reversed)
    {
        int n = reversed.Length;

        // Reorder: even indices get first half, odd indices get reversed second half
        // y[j] = x[2j] for j = 0..n/2-1, y[n-1-j] = x[2j+1] for j = 0..n/2-1
        var reordered = new System.Numerics.Complex[n];
        for (int j = 0; j < (n + 1) / 2; j++)
            reordered[j] = new System.Numerics.Complex(reversed[2 * j], 0);
        for (int j = 0; j < n / 2; j++)
            reordered[n - 1 - j] = new System.Numerics.Complex(reversed[2 * j + 1], 0);

        // In-place FFT
        MathNet.Numerics.IntegralTransforms.Fourier.Forward(
            reordered,
            MathNet.Numerics.IntegralTransforms.FourierOptions.NoScaling);

        // Extract DCT-II coefficients: coeffs[k] = 2/n * Re(exp(-i*pi*k/(2n)) * FFT[k])
        double[] coeffs = new double[n];
        for (int k = 0; k < n; k++)
        {
            double angle = -Math.PI * k / (2.0 * n);
            var twiddle = new System.Numerics.Complex(Math.Cos(angle), Math.Sin(angle));
            var val = twiddle * reordered[k];
            coeffs[k] = val.Real * 2.0 / n;
        }
        coeffs[0] /= 2.0;
        return coeffs;
    }
}
