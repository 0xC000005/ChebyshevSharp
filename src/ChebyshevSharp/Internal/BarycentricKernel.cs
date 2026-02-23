namespace ChebyshevSharp.Internal;

/// <summary>
/// Free functions for barycentric interpolation: weight computation,
/// differentiation matrices, and evaluation kernels.
/// </summary>
internal static class BarycentricKernel
{
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
        // chebpts1(n): cos(pi*(2k+1)/(2n)) for k=0..n-1, which is the same as
        // cos((2*(n-k)-1)*pi/(2n)) in ascending order.
        // numpy.polynomial.chebyshev.chebpts1 returns them in ascending order.
        double[] nodesStd = new double[n];
        for (int k = 0; k < n; k++)
        {
            // chebpts1 formula: cos(pi * (n - 1 - 2k) / (2n)) for k=0..n-1
            // Actually numpy uses: x_k = cos(pi*(2*(n-1-k)+1)/(2n)) for k=0..n-1 ascending
            // Simpler: generate descending then reverse
            nodesStd[k] = Math.Cos(Math.PI * (2 * k + 1) / (2 * n));
        }
        // nodesStd is in descending order, sort ascending
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
    /// Multiply the last axis of an N-D array (stored flat) by a vector or matrix.
    /// This is the C# equivalent of _matmul_last_axis in Python.
    /// </summary>
    internal static double[] MatmulLastAxis(double[] data, int[] shape, double[] rhs)
    {
        int ndim = shape.Length;
        int lastDim = shape[ndim - 1];
        int leadSize = 1;
        for (int i = 0; i < ndim - 1; i++)
            leadSize *= shape[i];

        double[] result = new double[leadSize];
        for (int i = 0; i < leadSize; i++)
        {
            double sum = 0.0;
            int offset = i * lastDim;
            for (int j = 0; j < lastDim; j++)
                sum += data[offset + j] * rhs[j];
            result[i] = sum;
        }
        return result;
    }

    /// <summary>
    /// Multiply the last axis of an N-D array (stored flat) by a matrix (columns of rhs).
    /// Result has last dimension = number of columns in rhs.
    /// </summary>
    internal static double[] MatmulLastAxisMatrix(double[] data, int[] shape, double[,] rhs)
    {
        int ndim = shape.Length;
        int lastDim = shape[ndim - 1];
        int rhsCols = rhs.GetLength(1);
        int leadSize = 1;
        for (int i = 0; i < ndim - 1; i++)
            leadSize *= shape[i];

        double[] result = new double[leadSize * rhsCols];
        for (int i = 0; i < leadSize; i++)
        {
            int srcOffset = i * lastDim;
            int dstOffset = i * rhsCols;
            for (int c = 0; c < rhsCols; c++)
            {
                double sum = 0.0;
                for (int j = 0; j < lastDim; j++)
                    sum += data[srcOffset + j] * rhs[j, c];
                result[dstOffset + c] = sum;
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

        int totalNew = 1;
        for (int i = 0; i < newShape.Length; i++)
            totalNew *= newShape[i];

        double[] result = new double[totalNew];

        // Compute strides for the original tensor
        int[] strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        // For each element in the result, compute the corresponding index in the source
        int[] newStrides = new int[newShape.Length];
        if (newShape.Length > 0)
        {
            newStrides[newShape.Length - 1] = 1;
            for (int i = newShape.Length - 2; i >= 0; i--)
                newStrides[i] = newStrides[i + 1] * newShape[i + 1];
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

        int totalNew = 1;
        for (int i = 0; i < newShape.Length; i++)
            totalNew *= newShape[i];

        double[] result = new double[totalNew];

        // Compute strides for the original tensor
        int[] strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        // Compute strides for the result
        int[] newStrides = new int[newShape.Length];
        if (newShape.Length > 0)
        {
            newStrides[newShape.Length - 1] = 1;
            for (int i = newShape.Length - 2; i >= 0; i--)
                newStrides[i] = newStrides[i + 1] * newShape[i + 1];
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
    /// </summary>
    internal static double[] ChebyshevCoefficients1D(double[] values)
    {
        int n = values.Length;
        // Reverse to decreasing-node order for DCT-II convention
        double[] reversed = new double[n];
        for (int i = 0; i < n; i++)
            reversed[i] = values[n - 1 - i];

        // DCT-II: coeffs[k] = sum_{j=0}^{n-1} reversed[j] * cos(pi*k*(2j+1)/(2n))
        double[] coeffs = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += reversed[j] * Math.Cos(Math.PI * k * (2 * j + 1) / (2.0 * n));
            coeffs[k] = sum * 2.0 / n; // scipy dct type 2 = 2 * sum * cos(...)
        }
        coeffs[0] /= 2.0;

        return coeffs;
    }
}
