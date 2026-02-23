namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for Chebyshev extrusion and slicing operations.
/// </summary>
internal static class ExtrudeSlice
{
    /// <summary>
    /// Validate and normalize extrusion parameters. Returns sorted ascending by dimIndex.
    /// </summary>
    internal static (int dimIndex, double[] bounds, int nNodes)[] NormalizeExtrusionParams(
        (int dimIndex, double[] bounds, int nNodes)[] param, int ndim)
    {
        int newNdim = ndim + param.Length;
        var seen = new HashSet<int>();

        foreach (var (dimIdx, bounds, n) in param)
        {
            if (dimIdx < 0 || dimIdx >= newNdim)
                throw new ArgumentException($"dim_index {dimIdx} out of range [0, {newNdim - 1}]");
            if (!seen.Add(dimIdx))
                throw new ArgumentException($"Duplicate dim_index {dimIdx}");
            if (bounds[0] >= bounds[1])
                throw new ArgumentException($"Domain bounds must satisfy lo < hi, got [{bounds[0]}, {bounds[1]}]");
            if (n < 2)
                throw new ArgumentException($"n_nodes must be int >= 2, got {n}");
        }

        return param.OrderBy(p => p.dimIndex).ToArray();
    }

    /// <summary>
    /// Validate and normalize slicing parameters. Returns sorted descending by dimIndex.
    /// </summary>
    internal static (int dimIndex, double value)[] NormalizeSlicingParams(
        (int dimIndex, double value)[] param, int ndim)
    {
        if (param.Length >= ndim)
            throw new ArgumentException($"Cannot slice all {ndim} dimensions (would produce 0D result)");

        var seen = new HashSet<int>();
        foreach (var (dimIdx, value) in param)
        {
            if (dimIdx < 0 || dimIdx >= ndim)
                throw new ArgumentException($"dim_index {dimIdx} out of range [0, {ndim - 1}]");
            if (!seen.Add(dimIdx))
                throw new ArgumentException($"Duplicate dim_index {dimIdx}");
        }

        return param.OrderByDescending(p => p.dimIndex).ToArray();
    }

    /// <summary>
    /// Insert a new axis in a flat tensor and replicate values.
    /// </summary>
    internal static double[] ExtrudeTensor(double[] data, int[] shape, int axis, int nNew)
    {
        int ndim = shape.Length;

        // Compute new shape
        var newShape = new List<int>(shape);
        newShape.Insert(axis, nNew);

        int newTotal = 1;
        foreach (int s in newShape) newTotal *= s;

        double[] result = new double[newTotal];

        // Compute strides for old and new shapes
        int[] oldStrides = ComputeStrides(shape);
        int[] newStrides = ComputeStrides(newShape.ToArray());

        // For each element in result, compute index in source
        int[] newIdx = new int[newShape.Count];
        for (int flat = 0; flat < newTotal; flat++)
        {
            // Decompose flat into multi-index
            int remaining = flat;
            for (int d = 0; d < newShape.Count; d++)
            {
                newIdx[d] = remaining / newStrides[d];
                remaining %= newStrides[d];
            }

            // Map to old index (skip the new axis)
            int oldFlat = 0;
            int oldDim = 0;
            for (int d = 0; d < newShape.Count; d++)
            {
                if (d == axis)
                    continue;
                oldFlat += newIdx[d] * oldStrides[oldDim];
                oldDim++;
            }

            result[flat] = data[oldFlat];
        }

        return result;
    }

    /// <summary>
    /// Contract tensor along axis at the given value via barycentric interpolation.
    /// </summary>
    internal static double[] SliceTensor(double[] data, int[] shape, int axis,
        double[] nodes, double[] weights, double value)
    {
        double[] diff = new double[nodes.Length];
        int exactIdx = -1;
        for (int i = 0; i < nodes.Length; i++)
        {
            diff[i] = value - nodes[i];
            if (Math.Abs(diff[i]) < 1e-14)
                exactIdx = i;
        }

        if (exactIdx >= 0)
        {
            // Fast path: exact node
            return BarycentricKernel.TakeAlongAxis(data, shape, axis, exactIdx);
        }

        // General path: barycentric contraction
        double[] wOverDiff = new double[nodes.Length];
        double sumW = 0.0;
        for (int i = 0; i < nodes.Length; i++)
        {
            wOverDiff[i] = weights[i] / diff[i];
            sumW += wOverDiff[i];
        }

        double[] wNorm = new double[nodes.Length];
        for (int i = 0; i < nodes.Length; i++)
            wNorm[i] = wOverDiff[i] / sumW;

        return BarycentricKernel.TensordotVector(data, shape, axis, wNorm);
    }

    private static int[] ComputeStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        if (shape.Length == 0) return strides;
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }
}
