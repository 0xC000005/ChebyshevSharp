using System.Diagnostics;
using System.Text;
using System.Text.Json;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp;

/// <summary>
/// Multi-dimensional Chebyshev tensor interpolation with analytical derivatives.
/// Uses barycentric interpolation with pre-computed weights.
/// </summary>
public class ChebyshevApproximation
{
    /// <summary>The function to approximate. Null after load or from_values.</summary>
    public Func<double[], object?, double>? Function { get; internal set; }

    /// <summary>Number of input dimensions.</summary>
    public int NumDimensions { get; internal set; }

    /// <summary>Domain bounds for each dimension, as list of [lo, hi].</summary>
    public double[][] Domain { get; internal set; } = Array.Empty<double[]>();

    /// <summary>Number of Chebyshev nodes per dimension.</summary>
    public int[] NNodes { get; internal set; } = Array.Empty<int>();

    /// <summary>Maximum supported derivative order.</summary>
    public int MaxDerivativeOrder { get; internal set; } = 2;

    /// <summary>Chebyshev nodes per dimension, each sorted ascending.</summary>
    public double[][] NodeArrays { get; internal set; } = Array.Empty<double[]>();

    /// <summary>Flat tensor of function values at all node combinations (C-order).</summary>
    public double[]? TensorValues { get; internal set; }

    /// <summary>Barycentric weights per dimension.</summary>
    public double[][]? Weights { get; internal set; }

    /// <summary>Spectral differentiation matrices per dimension.</summary>
    public double[][,]? DiffMatrices { get; internal set; }

    /// <summary>Pre-transposed diff matrices flattened to double[] for BLAS GEMM (row-major).</summary>
    internal double[][]? DiffMatricesTFlat { get; set; }

    /// <summary>Time taken by Build() in seconds.</summary>
    public double BuildTime { get; internal set; }

    /// <summary>Number of function evaluations during Build().</summary>
    public int NEvaluations { get; internal set; }

    private double? _cachedErrorEstimate;

    /// <summary>
    /// Create a new ChebyshevApproximation.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -> double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public ChebyshevApproximation(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxDerivativeOrder = 2)
    {
        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        NNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;

        // Generate Chebyshev nodes for each dimension
        NodeArrays = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
        {
            NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
        }
    }

    // Internal parameterless constructor for factories
    internal ChebyshevApproximation() { }

    /// <summary>
    /// Evaluate the function at all node combinations and pre-compute weights.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    public void Build(bool verbose = true)
    {
        if (Function == null)
            throw new InvalidOperationException(
                "Cannot build: no function assigned. " +
                "This object was created via FromValues() or Load().");

        int total = 1;
        for (int d = 0; d < NumDimensions; d++)
            total *= NNodes[d];

        if (verbose)
            Console.WriteLine($"Building {NumDimensions}D Chebyshev approximation ({total:N0} evaluations)...");

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        // Step 1: Evaluate at all node combinations (C-order / ndindex)
        TensorValues = new double[total];
        double[] point = new double[NumDimensions];
        int[] indices = new int[NumDimensions];

        for (int flat = 0; flat < total; flat++)
        {
            // Convert flat index to multi-index
            int rem = flat;
            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                indices[d] = rem % NNodes[d];
                rem /= NNodes[d];
            }

            for (int d = 0; d < NumDimensions; d++)
                point[d] = NodeArrays[d][indices[d]];

            TensorValues[flat] = Function(point, null);
        }
        NEvaluations = total;

        // Step 2: Pre-compute barycentric weights
        Weights = new double[NumDimensions][];
        for (int d = 0; d < NumDimensions; d++)
            Weights[d] = BarycentricKernel.ComputeBarycentricWeights(NodeArrays[d]);

        // Step 3: Pre-compute differentiation matrices
        DiffMatrices = new double[NumDimensions][,];
        for (int d = 0; d < NumDimensions; d++)
            DiffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(NodeArrays[d], Weights[d]);

        // Step 4: Pre-transpose diff matrices for VectorizedEval
        PrecomputeTransposedDiffMatrices();

        sw.Stop();
        BuildTime = sw.Elapsed.TotalSeconds;

        if (verbose)
        {
            int totalWeights = Weights.Sum(w => w.Length);
            Console.WriteLine($"  Built in {BuildTime:F3}s ({totalWeights} weights, {totalWeights * 8} bytes)");
        }
    }

    /// <summary>
    /// Evaluate using dimensional decomposition with barycentric interpolation.
    /// Loop-based implementation matching Python eval().
    /// </summary>
    /// <param name="point">Query point, one coordinate per dimension.</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Interpolated value or derivative at the query point.</returns>
    public double Eval(double[] point, int[] derivativeOrder)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        // Current working data and its shape
        double[] current = TensorValues;
        int[] currentShape = (int[])NNodes.Clone();

        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            double x = point[d];
            int deriv = derivativeOrder[d];
            double[] nodes = NodeArrays[d];
            double[] weights = Weights![d];
            double[,] diffMatrix = DiffMatrices![d];
            int nNodesD = NNodes[d];

            if (d == 0)
            {
                // Last dimension: current is 1D (length = nNodesD)
                if (deriv == 0)
                    return BarycentricKernel.BarycentricInterpolate(x, nodes, current, weights);
                else
                    return BarycentricKernel.BarycentricDerivativeAnalytical(x, nodes, current, weights, diffMatrix, deriv);
            }
            else
            {
                // Compute size of leading dimensions
                int leadSize = 1;
                for (int i = 0; i < d; i++)
                    leadSize *= currentShape[i];

                // Contract dimension d
                double[] next = new double[leadSize];
                for (int idx = 0; idx < leadSize; idx++)
                {
                    // Extract 1D slice along dimension d
                    double[] values1d = new double[nNodesD];

                    // The flat index for current[idx, :, 0, 0, ...] along dim d
                    // Shape of current: currentShape[0..d] (d+1 dimensions, dims after d are already contracted)
                    // Actually current at this stage has shape currentShape[0..d]
                    for (int k = 0; k < nNodesD; k++)
                    {
                        values1d[k] = current[idx * nNodesD + k];
                    }

                    if (deriv == 0)
                        next[idx] = BarycentricKernel.BarycentricInterpolate(x, nodes, values1d, weights);
                    else
                        next[idx] = BarycentricKernel.BarycentricDerivativeAnalytical(x, nodes, values1d, weights, diffMatrix, deriv);
                }

                current = next;
                // Update shape: remove dimension d
                int[] newShape = new int[d];
                for (int i = 0; i < d; i++)
                    newShape[i] = currentShape[i];
                currentShape = newShape;
            }
        }

        return current[0]; // Should not reach here normally
    }

    /// <summary>
    /// Fully vectorized evaluation using matrix operations.
    /// Replaces the Python loop with BLAS-style matrix-vector products.
    /// </summary>
    /// <param name="point">Query point, one coordinate per dimension.</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Interpolated value or derivative.</returns>
    public double VectorizedEval(double[] point, int[] derivativeOrder)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        double[] current = TensorValues;

        // Track tensor dimensions without shape array allocations.
        // leadSize = product of all dims before current last dim.
        // After contracting dim d, leadSize shrinks accordingly.
        int totalSize = current.Length;

        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            double x = point[d];
            int deriv = derivativeOrder[d];
            int lastDim = NNodes[d];
            int leadSize = totalSize / lastDim;

            // Apply differentiation matrix if derivative order > 0
            if (deriv > 0)
            {
                double[] dTFlat = DiffMatricesTFlat![d];
                for (int o = 0; o < deriv; o++)
                    current = BarycentricKernel.MatmulLastAxisMatrixFlat(current, leadSize, lastDim, dTFlat, lastDim);
            }

            // Barycentric contraction along last axis
            int exactIdx = -1;
            for (int i = 0; i < lastDim; i++)
            {
                if (Math.Abs(x - NodeArrays[d][i]) < 1e-14)
                {
                    exactIdx = i;
                    break;
                }
            }

            if (exactIdx >= 0)
            {
                // Exact node: extract every leadSize-th element
                double[] result = new double[leadSize];
                for (int i = 0; i < leadSize; i++)
                    result[i] = current[i * lastDim + exactIdx];
                current = result;
            }
            else
            {
                // Barycentric formula: compute normalized weights inline
                double[] wNorm = new double[lastDim];
                double sumW = 0.0;
                for (int i = 0; i < lastDim; i++)
                {
                    double wod = Weights![d][i] / (x - NodeArrays[d][i]);
                    wNorm[i] = wod;
                    sumW += wod;
                }
                double invSumW = 1.0 / sumW;
                for (int i = 0; i < lastDim; i++)
                    wNorm[i] *= invSumW;

                current = BarycentricKernel.MatmulLastAxis(current, leadSize, lastDim, wNorm);
            }

            totalSize = leadSize;
        }

        return current[0];
    }

    /// <summary>
    /// Evaluate at multiple points.
    /// </summary>
    /// <param name="points">Points as double[N][numDimensions].</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Results array of length N.</returns>
    public double[] VectorizedEvalBatch(double[][] points, int[] derivativeOrder)
    {
        double[] results = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
            results[i] = VectorizedEval(points[i], derivativeOrder);
        return results;
    }

    /// <summary>
    /// Evaluate multiple derivative orders at the same point, sharing barycentric weights.
    /// </summary>
    /// <param name="point">Query point.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.</param>
    /// <returns>One result per derivative order.</returns>
    public double[] VectorizedEvalMulti(double[] point, int[][] derivativeOrders)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        // Pre-compute dimension info (shared across all derivative orders)
        var dimInfo = new (bool isExact, int exactIdx, double[]? wNorm)[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            double x = point[d];
            int exactIdx = -1;
            for (int i = 0; i < NNodes[d]; i++)
            {
                if (Math.Abs(x - NodeArrays[d][i]) < 1e-14)
                {
                    exactIdx = i;
                    break;
                }
            }

            if (exactIdx >= 0)
            {
                dimInfo[d] = (true, exactIdx, null);
            }
            else
            {
                double[] diff = new double[NNodes[d]];
                for (int i = 0; i < NNodes[d]; i++)
                    diff[i] = x - NodeArrays[d][i];

                double[] wOverDiff = new double[NNodes[d]];
                double sumW = 0.0;
                for (int i = 0; i < NNodes[d]; i++)
                {
                    wOverDiff[i] = Weights![d][i] / diff[i];
                    sumW += wOverDiff[i];
                }
                double[] wNorm = new double[NNodes[d]];
                for (int i = 0; i < NNodes[d]; i++)
                    wNorm[i] = wOverDiff[i] / sumW;

                dimInfo[d] = (false, -1, wNorm);
            }
        }

        double[] results = new double[derivativeOrders.Length];
        int tensorSize = TensorValues.Length;

        for (int q = 0; q < derivativeOrders.Length; q++)
        {
            int[] derivOrder = derivativeOrders[q];
            double[] current = TensorValues;
            int totalSize = tensorSize;

            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                int deriv = derivOrder[d];
                int lastDim = NNodes[d];
                int leadSize = totalSize / lastDim;

                if (deriv > 0)
                {
                    double[] dTFlat = DiffMatricesTFlat![d];
                    for (int o = 0; o < deriv; o++)
                        current = BarycentricKernel.MatmulLastAxisMatrixFlat(current, leadSize, lastDim, dTFlat, lastDim);
                }

                var (isExact, exactIdx, wNorm) = dimInfo[d];
                if (isExact)
                {
                    double[] result = new double[leadSize];
                    for (int i = 0; i < leadSize; i++)
                        result[i] = current[i * lastDim + exactIdx];
                    current = result;
                }
                else
                {
                    current = BarycentricKernel.MatmulLastAxis(current, leadSize, lastDim, wNorm!);
                }

                totalSize = leadSize;
            }

            results[q] = current[0];
        }

        return results;
    }

    /// <summary>
    /// Return derivative order as-is (for API compatibility).
    /// </summary>
    public int[] GetDerivativeId(int[] derivativeOrder) => derivativeOrder;

    // ------------------------------------------------------------------
    // Error estimation
    // ------------------------------------------------------------------

    /// <summary>
    /// Estimate the supremum-norm interpolation error using Chebyshev coefficient decay.
    /// </summary>
    /// <returns>Estimated maximum interpolation error.</returns>
    public double ErrorEstimate()
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;

        double totalError = 0.0;
        for (int d = 0; d < NumDimensions; d++)
        {
            double maxErrThisDim = 0.0;

            // Iterate over all indices except dimension d
            int[] otherShape = NNodes.Where((_, i) => i != d).ToArray();
            int otherTotal = 1;
            for (int i = 0; i < otherShape.Length; i++)
                otherTotal *= otherShape[i];

            for (int otherFlat = 0; otherFlat < otherTotal; otherFlat++)
            {
                // Extract 1D slice along dimension d
                double[] values1d = Extract1DSlice(TensorValues, NNodes, d, otherFlat, otherShape);
                double[] coeffs = BarycentricKernel.ChebyshevCoefficients1D(values1d);
                double lastCoeff = Math.Abs(coeffs[coeffs.Length - 1]);
                if (lastCoeff > maxErrThisDim)
                    maxErrThisDim = lastCoeff;
            }
            totalError += maxErrThisDim;
        }

        _cachedErrorEstimate = totalError;
        return totalError;
    }

    /// <summary>
    /// Get Chebyshev coefficients for a 1D array of values at Type I nodes.
    /// Public for testing.
    /// </summary>
    public static double[] ChebyshevCoefficients1D(double[] values)
    {
        return BarycentricKernel.ChebyshevCoefficients1D(values);
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// <summary>
    /// Save the built interpolant to a file using JSON serialization.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    public void Save(string path)
    {
        if (TensorValues == null)
            throw new InvalidOperationException(
                "Cannot save an unbuilt interpolant. Call Build() first.");

        var state = new SerializationState
        {
            NumDimensions = NumDimensions,
            Domain = Domain,
            NNodes = NNodes,
            MaxDerivativeOrder = MaxDerivativeOrder,
            NodeArrays = NodeArrays,
            TensorValues = TensorValues,
            Weights = Weights!,
            DiffMatrices = DiffMatrices!.Select(Flatten2D).ToArray(),
            BuildTime = BuildTime,
            NEvaluations = NEvaluations,
            Version = "0.1.0"
        };

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load a previously saved interpolant from a file.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>The restored interpolant.</returns>
    public static ChebyshevApproximation Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize");

        var obj = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = state.NumDimensions,
            Domain = state.Domain,
            NNodes = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder,
            NodeArrays = state.NodeArrays,
            TensorValues = state.TensorValues,
            Weights = state.Weights,
            BuildTime = state.BuildTime,
            NEvaluations = state.NEvaluations,
            _cachedErrorEstimate = null,
        };

        // Reconstruct diff matrices from flat arrays
        obj.DiffMatrices = new double[state.NumDimensions][,];
        for (int d = 0; d < state.NumDimensions; d++)
        {
            int n = state.NNodes[d];
            obj.DiffMatrices[d] = Unflatten2D(state.DiffMatrices[d], n, n);
        }

        obj.PrecomputeTransposedDiffMatrices();

        return obj;
    }

    // ------------------------------------------------------------------
    // Static factories
    // ------------------------------------------------------------------

    /// <summary>
    /// Generate Chebyshev nodes without evaluating any function.
    /// </summary>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Lower and upper bounds for each dimension.</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <returns>Dictionary with "NodesPerDim", "FullGrid", and "Shape".</returns>
    public static NodeInfo Nodes(int numDimensions, double[][] domain, int[] nNodes)
    {
        if (domain.Length != numDimensions || nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"len(domain)={domain.Length} and len(nNodes)={nNodes.Length} " +
                $"must both equal numDimensions={numDimensions}");

        double[][] nodesPerDim = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            nodesPerDim[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);

        // Build full grid (Cartesian product, C-order)
        int totalPoints = 1;
        for (int d = 0; d < numDimensions; d++)
            totalPoints *= nNodes[d];

        double[][] fullGrid = new double[totalPoints][];
        int[] indices = new int[numDimensions];
        for (int flat = 0; flat < totalPoints; flat++)
        {
            double[] gridPoint = new double[numDimensions];
            int rem = flat;
            for (int d = numDimensions - 1; d >= 0; d--)
            {
                indices[d] = rem % nNodes[d];
                rem /= nNodes[d];
            }
            for (int d = 0; d < numDimensions; d++)
                gridPoint[d] = nodesPerDim[d][indices[d]];
            fullGrid[flat] = gridPoint;
        }

        return new NodeInfo
        {
            NodesPerDim = nodesPerDim,
            FullGrid = fullGrid,
            Shape = (int[])nNodes.Clone()
        };
    }

    /// <summary>
    /// Create an interpolant from pre-computed function values.
    /// </summary>
    public static ChebyshevApproximation FromValues(
        double[] tensorValues,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxDerivativeOrder = 2)
    {
        // Validation
        if (domain.Length != numDimensions || nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"len(domain)={domain.Length} and len(nNodes)={nNodes.Length} " +
                $"must both equal numDimensions={numDimensions}");

        int expectedTotal = 1;
        for (int d = 0; d < numDimensions; d++)
            expectedTotal *= nNodes[d];

        if (tensorValues.Length != expectedTotal)
            throw new ArgumentException(
                $"tensor_values.shape=({tensorValues.Length}) does not match " +
                $"n_nodes=({string.Join(", ", nNodes)})");

        for (int i = 0; i < tensorValues.Length; i++)
        {
            if (double.IsNaN(tensorValues[i]) || double.IsInfinity(tensorValues[i]))
                throw new ArgumentException("tensor_values contains NaN or Inf");
        }

        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d][0] >= domain[d][1])
                throw new ArgumentException(
                    $"domain[{d}]: lo={domain[d][0]} must be strictly less than hi={domain[d][1]}");
        }

        var obj = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = numDimensions,
            Domain = domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])nNodes.Clone(),
            MaxDerivativeOrder = maxDerivativeOrder,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };

        // Chebyshev nodes
        obj.NodeArrays = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            obj.NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);

        obj.TensorValues = (double[])tensorValues.Clone();

        // Barycentric weights
        obj.Weights = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            obj.Weights[d] = BarycentricKernel.ComputeBarycentricWeights(obj.NodeArrays[d]);

        // Differentiation matrices
        obj.DiffMatrices = new double[numDimensions][,];
        for (int d = 0; d < numDimensions; d++)
            obj.DiffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(obj.NodeArrays[d], obj.Weights[d]);

        obj.PrecomputeTransposedDiffMatrices();

        return obj;
    }

    /// <summary>
    /// Create a new instance sharing grid data from source with new tensor values.
    /// Internal factory for arithmetic operators.
    /// </summary>
    internal static ChebyshevApproximation FromGrid(ChebyshevApproximation source, double[] tensorValues)
    {
        return new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = source.NumDimensions,
            Domain = source.Domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])source.NNodes.Clone(),
            MaxDerivativeOrder = source.MaxDerivativeOrder,
            NodeArrays = source.NodeArrays,
            Weights = source.Weights,
            DiffMatrices = source.DiffMatrices,
            DiffMatricesTFlat = source.DiffMatricesTFlat,
            TensorValues = tensorValues,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };
    }

    // ------------------------------------------------------------------
    // Extrusion and slicing
    // ------------------------------------------------------------------

    /// <summary>
    /// Add new dimensions where the function is constant.
    /// </summary>
    public ChebyshevApproximation Extrude(params (int dimIndex, double[] bounds, int nNodes)[] extrudeParams)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var sorted = ExtrudeSlice.NormalizeExtrusionParams(extrudeParams, NumDimensions);

        double[] tensor = (double[])TensorValues.Clone();
        int[] shape = (int[])NNodes.Clone();
        var nodes = NodeArrays.ToList();
        var weights = Weights!.ToList();
        var diffMats = DiffMatrices!.ToList();
        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();

        foreach (var (dimIdx, bounds, n) in sorted)
        {
            tensor = ExtrudeSlice.ExtrudeTensor(tensor, shape, dimIdx, n);
            // Update shape
            var shapeList = shape.ToList();
            shapeList.Insert(dimIdx, n);
            shape = shapeList.ToArray();

            var newNodes = BarycentricKernel.MakeNodesForDim(bounds[0], bounds[1], n);
            var newWeights = BarycentricKernel.ComputeBarycentricWeights(newNodes);
            var newDiffMat = BarycentricKernel.ComputeDifferentiationMatrix(newNodes, newWeights);
            nodes.Insert(dimIdx, newNodes);
            weights.Insert(dimIdx, newWeights);
            diffMats.Insert(dimIdx, newDiffMat);
            domain.Insert(dimIdx, (double[])bounds.Clone());
            nNodes.Insert(dimIdx, n);
        }

        int newNdim = NumDimensions + sorted.Length;
        var result = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = newNdim,
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            NodeArrays = nodes.ToArray(),
            Weights = weights.ToArray(),
            DiffMatrices = diffMats.ToArray(),
            TensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };
        result.PrecomputeTransposedDiffMatrices();
        return result;
    }

    /// <summary>
    /// Fix one or more dimensions at given values, reducing dimensionality.
    /// </summary>
    public ChebyshevApproximation Slice(params (int dimIndex, double value)[] sliceParams)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var sorted = ExtrudeSlice.NormalizeSlicingParams(sliceParams, NumDimensions);

        // Validate values within domain
        foreach (var (dimIdx, value) in sorted)
        {
            double lo = Domain[dimIdx][0];
            double hi = Domain[dimIdx][1];
            if (value < lo || value > hi)
                throw new ArgumentException(
                    $"Slice value {value} for dim {dimIdx} is outside domain [{lo}, {hi}]");
        }

        double[] tensor = (double[])TensorValues.Clone();
        int[] shape = (int[])NNodes.Clone();
        var nodes = NodeArrays.ToList();
        var weights = Weights!.ToList();
        var diffMats = DiffMatrices!.ToList();
        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();

        foreach (var (dimIdx, value) in sorted)
        {
            tensor = ExtrudeSlice.SliceTensor(tensor, shape, dimIdx, nodes[dimIdx], weights[dimIdx], value);

            // Update shape
            var shapeList = shape.ToList();
            shapeList.RemoveAt(dimIdx);
            shape = shapeList.ToArray();

            nodes.RemoveAt(dimIdx);
            weights.RemoveAt(dimIdx);
            diffMats.RemoveAt(dimIdx);
            domain.RemoveAt(dimIdx);
            nNodes.RemoveAt(dimIdx);
        }

        int newNdim = NumDimensions - sorted.Length;
        var result = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = newNdim,
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            NodeArrays = nodes.ToArray(),
            Weights = weights.ToArray(),
            DiffMatrices = diffMats.ToArray(),
            TensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };
        result.PrecomputeTransposedDiffMatrices();
        return result;
    }

    // ------------------------------------------------------------------
    // Calculus: integration, roots, optimization
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the interpolant over one or more dimensions.
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all.</param>
    /// <param name="bounds">Sub-interval bounds per dim. Null = full domain.</param>
    /// <returns>Scalar if all dims integrated, otherwise a lower-dimensional interpolant.</returns>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        if (dims == null)
            dims = Enumerable.Range(0, NumDimensions).ToArray();

        var sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= NumDimensions)
                throw new ArgumentException($"dim {d} out of range [0, {NumDimensions - 1}]");
        }

        var perDimBounds = Calculus.NormalizeBounds(sortedDims, bounds, Domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        double[] tensor = (double[])TensorValues.Clone();
        int[] shape = (int[])NNodes.Clone();
        var nodes = NodeArrays.ToList();
        var wts = Weights!.ToList();
        var diffMats = DiffMatrices!.ToList();
        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();

        // Process dimensions in descending order
        foreach (int d in sortedDims.OrderByDescending(x => x))
        {
            double a = domain[d][0];
            double b = domain[d][1];
            double scale = (b - a) / 2.0;
            var bd = perDimBounds[dimToIdx[d]];

            double[] quadW;
            if (bd == null)
            {
                quadW = Calculus.ComputeFejer1Weights(nNodes[d]);
            }
            else
            {
                double tLo = 2.0 * (bd.Value.lo - a) / (b - a) - 1.0;
                double tHi = 2.0 * (bd.Value.hi - a) / (b - a) - 1.0;
                quadW = Calculus.ComputeSubIntervalWeights(nNodes[d], tLo, tHi);
            }

            // Scale weights
            double[] scaledW = new double[quadW.Length];
            for (int i = 0; i < quadW.Length; i++)
                scaledW[i] = quadW[i] * scale;

            tensor = BarycentricKernel.TensordotVector(tensor, shape, d, scaledW);

            // Update shape
            var shapeList = shape.ToList();
            shapeList.RemoveAt(d);
            shape = shapeList.ToArray();

            nodes.RemoveAt(d);
            wts.RemoveAt(d);
            diffMats.RemoveAt(d);
            domain.RemoveAt(d);
            nNodes.RemoveAt(d);
        }

        int newNdim = NumDimensions - sortedDims.Length;
        if (newNdim == 0)
            return tensor[0];

        return new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = newNdim,
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            NodeArrays = nodes.ToArray(),
            Weights = wts.ToArray(),
            DiffMatrices = diffMats.ToArray(),
            TensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };
    }

    /// <summary>
    /// Find all roots of the interpolant along a specified dimension.
    /// </summary>
    public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);

        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
        return Calculus.Roots1D(sliced.TensorValues!, sliced.Domain[0]);
    }

    /// <summary>
    /// Find the minimum value of the interpolant along a dimension.
    /// </summary>
    public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);
        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        return Calculus.Optimize1D(
            sliced.TensorValues!, sliced.NodeArrays[0], sliced.Weights![0],
            sliced.DiffMatrices![0], sliced.Domain[0], "min");
    }

    /// <summary>
    /// Find the maximum value of the interpolant along a dimension.
    /// </summary>
    public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);
        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        return Calculus.Optimize1D(
            sliced.TensorValues!, sliced.NodeArrays[0], sliced.Weights![0],
            sliced.DiffMatrices![0], sliced.Domain[0], "max");
    }

    // ------------------------------------------------------------------
    // Arithmetic operators
    // ------------------------------------------------------------------

    /// <summary>Add two interpolants with the same grid.</summary>
    public static ChebyshevApproximation operator +(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException("Cannot combine different types");
        Algebra.CheckCompatible(a, b);
        double[] newValues = new double[a.TensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a.TensorValues[i] + b.TensorValues![i];
        return FromGrid(a, newValues);
    }

    /// <summary>Subtract two interpolants with the same grid.</summary>
    public static ChebyshevApproximation operator -(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException("Cannot combine different types");
        Algebra.CheckCompatible(a, b);
        double[] newValues = new double[a.TensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a.TensorValues[i] - b.TensorValues![i];
        return FromGrid(a, newValues);
    }

    /// <summary>Multiply interpolant by a scalar.</summary>
    public static ChebyshevApproximation operator *(ChebyshevApproximation a, double scalar)
    {
        double[] newValues = new double[a.TensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a.TensorValues[i] * scalar;
        return FromGrid(a, newValues);
    }

    /// <summary>Multiply scalar by interpolant.</summary>
    public static ChebyshevApproximation operator *(double scalar, ChebyshevApproximation a)
    {
        return a * scalar;
    }

    /// <summary>Divide interpolant by a scalar.</summary>
    public static ChebyshevApproximation operator /(ChebyshevApproximation a, double scalar)
    {
        return a * (1.0 / scalar);
    }

    /// <summary>Negate interpolant.</summary>
    public static ChebyshevApproximation operator -(ChebyshevApproximation a)
    {
        return a * -1.0;
    }

    // ------------------------------------------------------------------
    // Printing
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    public override string ToString()
    {
        bool built = TensorValues != null;
        int totalNodes = 1;
        for (int d = 0; d < NumDimensions; d++)
            totalNodes *= NNodes[d];
        string status = built ? "built" : "not built";

        int maxDisplay = 6;
        string nodesStr, domainStr;
        if (NumDimensions > maxDisplay)
        {
            nodesStr = "[" + string.Join(", ", NNodes.Take(maxDisplay)) + ", ...]";
            domainStr = string.Join(" x ", Domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        }
        else
        {
            nodesStr = "[" + string.Join(", ", NNodes) + "]";
            domainStr = string.Join(" x ", Domain.Select(d => $"[{d[0]}, {d[1]}]"));
        }

        var sb = new StringBuilder();
        sb.AppendLine($"ChebyshevApproximation ({NumDimensions}D, {status})");
        sb.AppendLine($"  Nodes:       {nodesStr} ({totalNodes:N0} total)");
        sb.AppendLine($"  Domain:      {domainStr}");

        if (built)
        {
            sb.AppendLine($"  Build:       {BuildTime:F3}s, {NEvaluations:N0} evaluations");
            sb.AppendLine($"  Error est:   {ErrorEstimate():E2}");
        }

        sb.Append($"  Derivatives: up to order {MaxDerivativeOrder}");
        return sb.ToString();
    }

    /// <summary>
    /// Compact representation of the interpolant.
    /// </summary>
    public string ToReprString()
    {
        bool built = TensorValues != null;
        return $"ChebyshevApproximation(dims={NumDimensions}, nodes=[{string.Join(", ", NNodes)}], built={built})";
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// <summary>
    /// Pre-compute transposed diff matrices as flat arrays for BLAS GEMM.
    /// Called after DiffMatrices is set in Build, FromValues, Load, Extrude, Slice.
    /// </summary>
    private void PrecomputeTransposedDiffMatrices()
    {
        if (DiffMatrices == null) return;
        DiffMatricesTFlat = new double[DiffMatrices.Length][];
        for (int d = 0; d < DiffMatrices.Length; d++)
        {
            int rows = DiffMatrices[d].GetLength(0);
            int cols = DiffMatrices[d].GetLength(1);
            // Transpose and flatten in one pass (row-major)
            var flat = new double[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[j * rows + i] = DiffMatrices[d][i, j];
            DiffMatricesTFlat[d] = flat;
        }
    }

    private static double[] Extract1DSlice(double[] data, int[] shape, int dim, int otherFlat, int[] otherShape)
    {
        int ndim = shape.Length;
        int nDim = shape[dim];
        double[] slice = new double[nDim];

        // Compute strides
        int[] strides = new int[ndim];
        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        // Decompose otherFlat into multi-index for other dimensions
        int[] otherStrides = new int[otherShape.Length];
        if (otherShape.Length > 0)
        {
            otherStrides[otherShape.Length - 1] = 1;
            for (int i = otherShape.Length - 2; i >= 0; i--)
                otherStrides[i] = otherStrides[i + 1] * otherShape[i + 1];
        }

        int baseIdx = 0;
        int remaining = otherFlat;
        int otherDim = 0;
        for (int d = 0; d < ndim; d++)
        {
            if (d == dim)
                continue;
            int coord = remaining / otherStrides[otherDim];
            remaining %= otherStrides[otherDim];
            baseIdx += coord * strides[d];
            otherDim++;
        }

        for (int k = 0; k < nDim; k++)
            slice[k] = data[baseIdx + k * strides[dim]];

        return slice;
    }

    private static double[] Flatten2D(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = matrix[i, j];
        return flat;
    }

    private static double[,] Unflatten2D(double[] flat, int rows, int cols)
    {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = flat[i * cols + j];
        return matrix;
    }

    // ------------------------------------------------------------------
    // Serialization state
    // ------------------------------------------------------------------

    internal class SerializationState
    {
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int MaxDerivativeOrder { get; set; }
        public double[][] NodeArrays { get; set; } = Array.Empty<double[]>();
        public double[] TensorValues { get; set; } = Array.Empty<double>();
        public double[][] Weights { get; set; } = Array.Empty<double[]>();
        public double[][] DiffMatrices { get; set; } = Array.Empty<double[]>();
        public double BuildTime { get; set; }
        public int NEvaluations { get; set; }
        public string Version { get; set; } = "";
    }
}

/// <summary>
/// Information about Chebyshev nodes for a given configuration.
/// </summary>
public class NodeInfo
{
    /// <summary>Chebyshev nodes for each dimension, sorted ascending.</summary>
    public double[][] NodesPerDim { get; set; } = Array.Empty<double[]>();

    /// <summary>Full grid (Cartesian product of all nodes), shape (totalPoints, numDimensions).</summary>
    public double[][] FullGrid { get; set; } = Array.Empty<double[]>();

    /// <summary>Expected tensor shape (== nNodes).</summary>
    public int[] Shape { get; set; } = Array.Empty<int>();
}
