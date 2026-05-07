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
    private double[][] _domain = Array.Empty<double[]>();
    private int[] _nNodes = Array.Empty<int>();
    private double[][] _nodeArrays = Array.Empty<double[]>();
    private double[]? _tensorValues;
    private double[][]? _weights;
    private double[][,]? _diffMatrices;
    private double[][]? _diffMatricesTFlat;

    /// <summary>The function to approximate. Null after load or from_values.</summary>
    public Func<double[], object?, double>? Function { get; internal set; }

    /// <summary>Number of input dimensions.</summary>
    public int NumDimensions { get; internal set; }

    /// <summary>Domain bounds for each dimension, as list of [lo, hi].</summary>
    public double[][] Domain
    {
        get => CloneHelpers.DeepCopy(_domain)!;
        internal set => _domain = value ?? Array.Empty<double[]>();
    }

    /// <summary>Number of Chebyshev nodes per dimension.</summary>
    public int[] NNodes
    {
        get => CloneHelpers.DeepCopy(_nNodes)!;
        internal set => _nNodes = value ?? Array.Empty<int>();
    }

    /// <summary>Maximum supported derivative order.</summary>
    public int MaxDerivativeOrder { get; internal set; } = 2;

    /// <summary>Chebyshev nodes per dimension, each sorted ascending.</summary>
    public double[][] NodeArrays
    {
        get => CloneHelpers.DeepCopy(_nodeArrays)!;
        internal set => _nodeArrays = value ?? Array.Empty<double[]>();
    }

    /// <summary>Flat tensor of function values at all node combinations (C-order).</summary>
    public double[]? TensorValues
    {
        get => CloneHelpers.DeepCopy(_tensorValues);
        internal set => _tensorValues = value;
    }

    /// <summary>Barycentric weights per dimension.</summary>
    public double[][]? Weights
    {
        get => CloneHelpers.DeepCopy(_weights);
        internal set => _weights = value;
    }

    /// <summary>Spectral differentiation matrices per dimension.</summary>
    public double[][,]? DiffMatrices
    {
        get => CloneHelpers.DeepCopy(_diffMatrices);
        internal set => _diffMatrices = value;
    }

    /// <summary>Pre-transposed diff matrices flattened to double[] for BLAS GEMM (row-major).</summary>
    internal double[][]? DiffMatricesTFlat
    {
        get => _diffMatricesTFlat;
        set => _diffMatricesTFlat = value;
    }

    internal double[][] DomainStorage
    {
        get => _domain;
        set => _domain = value;
    }

    internal int[] NNodesStorage
    {
        get => _nNodes;
        set => _nNodes = value;
    }

    internal double[][] NodeArraysStorage
    {
        get => _nodeArrays;
        set => _nodeArrays = value;
    }

    internal double[]? TensorValuesStorage
    {
        get => _tensorValues;
        set => _tensorValues = value;
    }

    internal double[][]? WeightsStorage
    {
        get => _weights;
        set => _weights = value;
    }

    internal double[][,]? DiffMatricesStorage
    {
        get => _diffMatrices;
        set => _diffMatrices = value;
    }

    /// <summary>Time taken by Build() in seconds.</summary>
    public double BuildTime { get; internal set; }

    /// <summary>Number of function evaluations during Build().</summary>
    public int NEvaluations { get; internal set; }

    /// <summary>Target supremum-norm error for auto-N construction. Null in fixed-N mode.</summary>
    public double? ErrorThreshold { get; internal set; }

    /// <summary>Maximum nodes per dimension for the auto-N doubling loop. Default 64.</summary>
    public int MaxN { get; internal set; } = 64;

    /// <summary>Warning emitted by Build() if maxN was reached before errorThreshold was satisfied. Null otherwise.</summary>
    public string? BuildWarning { get; internal set; }

    /// <summary>The user's original nNodes argument with null sentinels intact, used to dispatch a re-run of the doubling loop on a second Build() call.</summary>
    internal int?[] OriginalNNodes { get; set; } = Array.Empty<int?>();

    private double? _cachedErrorEstimate;
    private string? _descriptor;
    private string _constructorType = "function";
    private bool _isConstructionFinished;
    private object? _additionalData;
    private int? _nWorkers;
    private IProgress<int>? _progress;
    private double[]? _evaluationPointsCache;
    private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
    private readonly List<int[]> _registeredDerivativeOrders = new();
#pragma warning disable CS0649  // Used by future Clone() and JSON round-trip
    private double[][]? _specialPoints;
#pragma warning restore CS0649

    /// <summary>Internal hook for AdaptiveBuild to seed the error-estimate cache after each iteration.</summary>
    internal void SetCachedErrorEstimate(double value) => _cachedErrorEstimate = value;

    /// <summary>
    /// Create a new ChebyshevApproximation.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -> double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="deferBuild">If true, skip eager node materialization. Call <see cref="SetOriginalFunctionValues"/> to finish construction.</param>
    /// <param name="nWorkers">Number of parallel workers for Build(): null (sequential), -1 (all cores), or positive int. Mirrors PyChebyshev v0.19 <c>n_workers</c>.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count 1..N during Build().</param>
    /// <remarks>
    /// When <paramref name="nWorkers"/> is non-null, <paramref name="function"/> may be
    /// invoked concurrently from multiple threads via <c>Parallel.For</c>. Functions that
    /// capture mutable state must use locks or external synchronization, or pass
    /// <c>nWorkers: null</c> (the default).
    /// </remarks>
    public ChebyshevApproximation(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        bool deferBuild = false,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        ValidateFixedGridArguments(nameof(ChebyshevApproximation), numDimensions, domain, nNodes);
        ArgumentOutOfRangeException.ThrowIfNegative(maxDerivativeOrder);

        Function = function;
        NumDimensions = numDimensions;
        _domain = domain.Select(d => (double[])d.Clone()).ToArray();
        _nNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;

        if (!deferBuild)
        {
            // Generate Chebyshev nodes for each dimension
            _nodeArrays = new double[numDimensions][];
            for (int d = 0; d < numDimensions; d++)
            {
                _nodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
            }
        }
        else
        {
            _nodeArrays = Array.Empty<double[]>();
        }
    }

    // Internal parameterless constructor for factories
    internal ChebyshevApproximation() { }

    /// <summary>
    /// Create a new ChebyshevApproximation with optional error-driven auto-N construction.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -&gt; double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension; null entries signal auto-N for that dim. Pass null to make every dim auto-N (requires errorThreshold).</param>
    /// <param name="errorThreshold">Finite positive target supremum-norm error. Required if any nNodes entry is null.</param>
    /// <param name="maxN">Cap on nodes per dimension during the doubling loop (default 64, must be at least 3).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="nWorkers">Number of parallel workers for Build(): null (sequential), -1 (all cores), or positive int. Mirrors PyChebyshev v0.19 <c>n_workers</c>.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count 1..N during Build().</param>
    /// <remarks>
    /// When <paramref name="nWorkers"/> is non-null, <paramref name="function"/> may be
    /// invoked concurrently from multiple threads via <c>Parallel.For</c>. Functions that
    /// capture mutable state must use locks or external synchronization, or pass
    /// <c>nWorkers: null</c> (the default).
    /// </remarks>
    public ChebyshevApproximation(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int?[]? nNodes = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        if (maxN < 3)
            throw new ArgumentException(
                $"maxN must be at least 3 (the initial N of the doubling loop), got maxN={maxN}. " +
                "For a grid smaller than 3 per dimension, pass nNodes explicitly.");
        ArgumentOutOfRangeException.ThrowIfNegative(maxDerivativeOrder);
        AdaptiveBuild.ValidateErrorThreshold(errorThreshold);

        ValidateDomainArguments(nameof(ChebyshevApproximation), numDimensions, domain);

        // Normalize nNodes: null array means "all dims auto-N"
        int?[] resolved;
        if (nNodes == null)
        {
            if (errorThreshold == null)
                throw new ArgumentException(
                    "Must provide either nNodes (explicit) or errorThreshold (auto-N). Got neither.");
            resolved = new int?[numDimensions];
        }
        else
        {
            resolved = (int?[])nNodes.Clone();
            if (resolved.Length != numDimensions)
                throw new ArgumentException(
                    $"len(nNodes)={resolved.Length} must equal numDimensions={numDimensions}");
            if (resolved.Any(n => n == null) && errorThreshold == null)
                throw new ArgumentException(
                    "Null entries in nNodes require errorThreshold to be set (auto-N mode).");
        }

        Function = function;
        NumDimensions = numDimensions;
        _domain = domain.Select(d => (double[])d.Clone()).ToArray();
        ErrorThreshold = errorThreshold;
        MaxN = maxN;
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        OriginalNNodes = (int?[])resolved.Clone();
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;

        // If all entries are non-null, populate resolved node counts immediately.
        if (resolved.All(n => n != null))
        {
            _nNodes = resolved.Select(n => n!.Value).ToArray();
            ValidateNodeCounts(nameof(ChebyshevApproximation), _nNodes);
            _nodeArrays = new double[numDimensions][];
            for (int d = 0; d < numDimensions; d++)
                _nodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], _nNodes[d]);
        }
        else
        {
            // Auto-N path: resolved node counts are empty until Build().
            _nNodes = Array.Empty<int>();
            _nodeArrays = Array.Empty<double[]>();
        }
    }

    /// <summary>
    /// Build the Chebyshev approximation. Dispatches to the doubling loop if any
    /// dimension was constructed with a null entry in nNodes (auto-N), otherwise
    /// builds on the resolved fixed grid.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    public void Build(bool verbose = true)
    {
        if (Function == null)
            throw new InvalidOperationException(
                "Cannot build: no function assigned. " +
                "This object was created via FromValues() or Load().");

        if (OriginalNNodes.Length > 0 && OriginalNNodes.Any(n => n == null))
        {
            AdaptiveBuild.RunDoublingLoop(this, verbose);
            return;
        }

        BuildFixedGrid(verbose);
    }

    /// <summary>
    /// Build on the already-resolved (all-int) grid. The original Build() body,
    /// extracted so the doubling loop can call it once per iteration.
    /// </summary>
    internal void BuildFixedGrid(bool verbose = true)
    {
        int total = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(_nNodes, nameof(BuildFixedGrid)),
            nameof(BuildFixedGrid),
            _nNodes);

        if (verbose)
            Console.WriteLine($"Building {NumDimensions}D Chebyshev approximation ({total:N0} evaluations)...");

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        // Step 1: Materialize the full points array (C-order / ndindex), then
        // evaluate sequentially or in parallel via ParallelBuild.
        var points = new double[total][];
        int[] indices = new int[NumDimensions];
        for (int flat = 0; flat < total; flat++)
        {
            int rem = flat;
            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                indices[d] = rem % _nNodes[d];
                rem /= _nNodes[d];
            }
            var pt = new double[NumDimensions];
            for (int d = 0; d < NumDimensions; d++)
                pt[d] = _nodeArrays[d][indices[d]];
            points[flat] = pt;
        }
        var tensorValues = Internal.ParallelBuild.EvaluateInParallel(
            Function!, points, _additionalData, _nWorkers, _progress);
        ValidateFiniteBuildValues(tensorValues);
        _tensorValues = tensorValues;
        NEvaluations = total;

        // Step 2: Pre-compute barycentric weights
        _weights = new double[NumDimensions][];
        for (int d = 0; d < NumDimensions; d++)
            _weights[d] = BarycentricKernel.ComputeBarycentricWeights(_nodeArrays[d]);

        // Step 3: Pre-compute differentiation matrices
        _diffMatrices = new double[NumDimensions][,];
        for (int d = 0; d < NumDimensions; d++)
            _diffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(_nodeArrays[d], _weights[d]);

        // Step 4: Pre-transpose diff matrices for VectorizedEval
        PrecomputeTransposedDiffMatrices();

        sw.Stop();
        BuildTime = sw.Elapsed.TotalSeconds;

        if (verbose)
        {
            int totalWeights = _weights.Sum(w => w.Length);
            Console.WriteLine($"  Built in {BuildTime:F3}s ({totalWeights} weights, {totalWeights * 8} bytes)");
        }

        _isConstructionFinished = true;
    }

    private static void ValidateFiniteBuildValues(double[] values)
    {
        int badCount = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (!double.IsFinite(values[i]))
                badCount++;
        }

        if (badCount > 0)
            throw new ArgumentException(
                $"function returned non-finite values at {badCount} grid point(s); " +
                "build cannot proceed with NaN/Infinity in TensorValues");
    }

    /// <summary>
    /// Evaluate using dimensional decomposition with barycentric interpolation.
    /// Loop-based implementation matching Python eval().
    /// </summary>
    /// <param name="point">Query point inside the declared domain, one coordinate per dimension.</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Interpolated value or derivative at the query point.</returns>
    public double Eval(double[] point, int[] derivativeOrder)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions, maxDerivativeOrder: MaxDerivativeOrder);

        // Current working data and its shape
        double[] current = _tensorValues;
        int[] currentShape = (int[])_nNodes.Clone();

        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            double x = point[d];
            int deriv = derivativeOrder[d];
            double[] nodes = _nodeArrays[d];
            double[] weights = _weights![d];
            double[,] diffMatrix = _diffMatrices![d];
            int nNodesD = _nNodes[d];

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
                int leadSize = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(currentShape.Take(d), nameof(Eval)),
                    nameof(Eval));

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
    /// Evaluate the function value (no derivatives) at the given point.
    /// Convenience overload equivalent to <c>Eval(point, new int[NumDimensions])</c>.
    /// </summary>
    /// <param name="point">Query point inside the declared domain, one coordinate per dimension.</param>
    /// <returns>Interpolated value at the query point.</returns>
    public double Eval(double[] point)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException(
                "Cannot evaluate an unbuilt interpolant. Call Build() or SetOriginalFunctionValues() first.");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        return Eval(point, new int[NumDimensions]);
    }

    /// <summary>
    /// Fully vectorized evaluation using matrix operations.
    /// Replaces the Python loop with BLAS-style matrix-vector products.
    /// </summary>
    /// <param name="point">Query point inside the declared domain, one coordinate per dimension.</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Interpolated value or derivative.</returns>
    public double VectorizedEval(double[] point, int[] derivativeOrder)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions, maxDerivativeOrder: MaxDerivativeOrder);

        double[] current = _tensorValues;

        // Track tensor dimensions without shape array allocations.
        // leadSize = product of all dims before current last dim.
        // After contracting dim d, leadSize shrinks accordingly.
        int totalSize = current.Length;

        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            double x = point[d];
            int deriv = derivativeOrder[d];
            int lastDim = _nNodes[d];
            int leadSize = totalSize / lastDim;

            // Apply differentiation matrix if derivative order > 0
            if (deriv > 0)
            {
                double[] dTFlat = _diffMatricesTFlat![d];
                for (int o = 0; o < deriv; o++)
                    current = BarycentricKernel.MatmulLastAxisMatrixFlat(current, leadSize, lastDim, dTFlat, lastDim);
            }

            // Barycentric contraction along last axis
            int exactIdx = -1;
            for (int i = 0; i < lastDim; i++)
            {
                if (Math.Abs(x - _nodeArrays[d][i]) < 1e-14)
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
                    double wod = _weights![d][i] / (x - _nodeArrays[d][i]);
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
    /// Hoists derivative-matrix matmuls outside the per-point loop (they are
    /// point-independent), then does only barycentric reductions per point.
    /// </summary>
    /// <param name="points">Points inside the declared domain as double[N][numDimensions].</param>
    /// <param name="derivativeOrder">Derivative order per dimension.</param>
    /// <returns>Results array of length N.</returns>
    public double[] VectorizedEvalBatch(double[][] points, int[] derivativeOrder)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");
        EvaluationArguments.ValidatePointsInDomain(points, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions, maxDerivativeOrder: MaxDerivativeOrder);

        // Hoist: apply all derivative-matrix matmuls once — they are point-independent.
        // Process from last dimension to first to match VectorizedEval ordering.
        double[] tensorWithDerivs = ApplyDerivativePasses(_tensorValues, _nNodes, derivativeOrder);

        double[] results = new double[points.Length];
        int totalSize = _tensorValues.Length;

        for (int i = 0; i < points.Length; i++)
        {
            double[] current = tensorWithDerivs;
            int curSize = totalSize;

            // Per-point: only the barycentric reduction (no derivative passes).
            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                double x = points[i][d];
                int lastDim = _nNodes[d];
                int leadSize = curSize / lastDim;

                // Barycentric contraction along last axis (no diff-matrix here — already hoisted)
                int exactIdx = -1;
                for (int j = 0; j < lastDim; j++)
                {
                    if (Math.Abs(x - _nodeArrays[d][j]) < 1e-14)
                    {
                        exactIdx = j;
                        break;
                    }
                }

                if (exactIdx >= 0)
                {
                    double[] res = new double[leadSize];
                    for (int j = 0; j < leadSize; j++)
                        res[j] = current[j * lastDim + exactIdx];
                    current = res;
                }
                else
                {
                    double[] wNorm = new double[lastDim];
                    double sumW = 0.0;
                    for (int j = 0; j < lastDim; j++)
                    {
                        double wod = _weights![d][j] / (x - _nodeArrays[d][j]);
                        wNorm[j] = wod;
                        sumW += wod;
                    }
                    double invSumW = 1.0 / sumW;
                    for (int j = 0; j < lastDim; j++)
                        wNorm[j] *= invSumW;

                    current = BarycentricKernel.MatmulLastAxis(current, leadSize, lastDim, wNorm);
                }

                curSize = leadSize;
            }

            results[i] = current[0];
        }

        return results;
    }

    /// <summary>
    /// Apply differentiation-matrix passes to the full coefficient tensor (all axes,
    /// shape unchanged). Used to hoist the point-independent part of
    /// <see cref="VectorizedEvalBatch"/> outside the per-point loop.
    /// Mirrors Python's <c>_apply_derivative_passes</c>.
    /// </summary>
    private double[] ApplyDerivativePasses(double[] tensor, int[] shape, int[] derivativeOrder)
    {
        double[] result = tensor;
        // Process from last dimension to first (matches VectorizedEval ordering).
        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            int deriv = derivativeOrder[d];
            if (deriv > 0)
            {
                double[,] dm = _diffMatrices![d];
                for (int o = 0; o < deriv; o++)
                    result = BarycentricKernel.MatmulAlongAxis(result, shape, d, dm);
            }
        }
        return result;
    }

    /// <summary>
    /// Evaluate multiple derivative orders at the same point, sharing barycentric weights.
    /// </summary>
    /// <param name="point">Query point inside the declared domain.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.</param>
    /// <returns>One result per derivative order.</returns>
    public double[] VectorizedEvalMulti(double[] point, int[][] derivativeOrders)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrders(derivativeOrders, NumDimensions, maxDerivativeOrder: MaxDerivativeOrder);

        // Pre-compute dimension info (shared across all derivative orders)
        var dimInfo = new (bool isExact, int exactIdx, double[]? wNorm)[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            double x = point[d];
            int exactIdx = -1;
            for (int i = 0; i < _nNodes[d]; i++)
            {
                if (Math.Abs(x - _nodeArrays[d][i]) < 1e-14)
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
                double[] diff = new double[_nNodes[d]];
                for (int i = 0; i < _nNodes[d]; i++)
                    diff[i] = x - _nodeArrays[d][i];

                double[] wOverDiff = new double[_nNodes[d]];
                double sumW = 0.0;
                for (int i = 0; i < _nNodes[d]; i++)
                {
                    wOverDiff[i] = _weights![d][i] / diff[i];
                    sumW += wOverDiff[i];
                }
                double[] wNorm = new double[_nNodes[d]];
                for (int i = 0; i < _nNodes[d]; i++)
                    wNorm[i] = wOverDiff[i] / sumW;

                dimInfo[d] = (false, -1, wNorm);
            }
        }

        double[] results = new double[derivativeOrders.Length];
        int tensorSize = _tensorValues.Length;

        for (int q = 0; q < derivativeOrders.Length; q++)
        {
            int[] derivOrder = derivativeOrders[q];
            double[] current = _tensorValues;
            int totalSize = tensorSize;

            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                int deriv = derivOrder[d];
                int lastDim = _nNodes[d];
                int leadSize = totalSize / lastDim;

                if (deriv > 0)
                {
                    double[] dTFlat = _diffMatricesTFlat![d];
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

    // ------------------------------------------------------------------
    // Error estimation
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute per-dimension max last-coefficient magnitudes.
    /// Returns one entry per dimension; ErrorEstimate() returns the sum.
    /// Used by the auto-N doubling loop to pick the worst-contributing dim.
    /// </summary>
    /// <returns>Per-dimension last-coefficient magnitudes, one entry per dim.</returns>
    public double[] ErrorEstimatePerDim()
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var perDim = new double[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            double maxErrThisDim = 0.0;
            int[] otherShape = _nNodes.Where((_, i) => i != d).ToArray();
            int otherTotal = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(otherShape, nameof(ErrorEstimatePerDim)),
                nameof(ErrorEstimatePerDim),
                otherShape);

            for (int otherFlat = 0; otherFlat < otherTotal; otherFlat++)
            {
                double[] values1d = Extract1DSlice(_tensorValues, _nNodes, d, otherFlat, otherShape);
                double[] coeffs = BarycentricKernel.ChebyshevCoefficients1D(values1d);
                double lastCoeff = Math.Abs(coeffs[^1]);
                if (lastCoeff > maxErrThisDim)
                    maxErrThisDim = lastCoeff;
            }
            perDim[d] = maxErrThisDim;
        }
        return perDim;
    }

    /// <summary>
    /// Estimate the supremum-norm interpolation error using Chebyshev coefficient decay.
    /// Sums per-dimension max last-coefficient magnitudes.
    /// </summary>
    /// <returns>Estimated maximum interpolation error.</returns>
    public double ErrorEstimate()
    {
        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;
        double total = ErrorEstimatePerDim().Sum();
        _cachedErrorEstimate = total;
        return total;
    }

    /// <summary>Return the error threshold passed to the constructor, or null in fixed-N mode.</summary>
    public double? GetErrorThreshold() => ErrorThreshold;

    /// <summary>
    /// 1-D capacity estimator: the smallest N at which a 1-D Chebyshev build
    /// over <paramref name="domain"/> hits <paramref name="errorThreshold"/>.
    /// Useful as a sizing pass before committing to a multi-dimensional build.
    /// </summary>
    /// <param name="function">Function to approximate; signature f(point[1], data) -&gt; double.</param>
    /// <param name="domain">(lo, hi) bounds for the single dimension.</param>
    /// <param name="errorThreshold">Finite positive target supremum-norm error.</param>
    /// <param name="maxN">Cap on the returned N. Default 64. If the doubling loop cannot achieve <paramref name="errorThreshold"/> within this cap, returns <paramref name="maxN"/> with BuildWarning set on the temporary internal interpolant.</param>
    /// <returns>Resolved N on the single dimension.</returns>
    public static int GetOptimalN1(
        Func<double[], object?, double> function,
        (double lo, double hi) domain,
        double errorThreshold,
        int maxN = 64)
    {
        var cheb = new ChebyshevApproximation(
            function, 1, new[] { new[] { domain.lo, domain.hi } },
            nNodes: null, errorThreshold: errorThreshold, maxN: maxN);
        cheb.Build(verbose: false);
        return cheb._nNodes[0];
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
    /// Save the built interpolant to a file.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    /// <param name="format">"json" (default) or "binary". Binary is the
    /// portable .pcb format readable by C/Rust/Julia consumers.</param>
    public void Save(string path, string format = "json")
    {
        if (_tensorValues == null)
            throw new InvalidOperationException(
                "Cannot save an unbuilt interpolant. Call Build() first.");

        switch (format)
        {
            case "json":
                SaveJson(path);
                break;
            case "binary":
                SaveBinary(path);
                break;
            default:
                throw new ArgumentException(
                    $"Unknown format '{format}'. Expected 'json' or 'binary'.",
                    nameof(format));
        }
    }

    private void SaveJson(string path)
    {
        var state = new SerializationState
        {
            NumDimensions = NumDimensions,
            Domain = _domain,
            NNodes = _nNodes,
            MaxDerivativeOrder = MaxDerivativeOrder,
            NodeArrays = _nodeArrays,
            TensorValues = _tensorValues!,
            Weights = _weights!,
            DiffMatrices = _diffMatrices!.Select(Flatten2D).ToArray(),
            BuildTime = BuildTime,
            NEvaluations = NEvaluations,
            OriginalNNodes = OriginalNNodes,
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            Version = "0.8.0",
            Descriptor = _descriptor,
            SpecialPoints = _specialPoints,
            RegisteredDerivativeOrders = _registeredDerivativeOrders.Count > 0
                ? _registeredDerivativeOrders.Select(o => (int[])o.Clone()).ToArray()
                : null,
        };

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    private void SaveBinary(string path)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        Internal.PcbFormat.WriteHeader(w, Internal.PcbFormat.ClassTagApproximation);
        Internal.PcbFormat.WriteApproximationBody(w, _domain, _nNodes, _tensorValues!);
    }

    /// <summary>
    /// Read the major version byte of a .pcb binary file without deserializing the body.
    /// Useful for forward-compat tooling.
    /// </summary>
    /// <param name="path">Path to a .pcb file.</param>
    /// <returns>The major format version (currently 1).</returns>
    /// <exception cref="FileNotFoundException">Thrown if the path does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown if the file is not a .pcb file
    /// (no magic header) or is shorter than 12 bytes.</exception>
    public static int PeekFormatVersion(string path)
        => Internal.PcbFormat.PeekFormatVersion(path);

    /// <summary>
    /// Load a previously saved interpolant. Auto-detects JSON vs binary .pcb
    /// by sniffing the first 4 bytes for the b"PCB\0" magic.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>The restored interpolant.</returns>
    public static ChebyshevApproximation Load(string path)
    {
        if (Internal.PcbFormat.IsBinary(path))
            return LoadBinary(path);
        return LoadJson(path);
    }

    private static ChebyshevApproximation LoadBinary(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        var header = Internal.PcbFormat.ReadHeader(r);
        if (header.ClassTag != Internal.PcbFormat.ClassTagApproximation)
            throw new InvalidDataException(
                $"binary file class_tag={header.ClassTag} is not ChebyshevApproximation " +
                $"(tag {Internal.PcbFormat.ClassTagApproximation}); " +
                $"call ChebyshevSpline.Load instead if class_tag={Internal.PcbFormat.ClassTagSpline}");

        var (domain, nNodes, tensor) = Internal.PcbFormat.ReadApproximationBody(r);
        var obj = FromValues(tensor, domain.Length, domain, nNodes);
        obj._constructorType = "load";
        return obj;
    }

    private static ChebyshevApproximation LoadJson(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize");

        ValidateJsonState(state);

        var obj = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = state.NumDimensions,
            _domain = state.Domain,
            _nNodes = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder ?? 2,
            _nodeArrays = state.NodeArrays,
            _tensorValues = state.TensorValues,
            _weights = state.Weights,
            BuildTime = state.BuildTime,
            NEvaluations = state.NEvaluations,
            _cachedErrorEstimate = null,
        };

        // Reconstruct diff matrices from flat arrays
        obj._diffMatrices = new double[state.NumDimensions][,];
        for (int d = 0; d < state.NumDimensions; d++)
        {
            int n = state.NNodes[d];
            obj._diffMatrices[d] = Unflatten2D(state.DiffMatrices[d], n, n);
        }

        obj.PrecomputeTransposedDiffMatrices();

        // v0.5.0 migration: OriginalNNodes / ErrorThreshold / MaxN may be absent in older files.
        if (state.OriginalNNodes != null)
            obj.OriginalNNodes = state.OriginalNNodes;
        else
            obj.OriginalNNodes = obj._nNodes.Select(n => (int?)n).ToArray();
        obj.ErrorThreshold = state.ErrorThreshold;
        obj.MaxN = state.MaxN ?? 64;

        // v0.8.0 migration: Descriptor, SpecialPoints, RegisteredDerivativeOrders may be absent in older files.
        obj._descriptor = state.Descriptor;
        obj._specialPoints = state.SpecialPoints;
        if (state.RegisteredDerivativeOrders != null)
        {
            foreach (var orders in state.RegisteredDerivativeOrders)
            {
                var key = new Internal.TupleKey(orders);
                int id = obj._registeredDerivativeOrders.Count;
                obj._registeredDerivativeOrders.Add((int[])orders.Clone());
                obj._derivativeIdRegistry[key] = id;
            }
        }

        // ConstructorType is intentionally NOT restored from state — Load always sets "load".
        obj._constructorType = "load";
        obj._isConstructionFinished = true;

        return obj;
    }

    private static void ValidateJsonState(SerializationState state)
    {
        int ndim = state.NumDimensions;
        if (ndim < 1)
            throw MalformedJson($"NumDimensions must be >= 1, got {ndim}");
        int maxDerivativeOrder = state.MaxDerivativeOrder ?? 2;
        if (maxDerivativeOrder < 0)
            throw MalformedJson($"MaxDerivativeOrder must be non-negative, got {maxDerivativeOrder}");

        RequireLength(state.Domain, ndim, nameof(SerializationState.Domain));
        RequireLength(state.NNodes, ndim, nameof(SerializationState.NNodes));
        RequireLength(state.NodeArrays, ndim, nameof(SerializationState.NodeArrays));
        RequireLength(state.Weights, ndim, nameof(SerializationState.Weights));
        RequireLength(state.DiffMatrices, ndim, nameof(SerializationState.DiffMatrices));

        int total = 1;
        for (int d = 0; d < ndim; d++)
        {
            double[] domain = state.Domain[d]
                ?? throw MalformedJson($"Domain[{d}] must not be null");
            if (domain.Length != 2)
                throw MalformedJson($"Domain[{d}] must have length 2, got {domain.Length}");
            if (!double.IsFinite(domain[0]) || !double.IsFinite(domain[1]) || domain[0] >= domain[1])
                throw MalformedJson($"Domain[{d}] must contain finite bounds with lo < hi");

            int n = state.NNodes[d];
            if (n < 1)
                throw MalformedJson($"NNodes[{d}] must be >= 1, got {n}");
            total = CheckedJsonProduct(total, n, "prod(NNodes)");

            RequireFiniteLength(state.NodeArrays[d], n, $"NodeArrays[{d}]");
            RequireFiniteLength(state.Weights[d], n, $"Weights[{d}]");
            RequireFiniteLength(
                state.DiffMatrices[d],
                CheckedJsonProduct(n, n, $"DiffMatrices[{d}] size"),
                $"DiffMatrices[{d}]");
        }

        RequireFiniteLength(state.TensorValues, total, nameof(SerializationState.TensorValues));

        if (state.OriginalNNodes != null)
        {
            if (state.OriginalNNodes.Length != 0)
                RequireLength(state.OriginalNNodes, ndim, nameof(SerializationState.OriginalNNodes));
            for (int d = 0; d < state.OriginalNNodes.Length; d++)
            {
                int? n = state.OriginalNNodes[d];
                if (n.HasValue && n.Value < 1)
                    throw MalformedJson($"OriginalNNodes[{d}] must be null or >= 1, got {n.Value}");
            }
        }

        if (state.SpecialPoints != null)
        {
            RequireLength(state.SpecialPoints, ndim, nameof(SerializationState.SpecialPoints));
            for (int d = 0; d < state.SpecialPoints.Length; d++)
                RequireFiniteArray(state.SpecialPoints[d], $"SpecialPoints[{d}]");
        }

        ValidateDerivativeRegistry(state.RegisteredDerivativeOrders, ndim, maxDerivativeOrder);
    }

    private static void ValidateDerivativeRegistry(
        int[][]? registeredDerivativeOrders,
        int numDimensions,
        int maxDerivativeOrder)
    {
        if (registeredDerivativeOrders is null) return;

        for (int i = 0; i < registeredDerivativeOrders.Length; i++)
        {
            int[]? orders = registeredDerivativeOrders[i];
            RequireLength(
                orders,
                numDimensions,
                $"{nameof(SerializationState.RegisteredDerivativeOrders)}[{i}]");
            for (int j = 0; j < orders!.Length; j++)
            {
                if (orders[j] < 0)
                    throw MalformedJson(
                        $"{nameof(SerializationState.RegisteredDerivativeOrders)}[{i}][{j}] must be non-negative, got {orders[j]}");
                if (orders[j] > maxDerivativeOrder)
                    throw MalformedJson(
                        $"{nameof(SerializationState.RegisteredDerivativeOrders)}[{i}][{j}]={orders[j]} exceeds MaxDerivativeOrder {maxDerivativeOrder}");
            }
        }
    }

    private static void RequireLength<T>(T[]? values, int expected, string name)
    {
        if (values == null)
            throw MalformedJson($"{name} must not be null");
        if (values.Length != expected)
            throw MalformedJson($"{name} length {values.Length} does not match NumDimensions {expected}");
    }

    private static void RequireFiniteLength(double[]? values, int expected, string name)
    {
        RequireLength(values, expected, name);
        RequireFiniteArray(values!, name);
    }

    private static void RequireFiniteArray(double[]? values, string name)
    {
        if (values == null)
            throw MalformedJson($"{name} must not be null");
        for (int i = 0; i < values.Length; i++)
        {
            if (!double.IsFinite(values[i]))
                throw MalformedJson($"{name}[{i}] must be finite");
        }
    }

    private static int CheckedJsonProduct(int a, int b, string name)
    {
        try { return checked(a * b); }
        catch (OverflowException)
        {
            throw MalformedJson($"{name} overflows int");
        }
    }

    private static InvalidDataException MalformedJson(string message)
        => new($"Malformed ChebyshevApproximation JSON: {message}");

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
        ValidateFixedGridArguments(nameof(Nodes), numDimensions, domain, nNodes);

        int totalPoints = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(nNodes, nameof(Nodes)),
            nameof(Nodes),
            nNodes);

        double[][] nodesPerDim = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            nodesPerDim[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);

        // Build full grid (Cartesian product, C-order)
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
        ValidateFixedGridArguments(nameof(FromValues), numDimensions, domain, nNodes);
        ArgumentOutOfRangeException.ThrowIfNegative(maxDerivativeOrder);

        int expectedTotal = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(nNodes, nameof(FromValues)),
            nameof(FromValues),
            nNodes);

        if (tensorValues.Length != expectedTotal)
            throw new ArgumentException(
                $"tensor_values.shape=({tensorValues.Length}) does not match " +
                $"n_nodes=({string.Join(", ", nNodes)})");

        for (int i = 0; i < tensorValues.Length; i++)
        {
            if (double.IsNaN(tensorValues[i]) || double.IsInfinity(tensorValues[i]))
                throw new ArgumentException("tensor_values contains NaN or Inf");
        }

        var obj = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = numDimensions,
            _domain = domain.Select(d => (double[])d.Clone()).ToArray(),
            _nNodes = (int[])nNodes.Clone(),
            MaxDerivativeOrder = maxDerivativeOrder,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
        };

        // Chebyshev nodes
        obj._nodeArrays = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            obj._nodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);

        obj._tensorValues = (double[])tensorValues.Clone();

        // Barycentric weights
        obj._weights = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            obj._weights[d] = BarycentricKernel.ComputeBarycentricWeights(obj._nodeArrays[d]);

        // Differentiation matrices
        obj._diffMatrices = new double[numDimensions][,];
        for (int d = 0; d < numDimensions; d++)
            obj._diffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(obj._nodeArrays[d], obj._weights[d]);

        obj.PrecomputeTransposedDiffMatrices();

        obj._constructorType = "from_values";
        obj._isConstructionFinished = true;

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
            _domain = source._domain.Select(d => (double[])d.Clone()).ToArray(),
            _nNodes = (int[])source._nNodes.Clone(),
            MaxDerivativeOrder = source.MaxDerivativeOrder,
            _nodeArrays = Internal.CloneHelpers.DeepCopy(source._nodeArrays)!,
            _weights = Internal.CloneHelpers.DeepCopy(source._weights),
            _diffMatrices = Internal.CloneHelpers.DeepCopy(source._diffMatrices),
            _diffMatricesTFlat = Internal.CloneHelpers.DeepCopy(source._diffMatricesTFlat),
            _tensorValues = tensorValues,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
            _isConstructionFinished = true,
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
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var sorted = ExtrudeSlice.NormalizeExtrusionParams(extrudeParams, NumDimensions);

        double[] tensor = (double[])_tensorValues.Clone();
        int[] shape = (int[])_nNodes.Clone();
        var nodes = _nodeArrays.ToList();
        var weights = _weights!.ToList();
        var diffMats = _diffMatrices!.ToList();
        var domain = _domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = _nNodes.ToList();

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
            _domain = domain.ToArray(),
            _nNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            _nodeArrays = nodes.ToArray(),
            _weights = weights.ToArray(),
            _diffMatrices = diffMats.ToArray(),
            _tensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
            _isConstructionFinished = true,
        };
        result.PrecomputeTransposedDiffMatrices();
        return result;
    }

    /// <summary>
    /// Fix one or more dimensions at given values, reducing dimensionality.
    /// </summary>
    public ChebyshevApproximation Slice(params (int dimIndex, double value)[] sliceParams)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var sorted = ExtrudeSlice.NormalizeSlicingParams(sliceParams, NumDimensions);

        // Validate values within domain
        foreach (var (dimIdx, value) in sorted)
        {
            double lo = _domain[dimIdx][0];
            double hi = _domain[dimIdx][1];
            if (value < lo || value > hi)
                throw new ArgumentException(
                    $"Slice value {value} for dim {dimIdx} is outside domain [{lo}, {hi}]");
        }

        double[] tensor = (double[])_tensorValues.Clone();
        int[] shape = (int[])_nNodes.Clone();
        var nodes = _nodeArrays.ToList();
        var weights = _weights!.ToList();
        var diffMats = _diffMatrices!.ToList();
        var domain = _domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = _nNodes.ToList();

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
            _domain = domain.ToArray(),
            _nNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            _nodeArrays = nodes.ToArray(),
            _weights = weights.ToArray(),
            _diffMatrices = diffMats.ToArray(),
            _tensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
            _isConstructionFinished = true,
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
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        if (dims == null)
            dims = Enumerable.Range(0, NumDimensions).ToArray();

        var sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= NumDimensions)
                throw new ArgumentException($"dim {d} out of range [0, {NumDimensions - 1}]");
        }

        var perDimBounds = Calculus.NormalizeBounds(sortedDims, bounds, _domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        double[] tensor = (double[])_tensorValues.Clone();
        int[] shape = (int[])_nNodes.Clone();
        var nodes = _nodeArrays.ToList();
        var wts = _weights!.ToList();
        var diffMats = _diffMatrices!.ToList();
        var domain = _domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = _nNodes.ToList();

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

        var result = new ChebyshevApproximation
        {
            Function = null,
            NumDimensions = newNdim,
            _domain = domain.ToArray(),
            _nNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            _nodeArrays = nodes.ToArray(),
            _weights = wts.ToArray(),
            _diffMatrices = diffMats.ToArray(),
            _tensorValues = tensor,
            BuildTime = 0.0,
            NEvaluations = 0,
            _cachedErrorEstimate = null,
            _isConstructionFinished = true,
        };
        result.PrecomputeTransposedDiffMatrices();
        return result;
    }

    /// <summary>
    /// Find all roots of the interpolant along a specified dimension.
    /// </summary>
    public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);

        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
        return Calculus.Roots1D(sliced._tensorValues!, sliced._domain[0]);
    }

    /// <summary>
    /// Find the minimum value of the interpolant along a dimension.
    /// </summary>
    public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);
        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        return Calculus.Optimize1D(
            sliced._tensorValues!, sliced._nodeArrays[0], sliced._weights![0],
            sliced._diffMatrices![0], sliced._domain[0], "min");
    }

    /// <summary>
    /// Find the maximum value of the interpolant along a dimension.
    /// </summary>
    public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (_tensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);
        ChebyshevApproximation sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        return Calculus.Optimize1D(
            sliced._tensorValues!, sliced._nodeArrays[0], sliced._weights![0],
            sliced._diffMatrices![0], sliced._domain[0], "max");
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
        double[] newValues = new double[a._tensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a._tensorValues[i] + b._tensorValues![i];
        return FromGrid(a, newValues);
    }

    /// <summary>Subtract two interpolants with the same grid.</summary>
    public static ChebyshevApproximation operator -(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException("Cannot combine different types");
        Algebra.CheckCompatible(a, b);
        double[] newValues = new double[a._tensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a._tensorValues[i] - b._tensorValues![i];
        return FromGrid(a, newValues);
    }

    /// <summary>Multiply interpolant by a scalar.</summary>
    public static ChebyshevApproximation operator *(ChebyshevApproximation a, double scalar)
    {
        if (a._tensorValues == null)
            throw new InvalidOperationException("Operand is not built. Call Build() first.");
        Algebra.ValidateFiniteScalar(scalar, nameof(scalar));

        double[] newValues = new double[a._tensorValues!.Length];
        for (int i = 0; i < newValues.Length; i++)
            newValues[i] = a._tensorValues[i] * scalar;
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
        if (a._tensorValues == null)
            throw new InvalidOperationException("Operand is not built. Call Build() first.");
        Algebra.ValidateFiniteNonZeroDivisor(scalar, nameof(scalar));
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
        bool built = _tensorValues != null;
        long totalNodes = TensorShape.CheckedProduct(_nNodes, nameof(ToString));
        string status = built ? "built" : "not built";

        int maxDisplay = 6;
        string nodesStr, domainStr;
        if (NumDimensions > maxDisplay)
        {
            nodesStr = "[" + string.Join(", ", _nNodes.Take(maxDisplay)) + ", ...]";
            domainStr = string.Join(" x ", _domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        }
        else
        {
            nodesStr = "[" + string.Join(", ", _nNodes) + "]";
            domainStr = string.Join(" x ", _domain.Select(d => $"[{d[0]}, {d[1]}]"));
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
        bool built = _tensorValues != null;
        return $"ChebyshevApproximation(dims={NumDimensions}, nodes=[{string.Join(", ", _nNodes)}], built={built})";
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private static void ValidateFixedGridArguments(
        string caller,
        int numDimensions,
        double[][] domain,
        int[] nNodes)
    {
        ValidateDomainArguments(caller, numDimensions, domain);
        ArgumentNullException.ThrowIfNull(nNodes);
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"len(nNodes)={nNodes.Length} must equal numDimensions={numDimensions}");
        ValidateNodeCounts(caller, nNodes);
    }

    private static void ValidateDomainArguments(string caller, int numDimensions, double[][] domain)
    {
        if (numDimensions <= 0)
            throw new ArgumentException($"{caller} requires numDimensions to be positive, got {numDimensions}");

        ArgumentNullException.ThrowIfNull(domain);

        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"len(domain)={domain.Length} must equal numDimensions={numDimensions}");

        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d] is null)
                throw new ArgumentException($"domain[{d}] must contain exactly two bounds [lo, hi]", nameof(domain));

            if (domain[d].Length != 2)
                throw new ArgumentException($"domain[{d}] must contain exactly two bounds [lo, hi]");

            double lo = domain[d][0];
            double hi = domain[d][1];
            if (!double.IsFinite(lo) || !double.IsFinite(hi) || lo >= hi)
                throw new ArgumentException(
                    $"domain[{d}]: lo={lo} must be strictly less than hi={hi}; both bounds must be finite");
        }
    }

    private static void ValidateNodeCounts(string caller, int[] nNodes)
    {
        for (int d = 0; d < nNodes.Length; d++)
        {
            if (nNodes[d] <= 0)
                throw new ArgumentException(
                    $"{caller} requires nNodes[{d}] to be positive, got {nNodes[d]}");
        }
    }

    /// <summary>
    /// Pre-compute transposed diff matrices as flat arrays for BLAS GEMM.
    /// Called after differentiation matrices are set in Build, FromValues, Load, Extrude, Slice, and Integrate.
    /// </summary>
    internal void PrecomputeTransposedDiffMatrices()
    {
        if (_diffMatrices == null) return;
        _diffMatricesTFlat = new double[_diffMatrices.Length][];
        for (int d = 0; d < _diffMatrices.Length; d++)
        {
            int rows = _diffMatrices[d].GetLength(0);
            int cols = _diffMatrices[d].GetLength(1);
            // Transpose and flatten in one pass (row-major)
            int flatLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { rows, cols }, nameof(PrecomputeTransposedDiffMatrices)),
                nameof(PrecomputeTransposedDiffMatrices),
                new[] { rows, cols });
            var flat = new double[flatLength];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[j * rows + i] = _diffMatrices[d][i, j];
            _diffMatricesTFlat[d] = flat;
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
            strides[i] = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(shape.Skip(i + 1), nameof(Extract1DSlice)),
                nameof(Extract1DSlice));

        // Decompose otherFlat into multi-index for other dimensions
        int[] otherStrides = new int[otherShape.Length];
        if (otherShape.Length > 0)
        {
            otherStrides[otherShape.Length - 1] = 1;
            for (int i = otherShape.Length - 2; i >= 0; i--)
                otherStrides[i] = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(otherShape.Skip(i + 1), nameof(Extract1DSlice)),
                    nameof(Extract1DSlice));
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

    internal static double[] Flatten2D(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        int flatLength = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { rows, cols }, nameof(Flatten2D)),
            nameof(Flatten2D),
            new[] { rows, cols });
        double[] flat = new double[flatLength];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = matrix[i, j];
        return flat;
    }

    internal static double[,] Unflatten2D(double[] flat, int rows, int cols)
    {
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = flat[i * cols + j];
        return matrix;
    }

    // ------------------------------------------------------------------
    // Phase 4 ergonomics — accessors
    // ------------------------------------------------------------------

    /// <summary>Set a free-form descriptor string for this interpolant.</summary>
    public void SetDescriptor(string descriptor) => _descriptor = descriptor;

    /// <summary>Get the descriptor previously set via <see cref="SetDescriptor"/>; null if unset.</summary>
    public string? GetDescriptor() => _descriptor;

    /// <summary>True if <see cref="Build"/>/<see cref="FromValues"/>/<see cref="Load"/> completed.</summary>
    public bool IsConstructionFinished() => _isConstructionFinished;

    /// <summary>Returns one of: "function" (Build), "from_values" (FromValues factory), "load" (Load), "clone" (Clone).</summary>
    public string GetConstructorType() => _constructorType;

    /// <summary>Per-dimension Chebyshev node counts actually used. After auto-N construction, these are the resolved values.</summary>
    public int[] GetUsedNs() => (int[])_nNodes.Clone();

    /// <summary>Maximum derivative order this approximation supports.</summary>
    public int GetMaxDerivativeOrder() => MaxDerivativeOrder;

    /// <summary>
    /// Returns the user-supplied <c>additionalData</c> object passed to the constructor,
    /// or null if none was provided. Same value is threaded through every <c>f(point, data)</c>
    /// call during <see cref="Build"/>.
    /// </summary>
    public object? GetAdditionalData() => _additionalData;

    /// <summary>Internal accessor for AdaptiveBuild to read _additionalData.</summary>
    internal object? AdditionalData => _additionalData;

    /// <summary>
    /// Total number of evaluation points (product of nNodes across all dimensions).
    /// </summary>
    /// <returns>The total count of Chebyshev nodes in the tensor grid.</returns>
    public int GetNumEvaluationPoints()
    {
        return TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(_nNodes, nameof(GetNumEvaluationPoints)),
            nameof(GetNumEvaluationPoints),
            _nNodes);
    }

    /// <summary>
    /// Flat row-major array of all Chebyshev node coordinates.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached internally.
    /// </summary>
    /// <returns>A snapshot of shape [numPoints, ndim] flattened to 1D in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return CloneHelpers.DeepCopy(_evaluationPointsCache)!;

        int num = GetNumEvaluationPoints();
        int ndim = NumDimensions;
        int coordinateCount = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { num, ndim }, nameof(GetEvaluationPoints)),
            nameof(GetEvaluationPoints),
            new[] { num, ndim });
        var points = new double[coordinateCount];
        var indices = new int[ndim];

        for (int flat = 0; flat < num; flat++)
        {
            int rem = flat;
            for (int d = ndim - 1; d >= 0; d--)
            {
                indices[d] = rem % _nNodes[d];
                rem /= _nNodes[d];
            }
            for (int d = 0; d < ndim; d++)
            {
                points[flat * ndim + d] = _nodeArrays[d][indices[d]];
            }
        }

        _evaluationPointsCache = points;
        return CloneHelpers.DeepCopy(points)!;
    }

    /// <summary>
    /// Get special points (e.g., knots or singularities) used in construction.
    /// </summary>
    /// <returns>Special points per dimension, or null if not applicable.</returns>
    public double[][]? GetSpecialPoints() => Internal.CloneHelpers.DeepCopy(_specialPoints);

    /// <summary>
    /// Populate this interpolant's tensor values from a precomputed flat array.
    /// Used after constructing with <c>deferBuild: true</c>. Bit-identical to
    /// the <see cref="FromValues"/> factory.
    /// </summary>
    /// <param name="values">Flat C-order tensor of length nNodes[0]*nNodes[1]*...</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when values length does not match the expected product of nNodes.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the interpolant is already constructed.</exception>
    public void SetOriginalFunctionValues(double[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (_isConstructionFinished || _tensorValues != null)
            throw new InvalidOperationException(
                "interpolant is already constructed; SetOriginalFunctionValues is for unconstructed deferred objects");

        int expected = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(_nNodes, nameof(SetOriginalFunctionValues)),
            nameof(SetOriginalFunctionValues),
            _nNodes);
        if (values.Length != expected)
            throw new ArgumentException(
                $"values has {values.Length} entries, expected {expected} for nNodes=[{string.Join(",", _nNodes)}]");
        for (int i = 0; i < values.Length; i++)
        {
            if (!double.IsFinite(values[i]))
                throw new ArgumentException("values contains NaN or Inf", nameof(values));
        }

        // Materialize node arrays now if deferred.
        if (_nodeArrays.Length == 0)
        {
            _nodeArrays = new double[NumDimensions][];
            for (int d = 0; d < NumDimensions; d++)
                _nodeArrays[d] = BarycentricKernel.MakeNodesForDim(_domain[d][0], _domain[d][1], _nNodes[d]);
        }

        // Mirror FromValues precomputation (bit-identical).
        _tensorValues = (double[])values.Clone();

        _weights = new double[NumDimensions][];
        for (int d = 0; d < NumDimensions; d++)
            _weights[d] = BarycentricKernel.ComputeBarycentricWeights(_nodeArrays[d]);

        _diffMatrices = new double[NumDimensions][,];
        for (int d = 0; d < NumDimensions; d++)
            _diffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(_nodeArrays[d], _weights[d]);

        PrecomputeTransposedDiffMatrices();

        _evaluationPointsCache = null;
        _isConstructionFinished = true;
        _constructorType = "from_values";
    }

    /// <summary>
    /// Register or look up a derivative-orders tuple. Returns a stable
    /// session-local int id for the same orders. Used in conjunction with
    /// the <c>Eval(point, derivativeId)</c> overload.
    /// </summary>
    /// <param name="orders">Derivative order per dimension.</param>
    /// <returns>A stable int id for this orders tuple (0-based, assigned in registration order).</returns>
    public int GetDerivativeId(int[] orders)
    {
        EvaluationArguments.ValidateDerivativeOrder(orders, NumDimensions, nameof(orders), MaxDerivativeOrder);
        var key = new Internal.TupleKey(orders);
        if (_derivativeIdRegistry.TryGetValue(key, out int existing))
            return existing;
        int id = _registeredDerivativeOrders.Count;
        _registeredDerivativeOrders.Add((int[])orders.Clone());
        _derivativeIdRegistry[key] = id;
        return id;
    }

    /// <summary>Evaluate at <paramref name="point"/> using a previously-registered derivative id.</summary>
    /// <param name="point">Evaluation point.</param>
    /// <param name="derivativeId">Id returned by <see cref="GetDerivativeId"/>.</param>
    /// <returns>Interpolated value at the given derivative order.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="derivativeId"/> has not been registered.</exception>
    public double Eval(double[] point, int derivativeId)
    {
        if (derivativeId < 0 || derivativeId >= _registeredDerivativeOrders.Count)
            throw new ArgumentOutOfRangeException(
                nameof(derivativeId),
                $"derivativeId {derivativeId} not registered. Call GetDerivativeId first.");
        return Eval(point, _registeredDerivativeOrders[derivativeId]);
    }

    internal Dictionary<Internal.TupleKey, int> DerivativeIdRegistry => _derivativeIdRegistry;
    internal List<int[]> RegisteredDerivativeOrders => _registeredDerivativeOrders;

    // ------------------------------------------------------------------
    // Clone
    // ------------------------------------------------------------------

    /// <summary>
    /// Returns a deep copy of this approximation. The source <see cref="Function"/>
    /// callable is NOT duplicated — clones cannot be rebuilt without re-supplying
    /// the function. All precomputed state, descriptor, derivative-id registry,
    /// and special points are deep-copied.
    /// </summary>
    /// <returns>A fully independent <see cref="ChebyshevApproximation"/> with <see cref="Function"/> set to null.</returns>
    public ChebyshevApproximation Clone()
    {
        var copy = new ChebyshevApproximation();
        copy.NumDimensions = NumDimensions;
        copy.NNodes = Internal.CloneHelpers.DeepCopy(_nNodes)!;
        copy.Domain = Internal.CloneHelpers.DeepCopy(_domain)!;
        copy.NodeArrays = Internal.CloneHelpers.DeepCopy(_nodeArrays)!;
        copy.TensorValues = Internal.CloneHelpers.DeepCopy(_tensorValues);
        copy.Weights = Internal.CloneHelpers.DeepCopy(_weights);
        copy.DiffMatrices = Internal.CloneHelpers.DeepCopy(_diffMatrices);
        copy.DiffMatricesTFlat = Internal.CloneHelpers.DeepCopy(_diffMatricesTFlat);
        copy.MaxDerivativeOrder = MaxDerivativeOrder;
        copy.MaxN = MaxN;
        copy.ErrorThreshold = ErrorThreshold;
        copy.BuildWarning = BuildWarning;
        copy.OriginalNNodes = Internal.CloneHelpers.DeepCopy(OriginalNNodes)!;
        copy.NEvaluations = NEvaluations;
        copy.BuildTime = BuildTime;
        copy._descriptor = _descriptor;
        copy._additionalData = _additionalData;
        copy._specialPoints = Internal.CloneHelpers.DeepCopy(_specialPoints);
        copy._isConstructionFinished = _isConstructionFinished;
        copy._constructorType = "clone";
        copy._evaluationPointsCache = null;
        foreach (var kv in _derivativeIdRegistry)
            copy._derivativeIdRegistry[kv.Key] = kv.Value;
        foreach (var orders in _registeredDerivativeOrders)
            copy._registeredDerivativeOrders.Add((int[])orders.Clone());
        return copy;
    }

    // ------------------------------------------------------------------
    // Sobol sensitivity indices
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute first- and total-order variance-based sensitivity indices directly
    /// from this approximation's spectral Chebyshev coefficients. The variance is
    /// with respect to the Chebyshev orthogonality weight on the normalized domain,
    /// not a Monte Carlo estimate under a uniform input distribution.
    /// </summary>
    /// <returns>A <see cref="SobolResult"/> with per-dim FirstOrder, TotalOrder, and Chebyshev-weighted total Variance.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public SobolResult SobolIndices()
    {
        if (_tensorValues == null)
            throw new InvalidOperationException(
                "SobolIndices requires a built ChebyshevApproximation. Call Build() first.");
        var coeffs = Internal.Sensitivity.ChebyshevCoefficientsND(_tensorValues, _nNodes);
        return Internal.Sensitivity.ComputeSobolFromCoeffs(coeffs, _nNodes);
    }

    // ------------------------------------------------------------------
    // Serialization state
    // ------------------------------------------------------------------

    internal class SerializationState
    {
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int? MaxDerivativeOrder { get; set; }
        public double[][] NodeArrays { get; set; } = Array.Empty<double[]>();
        public double[] TensorValues { get; set; } = Array.Empty<double>();
        public double[][] Weights { get; set; } = Array.Empty<double[]>();
        public double[][] DiffMatrices { get; set; } = Array.Empty<double[]>();
        public double BuildTime { get; set; }
        public int NEvaluations { get; set; }
        public int?[]? OriginalNNodes { get; set; }
        public double? ErrorThreshold { get; set; }
        public int? MaxN { get; set; }
        public string Version { get; set; } = "";
        // v0.8.0 ergonomics fields (absent in pre-v0.8.0 JSON; null == not set)
        public string? Descriptor { get; set; }
        public string? ConstructorType { get; set; }
        public double[][]? SpecialPoints { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
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
