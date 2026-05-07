using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp;

/// <summary>
/// Chebyshev interpolation in Tensor Train (TT) format.
/// For functions of 5+ dimensions where full tensor interpolation is infeasible.
/// Uses TT-Cross to build from O(d * n * r^2) function evaluations instead of O(n^d),
/// then evaluates via TT inner product with Chebyshev polynomial basis.
/// </summary>
public class ChebyshevTT
{
    private readonly Func<double[], double>? _function;
    private readonly int _numDimensions;
    private readonly double[][] _domain;
    private readonly int[] _nNodes;
    private readonly int _maxRank;
    private readonly double _tolerance;
    private readonly int _maxSweeps;

    // Build-time state
    private TensorTrainKernel.TtCore[]? _coeffCores;
    private bool _built;
    private int[]? _ttRanks;
    private double _buildTime;
    private int _totalBuildEvals;
    private double? _cachedErrorEstimate;
    private string? _descriptor;
    private int _maxDerivativeOrder = 2;
    private object? _additionalData;
    private int[] _dimOrder = Array.Empty<int>();
    private double[]? _evaluationPointsCache;
    private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
    private readonly List<int[]> _registeredDerivativeOrders = new();
    private readonly int? _nWorkers;   // accepted for API symmetry; ignored (D10).
    private readonly IProgress<int>? _progress;

    private int EffectiveMaxDerivativeOrder => Math.Min(_maxDerivativeOrder, 2);

    /// <summary>Warning message set when loading from a different library version.</summary>
    public string? LoadWarning { get; private set; }

    /// <summary>Warning emitted by Build() if maxRank was reached before tolerance was satisfied during ALS. Null otherwise.</summary>
    public string? BuildWarning { get; private set; }

    /// <summary>
    /// Build method that produced the current cores: <c>"cross"</c>, <c>"svd"</c>, or <c>"als"</c>.
    /// <c>null</c> only before <see cref="Build"/> is called or after <see cref="Load"/> from a
    /// pre-v0.6.0 JSON file that predates this property.
    /// </summary>
    public string? Method { get; private set; }

    // Overrides GetConstructorType() when set explicitly (e.g. "clone"). null means fall back to Method.
    private string? _constructorType;

    /// <summary>Number of input dimensions.</summary>
    public int NumDimensions => _numDimensions;

    /// <summary>Bounds [(lo, hi), ...] for each dimension.</summary>
    public double[][] Domain => _domain.Select(d => (double[])d.Clone()).ToArray();

    /// <summary>Number of Chebyshev nodes per dimension.</summary>
    public int[] NNodes => (int[])_nNodes.Clone();

    /// <summary>Maximum TT rank.</summary>
    public int MaxRank => _maxRank;

    /// <summary>Total number of function evaluations used during build.</summary>
    public int TotalBuildEvals => _totalBuildEvals;

    /// <summary>
    /// TT ranks [1, r_1, r_2, ..., r_{d-1}, 1]. Only available after <see cref="Build"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public int[] TtRanks
    {
        get
        {
            CheckBuilt();
            return (int[])_ttRanks!.Clone();
        }
    }

    /// <summary>
    /// Ratio of full tensor elements to TT storage elements.
    /// </summary>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double CompressionRatio
    {
        get
        {
            CheckBuilt();
            double fullSize = FullGridSizeAsDouble();
            int ttSize = 0;
            for (int i = 0; i < _coeffCores!.Length; i++) ttSize += _coeffCores[i].Size;
            return fullSize / ttSize;
        }
    }

    /// <summary>
    /// Create a new ChebyshevTT interpolant.
    /// </summary>
    /// <param name="function">Function to approximate. Signature: f(point) -> double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] for each dimension.</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <param name="maxRank">Maximum positive TT rank. Default is 10.</param>
    /// <param name="tolerance">Finite positive convergence tolerance for TT-Cross/ALS. Default is 1e-6.</param>
    /// <param name="maxSweeps">Maximum positive number of TT-Cross sweeps. Default is 10.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support. Default is 2.</param>
    /// <param name="additionalData">Optional user data object stored for introspection via <see cref="GetAdditionalData"/>. NOT threaded through build calls (TT function signature has no data arg).</param>
    /// <param name="nWorkers">Accepted for API symmetry with the other classes but
    /// ignored: TT-Cross is adaptive sampling, not pre-grid evaluation. Pass null.</param>
    /// <param name="progress">Optional progress reporter; receives the cumulative
    /// sweep count after each TT-Cross sweep.</param>
    /// <remarks>Thread safety: TT-Cross is inherently sequential; <paramref name="nWorkers"/> is accepted but has no effect.</remarks>
    public ChebyshevTT(
        Func<double[], double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6,
        int maxSweeps = 10,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(function);
        ValidateFixedGridArguments(numDimensions, domain, nNodes);
        ValidatePositiveRank(maxRank, nameof(maxRank));
        ValidatePositiveFiniteTolerance(tolerance, nameof(tolerance));
        ValidatePositiveInteger(maxSweeps, nameof(maxSweeps));
        ArgumentOutOfRangeException.ThrowIfNegative(maxDerivativeOrder);

        _function = function;
        _numDimensions = numDimensions;
        _domain = CloneDomain(domain);
        _nNodes = (int[])nNodes.Clone();
        _maxRank = maxRank;
        _tolerance = tolerance;
        _maxSweeps = maxSweeps;
        _maxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;
        _dimOrder = Enumerable.Range(0, numDimensions).ToArray();
    }

    // Private constructor for deserialization
    private ChebyshevTT(
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank,
        double tolerance,
        int maxSweeps,
        TensorTrainKernel.TtCore[] coeffCores,
        int[] ttRanks,
        double buildTime,
        int totalBuildEvals,
        int maxDerivativeOrder = 2)
    {
        _function = null;
        _numDimensions = numDimensions;
        _domain = domain;
        _nNodes = nNodes;
        _maxRank = maxRank;
        _tolerance = tolerance;
        _maxSweeps = maxSweeps;
        _coeffCores = coeffCores;
        _ttRanks = ttRanks;
        _buildTime = buildTime;
        _totalBuildEvals = totalBuildEvals;
        _maxDerivativeOrder = maxDerivativeOrder;
        _built = true;
        _dimOrder = Enumerable.Range(0, numDimensions).ToArray();  // overwritten by Load's v2 deserialization
    }

    private static void ValidateFixedGridArguments(int numDimensions, double[][] domain, int[] nNodes)
    {
        if (numDimensions <= 0)
            throw new ArgumentException(
                $"numDimensions must be positive, got {numDimensions}.",
                nameof(numDimensions));

        ArgumentNullException.ThrowIfNull(domain);
        ArgumentNullException.ThrowIfNull(nNodes);

        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}", nameof(domain));
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}", nameof(nNodes));

        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d] is null)
                throw new ArgumentException($"domain[{d}] must not be null.", nameof(domain));
            if (domain[d].Length != 2)
                throw new ArgumentException($"domain[{d}] must contain exactly two bounds [lo, hi].", nameof(domain));

            double lo = domain[d][0];
            double hi = domain[d][1];
            if (!double.IsFinite(lo) || !double.IsFinite(hi) || lo >= hi)
                throw new ArgumentException(
                    $"domain[{d}] must contain finite bounds with lo < hi; got [{lo}, {hi}].",
                    nameof(domain));

            if (nNodes[d] <= 0)
                throw new ArgumentException($"nNodes[{d}] must be positive; got {nNodes[d]}.", nameof(nNodes));
        }
    }

    private static double[][] CloneDomain(double[][] domain)
    {
        return domain.Select(d => (double[])d.Clone()).ToArray();
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// <summary>
    /// Build TT approximation and convert to Chebyshev coefficient cores.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    /// <param name="seed">Random seed for TT-Cross/ALS initialization. Ignored for method="svd".</param>
    /// <param name="method">"cross" (default), "svd", or "als".</param>
    /// <exception cref="ArgumentException">If method is not "cross", "svd", or "als", or if the function returns NaN or Infinity at a sampled grid point.</exception>
    /// <exception cref="InvalidOperationException">If this TT was loaded or created from values without the original function.</exception>
    public void Build(bool verbose = true, int? seed = null, string method = "cross")
    {
        if (method != "cross" && method != "svd" && method != "als")
            throw new ArgumentException($"method must be 'cross', 'svd', or 'als', got '{method}'");
        var function = GetRequiredFunction(nameof(Build));
        Method = method;
        BuildWarning = null;

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        double fullTensorSize = FullGridSizeAsDouble();

        if (verbose)
        {
            Console.WriteLine($"Building {_numDimensions}D ChebyshevTT (max_rank={_maxRank}, method='{method}')...");
            Console.WriteLine($"  Full tensor would need {fullTensorSize:N0} evaluations");
        }

        // Step 1: Generate Chebyshev Type I nodes per dimension
        double[][] grids = new double[_numDimensions][];
        for (int d = 0; d < _numDimensions; d++)
            grids[d] = BarycentricKernel.MakeNodesForDim(_domain[d][0], _domain[d][1], _nNodes[d]);

        // Step 2: Build value cores
        TensorTrainKernel.TtCore[] valueCores;
        int nEvals;
        Func<double[], double> finiteFunction = point => EvaluateFiniteFunction(function, point, nameof(Build));

        if (method == "cross")
        {
            if (verbose) Console.WriteLine("  Running TT-Cross...");
            (valueCores, nEvals) = TensorTrainKernel.TtCross(
                finiteFunction, grids, _maxRank, _tolerance, _maxSweeps, verbose, seed, _progress);
        }
        else if (method == "svd")
        {
            (valueCores, nEvals) = TensorTrainKernel.TtSvd(
                finiteFunction, grids, _maxRank, _tolerance, verbose);
        }
        else  // method == "als"
        {
            if (verbose) Console.WriteLine("  Running TT-ALS...");
            bool hitCap;
            (valueCores, nEvals, hitCap) = TensorTrainKernel.AlsAdaptiveRank(
                finiteFunction, grids, _maxRank, _tolerance, seed, verbose);
            if (hitCap)
                BuildWarning =
                    $"maxRank={_maxRank} reached before ALS tolerance={_tolerance:e2} satisfied. " +
                    "Increase maxRank or relax tolerance.";
        }
        _totalBuildEvals = nEvals;

        // Step 3: Convert value cores to coefficient cores via DCT-II
        _coeffCores = TensorTrainKernel.ValueToCoeffCores(valueCores);

        // Step 4: Extract TT ranks
        _ttRanks = new int[_numDimensions + 1];
        _ttRanks[0] = 1;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks[i + 1] = _coeffCores[i].RRight;

        sw.Stop();
        _buildTime = sw.Elapsed.TotalSeconds;
        _built = true;

        if (verbose)
        {
            int ttStorage = 0;
            for (int i = 0; i < _coeffCores.Length; i++) ttStorage += _coeffCores[i].Size;
            Console.WriteLine($"  Built in {_buildTime:F3}s ({nEvals:N0} function evaluations)");
            Console.WriteLine($"  TT ranks: [{string.Join(", ", _ttRanks)}]");
            Console.WriteLine($"  Compression: {fullTensorSize:N0} -> {ttStorage:N0} elements ({fullTensorSize / ttStorage:F1}x)");
        }
    }

    private double FullGridSizeAsDouble()
    {
        double total = 1.0;
        for (int i = 0; i < _numDimensions; i++) total *= _nNodes[i];
        return total;
    }

    private int FullGridSizeAsIntForMaterialization(string caller)
    {
        long total = TensorShape.ProductAtMost(_nNodes, int.MaxValue);
        if (total > int.MaxValue)
        {
            throw new OverflowException(
                $"{caller} requires materializing the full Chebyshev grid, but the grid is too large. " +
                "Use sparse TT evaluation for high-dimensional tensors.");
        }
        return (int)total;
    }

    private Func<double[], double> GetRequiredFunction(string caller)
    {
        if (_function == null)
        {
            throw new InvalidOperationException(
                $"{caller} requires Function to be callable; this TT was loaded or created from values without the original function.");
        }
        return _function;
    }

    private static double EvaluateFiniteFunction(Func<double[], double> function, double[] point, string caller)
    {
        double value = function(point);
        if (!double.IsFinite(value))
        {
            throw new ArgumentException(
                $"{caller} function returned a non-finite value at a Chebyshev grid point. " +
                "ChebyshevTT build and completion require finite function values.",
                "function");
        }
        return value;
    }

    private void CheckBuilt()
    {
        if (!_built)
            throw new InvalidOperationException("Call Build() before using this method.");
    }

    private static void ValidatePositiveRank(int maxRank, string paramName)
    {
        if (maxRank <= 0)
            throw new ArgumentOutOfRangeException(paramName, maxRank, $"{paramName} must be positive.");
    }

    private static void ValidatePositiveInteger(int value, string paramName)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(paramName, value, $"{paramName} must be positive.");
    }

    private static void ValidatePositiveFiniteTolerance(double tolerance, string paramName)
    {
        if (!double.IsFinite(tolerance) || tolerance <= 0.0)
            throw new ArgumentOutOfRangeException(
                paramName,
                tolerance,
                $"{paramName} must be finite and positive.");
    }

    private static void ValidateNonNegativeFiniteTolerance(double tolerance, string paramName)
    {
        if (!double.IsFinite(tolerance) || tolerance < 0.0)
            throw new ArgumentOutOfRangeException(
                paramName,
                tolerance,
                $"{paramName} must be finite and non-negative.");
    }

    /// <summary>Returns true if _dimOrder is the identity permutation [0, 1, ..., d-1].</summary>
    private bool IsIdentityDimOrder()
    {
        for (int i = 0; i < _dimOrder.Length; i++)
            if (_dimOrder[i] != i) return false;
        return true;
    }

    // ------------------------------------------------------------------
    // Eval
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate at a single point via TT inner product with Chebyshev polynomial basis.
    /// Cost: O(d * n * r^2) per point.
    /// </summary>
    /// <param name="point">Query point inside the declared domain, one coordinate per dimension.</param>
    /// <returns>Interpolated value.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double Eval(double[] point)
    {
        CheckBuilt();
        EvaluationArguments.ValidatePointInDomain(point, _numDimensions, UserFrameDomain());
        // Remap user coordinates to internal TT storage order when a non-identity
        // dim_order was set by Reorder() or WithAutoOrder(). Identity order is a no-op.
        if (!IsIdentityDimOrder())
        {
            var permPoint = new double[_numDimensions];
            for (int k = 0; k < _numDimensions; k++) permPoint[k] = point[_dimOrder[k]];
            point = permPoint;
        }
        return EvalCore(point);
    }

    /// <summary>
    /// Core evaluation without dim_order remapping. Used by EvalMulti's FD machinery
    /// (which operates in storage frame after EvalMulti does the top-level remap).
    /// </summary>
    private double EvalCore(double[] point)
    {
        // result starts as 1x1 identity
        double[] result = { 1.0 };
        int resultRows = 1;

        for (int d = 0; d < _numDimensions; d++)
        {
            double a = _domain[d][0], b = _domain[d][1];
            double scaled = 2.0 * (point[d] - a) / (b - a) - 1.0;

            // Evaluate Chebyshev polynomials T_0..T_{n-1} via recurrence
            int nk = _nNodes[d];
            double[] q = ChebyshevPolynomials(scaled, nk);

            // Contract: v[i,k] = sum_j q[j] * core[i,j,k]
            var core = _coeffCores![d];
            int rRight = core.RRight;
            int vLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { resultRows, rRight }, nameof(Eval)),
                nameof(Eval),
                new[] { resultRows, rRight });
            double[] v = new double[vLength];

            for (int i = 0; i < resultRows; i++)
                for (int k = 0; k < rRight; k++)
                {
                    double sum = 0;
                    for (int j = 0; j < nk; j++)
                        sum += q[j] * core[i, j, k];
                    v[i * rRight + k] = sum;
                }

            // Chain multiply: newResult[i,k] = sum_j result[i,j] * v[j,k]
            // But result is 1D (resultRows entries = 1 x resultRows), v is (resultRows x rRight)
            // After first iteration: result is (1 x rRight), etc.
            // Actually: result is flat [resultCols], v is [resultCols x rRight]
            // newResult[k] = sum_j result[j] * v[j * rRight + k]
            // Wait — let me think about this more carefully.
            //
            // Python: result = np.ones((1,1)), then result = result @ v where v is (r_{d-1}, r_d)
            // So result is always (1, r_d). We track as flat array of length r_d.
            double[] newResult = new double[rRight];
            for (int k = 0; k < rRight; k++)
            {
                double sum = 0;
                for (int j = 0; j < resultRows; j++)
                    sum += result[j] * v[j * rRight + k];
                newResult[k] = sum;
            }

            result = newResult;
            resultRows = rRight;
        }

        return result[0];
    }

    /// <summary>
    /// Evaluate at multiple points simultaneously.
    /// Vectorized TT inner product: 15-20x speedup over calling Eval in a loop.
    /// </summary>
    /// <param name="points">Query points inside the declared domain, shape (N, numDimensions).</param>
    /// <returns>Interpolated values, length N.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double[] EvalBatch(double[,] points)
    {
        CheckBuilt();
        EvaluationArguments.ValidatePointBatchInDomain(points, _numDimensions, UserFrameDomain());

        // Remap columns from user's original dim order to internal storage order.
        if (!IsIdentityDimOrder())
        {
            int n = points.GetLength(0);
            var permPoints = new double[n, _numDimensions];
            for (int i = 0; i < n; i++)
                for (int k = 0; k < _numDimensions; k++)
                    permPoints[i, k] = points[i, _dimOrder[k]];
            points = permPoints;
        }

        int N = points.GetLength(0);
        // result[n] is a flat vector of length resultCols (starts at 1)
        // result shape: (N, 1, 1) → after each dim: (N, 1, r_d)
        // We store as flat: result[n * resultCols + col]
        double[] result = new double[N];
        for (int i = 0; i < N; i++) result[i] = 1.0;
        int resultCols = 1;

        for (int d = 0; d < _numDimensions; d++)
        {
            double a = _domain[d][0], b = _domain[d][1];
            int nk = _nNodes[d];
            var core = _coeffCores![d];
            int rLeft = core.RLeft; // should == resultCols
            int rRight = core.RRight;

            // Compute Q[n, j] = T_j(scaled_n) for all n
            int qLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { N, nk }, nameof(EvalBatch)),
                nameof(EvalBatch),
                new[] { N, nk });
            double[] Q = new double[qLength]; // Q[n * nk + j]
            for (int nn = 0; nn < N; nn++)
            {
                double scaled = 2.0 * (points[nn, d] - a) / (b - a) - 1.0;
                double[] q = ChebyshevPolynomials(scaled, nk);
                for (int j = 0; j < nk; j++)
                    Q[nn * nk + j] = q[j];
            }

            // V[n,i,k] = sum_j Q[n,j] * core[i,j,k]
            int vLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { N, rLeft, rRight }, nameof(EvalBatch)),
                nameof(EvalBatch),
                new[] { N, rLeft, rRight });
            double[] V = new double[vLength];
            for (int nn = 0; nn < N; nn++)
                for (int i = 0; i < rLeft; i++)
                    for (int k = 0; k < rRight; k++)
                    {
                        double sum = 0;
                        for (int j = 0; j < nk; j++)
                            sum += Q[nn * nk + j] * core[i, j, k];
                        V[nn * rLeft * rRight + i * rRight + k] = sum;
                    }

            // newResult[n, k] = sum_j result[n, j] * V[n, j, k]
            // result is (N, resultCols), V is (N, rLeft, rRight), rLeft == resultCols
            int newResultLength = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(new[] { N, rRight }, nameof(EvalBatch)),
                nameof(EvalBatch),
                new[] { N, rRight });
            double[] newResult = new double[newResultLength];
            for (int nn = 0; nn < N; nn++)
                for (int k = 0; k < rRight; k++)
                {
                    double sum = 0;
                    for (int j = 0; j < resultCols; j++)
                        sum += result[nn * resultCols + j] * V[nn * rLeft * rRight + j * rRight + k];
                    newResult[nn * rRight + k] = sum;
                }

            result = newResult;
            resultCols = rRight;
        }

        // Extract scalar results
        double[] output = new double[N];
        for (int i = 0; i < N; i++)
            output[i] = result[i]; // resultCols should be 1 at this point
        return output;
    }

    // ------------------------------------------------------------------
    // EvalMulti — finite-difference derivatives
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate with finite-difference derivatives at a single point.
    /// </summary>
    /// <param name="point">Evaluation point inside the declared domain.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.
    /// Supports 0 (value), 1 (first), and 2 (second).</param>
    /// <returns>One result per derivative order specification.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double[] EvalMulti(double[] point, int[][] derivativeOrders)
    {
        CheckBuilt();
        EvaluationArguments.ValidatePointInDomain(point, _numDimensions, UserFrameDomain());
        EvaluationArguments.ValidateDerivativeOrders(
            derivativeOrders,
            _numDimensions,
            maxDerivativeOrder: EffectiveMaxDerivativeOrder);

        // v0.21.1: race-safe via EvalStorageFrame helper that always operates in
        // storage frame. Public EvalMulti permutes user-frame inputs once into
        // local arrays — no mutation of self._dimOrder.
        // Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:2172-2215.
        double[] storagePoint = point;
        int[][] storageOrders = derivativeOrders;

        if (!IsIdentityDimOrder())
        {
            storagePoint = new double[_numDimensions];
            for (int s = 0; s < _numDimensions; s++)
                storagePoint[s] = point[_dimOrder[s]];

            storageOrders = new int[derivativeOrders.Length][];
            for (int i = 0; i < derivativeOrders.Length; i++)
            {
                storageOrders[i] = new int[_numDimensions];
                for (int s = 0; s < _numDimensions; s++)
                    storageOrders[i][s] = derivativeOrders[i][_dimOrder[s]];
            }
        }

        var results = new double[storageOrders.Length];
        for (int i = 0; i < storageOrders.Length; i++)
            results[i] = EvalStorageFrame(storagePoint, storageOrders[i]);
        return results;
    }

    /// <summary>
    /// Evaluate at a single point assuming storage-frame inputs (no _dimOrder
    /// remapping). The structural workhorse for Eval and EvalMulti.
    /// </summary>
    /// <param name="storagePoint">Point in storage frame.</param>
    /// <param name="derivativeOrderStorage">Derivative orders in storage frame.
    /// All-zero triggers the value path; otherwise FD machinery.</param>
    /// <returns>Interpolated value (or FD derivative).</returns>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:2172-2215</c>.
    /// Does not mutate <see cref="_dimOrder"/>; safe under concurrent invocation.
    /// </remarks>
    private double EvalStorageFrame(double[] storagePoint, int[] derivativeOrderStorage)
    {
        bool allZero = true;
        for (int d = 0; d < derivativeOrderStorage.Length; d++)
            if (derivativeOrderStorage[d] != 0) { allZero = false; break; }

        if (allZero)
            return EvalCore(storagePoint);
        return FdDerivative(storagePoint, derivativeOrderStorage);
    }

    private double FdDerivative(double[] point, int[] derivOrder)
    {
        var activeDims = new List<(int dim, int order)>();
        for (int d = 0; d < derivOrder.Length; d++)
            if (derivOrder[d] > 0)
                activeDims.Add((d, derivOrder[d]));

        if (activeDims.Count == 1)
        {
            var (d, order) = activeDims[0];
            return FdSingleDim(point, d, order);
        }
        else if (activeDims.Count == 2)
        {
            var (d1, o1) = activeDims[0];
            var (d2, o2) = activeDims[1];
            if (o1 == 1 && o2 == 1)
                return FdCrossDerivative(point, d1, d2);
            else
                return FdNested(point, activeDims, 0);
        }
        else
        {
            return FdNested(point, activeDims, 0);
        }
    }

    private double FdStep(int d)
    {
        return (_domain[d][1] - _domain[d][0]) * 1e-4;
    }

    private double[] NudgePoint(double[] point, int d, double h)
    {
        double[] pt = (double[])point.Clone();
        double a = _domain[d][0], b = _domain[d][1];
        double needed = h * 1.5;
        if (pt[d] - a < needed) pt[d] = a + needed;
        if (b - pt[d] < needed) pt[d] = b - needed;
        return pt;
    }

    private double FdSingleDim(double[] point, int d, int order)
    {
        double h = FdStep(d);
        double[] pt = NudgePoint(point, d, h);

        if (order == 1)
        {
            double[] ptPlus = (double[])pt.Clone();
            double[] ptMinus = (double[])pt.Clone();
            ptPlus[d] += h;
            ptMinus[d] -= h;
            return (EvalCore(ptPlus) - EvalCore(ptMinus)) / (2.0 * h);
        }
        else if (order == 2)
        {
            double[] ptPlus = (double[])pt.Clone();
            double[] ptMinus = (double[])pt.Clone();
            ptPlus[d] += h;
            ptMinus[d] -= h;
            return (EvalCore(ptPlus) - 2.0 * EvalCore(pt) + EvalCore(ptMinus)) / (h * h);
        }
        else
        {
            throw new ArgumentException($"Derivative order {order} not supported (use 1 or 2)");
        }
    }

    private double FdCrossDerivative(double[] point, int d1, int d2)
    {
        double h1 = FdStep(d1);
        double h2 = FdStep(d2);
        double[] pt = NudgePoint(point, d1, h1);
        pt = NudgePoint(pt, d2, h2);

        double[] MakePt(double delta1, double delta2)
        {
            double[] p = (double[])pt.Clone();
            p[d1] += delta1;
            p[d2] += delta2;
            return p;
        }

        double fpp = EvalCore(MakePt(+h1, +h2));
        double fpm = EvalCore(MakePt(+h1, -h2));
        double fmp = EvalCore(MakePt(-h1, +h2));
        double fmm = EvalCore(MakePt(-h1, -h2));
        return (fpp - fpm - fmp + fmm) / (4.0 * h1 * h2);
    }

    private double FdNested(double[] point, List<(int dim, int order)> activeDims, int startIdx)
    {
        if (startIdx >= activeDims.Count)
            return EvalCore(point);

        var (d, order) = activeDims[startIdx];
        double h = FdStep(d);
        double[] pt = NudgePoint(point, d, h);

        if (order == 1)
        {
            double[] ptPlus = (double[])pt.Clone();
            double[] ptMinus = (double[])pt.Clone();
            ptPlus[d] += h;
            ptMinus[d] -= h;
            double fPlus = FdNested(ptPlus, activeDims, startIdx + 1);
            double fMinus = FdNested(ptMinus, activeDims, startIdx + 1);
            return (fPlus - fMinus) / (2.0 * h);
        }
        else if (order == 2)
        {
            double[] ptPlus = (double[])pt.Clone();
            double[] ptMinus = (double[])pt.Clone();
            ptPlus[d] += h;
            ptMinus[d] -= h;
            double fPlus = FdNested(ptPlus, activeDims, startIdx + 1);
            double fCenter = FdNested(pt, activeDims, startIdx + 1);
            double fMinus = FdNested(ptMinus, activeDims, startIdx + 1);
            return (fPlus - 2.0 * fCenter + fMinus) / (h * h);
        }
        else
        {
            throw new ArgumentException($"Derivative order {order} not supported (use 1 or 2)");
        }
    }

    // ------------------------------------------------------------------
    // Error estimation
    // ------------------------------------------------------------------

    /// <summary>
    /// Estimate interpolation error from Chebyshev coefficient cores.
    /// Sum of max|core[:, -1, :]| per dimension.
    /// </summary>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double ErrorEstimate()
    {
        CheckBuilt();

        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;

        double totalError = 0.0;
        for (int d = 0; d < _numDimensions; d++)
        {
            var core = _coeffCores![d];
            int lastNode = core.NNodes - 1;
            double maxLast = 0;
            for (int i = 0; i < core.RLeft; i++)
                for (int k = 0; k < core.RRight; k++)
                {
                    double v = Math.Abs(core[i, lastNode, k]);
                    if (v > maxLast) maxLast = v;
                }
            totalError += maxLast;
        }

        _cachedErrorEstimate = totalError;
        return totalError;
    }

    // ------------------------------------------------------------------
    // Integration (Phase 5 — PyChebyshev v0.17)
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the TT-approximated function over selected dimensions.
    /// Per-dim Fejér-1 quadrature is applied to the value-space cores
    /// (Chebyshev coefficient cores are converted to value cores via
    /// <see cref="TensorTrainKernel.CoeffCoreToValueCore"/> before contraction).
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all (full integration → scalar).</param>
    /// <param name="bounds">Sub-interval bounds per dim (positional with sorted dims). Null = full domain.</param>
    /// <returns>A boxed <c>double</c> when every dim is integrated; otherwise a new <see cref="ChebyshevTT"/> over surviving dims.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If <paramref name="dims"/> contains out-of-range indices, or <paramref name="bounds"/> are invalid. Duplicate <paramref name="dims"/> entries are silently deduplicated (matches <see cref="ChebyshevApproximation.Integrate"/>).</exception>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        CheckBuilt();

        // Normalize dims: null = all, sort + deduplicate, validate range.
        // dims are in USER (original-dim) frame; translate to storage positions.
        int[] sortedUserDims;
        if (dims == null)
            sortedUserDims = Enumerable.Range(0, _numDimensions).ToArray();
        else
            sortedUserDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedUserDims)
        {
            if (d < 0 || d >= _numDimensions)
                throw new ArgumentException(
                    $"dim {d} out-of-range [0, {_numDimensions - 1}]");
        }

        // Translate user dims to storage positions.
        // Cores/_domain/_nNodes are all in storage frame after Reorder().
        // Two distinct arrays are needed:
        //   - storagePosForBounds: positional with `bounds` (preserves user-sorted order
        //     so bounds[i] still pairs with the storage position derived from sortedUserDims[i]).
        //   - sortedStoragePos: storage-position-ascending, used for chain-walk iteration.
        // Mirrors Python tensor_train.py:1565-1568 which keeps these two arrays distinct.
        int[] storagePosForBounds;
        int[] sortedStoragePos;
        if (IsIdentityDimOrder())
        {
            storagePosForBounds = sortedUserDims;
            sortedStoragePos = sortedUserDims;
        }
        else
        {
            storagePosForBounds = sortedUserDims
                .Select(ud => Array.IndexOf(_dimOrder, ud))
                .ToArray();
            sortedStoragePos = (int[])storagePosForBounds.Clone();
            Array.Sort(sortedStoragePos);
        }

        // v0.21.1: pre-validate bounds against user-frame domain so error messages
        // reference user-frame dim indices (issue #20). The downstream NormalizeBounds
        // would otherwise report storage-frame indices when _dimOrder is non-identity.
        // Python source: spec §4.5.
        if (bounds != null && bounds.Length > 0)
        {
            if (bounds.Length != sortedUserDims.Length)
                throw new ArgumentException(
                    $"bounds length {bounds.Length} != dims length {sortedUserDims.Length}");
            for (int i = 0; i < bounds.Length; i++)
            {
                int userDim = sortedUserDims[i];
                int storageDim = Array.IndexOf(_dimOrder, userDim);
                double lo = _domain[storageDim][0], hi = _domain[storageDim][1];
                var bd = bounds[i];
                if (bd.lo > bd.hi)
                    throw new ArgumentException($"bounds lo={bd.lo} > hi={bd.hi} for dim {userDim}");
                if (bd.lo < lo - 1e-14 || bd.hi > hi + 1e-14)
                    throw new ArgumentException(
                        $"bounds ({bd.lo}, {bd.hi}) outside domain [{lo}, {hi}] for dim {userDim}");
            }
        }

        // Validate bounds against storage-frame domain.
        // NormalizeBounds is positional with `bounds`, so we pass storagePosForBounds
        // (NOT sortedStoragePos — re-sorting would break the bounds[i] ↔ dim[i] pairing
        // when _dimOrder is non-identity).
        var perDimBounds = Internal.Calculus.NormalizeBounds(storagePosForBounds, bounds, _domain);
        var storagePosToIdx = new Dictionary<int, int>();
        for (int i = 0; i < storagePosForBounds.Length; i++)
            storagePosToIdx[storagePosForBounds[i]] = i;

        // Compute scaled quadrature weights per integrated storage position.
        // Cores live in coefficient space — convert each integrated core to
        // value space before applying weights.
        var weightsPerStorage = new Dictionary<int, double[]>();
        foreach (int sp in sortedStoragePos)
        {
            int n = _nNodes[sp];
            double a = _domain[sp][0], b = _domain[sp][1];
            double scale = (b - a) / 2.0;
            var bd = perDimBounds[storagePosToIdx[sp]];
            double[] w;
            if (bd == null)
            {
                w = Internal.Calculus.ComputeFejer1Weights(n);
            }
            else
            {
                double tLo = 2.0 * (bd.Value.lo - a) / (b - a) - 1.0;
                double tHi = 2.0 * (bd.Value.hi - a) / (b - a) - 1.0;
                w = Internal.Calculus.ComputeSubIntervalWeights(n, tLo, tHi);
            }
            for (int i = 0; i < w.Length; i++) w[i] *= scale;
            weightsPerStorage[sp] = w;
        }

        // Per-integrated-dim contraction: coefficient core -> value core -> M_k.
        var contracted = new Dictionary<int, double[,]>();
        foreach (int sp in sortedStoragePos)
        {
            var valueCore = Internal.TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![sp]);
            contracted[sp] = Internal.Calculus.IntegrateTtAlongDim(valueCore, weightsPerStorage[sp]);
        }

        if (sortedStoragePos.Length == _numDimensions)
        {
            // Full integration: chain-multiply all M_k matrices left-to-right.
            // contracted[sortedStoragePos[0]] is shape (rL_0=1, rR_0); after all multiplications,
            // result is (1, 1).
            double[,] result = contracted[sortedStoragePos[0]];
            for (int i = 1; i < sortedStoragePos.Length; i++)
                result = MatMul(result, contracted[sortedStoragePos[i]]);
            return result[0, 0];
        }

        // Partial integration: walk the TT chain, absorbing each contracted
        // matrix into a neighboring kept core's left rank dim (Python
        // tensor_train.py:1582-1608).
        var integratedSet = new HashSet<int>(sortedStoragePos);
        var newCores = new List<Internal.TensorTrainKernel.TtCore>();
        double[,]? pending = null;

        for (int k = 0; k < _numDimensions; k++)
        {
            if (integratedSet.Contains(k))
            {
                var M = contracted[k];
                if (pending != null) M = MatMul(pending, M);
                pending = M;
                continue;
            }
            // k is a kept storage position — absorb any pending matrix into this core's left rank.
            var core = _coeffCores![k].Copy();
            if (pending != null)
            {
                core = AbsorbLeft(pending, core);
                pending = null;
            }
            newCores.Add(core);
        }

        // Trailing pending: absorb into the last kept core's right rank.
        if (pending != null && newCores.Count > 0)
            newCores[newCores.Count - 1] = AbsorbRight(newCores[newCores.Count - 1], pending);

        // Construct result TT.
        int[] keptStoragePositions = Enumerable.Range(0, _numDimensions)
            .Where(k => !integratedSet.Contains(k))
            .ToArray();
        var newDomain = keptStoragePositions.Select(k => (double[])_domain[k].Clone()).ToArray();
        var newNNodes = keptStoragePositions.Select(k => _nNodes[k]).ToArray();

        // Build result _dimOrder: keptStoragePositions[i] was storage position k,
        // which holds original dim _dimOrder[k]. Renumber surviving original dims to [0..m-1].
        var integratedUserDimsSet = new HashSet<int>(sortedUserDims);
        var survivingOrigDims = keptStoragePositions.Select(k => _dimOrder[k]).ToArray();
        // Renumber: sort surviving original dims ascending, assign 0..m-1.
        var sortedSurvivors = survivingOrigDims.Distinct().OrderBy(d => d).ToArray();
        var dimIndex = new Dictionary<int, int>();
        for (int i = 0; i < sortedSurvivors.Length; i++) dimIndex[sortedSurvivors[i]] = i;
        var newDimOrder = survivingOrigDims.Select(d => dimIndex[d]).ToArray();

        var partialResult = BuildIntegrateResult(newCores.ToArray(), newDomain, newNNodes);
        partialResult._dimOrder = newDimOrder;
        return partialResult;
    }

    /// <summary>
    /// Matrix-times-core contraction along the core's left rank dim:
    /// <c>result[l, j, s] = Σ_r M[l, r] * core[r, j, s]</c>.
    /// Used by partial Integrate to absorb a pending matrix into the next kept core.
    /// </summary>
    private static Internal.TensorTrainKernel.TtCore AbsorbLeft(
        double[,] M, Internal.TensorTrainKernel.TtCore core)
    {
        int newRLeft = M.GetLength(0);
        int absorbed = M.GetLength(1);
        if (absorbed != core.RLeft)
            throw new ArgumentException(
                $"AbsorbLeft shape mismatch: M is ({newRLeft}, {absorbed}); core.RLeft={core.RLeft}");
        int n = core.NNodes, rR = core.RRight;
        var result = new Internal.TensorTrainKernel.TtCore(newRLeft, n, rR);
        for (int l = 0; l < newRLeft; l++)
            for (int j = 0; j < n; j++)
                for (int s = 0; s < rR; s++)
                {
                    double acc = 0;
                    for (int r = 0; r < absorbed; r++)
                        acc += M[l, r] * core[r, j, s];
                    result[l, j, s] = acc;
                }
        return result;
    }

    /// <summary>
    /// Core-times-matrix contraction along the core's right rank dim:
    /// <c>result[l, j, r] = Σ_s core[l, j, s] * M[s, r]</c>.
    /// Used by partial Integrate to absorb a trailing pending matrix into the last kept core.
    /// </summary>
    private static Internal.TensorTrainKernel.TtCore AbsorbRight(
        Internal.TensorTrainKernel.TtCore core, double[,] M)
    {
        int absorbed = M.GetLength(0);
        int newRRight = M.GetLength(1);
        if (absorbed != core.RRight)
            throw new ArgumentException(
                $"AbsorbRight shape mismatch: core.RRight={core.RRight}; M is ({absorbed}, {newRRight})");
        int rL = core.RLeft, n = core.NNodes;
        var result = new Internal.TensorTrainKernel.TtCore(rL, n, newRRight);
        for (int l = 0; l < rL; l++)
            for (int j = 0; j < n; j++)
                for (int r = 0; r < newRRight; r++)
                {
                    double acc = 0;
                    for (int s = 0; s < absorbed; s++)
                        acc += core[l, j, s] * M[s, r];
                    result[l, j, r] = acc;
                }
        return result;
    }

    /// <summary>
    /// Construct a partial-integrate result TT, inheriting all Phase 4 ergonomics
    /// fields (descriptor, additionalData, maxDerivativeOrder) and Method (D3, D6).
    /// Mirrors <see cref="BuildResultFromCores"/> with extra inheritance.
    /// </summary>
    private ChebyshevTT BuildIntegrateResult(
        Internal.TensorTrainKernel.TtCore[] cores, double[][] newDomain, int[] newNNodes)
    {
        // Use the existing BuildResultFromCores then patch in ergonomics fields.
        var result = BuildResultFromCores(cores, newDomain, newNNodes);
        // Phase 4 ergonomics passthrough (D6).
        result._descriptor = _descriptor;
        result._additionalData = _additionalData;
        result._maxDerivativeOrder = _maxDerivativeOrder;
        return result;
    }

    /// <summary>
    /// Plain (m, k) x (k, n) -> (m, n) matrix multiply for the Integrate path.
    /// </summary>
    private static double[,] MatMul(double[,] a, double[,] b)
    {
        int m = a.GetLength(0);
        int k = a.GetLength(1);
        int kB = b.GetLength(0);
        int n = b.GetLength(1);
        if (k != kB)
            throw new ArgumentException(
                $"MatMul shape mismatch: ({m}, {k}) x ({kB}, {n})");
        var result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int l = 0; l < k; l++) s += a[i, l] * b[l, j];
                result[i, j] = s;
            }
        return result;
    }

    // ------------------------------------------------------------------
    // Roots / Minimize / Maximize (Phase 7 — PyChebyshev v0.21.1)
    // ------------------------------------------------------------------

    /// <summary>
    /// Return Domain reordered into user-frame indexing. For canonical
    /// _dimOrder, this returns an array semantically equivalent to Domain.
    /// For non-identity _dimOrder, _domain[s] is the storage-frame domain at
    /// storage position s; user-frame dim u lives at storage position
    /// Array.IndexOf(_dimOrder, u).
    /// </summary>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1737-1747</c>.
    /// </remarks>
    private double[][] UserFrameDomain()
    {
        var result = new double[_numDimensions][];
        for (int u = 0; u < _numDimensions; u++)
        {
            int s = Array.IndexOf(_dimOrder, u);
            result[u] = _domain[s];
        }
        return result;
    }

    /// <summary>
    /// Build a 1-D ChebyshevApproximation from this 1-D TT. Uses ToDense() to
    /// extract the values vector (which already applies the inverse permutation
    /// so values are in user frame), then constructs a ChebyshevApproximation
    /// via FromValues.
    /// </summary>
    /// <remarks>
    /// Precondition: this TT must be 1-D. Call Slice() to reduce a multi-D
    /// TT to 1-D before calling this helper.
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1704-1735</c>.
    /// </remarks>
    private ChebyshevApproximation To1DChebyshev()
    {
        if (_numDimensions != 1)
            throw new InvalidOperationException(
                $"To1DChebyshev requires a 1-D TT, got {_numDimensions}-D");

        double[] values = ToDense();
        double a = _domain[0][0];
        double b = _domain[0][1];
        return ChebyshevApproximation.FromValues(
            values,
            numDimensions: 1,
            domain: new[] { new[] { a, b } },
            nNodes: new[] { _nNodes[0] });
    }

    /// <summary>
    /// Find all real roots of the TT-approximated function along a specified dimension.
    /// Reduces to a 1-D problem by slicing all other dimensions to their fixed
    /// values, then delegates to <see cref="ChebyshevApproximation.Roots"/>.
    /// </summary>
    /// <param name="dim">User-frame dimension. For 1-D TTs, defaults to 0.</param>
    /// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
    /// user-frame dims except <paramref name="dim"/>. Validated against
    /// user-frame domain.</param>
    /// <returns>Sorted real root locations in the physical domain. Empty if no roots.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If validation fails.</exception>
    /// <remarks>
    /// Under non-identity <see cref="DimOrder"/>, dim and fixedDims keys translate
    /// to storage frame transparently inside <see cref="Slice"/> and
    /// <see cref="ToDense"/>.
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1749-1790</c>.
    /// </remarks>
    public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        CheckBuilt();

        var (_, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

        // Sort descending by dim index so that each sequential Slice call sees
        // valid user-frame indices even after earlier dims are removed.
        // This mirrors the Python approach of sorting by descending storage
        // position in tensor_train.py:2078.
        var sortedParams = ((int dimIndex, double value)[])sliceParams.Clone();
        Array.Sort(sortedParams, (a, b) => b.dimIndex.CompareTo(a.dimIndex));

        ChebyshevTT sliced = this;
        foreach (var (sliceDim, sliceValue) in sortedParams)
            sliced = sliced.Slice(sliceDim, sliceValue);

        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Roots();
    }

    /// <summary>
    /// Find the minimum value of the TT along a user-frame dimension.
    /// </summary>
    /// <param name="dim">User-frame dimension. For 1-D TTs, defaults to 0.</param>
    /// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
    /// user-frame dims except <paramref name="dim"/>. Validated against
    /// user-frame domain.</param>
    /// <returns>Tuple of (minimum value, location where minimum is achieved).</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If validation fails.</exception>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1792-1831</c>.
    /// </remarks>
    public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        CheckBuilt();

        var (_, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

        var sortedParams = ((int dimIndex, double value)[])sliceParams.Clone();
        Array.Sort(sortedParams, (a, b) => b.dimIndex.CompareTo(a.dimIndex));

        ChebyshevTT sliced = this;
        foreach (var (sliceDim, sliceValue) in sortedParams)
            sliced = sliced.Slice(sliceDim, sliceValue);

        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Minimize();
    }

    /// <summary>
    /// Find the maximum value of the TT along a user-frame dimension.
    /// </summary>
    /// <param name="dim">User-frame dimension. For 1-D TTs, defaults to 0.</param>
    /// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
    /// user-frame dims except <paramref name="dim"/>. Validated against
    /// user-frame domain.</param>
    /// <returns>Tuple of (maximum value, location where maximum is achieved).</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If validation fails.</exception>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1833-1872</c>.
    /// </remarks>
    public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        CheckBuilt();

        var (_, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

        var sortedParams = ((int dimIndex, double value)[])sliceParams.Clone();
        Array.Sort(sortedParams, (a, b) => b.dimIndex.CompareTo(a.dimIndex));

        ChebyshevTT sliced = this;
        foreach (var (sliceDim, sliceValue) in sortedParams)
            sliced = sliced.Slice(sliceDim, sliceValue);

        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Maximize();
    }

    // ------------------------------------------------------------------
    // Canonicalization (Phase 2 — PyChebyshev v0.13)
    // ------------------------------------------------------------------

    /// <summary>
    /// Left-orthogonalize cores [0..position-1] in place by absorbing each
    /// core's R factor into the next core's left bond. The represented tensor
    /// is unchanged.
    /// </summary>
    /// <param name="position">Pivot index, must be in [1, NumDimensions - 1].</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If position is outside [1, NumDimensions - 1].</exception>
    public void OrthLeft(int position)
    {
        CheckBuilt();
        if (position < 1 || position >= _numDimensions)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"position must be in [1, {_numDimensions - 1}] for OrthLeft, got {position}");
        TensorTrainKernel.OrthLeftSweep(_coeffCores!, position);
        _cachedErrorEstimate = null;
        // TT ranks may change (QR reduces rank to min(rL*n, rR)); refresh.
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores![i].RRight;
    }

    /// <summary>
    /// Right-orthogonalize cores [position+1..NumDimensions-1] in place by
    /// absorbing each core's L factor into the previous core's right bond.
    /// The represented tensor is unchanged.
    /// </summary>
    /// <param name="position">Pivot index, must be in [0, NumDimensions - 2].</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If position is outside [0, NumDimensions - 2].</exception>
    public void OrthRight(int position)
    {
        CheckBuilt();
        if (position < 0 || position >= _numDimensions - 1)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"position must be in [0, {_numDimensions - 2}] for OrthRight, got {position}");
        TensorTrainKernel.OrthRightSweep(_coeffCores!, position);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores![i].RRight;
    }

    /// <summary>
    /// Frobenius inner product of the Chebyshev coefficient tensors of two TTs.
    /// Both TTs must share the same <see cref="NumDimensions"/>, <see cref="Domain"/>,
    /// and <see cref="NNodes"/>.
    /// </summary>
    /// <param name="other">The other TT.</param>
    /// <returns>Σ_{i_1,…,i_d} C_self[i] * C_other[i].</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="other"/> is null.</exception>
    /// <exception cref="InvalidOperationException">If either TT has not been built.</exception>
    /// <exception cref="ArgumentException">If domain or nNodes do not match.</exception>
    public double InnerProduct(ChebyshevTT other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        CheckBuilt();
        other.CheckBuilt();
        if (other._numDimensions != _numDimensions)
            throw new ArgumentException(
                $"InnerProduct requires matching numDimensions; got {_numDimensions} vs {other._numDimensions}");
        for (int d = 0; d < _numDimensions; d++)
        {
            if (other._nNodes[d] != _nNodes[d])
                throw new ArgumentException(
                    $"InnerProduct requires matching nNodes; got [{string.Join(", ", _nNodes)}] vs [{string.Join(", ", other._nNodes)}]");
            if (other._domain[d][0] != _domain[d][0] || other._domain[d][1] != _domain[d][1])
                throw new ArgumentException(
                    $"InnerProduct requires matching domain at dim {d}; got [{_domain[d][0]}, {_domain[d][1]}] vs [{other._domain[d][0]}, {other._domain[d][1]}]");
        }
        // v0.21.1: strict _dimOrder check. Two TTs with different _dimOrder represent
        // the same underlying interpolant under different storage permutations; the
        // raw core-by-core contraction is not the inner product of the interpolants.
        // Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:1488-1495.
        if (!_dimOrder.SequenceEqual(other._dimOrder))
            throw new ArgumentException(
                $"InnerProduct requires matching _dimOrder; " +
                $"got [{string.Join(", ", _dimOrder)}] vs [{string.Join(", ", other._dimOrder)}]. " +
                $"Call other.Reorder(self.DimOrder) first.");
        return TensorTrainAlgebra.InnerProductCores(_coeffCores!, other._coeffCores!);
    }

    /// <summary>
    /// Refine the TT at its current rank via ALS sweeps. Works on any built TT
    /// (from "cross", "svd", or "als"). Rank does not grow; only per-core
    /// coefficients are refined.
    /// </summary>
    /// <param name="tolerance">Stop when inner-sweep relative change falls below this.</param>
    /// <param name="maxIter">Maximum number of outer ALS sweeps.</param>
    /// <param name="verbose">Print per-sweep residuals.</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called or if <c>Function</c> is null (loaded TT).</exception>
    /// <exception cref="ArgumentException">If the function returns NaN or Infinity at a sampled grid point.</exception>
    public void RunCompletion(double tolerance = 1e-8, int maxIter = 50, bool verbose = false)
    {
        CheckBuilt();
        ValidatePositiveFiniteTolerance(tolerance, nameof(tolerance));
        ValidatePositiveInteger(maxIter, nameof(maxIter));
        var function = GetRequiredFunction(nameof(RunCompletion));

        // Convert coefficient cores back to value cores at Chebyshev Type I nodes.
        var valueCores = new TensorTrainKernel.TtCore[_numDimensions];
        for (int k = 0; k < _numDimensions; k++)
            valueCores[k] = TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![k]);

        // Rebuild the grids that Build() used.
        var grids = new double[_numDimensions][];
        for (int k = 0; k < _numDimensions; k++)
            grids[k] = BarycentricKernel.MakeNodesForDim(_domain[k][0], _domain[k][1], _nNodes[k]);

        var cache = new Dictionary<Internal.TupleKey, double>();

        Func<int[], double> evalsAt = idx =>
        {
            var key = new Internal.TupleKey(idx);
            if (!cache.TryGetValue(key, out double v))
            {
                var pt = new double[_numDimensions];
                for (int i = 0; i < _numDimensions; i++) pt[i] = grids[i][idx[i]];
                v = EvaluateFiniteFunction(function, pt, nameof(RunCompletion));
                cache[key] = v;
            }
            return v;
        };

        TensorTrainKernel.AlsFixedRankSweep(
            valueCores, evalsAt, _nNodes, tolerance: tolerance, maxIter: maxIter, verbose: verbose);

        // Convert back to coefficient cores.
        _coeffCores = TensorTrainKernel.ValueToCoeffCores(valueCores);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    // ------------------------------------------------------------------
    // Static factories (Phase 2 — PyChebyshev v0.18)
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute the Chebyshev Type I node positions per dimension scaled to
    /// the user's domain. Static factory matching <see cref="ChebyshevApproximation.Nodes"/>.
    /// </summary>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension.</param>
    /// <returns>(NodesPerDim[d][j], Shape[d]) — node arrays in ascending order.</returns>
    public static (double[][] NodesPerDim, int[] Shape) Nodes(
        int numDimensions, double[][] domain, int[] nNodes)
    {
        ValidateFixedGridArguments(numDimensions, domain, nNodes);

        var nodesPerDim = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            nodesPerDim[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
        return (nodesPerDim, (int[])nNodes.Clone());
    }

    /// <summary>
    /// Build a TT directly from a precomputed dense tensor (skips function evaluation).
    /// Uses TT-SVD for compression. The resulting TT has <c>Function = null</c>.
    /// </summary>
    /// <param name="tensorValues">Flat row-major dense tensor of length Π nNodes.</param>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension.</param>
    /// <param name="maxRank">Maximum positive TT rank (default 10).</param>
    /// <param name="tolerance">Finite non-negative SVD truncation tolerance (default 1e-6).</param>
    /// <exception cref="ArgumentException">If tensorValues length doesn't match Π nNodes, or contains NaN/Infinity.</exception>
    public static ChebyshevTT FromValues(
        double[] tensorValues,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6)
    {
        ArgumentNullException.ThrowIfNull(tensorValues);
        ValidateFixedGridArguments(numDimensions, domain, nNodes);
        ValidatePositiveRank(maxRank, nameof(maxRank));
        ValidateNonNegativeFiniteTolerance(tolerance, nameof(tolerance));
        long expected = TensorShape.CheckedProduct(nNodes, nameof(FromValues));
        if (tensorValues.LongLength != expected)
            throw new ArgumentException(
                $"tensorValues has shape mismatch: length {tensorValues.LongLength} but expected Π nNodes = {expected}");
        for (int i = 0; i < tensorValues.Length; i++)
            if (!double.IsFinite(tensorValues[i]))
                throw new ArgumentException($"tensorValues[{i}] is NaN or Infinity (must be finite)");

        var valueCores = TensorTrainExtrude.FromValuesTtSvd(tensorValues, nNodes, maxRank, tolerance);
        var coeffCores = TensorTrainKernel.ValueToCoeffCores(valueCores);

        var ttRanks = new int[numDimensions + 1];
        ttRanks[0] = 1;
        for (int i = 0; i < numDimensions; i++) ttRanks[i + 1] = coeffCores[i].RRight;

        var tt = new ChebyshevTT(
            numDimensions: numDimensions,
            domain: CloneDomain(domain),
            nNodes: (int[])nNodes.Clone(),
            maxRank: maxRank,
            tolerance: tolerance,
            maxSweeps: 0,
            coeffCores: coeffCores,
            ttRanks: ttRanks,
            buildTime: 0.0,
            totalBuildEvals: 0);
        tt.Method = "svd";
        return tt;
    }

    /// <summary>
    /// Internal accessor for tests: return (rLeft, nNodes, rRight, flat data) of
    /// core <paramref name="k"/>. Exposes the live data buffer (not a copy).
    /// </summary>
    internal (int RLeft, int NNodes, int RRight, double[] Data) GetCoreShape(int k)
    {
        CheckBuilt();
        var c = _coeffCores![k];
        return (c.RLeft, c.NNodes, c.RRight, c.Data);
    }

    // ------------------------------------------------------------------
    // Materialization, extrusion, slicing (Phase 2 — PyChebyshev v0.18)
    // ------------------------------------------------------------------

    /// <summary>
    /// Materialize the TT chain into a full row-major dense tensor.
    /// Length is Π NNodes; <c>dense[flat]</c> equals <c>Eval(point_at_grid_idx)</c>
    /// where flat is the row-major index into the grid shape.
    /// Use sparingly: storage is Π NNodes doubles.
    /// </summary>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="OverflowException">If Π NNodes * 8 exceeds <c>int.MaxValue</c>.</exception>
    public double[] ToDense()
    {
        CheckBuilt();
        int total = FullGridSizeAsIntForMaterialization(nameof(ToDense));
        long byteSize = TensorShape.CheckedByteSize(total, sizeof(double), nameof(ToDense));
        if (byteSize > int.MaxValue)
            throw new OverflowException(
                $"ToDense would allocate {total} doubles ({byteSize} bytes), exceeding int.MaxValue. " +
                "Use ToDense for low-dimensional inspection only.");

        var dense = TensorTrainExtrude.ToDenseEinsumChain(_coeffCores!, _nNodes);
        if (IsIdentityDimOrder()) return dense;

        // dense has axes in storage order (_nNodes[k] gives size of storage axis k).
        // Transpose into original-dim order: output[flat over orig dims] = dense[flat over storage dims]
        // where storageIdx[k] = origIdx[_dimOrder[k]] (storage pos k holds orig dim _dimOrder[k]).
        int n = _numDimensions;
        // Build origNNodes: size of each original-dim axis.
        var origNNodes = new int[n];
        for (int k = 0; k < n; k++) origNNodes[_dimOrder[k]] = _nNodes[k];

        var result = new double[total];
        var origIdx = new int[n];
        var storageIdx = new int[n];
        for (long flat = 0; flat < total; flat++)
        {
            long rem = flat;
            for (int k = n - 1; k >= 0; k--)
            {
                origIdx[k] = (int)(rem % origNNodes[k]);
                rem /= origNNodes[k];
            }
            for (int k = 0; k < n; k++) storageIdx[k] = origIdx[_dimOrder[k]];
            long storageFlat = 0;
            for (int k = 0; k < n; k++) storageFlat = storageFlat * _nNodes[k] + storageIdx[k];
            result[flat] = dense[storageFlat];
        }
        return result;
    }

    /// <summary>
    /// Insert a new dimension at index <paramref name="dim"/> where the function
    /// is constant. The extruded TT evaluates identically to the original over
    /// the existing dimensions, regardless of the new dimension's coordinate.
    /// </summary>
    /// <param name="dim">Insertion index, 0 &lt;= dim &lt;= NumDimensions.</param>
    /// <param name="newDomain">Domain (lo, hi) for the new dimension.</param>
    /// <param name="newN">Number of nodes for the new dimension.</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If dim is outside [0, NumDimensions].</exception>
    /// <exception cref="ArgumentException">If newDomain.Lo >= newDomain.Hi, or newN &lt; 2.</exception>
    public ChebyshevTT Extrude(int dim, (double Lo, double Hi) newDomain, int newN)
    {
        CheckBuilt();
        if (!double.IsFinite(newDomain.Lo) || !double.IsFinite(newDomain.Hi))
            throw new ArgumentException(
                $"newDomain bounds must be finite; got ({newDomain.Lo}, {newDomain.Hi})",
                nameof(newDomain));
        if (newDomain.Lo >= newDomain.Hi)
            throw new ArgumentException(
                $"newDomain bounds must satisfy lo < hi; got ({newDomain.Lo}, {newDomain.Hi})",
                nameof(newDomain));

        TensorTrainKernel.TtCore[] newCores;
        double[][] newDomainArr;
        int[] newNNodes;
        int[] newDimOrder;

        if (IsIdentityDimOrder())
        {
            // Identity path: insert at storage position dim (preserves v0.18 canonical behavior).
            newCores = TensorTrainExtrude.ExtrudeCores(_coeffCores!, dim, newN);
            newDomainArr = new double[_numDimensions + 1][];
            for (int k = 0; k < dim; k++) newDomainArr[k] = (double[])_domain[k].Clone();
            newDomainArr[dim] = new[] { newDomain.Lo, newDomain.Hi };
            for (int k = dim; k < _numDimensions; k++) newDomainArr[k + 1] = (double[])_domain[k].Clone();
            newNNodes = new int[_numDimensions + 1];
            for (int k = 0; k < dim; k++) newNNodes[k] = _nNodes[k];
            newNNodes[dim] = newN;
            for (int k = dim; k < _numDimensions; k++) newNNodes[k + 1] = _nNodes[k];
            newDimOrder = Enumerable.Range(0, _numDimensions + 1).ToArray();
        }
        else
        {
            // Non-identity path: append the new core at storage end; encode
            // user's dim via _dimOrder (mirrors Python tensor_train.py:1793-1804).
            int storagePos = _numDimensions; // append at end
            newCores = TensorTrainExtrude.ExtrudeCores(_coeffCores!, storagePos, newN);
            newDomainArr = new double[_numDimensions + 1][];
            for (int k = 0; k < _numDimensions; k++) newDomainArr[k] = (double[])_domain[k].Clone();
            newDomainArr[storagePos] = new[] { newDomain.Lo, newDomain.Hi };
            newNNodes = new int[_numDimensions + 1];
            for (int k = 0; k < _numDimensions; k++) newNNodes[k] = _nNodes[k];
            newNNodes[storagePos] = newN;
            // Update _dimOrder: increment existing entries >= dim by 1, then append dim.
            newDimOrder = new int[_numDimensions + 1];
            for (int k = 0; k < _numDimensions; k++)
                newDimOrder[k] = _dimOrder[k] < dim ? _dimOrder[k] : _dimOrder[k] + 1;
            newDimOrder[_numDimensions] = dim;
        }

        var extruded = BuildResultFromCores(newCores, newDomainArr, newNNodes);
        extruded._dimOrder = newDimOrder;
        return extruded;
    }

    /// <summary>
    /// Fix dimension <paramref name="dim"/> at <paramref name="value"/>, returning
    /// a TT over the remaining (NumDimensions - 1) dimensions.
    /// </summary>
    /// <param name="dim">Dimension to slice, 0 &lt;= dim &lt; NumDimensions.</param>
    /// <param name="value">Value at which to fix the dimension; must lie within the domain.</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If dim is out of range or value is outside the domain.</exception>
    /// <exception cref="InvalidOperationException">If NumDimensions == 1 (would produce 0D result).</exception>
    public ChebyshevTT Slice(int dim, double value)
    {
        CheckBuilt();
        if (dim < 0 || dim >= _numDimensions)
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"dim={dim} out of range [0, {_numDimensions - 1}]");

        // dim is in user (original) frame; translate to storage position.
        // Cores/_domain/_nNodes are all in storage frame after Reorder().
        int storagePos = !IsIdentityDimOrder() ? Array.IndexOf(_dimOrder, dim) : dim;

        double lo = _domain[storagePos][0], hi = _domain[storagePos][1];
        if (!double.IsFinite(value))
            throw new ArgumentOutOfRangeException(nameof(value),
                $"Slice value {value} for dim {dim} must be finite");
        if (value < lo || value > hi)
            throw new ArgumentOutOfRangeException(nameof(value),
                $"Slice value {value} for dim {dim} is outside domain [{lo}, {hi}]");

        if (_numDimensions == 1)
            throw new InvalidOperationException("Cannot slice a 1D TT (would produce 0D result).");

        double[] nodes = BarycentricKernel.MakeNodesForDim(lo, hi, _nNodes[storagePos]);
        var newCores = TensorTrainExtrude.SliceCores(_coeffCores!, storagePos, value, nodes);

        var newDomain = new double[_numDimensions - 1][];
        var newNNodes = new int[_numDimensions - 1];
        int writeIdx = 0;
        for (int k = 0; k < _numDimensions; k++)
        {
            if (k == storagePos) continue;
            newDomain[writeIdx] = (double[])_domain[k].Clone();
            newNNodes[writeIdx] = _nNodes[k];
            writeIdx++;
        }

        // Build result _dimOrder: drop storagePos from _dimOrder, then renumber so
        // surviving original-dim indices form a permutation of [0, n-2].
        // dropped original-dim index: _dimOrder[storagePos]
        int droppedOrigDim = _dimOrder[storagePos];
        var survivingOrigDims = new int[_numDimensions - 1];
        int si = 0;
        for (int k = 0; k < _numDimensions; k++)
            if (k != storagePos) survivingOrigDims[si++] = _dimOrder[k];

        // Renumber: for each surviving original-dim, its new index = count of
        // original-dims < it that are NOT the dropped one.
        int counter = 0;
        var newDimIndex = new int[_numDimensions];
        for (int origDim = 0; origDim < _numDimensions; origDim++)
        {
            if (origDim == droppedOrigDim) continue;
            newDimIndex[origDim] = counter++;
        }
        var newDimOrder = new int[_numDimensions - 1];
        for (int k = 0; k < newDimOrder.Length; k++)
            newDimOrder[k] = newDimIndex[survivingOrigDims[k]];

        var sliced = BuildResultFromCores(newCores, newDomain, newNNodes);
        sliced._dimOrder = newDimOrder;
        return sliced;
    }

    /// <summary>
    /// Internal helper: assemble a fresh ChebyshevTT from a set of coefficient cores.
    /// Used by Extrude, Slice, and the algebra operators (Tasks 9 + 10).
    /// </summary>
    internal ChebyshevTT BuildResultFromCores(
        TensorTrainKernel.TtCore[] cores, double[][] newDomain, int[] newNNodes)
    {
        int newD = newNNodes.Length;
        var ttRanks = new int[newD + 1];
        ttRanks[0] = 1;
        for (int i = 0; i < newD; i++) ttRanks[i + 1] = cores[i].RRight;
        var tt = new ChebyshevTT(
            numDimensions: newD,
            domain: newDomain,
            nNodes: newNNodes,
            maxRank: _maxRank,
            tolerance: _tolerance,
            maxSweeps: _maxSweeps,
            coeffCores: cores,
            ttRanks: ttRanks,
            buildTime: 0.0,
            totalBuildEvals: 0);
        tt.Method = Method;
        return tt;
    }

    // ------------------------------------------------------------------
    // Scalar algebra (Phase 2 — PyChebyshev v0.18.c)
    // ------------------------------------------------------------------

    /// <summary>Scalar multiplication: <c>tt * scalar</c>.</summary>
    public static ChebyshevTT operator *(ChebyshevTT tt, double scalar)
    {
        if (tt is null) throw new ArgumentNullException(nameof(tt));
        tt.CheckBuilt();
        var newCores = TensorTrainAlgebra.ScalarMulCores(tt._coeffCores!, scalar);
        var domainCopy = tt._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])tt._nNodes.Clone();
        var result = tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
        result._dimOrder = (int[])tt._dimOrder.Clone();
        return result;
    }

    /// <summary>Scalar multiplication: <c>scalar * tt</c>.</summary>
    public static ChebyshevTT operator *(double scalar, ChebyshevTT tt) => tt * scalar;

    /// <summary>Scalar division: <c>tt / scalar</c>.</summary>
    /// <exception cref="DivideByZeroException">If <paramref name="scalar"/> is zero.</exception>
    public static ChebyshevTT operator /(ChebyshevTT tt, double scalar)
    {
        if (scalar == 0.0)
            throw new DivideByZeroException("Cannot divide ChebyshevTT by zero.");
        return tt * (1.0 / scalar);
    }

    /// <summary>Unary negation: <c>-tt</c>.</summary>
    public static ChebyshevTT operator -(ChebyshevTT tt)
    {
        if (tt is null) throw new ArgumentNullException(nameof(tt));
        tt.CheckBuilt();
        var newCores = TensorTrainAlgebra.NegateCores(tt._coeffCores!);
        var domainCopy = tt._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])tt._nNodes.Clone();
        var result = tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
        result._dimOrder = (int[])tt._dimOrder.Clone();
        return result;
    }

    /// <summary>Scale this TT in place by <paramref name="scalar"/>.</summary>
    public void ScalarMulInPlace(double scalar)
    {
        CheckBuilt();
        TensorTrainAlgebra.ScalarMulCoresInPlace(_coeffCores!, scalar);
        _cachedErrorEstimate = null;
    }

    /// <summary>Divide this TT in place by <paramref name="scalar"/>.</summary>
    /// <exception cref="DivideByZeroException">If <paramref name="scalar"/> is zero.</exception>
    public void ScalarDivInPlace(double scalar)
    {
        if (scalar == 0.0)
            throw new DivideByZeroException("Cannot divide ChebyshevTT by zero.");
        ScalarMulInPlace(1.0 / scalar);
    }

    /// <summary>Negate this TT in place.</summary>
    public void NegateInPlace()
    {
        CheckBuilt();
        TensorTrainAlgebra.NegateCoresInPlace(_coeffCores!);
        _cachedErrorEstimate = null;
    }

    // ------------------------------------------------------------------
    // Binary algebra (Phase 2 — PyChebyshev v0.18.d)
    // ------------------------------------------------------------------

    /// <summary>Default tolerance for TT-SVD rounding after addition/subtraction.</summary>
    public const double DefaultRoundTolerance = 1e-12;

    /// <summary>Validate two TTs share the same grid (numDim, domain, nNodes) and dim_order.</summary>
    private static void CheckCompatible(ChebyshevTT a, ChebyshevTT b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        a.CheckBuilt();
        b.CheckBuilt();
        if (a._numDimensions != b._numDimensions)
            throw new ArgumentException(
                $"Dimension mismatch: {a._numDimensions} vs {b._numDimensions}");
        for (int d = 0; d < a._numDimensions; d++)
        {
            if (a._nNodes[d] != b._nNodes[d])
                throw new ArgumentException(
                    $"nNodes mismatch at dim {d}: {a._nNodes[d]} vs {b._nNodes[d]}");
            if (a._domain[d][0] != b._domain[d][0] || a._domain[d][1] != b._domain[d][1])
                throw new ArgumentException(
                    $"Domain mismatch at dim {d}: [{a._domain[d][0]}, {a._domain[d][1]}] vs [{b._domain[d][0]}, {b._domain[d][1]}]");
        }
        // _dimOrder mismatch: refuse with a hint at Reorder (mirrors Python v0.20.1).
        for (int k = 0; k < a._numDimensions; k++)
            if (a._dimOrder[k] != b._dimOrder[k])
                throw new ArgumentException(
                    $"dim_order mismatch at storage position {k}: {a._dimOrder[k]} vs {b._dimOrder[k]}. " +
                    "Call Reorder() on one operand to align before adding/subtracting.");
    }

    /// <summary>Binary addition: <c>a + b</c>. Result is rounded to the larger of the two TTs' maxRank.</summary>
    public static ChebyshevTT operator +(ChebyshevTT a, ChebyshevTT b)
    {
        CheckCompatible(a, b);
        var summed = TensorTrainAlgebra.AddCores(a._coeffCores!, b._coeffCores!);
        int mr = Math.Max(a._maxRank, b._maxRank);
        var rounded = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        var domainCopy = a._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])a._nNodes.Clone();
        var result = a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
        result._dimOrder = (int[])a._dimOrder.Clone();
        return result;
    }

    /// <summary>Binary subtraction: <c>a - b</c>.</summary>
    public static ChebyshevTT operator -(ChebyshevTT a, ChebyshevTT b)
    {
        CheckCompatible(a, b);
        var negB = TensorTrainAlgebra.NegateCores(b._coeffCores!);
        var summed = TensorTrainAlgebra.AddCores(a._coeffCores!, negB);
        int mr = Math.Max(a._maxRank, b._maxRank);
        var rounded = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        var domainCopy = a._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])a._nNodes.Clone();
        var result = a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
        result._dimOrder = (int[])a._dimOrder.Clone();
        return result;
    }

    /// <summary>In-place addition: <c>this += other</c> followed by TT-SVD rounding.</summary>
    public void AddInPlace(ChebyshevTT other)
    {
        CheckCompatible(this, other);
        var summed = TensorTrainAlgebra.AddCores(_coeffCores!, other._coeffCores!);
        int mr = Math.Max(_maxRank, other._maxRank);
        _coeffCores = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    /// <summary>In-place subtraction: <c>this -= other</c> followed by TT-SVD rounding.</summary>
    public void SubInPlace(ChebyshevTT other)
    {
        CheckCompatible(this, other);
        var negOther = TensorTrainAlgebra.NegateCores(other._coeffCores!);
        var summed = TensorTrainAlgebra.AddCores(_coeffCores!, negOther);
        int mr = Math.Max(_maxRank, other._maxRank);
        _coeffCores = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    /// <summary>Round TT to lower rank in place via TT-SVD recompression.</summary>
    public void RoundInPlace(double tolerance)
    {
        CheckBuilt();
        _coeffCores = TensorTrainAlgebra.RoundCores(_coeffCores!, _maxRank, tolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    // ------------------------------------------------------------------
    // Chebyshev polynomial evaluation
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate Chebyshev polynomials T_0(x), T_1(x), ..., T_{n-1}(x) via three-term recurrence.
    /// </summary>
    private static double[] ChebyshevPolynomials(double x, int n)
    {
        double[] T = new double[n];
        if (n == 0) return T;
        T[0] = 1.0;
        if (n == 1) return T;
        T[1] = x;
        for (int k = 2; k < n; k++)
            T[k] = 2.0 * x * T[k - 1] - T[k - 2];
        return T;
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// <summary>
    /// Save the built TT interpolant to a JSON file.
    /// The original function is not saved — only numerical data needed for evaluation.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public void Save(string path)
    {
        CheckBuilt();

        var state = new TTSerializationState
        {
            Version = GetLibraryVersion(),
            Method = Method,
            NumDimensions = _numDimensions,
            Domain = _domain,
            NNodes = _nNodes,
            MaxRank = _maxRank,
            Tolerance = _tolerance,
            MaxSweeps = _maxSweeps,
            TtRanks = _ttRanks!,
            BuildTime = _buildTime,
            TotalBuildEvals = _totalBuildEvals,
            Cores = new CoreData[_numDimensions],
            Descriptor = _descriptor,
            MaxDerivativeOrder = _maxDerivativeOrder,
            RegisteredDerivativeOrders = _registeredDerivativeOrders.Count > 0
                ? _registeredDerivativeOrders.ToArray()
                : null,
            JsonVersion = 2,
            DimOrder = (int[])_dimOrder.Clone(),
        };

        for (int i = 0; i < _numDimensions; i++)
        {
            var core = _coeffCores![i];
            state.Cores[i] = new CoreData
            {
                RLeft = core.RLeft,
                NNodes = core.NNodes,
                RRight = core.RRight,
                Data = core.Data,
            };
        }

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load a previously saved TT interpolant from a JSON file.
    /// The loaded object can evaluate immediately; no rebuild is needed.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>The restored TT interpolant.</returns>
    /// <exception cref="InvalidOperationException">If the file cannot be deserialized as a ChebyshevTT state.</exception>
    /// <exception cref="InvalidDataException">If the file contains a malformed ChebyshevTT state.</exception>
    public static ChebyshevTT Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<TTSerializationState>(json)
                    ?? throw new InvalidOperationException("Failed to deserialize ChebyshevTT state.");

        int jsonVersion = state.JsonVersion ?? 1;
        int[] dimOrder = state.DimOrder ?? Enumerable.Range(0, state.NumDimensions).ToArray();
        ValidateSerializedState(state, dimOrder);

        var cores = new TensorTrainKernel.TtCore[state.NumDimensions];
        for (int i = 0; i < state.NumDimensions; i++)
        {
            var cd = state.Cores[i];
            cores[i] = new TensorTrainKernel.TtCore(cd.RLeft, cd.NNodes, cd.RRight, cd.Data);
        }

        var tt = new ChebyshevTT(
            state.NumDimensions,
            state.Domain,
            state.NNodes,
            state.MaxRank,
            state.Tolerance,
            state.MaxSweeps,
            cores,
            state.TtRanks,
            state.BuildTime,
            state.TotalBuildEvals,
            // v0.8.0 migration: MaxDerivativeOrder absent in pre-v0.8.0 JSON => null => default 2
            maxDerivativeOrder: state.MaxDerivativeOrder ?? 2);

        tt.Method = state.Method;

        // v0.8.0 migration: Descriptor may be absent in older files.
        tt._descriptor = state.Descriptor;

        // Restore derivative-id registry (absent in older JSON; skip if null).
        if (state.RegisteredDerivativeOrders != null)
        {
            foreach (var orders in state.RegisteredDerivativeOrders)
            {
                var key = new Internal.TupleKey(orders);
                int id = tt._registeredDerivativeOrders.Count;
                tt._registeredDerivativeOrders.Add((int[])orders.Clone());
                tt._derivativeIdRegistry[key] = id;
            }
        }

        // v2 migration: restore _dimOrder (backfill identity for v1 files).
        _ = jsonVersion; // consumed above via dimOrder derivation
        tt._dimOrder = (int[])dimOrder.Clone();

        string currentVersion = GetLibraryVersion();
        if (state.Version != null && state.Version != currentVersion)
        {
            tt.LoadWarning = $"This object was saved with ChebyshevSharp {state.Version}, " +
                             $"but you are loading it with {currentVersion}. " +
                             "Evaluation results may differ if internal data layout changed.";
        }

        return tt;
    }

    /// <summary>
    /// Load from a JSON file, raising <see cref="InvalidOperationException"/> if the type doesn't match.
    /// Used for testing wrong-type load scenarios.
    /// </summary>
    internal static ChebyshevTT LoadStrict(string path)
    {
        string json = File.ReadAllText(path);
        // Check if it's a ChebyshevTT file by checking for "Cores" key
        if (!json.Contains("\"Cores\""))
            throw new InvalidOperationException(
                $"Expected a ChebyshevTT file, got a different type.");
        return Load(path);
    }

    private static void ValidateSerializedState(TTSerializationState state, int[] dimOrder)
    {
        int d = state.NumDimensions;
        if (d <= 0)
            throw new InvalidDataException($"NumDimensions must be positive, got {d}.");

        ValidateDomain(state.Domain, d);
        ValidatePositiveVector(state.NNodes, d, nameof(TTSerializationState.NNodes));

        if (state.MaxRank <= 0)
            throw new InvalidDataException($"MaxRank must be positive, got {state.MaxRank}.");
        if (!double.IsFinite(state.Tolerance) || state.Tolerance < 0.0)
            throw new InvalidDataException($"Tolerance must be finite and non-negative, got {state.Tolerance}.");
        if (state.MaxSweeps < 0)
            throw new InvalidDataException($"MaxSweeps must be non-negative, got {state.MaxSweeps}.");
        if (!double.IsFinite(state.BuildTime) || state.BuildTime < 0.0)
            throw new InvalidDataException($"BuildTime must be finite and non-negative, got {state.BuildTime}.");
        if (state.TotalBuildEvals < 0)
            throw new InvalidDataException($"TotalBuildEvals must be non-negative, got {state.TotalBuildEvals}.");
        if (state.MaxDerivativeOrder is < 0)
            throw new InvalidDataException($"MaxDerivativeOrder must be non-negative, got {state.MaxDerivativeOrder}.");

        ValidatePositiveVector(state.TtRanks, d + 1, nameof(TTSerializationState.TtRanks));
        if (state.TtRanks[0] != 1 || state.TtRanks[^1] != 1)
            throw new InvalidDataException(
                $"TtRanks endpoints must be 1, got [{string.Join(",", state.TtRanks)}].");

        ValidateDimOrder(dimOrder, d);
        ValidateDerivativeRegistry(state.RegisteredDerivativeOrders, d, Math.Min(state.MaxDerivativeOrder ?? 2, 2));

        if (state.Cores is null)
            throw new InvalidDataException("Cores must be present.");
        if (state.Cores.Length != d)
            throw new InvalidDataException($"Cores has length {state.Cores.Length}, expected {d}.");

        for (int i = 0; i < d; i++)
        {
            CoreData core = state.Cores[i]
                ?? throw new InvalidDataException($"Cores[{i}] must be present.");

            if (core.RLeft <= 0 || core.NNodes <= 0 || core.RRight <= 0)
                throw new InvalidDataException(
                    $"Cores[{i}] dimensions must be positive, got ({core.RLeft}, {core.NNodes}, {core.RRight}).");
            if (core.RLeft != state.TtRanks[i] || core.RRight != state.TtRanks[i + 1])
                throw new InvalidDataException(
                    $"Cores[{i}] rank shape ({core.RLeft}, {core.RRight}) does not match " +
                    $"TtRanks[{i}:{i + 2}] = ({state.TtRanks[i]}, {state.TtRanks[i + 1]}).");
            if (core.NNodes != state.NNodes[i])
                throw new InvalidDataException(
                    $"Cores[{i}].NNodes={core.NNodes} does not match NNodes[{i}]={state.NNodes[i]}.");
            if (core.Data is null)
                throw new InvalidDataException($"Cores[{i}].Data must be present.");

            int expected = CheckedArrayLengthForInvalidData(
                new[] { core.RLeft, core.NNodes, core.RRight },
                $"Cores[{i}].Data");
            if (core.Data.Length != expected)
                throw new InvalidDataException(
                    $"Cores[{i}].Data has length {core.Data.Length}, expected {expected}.");

            for (int j = 0; j < core.Data.Length; j++)
                if (!double.IsFinite(core.Data[j]))
                    throw new InvalidDataException($"Cores[{i}].Data[{j}] must be finite.");
        }
    }

    private static void ValidateDomain(double[][]? domain, int numDimensions)
    {
        if (domain is null)
            throw new InvalidDataException("Domain must be present.");
        if (domain.Length != numDimensions)
            throw new InvalidDataException($"Domain has length {domain.Length}, expected {numDimensions}.");

        for (int i = 0; i < numDimensions; i++)
        {
            double[] bounds = domain[i]
                ?? throw new InvalidDataException($"Domain[{i}] must be present.");
            if (bounds.Length != 2)
                throw new InvalidDataException($"Domain[{i}] must contain exactly two bounds.");

            double lo = bounds[0];
            double hi = bounds[1];
            if (!double.IsFinite(lo) || !double.IsFinite(hi))
                throw new InvalidDataException($"Domain[{i}] bounds must be finite.");
            if (lo >= hi)
                throw new InvalidDataException($"Domain[{i}] lower bound must be less than upper bound.");
        }
    }

    private static void ValidatePositiveVector(int[]? values, int expectedLength, string name)
    {
        if (values is null)
            throw new InvalidDataException($"{name} must be present.");
        if (values.Length != expectedLength)
            throw new InvalidDataException($"{name} has length {values.Length}, expected {expectedLength}.");

        for (int i = 0; i < values.Length; i++)
            if (values[i] <= 0)
                throw new InvalidDataException($"{name}[{i}] must be positive, got {values[i]}.");
    }

    private static void ValidateDimOrder(int[] dimOrder, int numDimensions)
    {
        if (dimOrder.Length != numDimensions)
            throw new InvalidDataException($"DimOrder has length {dimOrder.Length}, expected {numDimensions}.");

        var seen = new bool[numDimensions];
        for (int i = 0; i < dimOrder.Length; i++)
        {
            int value = dimOrder[i];
            if (value < 0 || value >= numDimensions || seen[value])
                throw new InvalidDataException(
                    $"DimOrder must be a permutation of [0,{numDimensions - 1}], got [{string.Join(",", dimOrder)}].");
            seen[value] = true;
        }
    }

    private static void ValidateDerivativeRegistry(
        int[][]? registeredDerivativeOrders,
        int numDimensions,
        int maxDerivativeOrder)
    {
        if (registeredDerivativeOrders is null) return;

        for (int i = 0; i < registeredDerivativeOrders.Length; i++)
        {
            int[] orders = registeredDerivativeOrders[i]
                ?? throw new InvalidDataException($"RegisteredDerivativeOrders[{i}] must be present.");
            if (orders.Length != numDimensions)
                throw new InvalidDataException(
                    $"RegisteredDerivativeOrders[{i}] has length {orders.Length}, expected {numDimensions}.");
            for (int j = 0; j < orders.Length; j++)
            {
                if (orders[j] < 0)
                    throw new InvalidDataException(
                        $"RegisteredDerivativeOrders[{i}][{j}] must be non-negative, got {orders[j]}.");
                if (orders[j] > maxDerivativeOrder)
                    throw new InvalidDataException(
                        $"RegisteredDerivativeOrders[{i}][{j}]={orders[j]} exceeds maximum supported derivative order {maxDerivativeOrder}.");
            }
        }
    }

    private static int CheckedArrayLengthForInvalidData(int[] shape, string name)
    {
        try
        {
            long product = TensorShape.CheckedProduct(shape, name);
            return TensorShape.RequireArrayLength(product, name, shape);
        }
        catch (Exception ex) when (ex is ArgumentException or OverflowException or ArgumentOutOfRangeException)
        {
            throw new InvalidDataException($"{name} shape [{string.Join(",", shape)}] is invalid.", ex);
        }
    }

    private static string GetLibraryVersion()
    {
        var asm = typeof(ChebyshevTT).Assembly;
        var ver = asm.GetName().Version;
        // Use 3-part Major.Minor.Build form to match JSON serialization convention
        // (csproj <Version> is 3-part; .NET pads AssemblyVersion to 4 parts internally).
        return ver != null ? ver.ToString(3) : "0.0.0";
    }

    // ------------------------------------------------------------------
    // ToString
    // ------------------------------------------------------------------

    /// <inheritdoc/>
    public override string ToString()
    {
        string status = _built ? "built" : "not built";
        double fullTensorSize = FullGridSizeAsDouble();

        int maxDisplay = 6;
        string nodesStr, domainStr;

        if (_numDimensions > maxDisplay)
        {
            nodesStr = "[" + string.Join(", ", _nNodes.Take(maxDisplay)) + ", ...]";
            domainStr = string.Join(" x ",
                _domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        }
        else
        {
            nodesStr = "[" + string.Join(", ", _nNodes) + "]";
            domainStr = string.Join(" x ", _domain.Select(d => $"[{d[0]}, {d[1]}]"));
        }

        var sb = new StringBuilder();
        sb.AppendLine($"ChebyshevTT ({_numDimensions}D, {status})");
        sb.AppendLine($"  Nodes:       {nodesStr}");

        if (_built)
        {
            int ttStorage = 0;
            for (int i = 0; i < _coeffCores!.Length; i++) ttStorage += _coeffCores[i].Size;

            sb.AppendLine($"  TT ranks:    [{string.Join(", ", _ttRanks!)}]");
            sb.AppendLine($"  Compression: {fullTensorSize:N0} -> {ttStorage:N0} elements ({(double)fullTensorSize / ttStorage:F1}x)");
            sb.AppendLine($"  Build:       {_buildTime:F3}s ({_totalBuildEvals:N0} function evals)");
            sb.AppendLine($"  Domain:      {domainStr}");
            sb.Append($"  Error est:   {ErrorEstimate():E2}");
        }
        else
        {
            sb.Append($"  Domain:      {domainStr}");
        }

        return sb.ToString();
    }

    // ------------------------------------------------------------------
    // Phase 4 ergonomics — accessors
    // ------------------------------------------------------------------

    /// <summary>Set a free-form descriptor string for this tensor train.</summary>
    public void SetDescriptor(string descriptor) => _descriptor = descriptor;

    /// <summary>Get the descriptor previously set via <see cref="SetDescriptor"/>; null if unset.</summary>
    public string? GetDescriptor() => _descriptor;

    /// <summary>True if <see cref="Build"/>/<see cref="Load"/> completed.</summary>
    public bool IsConstructionFinished() => _built;

    /// <summary>Returns one of: "clone" (if cloned), "cross"/"svd"/"als" (build method used), or "function" if not yet built.</summary>
    public string GetConstructorType() => _constructorType ?? Method ?? "function";

    /// <summary>Per-dimension Chebyshev node counts actually used.</summary>
    public int[] GetUsedNs() => (int[])_nNodes.Clone();

    /// <summary>Maximum derivative order this tensor train supports.</summary>
    public int GetMaxDerivativeOrder() => _maxDerivativeOrder;

    /// <summary>
    /// Returns the user-supplied <c>additionalData</c> object passed to the constructor,
    /// or null if none was provided. Stored for introspection only — TT's function signature
    /// is <c>Func&lt;double[], double&gt;</c> (no data arg); wrap with a closure if you need data threading.
    /// </summary>
    public object? GetAdditionalData() => _additionalData;

    /// <summary>
    /// Total number of evaluation points in the full Chebyshev grid (product of nNodes).
    /// </summary>
    /// <returns>The total count of Chebyshev nodes across all dimensions.</returns>
    public int GetNumEvaluationPoints()
    {
        return FullGridSizeAsIntForMaterialization(nameof(GetNumEvaluationPoints));
    }

    /// <summary>
    /// Flat row-major array of the full Chebyshev grid coordinates.
    /// Generated on-demand using the domain and nNodes, independent of the sparse TT sampling.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached internally.
    /// </summary>
    /// <returns>A snapshot of full-grid node coordinates, flattened in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return CloneHelpers.DeepCopy(_evaluationPointsCache)!;

        int num = GetNumEvaluationPoints();
        int ndim = _numDimensions;

        var nodeArrays = new double[ndim][];
        for (int d = 0; d < ndim; d++)
            nodeArrays[d] = BarycentricKernel.MakeNodesForDim(_domain[d][0], _domain[d][1], _nNodes[d]);

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
                points[flat * ndim + d] = nodeArrays[d][indices[d]];
        }

        // v0.21.1: permute columns by inverse _dimOrder so column k is the user-frame
        // k-th coord (matches Approximation/Spline/Slider behavior).
        // Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:2775-2800.
        if (!IsIdentityDimOrder())
        {
            var inv = new int[ndim];
            for (int s = 0; s < ndim; s++) inv[_dimOrder[s]] = s;
            var permuted = new double[coordinateCount];
            for (int i = 0; i < num; i++)
                for (int u = 0; u < ndim; u++)
                    permuted[i * ndim + u] = points[i * ndim + inv[u]];
            points = permuted;
        }

        _evaluationPointsCache = points;
        return CloneHelpers.DeepCopy(points)!;
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
        EvaluationArguments.ValidateDerivativeOrder(
            orders,
            _numDimensions,
            nameof(orders),
            EffectiveMaxDerivativeOrder);
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
        var orders = _registeredDerivativeOrders[derivativeId];
        bool allZero = orders.All(o => o == 0);
        if (allZero) return Eval(point);
        return EvalMulti(point, new[] { orders })[0];
    }

    internal Dictionary<Internal.TupleKey, int> DerivativeIdRegistry => _derivativeIdRegistry;
    internal List<int[]> RegisteredDerivativeOrders => _registeredDerivativeOrders;

    // ------------------------------------------------------------------
    // DimOrder + Reorder (Phase 6 Task 9)
    // ------------------------------------------------------------------

    /// <summary>
    /// Storage permutation: <c>DimOrder[k]</c> is the original-dimension index stored
    /// at TT position k. Identity by default; non-identity for TTs produced by
    /// <see cref="Reorder"/>. Returns a defensive clone;
    /// mutating the returned array does not affect this TT.
    /// </summary>
    public int[] DimOrder => (int[])_dimOrder.Clone();

    /// <summary>
    /// Realign storage to a target permutation via TT-swap (adjacent-axis SVDs in
    /// coefficient space). Functional API; returns a new TT. Inherits all build
    /// parameters (maxRank, tolerance, maxSweeps, descriptor, additionalData,
    /// maxDerivativeOrder, Method) from this TT.
    /// </summary>
    /// <param name="newOrder">Target permutation; must be a permutation of [0, NumDimensions-1].</param>
    /// <param name="maxRank">Optional override for swap-time SVD truncation. Default: this TT's maxRank.</param>
    /// <param name="tolerance">Optional relative-tolerance cutoff. Default: this TT's tolerance.</param>
    /// <returns>A new TT with <c>DimOrder == newOrder</c>.</returns>
    /// <exception cref="ArgumentException">If <paramref name="newOrder"/> is not a valid permutation.</exception>
    public ChebyshevTT Reorder(int[] newOrder, int? maxRank = null, double? tolerance = null)
    {
        CheckBuilt();
        ValidatePermutation(newOrder, _numDimensions);
        int rank = maxRank ?? _maxRank;
        double tol = tolerance ?? _tolerance;
        ValidatePositiveRank(rank, nameof(maxRank));
        ValidateNonNegativeFiniteTolerance(tol, nameof(tolerance));

        // Short-circuit: reorder to current dim_order is just a clone (matches Python tensor_train.py:2397).
        if (newOrder.SequenceEqual(_dimOrder))
            return Clone();

        // Bubble-sort current_order DIRECTLY into newOrder (Python tensor_train.py:2403-2425).
        // newOrder is the absolute target: result._dim_order == list(new_order).
        var currentOrder = (int[])_dimOrder.Clone();
        var cores = new Internal.TensorTrainKernel.TtCore[_coeffCores!.Length];
        for (int k = 0; k < cores.Length; k++) cores[k] = _coeffCores[k].Copy();

        // Track domain and nNodes in storage order; swap alongside currentOrder
        // (Python lines 2421-2422 swap n_nodes/domain in lockstep with currentOrder).
        var domain = _domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodes = (int[])_nNodes.Clone();

        for (int k = 0; k < _numDimensions; k++)
        {
            int targetOrig = newOrder[k];
            int j = Array.IndexOf(currentOrder, targetOrig);
            while (j > k)
            {
                cores = Internal.TensorTrainAlgebra.TtSwapAdjacent(cores, j - 1, rank, tol);
                (currentOrder[j - 1], currentOrder[j]) = (currentOrder[j], currentOrder[j - 1]);
                (nNodes[j - 1], nNodes[j]) = (nNodes[j], nNodes[j - 1]);
                (domain[j - 1], domain[j]) = (domain[j], domain[j - 1]);
                j--;
            }
        }

        // Sanity check (matches Python's `assert current_order == new_order` at line 2425).
        if (!currentOrder.SequenceEqual(newOrder))
            throw new InvalidOperationException(
                "Reorder bubble-sort failed to converge to target permutation");

        var result = BuildResultFromCores(cores, domain, nNodes);
        result._dimOrder = (int[])newOrder.Clone();
        result._descriptor = _descriptor;
        result._additionalData = _additionalData;
        return result;
    }

    private static void ValidatePermutation(int[] perm, int n)
    {
        if (perm == null) throw new ArgumentNullException(nameof(perm));
        if (perm.Length != n)
            throw new ArgumentException(
                $"Permutation length {perm.Length} != numDimensions {n}", nameof(perm));
        var seen = new bool[n];
        foreach (int v in perm)
        {
            if (v < 0 || v >= n)
                throw new ArgumentException(
                    $"Permutation entry {v} out of range [0, {n - 1}]", nameof(perm));
            if (seen[v])
                throw new ArgumentException($"Duplicate entry {v} in permutation", nameof(perm));
            seen[v] = true;
        }
    }

    // ------------------------------------------------------------------
    // SobolIndices (TT-native)
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute first-order + total-order variance-based sensitivity indices natively
    /// from the TT coefficient cores. The variance is with respect to the Chebyshev
    /// orthogonality weight on the normalized domain. O(d · n · r²) per dim,
    /// no dense materialization.
    /// </summary>
    /// <returns><see cref="SobolResult"/> with arrays keyed by user-frame dim indices and Chebyshev-weighted total Variance.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <remarks>
    /// Mathematically equivalent to <see cref="ChebyshevApproximation.SobolIndices"/>
    /// applied to the dense version of the same function, but skips the O(n^d)
    /// materialization. Under non-identity <see cref="DimOrder"/>, result keys are
    /// translated from storage frame to user frame internally.
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:2823-2868</c>.
    /// </remarks>
    public SobolResult SobolIndices()
    {
        CheckBuilt();

        // Compute indices in storage frame (0..d-1 positional in _coeffCores).
        var storage = Internal.Sensitivity.ComputeSobolFromTtCores(_coeffCores!);

        // Fast path: identity _dimOrder needs no translation.
        if (IsIdentityDimOrder())
            return storage;

        // Translate storage-frame keys → user-frame keys.
        // _dimOrder[s] = u means storage position s holds original-dim u.
        // This is the inverse permutation by _dimOrder.
        var userFirst = new double[_numDimensions];
        var userTotal = new double[_numDimensions];
        for (int s = 0; s < _numDimensions; s++)
        {
            int u = _dimOrder[s];
            userFirst[u] = storage.FirstOrder[s];
            userTotal[u] = storage.TotalOrder[s];
        }
        return new SobolResult(userFirst, userTotal, storage.Variance);
    }

    /// <summary>
    /// Test-only accessor: returns the raw coefficient cores array.
    /// Visible to the test project via InternalsVisibleTo.
    /// </summary>
    internal Internal.TensorTrainKernel.TtCore[] GetCoeffCoresForTest() => _coeffCores!;

    // ------------------------------------------------------------------
    // WithAutoOrder
    // ------------------------------------------------------------------

    /// <summary>
    /// Build a TT trying multiple dim orderings, returning the lowest-rank result.
    /// TT-Cross compression depends on dim order; different orderings yield different
    /// ranks for the same function. Mirrors PyChebyshev <c>tensor_train.py:2687</c>.
    /// </summary>
    /// <param name="function">f(point) → double in the original (user) dim order.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension in original order.</param>
    /// <param name="numNodes">Node counts per dimension in original order.</param>
    /// <param name="maxRank">Maximum positive TT rank passed to each trial. Default 10.</param>
    /// <param name="tolerance">Finite positive convergence tolerance for each trial. Default 1e-6.</param>
    /// <param name="maxSweeps">Maximum positive TT-Cross sweeps per trial. Default 10.</param>
    /// <param name="additionalData">Stored on the result for introspection; not threaded into f.</param>
    /// <param name="nTrials">Number of swap iterations / random samples. Default 5.</param>
    /// <param name="method">"greedy_swap" (default, deterministic) or "random".</param>
    /// <param name="seed">Optional seed for "random". Ignored by "greedy_swap".</param>
    /// <param name="progress">Optional per-sweep progress reporter (forwarded to each trial's Build).</param>
    /// <param name="verbose">If true, print per-trial diagnostics.</param>
    /// <returns>The lowest-total-rank TT among the tried permutations, with <c>DimOrder</c> set.</returns>
    public static ChebyshevTT WithAutoOrder(
        Func<double[], double> function,
        int numDimensions,
        double[][] domain,
        int[] numNodes,
        int maxRank = 10,
        double tolerance = 1e-6,
        int maxSweeps = 10,
        object? additionalData = null,
        int nTrials = 5,
        string method = "greedy_swap",
        int? seed = null,
        IProgress<int>? progress = null,
        bool verbose = false)
    {
        if (method != "greedy_swap" && method != "random")
            throw new ArgumentException(
                $"unknown method: '{method}' (use 'greedy_swap' or 'random')", nameof(method));

        ChebyshevTT BuildWith(int[] order)
        {
            var permDomain = order.Select(d => domain[d]).ToArray();
            var permNNodes = order.Select(d => numNodes[d]).ToArray();
            // Permuted f: caller passes a point in PERMUTED order; map back to original.
            Func<double[], double> permF = (point) =>
            {
                var orig = new double[numDimensions];
                for (int k = 0; k < numDimensions; k++) orig[order[k]] = point[k];
                return function(orig);
            };
            var tt = new ChebyshevTT(permF, numDimensions, permDomain, permNNodes,
                maxRank: maxRank, tolerance: tolerance, maxSweeps: maxSweeps,
                additionalData: additionalData, progress: progress);
            tt.Build(verbose: verbose, seed: seed);
            tt._dimOrder = (int[])order.Clone();
            return tt;
        }

        int RankSum(ChebyshevTT t)
        {
            int sum = 0;
            foreach (int r in t.TtRanks) sum += r;
            return sum;
        }

        int[] canonical = Enumerable.Range(0, numDimensions).ToArray();
        var bestTt = BuildWith(canonical);
        int bestScore = RankSum(bestTt);

        if (nTrials <= 0) return bestTt;

        if (method == "greedy_swap")
        {
            bool improved = true;
            int iter = 0;
            while (improved && iter < nTrials)
            {
                improved = false;
                var currentDimOrder = (int[])bestTt.DimOrder.Clone();   // capture ONCE per outer iter
                for (int i = 0; i < numDimensions - 1; i++)
                {
                    var trial = (int[])currentDimOrder.Clone();          // same baseline every inner iter
                    (trial[i], trial[i + 1]) = (trial[i + 1], trial[i]);
                    var candidateTt = BuildWith(trial);
                    int candidateScore = RankSum(candidateTt);
                    if (candidateScore < bestScore)
                    {
                        bestTt = candidateTt;
                        bestScore = candidateScore;
                        improved = true;
                        break;                                            // restart outer loop on first improvement
                    }
                }
                iter++;
            }
        }
        else  // method == "random"
        {
            var rng = new Random(seed ?? 42);
            for (int t = 0; t < nTrials; t++)
            {
                // Fisher-Yates shuffle of canonical order.
                var trial = (int[])canonical.Clone();
                for (int i = numDimensions - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (trial[i], trial[j]) = (trial[j], trial[i]);
                }
                var candidateTt = BuildWith(trial);
                int candidateScore = RankSum(candidateTt);
                if (candidateScore < bestScore)
                {
                    bestTt = candidateTt;
                    bestScore = candidateScore;
                }
            }
        }

        return bestTt;
    }

    // ------------------------------------------------------------------
    // Clone
    // ------------------------------------------------------------------

    /// <summary>
    /// Returns a deep copy of this tensor train. The source function callable is
    /// NOT duplicated — clones cannot be rebuilt without re-supplying the function.
    /// All TT cores and state are deep-copied.
    /// </summary>
    /// <returns>A fully independent <see cref="ChebyshevTT"/>.</returns>
    public ChebyshevTT Clone()
    {
        TensorTrainKernel.TtCore[]? clonedCores = null;
        if (_coeffCores != null)
        {
            clonedCores = new TensorTrainKernel.TtCore[_coeffCores.Length];
            for (int i = 0; i < _coeffCores.Length; i++)
                clonedCores[i] = _coeffCores[i].Copy();
        }

        var copy = new ChebyshevTT(
            numDimensions: _numDimensions,
            domain: Internal.CloneHelpers.DeepCopy(_domain)!,
            nNodes: Internal.CloneHelpers.DeepCopy(_nNodes)!,
            maxRank: _maxRank,
            tolerance: _tolerance,
            maxSweeps: _maxSweeps,
            coeffCores: clonedCores ?? System.Array.Empty<TensorTrainKernel.TtCore>(),
            ttRanks: _ttRanks != null ? (int[])_ttRanks.Clone() : System.Array.Empty<int>(),
            buildTime: _buildTime,
            totalBuildEvals: _totalBuildEvals,
            maxDerivativeOrder: _maxDerivativeOrder);
        copy.Method = Method;
        copy._constructorType = "clone";
        copy.BuildWarning = BuildWarning;
        copy.LoadWarning = LoadWarning;
        copy._descriptor = _descriptor;
        copy._additionalData = _additionalData;
        copy._dimOrder = (int[])_dimOrder.Clone();
        copy._evaluationPointsCache = null;
        foreach (var kv in _derivativeIdRegistry)
            copy._derivativeIdRegistry[kv.Key] = kv.Value;
        foreach (var orders in _registeredDerivativeOrders)
            copy._registeredDerivativeOrders.Add((int[])orders.Clone());
        return copy;
    }

    // ------------------------------------------------------------------
    // Serialization DTO
    // ------------------------------------------------------------------

    internal class TTSerializationState
    {
        public string? Version { get; set; }
        public string? Method { get; set; }
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = null!;
        public int[] NNodes { get; set; } = null!;
        public int MaxRank { get; set; }
        public double Tolerance { get; set; }
        public int MaxSweeps { get; set; }
        public int[] TtRanks { get; set; } = null!;
        public double BuildTime { get; set; }
        public int TotalBuildEvals { get; set; }
        public CoreData[] Cores { get; set; } = null!;
        // v0.8.0 ergonomics fields (absent in pre-v0.8.0 JSON; null == not set)
        public string? Descriptor { get; set; }
        public string? ConstructorType { get; set; }
        // Nullable so pre-v0.8.0 files (which lack this field) default to null => 2
        public int? MaxDerivativeOrder { get; set; }
        // Derivative-id registry (absent in older JSON; null == not set)
        public int[][]? RegisteredDerivativeOrders { get; set; }
        // v2 (v0.10.0+): JsonVersion and DimOrder; null/absent => v1 (backfill identity)
        public int? JsonVersion { get; set; }
        public int[]? DimOrder { get; set; }
    }

    internal class CoreData
    {
        public int RLeft { get; set; }
        public int NNodes { get; set; }
        public int RRight { get; set; }
        public double[] Data { get; set; } = null!;
    }
}
