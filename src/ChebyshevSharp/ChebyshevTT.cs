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
    private double[]? _evaluationPointsCache;
    private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
    private readonly List<int[]> _registeredDerivativeOrders = new();

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
    public double[][] Domain => _domain;

    /// <summary>Number of Chebyshev nodes per dimension.</summary>
    public int[] NNodes => _nNodes;

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
            long fullSize = 1;
            for (int i = 0; i < _numDimensions; i++) fullSize *= _nNodes[i];
            int ttSize = 0;
            for (int i = 0; i < _coeffCores!.Length; i++) ttSize += _coeffCores[i].Size;
            return (double)fullSize / ttSize;
        }
    }

    /// <summary>
    /// Create a new ChebyshevTT interpolant.
    /// </summary>
    /// <param name="function">Function to approximate. Signature: f(point) -> double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] for each dimension.</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <param name="maxRank">Maximum TT rank. Default is 10.</param>
    /// <param name="tolerance">Convergence tolerance for TT-Cross. Default is 1e-6.</param>
    /// <param name="maxSweeps">Maximum number of TT-Cross sweeps. Default is 10.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support. Default is 2.</param>
    /// <param name="additionalData">Optional user data object stored for introspection via <see cref="GetAdditionalData"/>. NOT threaded through build calls (TT function signature has no data arg).</param>
    public ChebyshevTT(
        Func<double[], double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6,
        int maxSweeps = 10,
        int maxDerivativeOrder = 2,
        object? additionalData = null)
    {
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}");
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}");

        _function = function;
        _numDimensions = numDimensions;
        _domain = domain;
        _nNodes = nNodes;
        _maxRank = maxRank;
        _tolerance = tolerance;
        _maxSweeps = maxSweeps;
        _maxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
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
    /// <exception cref="ArgumentException">If method is not "cross", "svd", or "als".</exception>
    public void Build(bool verbose = true, int? seed = null, string method = "cross")
    {
        if (method != "cross" && method != "svd" && method != "als")
            throw new ArgumentException($"method must be 'cross', 'svd', or 'als', got '{method}'");
        Method = method;
        BuildWarning = null;

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        long fullTensorSize = 1;
        for (int i = 0; i < _numDimensions; i++) fullTensorSize *= _nNodes[i];

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

        if (method == "cross")
        {
            if (verbose) Console.WriteLine("  Running TT-Cross...");
            (valueCores, nEvals) = TensorTrainKernel.TtCross(
                _function!, grids, _maxRank, _tolerance, _maxSweeps, verbose, seed);
        }
        else if (method == "svd")
        {
            (valueCores, nEvals) = TensorTrainKernel.TtSvd(
                _function!, grids, _maxRank, _tolerance, verbose);
        }
        else  // method == "als"
        {
            if (verbose) Console.WriteLine("  Running TT-ALS...");
            bool hitCap;
            (valueCores, nEvals, hitCap) = TensorTrainKernel.AlsAdaptiveRank(
                _function!, grids, _maxRank, _tolerance, seed, verbose);
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
            Console.WriteLine($"  Compression: {fullTensorSize:N0} -> {ttStorage:N0} elements ({(double)fullTensorSize / ttStorage:F1}x)");
        }
    }

    private void CheckBuilt()
    {
        if (!_built)
            throw new InvalidOperationException("Call Build() before using this method.");
    }

    // ------------------------------------------------------------------
    // Eval
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate at a single point via TT inner product with Chebyshev polynomial basis.
    /// Cost: O(d * n * r^2) per point.
    /// </summary>
    /// <param name="point">Query point, one coordinate per dimension.</param>
    /// <returns>Interpolated value.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double Eval(double[] point)
    {
        CheckBuilt();

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
            double[] v = new double[resultRows * rRight];

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
    /// <param name="points">Query points, shape (N, numDimensions).</param>
    /// <returns>Interpolated values, length N.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double[] EvalBatch(double[,] points)
    {
        CheckBuilt();

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
            double[] Q = new double[N * nk]; // Q[n * nk + j]
            for (int nn = 0; nn < N; nn++)
            {
                double scaled = 2.0 * (points[nn, d] - a) / (b - a) - 1.0;
                double[] q = ChebyshevPolynomials(scaled, nk);
                for (int j = 0; j < nk; j++)
                    Q[nn * nk + j] = q[j];
            }

            // V[n,i,k] = sum_j Q[n,j] * core[i,j,k]
            double[] V = new double[N * rLeft * rRight];
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
            double[] newResult = new double[N * rRight];
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
    /// <param name="point">Evaluation point.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.
    /// Supports 0 (value), 1 (first), and 2 (second).</param>
    /// <returns>One result per derivative order specification.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public double[] EvalMulti(double[] point, int[][] derivativeOrders)
    {
        CheckBuilt();

        double[] results = new double[derivativeOrders.Length];
        for (int i = 0; i < derivativeOrders.Length; i++)
        {
            bool allZero = true;
            for (int d = 0; d < derivativeOrders[i].Length; d++)
                if (derivativeOrders[i][d] != 0) { allZero = false; break; }

            results[i] = allZero ? Eval(point) : FdDerivative(point, derivativeOrders[i]);
        }
        return results;
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
            return (Eval(ptPlus) - Eval(ptMinus)) / (2.0 * h);
        }
        else if (order == 2)
        {
            double[] ptPlus = (double[])pt.Clone();
            double[] ptMinus = (double[])pt.Clone();
            ptPlus[d] += h;
            ptMinus[d] -= h;
            return (Eval(ptPlus) - 2.0 * Eval(pt) + Eval(ptMinus)) / (h * h);
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

        double fpp = Eval(MakePt(+h1, +h2));
        double fpm = Eval(MakePt(+h1, -h2));
        double fmp = Eval(MakePt(-h1, +h2));
        double fmm = Eval(MakePt(-h1, -h2));
        return (fpp - fpm - fmp + fmm) / (4.0 * h1 * h2);
    }

    private double FdNested(double[] point, List<(int dim, int order)> activeDims, int startIdx)
    {
        if (startIdx >= activeDims.Count)
            return Eval(point);

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
    /// <exception cref="ArgumentException">If <paramref name="dims"/> contains out-of-range or duplicated indices, or <paramref name="bounds"/> are invalid.</exception>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        CheckBuilt();

        // Normalize dims: null = all, sort + deduplicate, validate range.
        int[] sortedDims;
        if (dims == null)
            sortedDims = Enumerable.Range(0, _numDimensions).ToArray();
        else
            sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= _numDimensions)
                throw new ArgumentException(
                    $"dim {d} out-of-range [0, {_numDimensions - 1}]");
        }

        var perDimBounds = Internal.Calculus.NormalizeBounds(sortedDims, bounds, _domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        // Compute scaled quadrature weights per integrated dim.
        // Cores live in coefficient space — convert each integrated core to
        // value space before applying weights.
        var weightsPerDim = new Dictionary<int, double[]>();
        foreach (int d in sortedDims)
        {
            int n = _nNodes[d];
            double a = _domain[d][0], b = _domain[d][1];
            double scale = (b - a) / 2.0;
            var bd = perDimBounds[dimToIdx[d]];
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
            weightsPerDim[d] = w;
        }

        // Per-integrated-dim contraction: coefficient core -> value core -> M_k.
        var contracted = new Dictionary<int, double[,]>();
        foreach (int d in sortedDims)
        {
            var valueCore = Internal.TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![d]);
            contracted[d] = Internal.Calculus.IntegrateTtAlongDim(valueCore, weightsPerDim[d]);
        }

        if (sortedDims.Length == _numDimensions)
        {
            // Full integration: chain-multiply all M_k matrices left-to-right.
            // contracted[sortedDims[0]] is shape (rL_0=1, rR_0); after all multiplications,
            // result is (1, 1).
            double[,] result = contracted[sortedDims[0]];
            for (int i = 1; i < sortedDims.Length; i++)
                result = MatMul(result, contracted[sortedDims[i]]);
            return result[0, 0];
        }

        // Partial integration: walk the TT chain, absorbing each contracted
        // matrix into a neighboring kept core's left rank dim (Python
        // tensor_train.py:1582-1608).
        var integratedSet = new HashSet<int>(sortedDims);
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
            // k is a kept dim — absorb any pending matrix into this core's left rank.
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
        int[] keptDims = Enumerable.Range(0, _numDimensions)
            .Where(d => !integratedSet.Contains(d))
            .ToArray();
        var newDomain = keptDims.Select(d => (double[])_domain[d].Clone()).ToArray();
        var newNNodes = keptDims.Select(d => _nNodes[d]).ToArray();

        return BuildIntegrateResult(newCores.ToArray(), newDomain, newNNodes);
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
    public void RunCompletion(double tolerance = 1e-8, int maxIter = 50, bool verbose = false)
    {
        CheckBuilt();
        if (_function == null)
            throw new InvalidOperationException(
                "RunCompletion requires Function to be callable; the TT was loaded from a source without the original function.");

        // Convert coefficient cores back to value cores at Chebyshev Type I nodes.
        var valueCores = new TensorTrainKernel.TtCore[_numDimensions];
        for (int k = 0; k < _numDimensions; k++)
            valueCores[k] = TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![k]);

        // Rebuild the grids that Build() used.
        var grids = new double[_numDimensions][];
        for (int k = 0; k < _numDimensions; k++)
            grids[k] = BarycentricKernel.MakeNodesForDim(_domain[k][0], _domain[k][1], _nNodes[k]);

        // Cache by mixed-radix flat index.
        var cache = new Dictionary<long, double>();
        long[] strides = new long[_numDimensions];
        strides[_numDimensions - 1] = 1;
        for (int i = _numDimensions - 2; i >= 0; i--) strides[i] = strides[i + 1] * _nNodes[i + 1];

        Func<int[], double> evalsAt = idx =>
        {
            long key = 0;
            for (int i = 0; i < _numDimensions; i++) key += idx[i] * strides[i];
            if (!cache.TryGetValue(key, out double v))
            {
                var pt = new double[_numDimensions];
                for (int i = 0; i < _numDimensions; i++) pt[i] = grids[i][idx[i]];
                v = _function(pt);
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
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}");
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}");

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
    /// <param name="maxRank">Maximum TT rank (default 10).</param>
    /// <param name="tolerance">SVD truncation tolerance (default 1e-6).</param>
    /// <exception cref="ArgumentException">If tensorValues length doesn't match Π nNodes, or contains NaN/Infinity.</exception>
    public static ChebyshevTT FromValues(
        double[] tensorValues,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6)
    {
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}");
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}");
        long expected = 1;
        for (int i = 0; i < numDimensions; i++) expected = checked(expected * nNodes[i]);
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
            domain: domain.Select(d => (double[])d.Clone()).ToArray(),
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
        long total = 1;
        for (int i = 0; i < _numDimensions; i++)
            total = checked(total * _nNodes[i]);
        if (total * 8 > int.MaxValue)
            throw new OverflowException(
                $"ToDense would allocate {total} doubles ({total * 8} bytes), exceeding int.MaxValue. " +
                "Use ToDense for low-dimensional inspection only.");
        return TensorTrainExtrude.ToDenseEinsumChain(_coeffCores!, _nNodes);
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
        if (newDomain.Lo >= newDomain.Hi)
            throw new ArgumentException(
                $"newDomain bounds must satisfy lo < hi; got ({newDomain.Lo}, {newDomain.Hi})",
                nameof(newDomain));
        var newCores = TensorTrainExtrude.ExtrudeCores(_coeffCores!, dim, newN);
        var newDomainArr = new double[_numDimensions + 1][];
        for (int k = 0; k < dim; k++) newDomainArr[k] = (double[])_domain[k].Clone();
        newDomainArr[dim] = new[] { newDomain.Lo, newDomain.Hi };
        for (int k = dim; k < _numDimensions; k++) newDomainArr[k + 1] = (double[])_domain[k].Clone();

        var newNNodes = new int[_numDimensions + 1];
        for (int k = 0; k < dim; k++) newNNodes[k] = _nNodes[k];
        newNNodes[dim] = newN;
        for (int k = dim; k < _numDimensions; k++) newNNodes[k + 1] = _nNodes[k];

        return BuildResultFromCores(newCores, newDomainArr, newNNodes);
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
        double lo = _domain[dim][0], hi = _domain[dim][1];
        if (value < lo || value > hi)
            throw new ArgumentOutOfRangeException(nameof(value),
                $"Slice value {value} for dim {dim} is outside domain [{lo}, {hi}]");
        if (_numDimensions == 1)
            throw new InvalidOperationException("Cannot slice a 1D TT (would produce 0D result).");

        double[] nodes = BarycentricKernel.MakeNodesForDim(lo, hi, _nNodes[dim]);
        var newCores = TensorTrainExtrude.SliceCores(_coeffCores!, dim, value, nodes);

        var newDomain = new double[_numDimensions - 1][];
        var newNNodes = new int[_numDimensions - 1];
        int writeIdx = 0;
        for (int k = 0; k < _numDimensions; k++)
        {
            if (k == dim) continue;
            newDomain[writeIdx] = (double[])_domain[k].Clone();
            newNNodes[writeIdx] = _nNodes[k];
            writeIdx++;
        }

        return BuildResultFromCores(newCores, newDomain, newNNodes);
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
        return tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
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
        return tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
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

    /// <summary>Validate two TTs share the same grid (numDim, domain, nNodes).</summary>
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
        return a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
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
        return a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
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
    /// <exception cref="InvalidOperationException">If the file does not contain a valid ChebyshevTT.</exception>
    public static ChebyshevTT Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<TTSerializationState>(json)
                    ?? throw new InvalidOperationException("Failed to deserialize ChebyshevTT state.");

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
        long fullTensorSize = 1;
        for (int i = 0; i < _numDimensions; i++) fullTensorSize *= _nNodes[i];

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
        int total = 1;
        foreach (int n in _nNodes) total *= n;
        return total;
    }

    /// <summary>
    /// Flat row-major array of the full Chebyshev grid coordinates.
    /// Generated on-demand using the domain and nNodes, independent of the sparse TT sampling.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached.
    /// </summary>
    /// <returns>Double array of full-grid node coordinates, flattened in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return _evaluationPointsCache;

        int num = GetNumEvaluationPoints();
        int ndim = _numDimensions;

        var nodeArrays = new double[ndim][];
        for (int d = 0; d < ndim; d++)
            nodeArrays[d] = BarycentricKernel.MakeNodesForDim(_domain[d][0], _domain[d][1], _nNodes[d]);

        var points = new double[num * ndim];
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

        _evaluationPointsCache = points;
        return points;
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
    }

    internal class CoreData
    {
        public int RLeft { get; set; }
        public int NNodes { get; set; }
        public int RRight { get; set; }
        public double[] Data { get; set; } = null!;
    }
}
