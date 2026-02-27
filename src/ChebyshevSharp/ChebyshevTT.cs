using System.Diagnostics;
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

    /// <summary>Warning message set when loading from a different library version.</summary>
    public string? LoadWarning { get; private set; }

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
    public ChebyshevTT(
        Func<double[], double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6,
        int maxSweeps = 10)
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
        int totalBuildEvals)
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
        _built = true;
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// <summary>
    /// Build TT approximation and convert to Chebyshev coefficient cores.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    /// <param name="seed">Random seed for TT-Cross initialization. Ignored for method="svd".</param>
    /// <param name="method">"cross" (default) or "svd".</param>
    /// <exception cref="ArgumentException">If method is not "cross" or "svd".</exception>
    public void Build(bool verbose = true, int? seed = null, string method = "cross")
    {
        if (method != "cross" && method != "svd")
            throw new ArgumentException($"method must be 'cross' or 'svd', got '{method}'");

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
        else
        {
            (valueCores, nEvals) = TensorTrainKernel.TtSvd(
                _function!, grids, _maxRank, _tolerance, verbose);
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
            state.TotalBuildEvals);

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
        return ver != null ? ver.ToString() : "0.0.0";
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
    // Serialization DTO
    // ------------------------------------------------------------------

    internal class TTSerializationState
    {
        public string? Version { get; set; }
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
    }

    internal class CoreData
    {
        public int RLeft { get; set; }
        public int NNodes { get; set; }
        public int RRight { get; set; }
        public double[] Data { get; set; } = null!;
    }
}
