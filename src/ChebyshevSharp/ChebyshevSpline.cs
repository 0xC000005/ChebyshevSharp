using System.Diagnostics;
using System.Text;
using System.Text.Json;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp;

/// <summary>
/// Piecewise Chebyshev interpolation with user-specified knots.
/// Partitions the domain into sub-intervals at interior knots and builds
/// an independent <see cref="ChebyshevApproximation"/> on each piece.
/// Query points are routed to the appropriate piece for evaluation.
/// </summary>
/// <remarks>
/// This is the correct approach when the target function has known
/// singularities (kinks, discontinuities) at specific locations: place
/// knots at those locations so that each piece is smooth, restoring
/// spectral convergence.
/// </remarks>
public class ChebyshevSpline
{
    /// <summary>The function to approximate. Null after load or from_values.</summary>
    public Func<double[], object?, double>? Function { get; internal set; }

    /// <summary>Number of input dimensions.</summary>
    public int NumDimensions { get; internal set; }

    /// <summary>Domain bounds for each dimension, as list of [lo, hi].</summary>
    public double[][] Domain { get; internal set; } = Array.Empty<double[]>();

    /// <summary>Number of Chebyshev nodes per dimension per piece.</summary>
    public int[] NNodes { get; internal set; } = Array.Empty<int>();

    /// <summary>Maximum supported derivative order.</summary>
    public int MaxDerivativeOrder { get; internal set; } = 2;

    /// <summary>Interior knots per dimension. Each sub-array is sorted ascending.</summary>
    public double[][] Knots { get; internal set; } = Array.Empty<double[]>();

    /// <summary>Per-dimension intervals. intervals[d] = [(lo, k1), (k1, k2), ..., (kn, hi)].</summary>
    internal (double lo, double hi)[][] Intervals { get; set; } = Array.Empty<(double, double)[]>();

    /// <summary>Per-dimension piece counts.</summary>
    internal int[] Shape { get; set; } = Array.Empty<int>();

    /// <summary>Flat array of pieces in C-order (row-major).</summary>
    internal ChebyshevApproximation?[] Pieces { get; set; } = Array.Empty<ChebyshevApproximation>();

    /// <summary>Whether Build() has been called.</summary>
    public bool Built { get; internal set; }

    /// <summary>Wall-clock time (seconds) for the most recent Build() call.</summary>
    public double BuildTime { get; internal set; }

    /// <summary>Target supremum-norm error for per-piece auto-N construction. Null in fixed-N mode.</summary>
    public double? ErrorThreshold { get; internal set; }

    /// <summary>Maximum nodes per dimension per piece for the auto-N doubling loop. Default 64.</summary>
    public int MaxN { get; internal set; } = 64;

    /// <summary>The user's original nNodes argument with null sentinels intact.</summary>
    internal int?[] OriginalNNodes { get; set; } = Array.Empty<int?>();

    /// <summary>Per-piece, per-dim node counts (when constructed with nested nNodesNested form). Null otherwise.</summary>
    internal int[][]? NestedNNodes { get; set; }

    private double? _cachedErrorEstimate;
    private string? _descriptor;
    private string _constructorType = "function";
    private object? _additionalData;
    private double[]? _evaluationPointsCache;
    private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
    private readonly List<int[]> _registeredDerivativeOrders = new();
    private int? _nWorkers;
    private IProgress<int>? _progress;

    /// <summary>
    /// Create a new ChebyshevSpline.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -&gt; double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension per piece.</param>
    /// <param name="knots">Interior knots for each dimension. Empty array for no knots.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="deferBuild">If true, skip eager build. Call <see cref="SetOriginalFunctionValues"/> to finish construction.</param>
    /// <param name="nWorkers">Number of parallel workers for function evaluation. null = sequential; -1 = <see cref="Environment.ProcessorCount"/>; positive = exact count.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count across all pieces.</param>
    /// <remarks>Thread safety: the user-supplied function must be thread-safe when <paramref name="nWorkers"/> is non-null.</remarks>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        double[][] knots,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        bool deferBuild = false,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        NNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;

        // Validate and store knots
        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        // Compute per-dimension intervals
        Intervals = ComputeIntervals(numDimensions, domain, knots);

        // Shape: per-dimension piece counts
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        // Allocate flat piece storage
        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(Shape, "spline pieces"),
            "ChebyshevSpline constructor",
            Shape);
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }

    /// <summary>
    /// Create a piecewise Chebyshev spline with optional error-driven auto-N construction.
    /// </summary>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension; null entries signal auto-N. Pass null to make every dim auto-N (requires errorThreshold).</param>
    /// <param name="knots">Interior knots per dimension. Null defaults to empty arrays (single piece per dim).</param>
    /// <param name="errorThreshold">Target supremum-norm error per piece. Required if any nNodes entry is null.</param>
    /// <param name="maxN">Cap on nodes per dimension during the doubling loop (default 64, must be at least 3).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="deferBuild">If true, skip eager build. Call <see cref="SetOriginalFunctionValues"/> to finish construction.</param>
    /// <param name="nWorkers">Number of parallel workers for function evaluation. null = sequential; -1 = <see cref="Environment.ProcessorCount"/>; positive = exact count.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count across all pieces.</param>
    /// <remarks>Thread safety: the user-supplied function must be thread-safe when <paramref name="nWorkers"/> is non-null.</remarks>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int?[]? nNodes = null,
        double[][]? knots = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        bool deferBuild = false,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        if (maxN < 3)
            throw new ArgumentException(
                $"maxN must be at least 3 (the initial N of the doubling loop), got maxN={maxN}.");

        knots ??= Enumerable.Range(0, numDimensions).Select(_ => Array.Empty<double>()).ToArray();

        // Normalize nNodes
        int?[] resolvedOriginal;
        if (nNodes == null)
        {
            if (errorThreshold == null)
                throw new ArgumentException(
                    "Must provide either nNodes (explicit) or errorThreshold (auto-N). Got neither.");
            resolvedOriginal = new int?[numDimensions];
        }
        else
        {
            resolvedOriginal = (int?[])nNodes.Clone();
            if (resolvedOriginal.Any(n => n == null) && errorThreshold == null)
                throw new ArgumentException(
                    "Null entries in nNodes require errorThreshold to be set (auto-N mode).");
        }

        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        ErrorThreshold = errorThreshold;
        MaxN = maxN;
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;
        OriginalNNodes = (int?[])resolvedOriginal.Clone();

        // Public NNodes is meaningful only after Build resolves the auto-N values.
        // For now, fill with 0 as placeholders (will be populated per-piece after Build).
        NNodes = resolvedOriginal.Select(n => n ?? 0).ToArray();

        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        Intervals = ComputeIntervals(numDimensions, domain, knots);
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(Shape, nameof(ChebyshevSpline)),
            nameof(ChebyshevSpline),
            Shape);
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }

    /// <summary>
    /// Create a piecewise Chebyshev spline with per-sub-interval node counts.
    /// </summary>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="nNodesNested">Nested array: nNodesNested[d][i] is the node count for piece i along dim d. Length per dim must equal knots[d].Length + 1.</param>
    /// <param name="knots">Interior knots per dimension. Required (no default) when using nested form.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="deferBuild">If true, skip eager build. Call <see cref="SetOriginalFunctionValues"/> to finish construction.</param>
    /// <param name="nWorkers">Number of parallel workers for function evaluation. null = sequential; -1 = <see cref="Environment.ProcessorCount"/>; positive = exact count.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count across all pieces.</param>
    /// <remarks>Thread safety: the user-supplied function must be thread-safe when <paramref name="nWorkers"/> is non-null.</remarks>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[][] nNodesNested,
        double[][] knots,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        bool deferBuild = false,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        if (nNodesNested.Length != numDimensions)
            throw new ArgumentException(
                $"nNodesNested must have {numDimensions} entries (one list per dim), got {nNodesNested.Length}");
        for (int d = 0; d < numDimensions; d++)
        {
            int expected = knots[d].Length + 1;
            if (nNodesNested[d].Length != expected)
                throw new ArgumentException(
                    $"nNodesNested[{d}] must have {expected} entries (one per piece), got {nNodesNested[d].Length}");
        }

        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;
        MaxN = 64;
        ErrorThreshold = null;
        OriginalNNodes = Array.Empty<int?>();
        NestedNNodes = nNodesNested.Select(row => (int[])row.Clone()).ToArray();

        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        Intervals = ComputeIntervals(numDimensions, domain, knots);
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        // Public NNodes surfaces piece 0's counts as a representative summary;
        // full per-piece data lives in NestedNNodes.
        NNodes = nNodesNested.Select(row => row[0]).ToArray();

        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(Shape, nameof(ChebyshevSpline)),
            nameof(ChebyshevSpline),
            Shape);
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }

    /// <summary>Return the error threshold passed to the constructor, or null in fixed-N mode.</summary>
    public double? GetErrorThreshold() => ErrorThreshold;

    // Internal parameterless constructor for factories
    internal ChebyshevSpline() { }

    // ------------------------------------------------------------------
    // Validation helpers
    // ------------------------------------------------------------------

    internal static void ValidateKnots(int numDimensions, double[][] domain, double[][] knots)
    {
        if (knots.Length != numDimensions)
            throw new ArgumentException(
                $"knots length {knots.Length} != numDimensions {numDimensions}");

        for (int d = 0; d < numDimensions; d++)
        {
            double lo = domain[d][0], hi = domain[d][1];
            for (int i = 0; i < knots[d].Length; i++)
            {
                double k = knots[d][i];
                if (!(lo < k && k < hi))
                    throw new ArgumentException(
                        $"Knot {k} for dimension {d} is not strictly inside domain [{lo}, {hi}]");
            }

            // Check sorted
            for (int i = 1; i < knots[d].Length; i++)
            {
                if (knots[d][i] <= knots[d][i - 1])
                    throw new ArgumentException($"Knots for dimension {d} must be sorted");
            }

            // Check unique
            if (knots[d].Distinct().Count() != knots[d].Length)
                throw new ArgumentException($"Knots for dimension {d} contain duplicates");
        }
    }

    internal static (double lo, double hi)[][] ComputeIntervals(
        int numDimensions, double[][] domain, double[][] knots)
    {
        var intervals = new (double lo, double hi)[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
        {
            double lo = domain[d][0], hi = domain[d][1];
            int nKnots = knots[d].Length;
            var dimIntervals = new (double, double)[nKnots + 1];
            if (nKnots == 0)
            {
                dimIntervals[0] = (lo, hi);
            }
            else
            {
                dimIntervals[0] = (lo, knots[d][0]);
                for (int i = 0; i < nKnots - 1; i++)
                    dimIntervals[i + 1] = (knots[d][i], knots[d][i + 1]);
                dimIntervals[nKnots] = (knots[d][nKnots - 1], hi);
            }
            intervals[d] = dimIntervals;
        }
        return intervals;
    }

    private static int[][]? CloneNestedNNodes(int[][]? nestedNNodes)
    {
        return nestedNNodes == null
            ? null
            : nestedNNodes.Select(row => (int[])row.Clone()).ToArray();
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// <summary>
    /// Build all pieces by evaluating the function on each sub-domain.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    public void Build(bool verbose = true)
    {
        if (Function == null)
            throw new InvalidOperationException(
                "Cannot build: no function assigned. " +
                "This object was created via FromValues() or Load().");

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        int totalPieces = NumPieces;
        string totalEvalsText = NNodes.Any(n => n <= 0)
            ? "adaptive"
            : $"{TotalBuildEvals:N0}";

        if (verbose)
            Console.WriteLine(
                $"Building {NumDimensions}D Chebyshev Spline " +
                $"({totalPieces} pieces, {totalEvalsText} total evaluations)...");

        int flatIdx = 0;
        int progressOffset = 0;
        foreach (var multiIdx in NdIndex(Shape))
        {
            // Compute sub-domain for this piece
            double[][] subDomain = new double[NumDimensions][];
            for (int d = 0; d < NumDimensions; d++)
            {
                var iv = Intervals[d][multiIdx[d]];
                subDomain[d] = new[] { iv.lo, iv.hi };
            }

            // Build per-piece progress shim that offsets reported values by cumulative evals so far.
            int capturedOffset = progressOffset;
            IProgress<int>? pieceProgress = _progress is null ? null
                : new Internal.OffsetProgress(_progress, capturedOffset);

            ChebyshevApproximation piece;
            if (NestedNNodes != null)
            {
                // Per-piece nested node counts: look up nNodes from the piece's multi-index
                int[] pieceN = new int[NumDimensions];
                for (int d = 0; d < NumDimensions; d++)
                    pieceN[d] = NestedNNodes[d][multiIdx[d]];
                piece = new ChebyshevApproximation(
                    Function!, NumDimensions, subDomain, pieceN,
                    maxDerivativeOrder: MaxDerivativeOrder,
                    additionalData: _additionalData,
                    nWorkers: _nWorkers, progress: pieceProgress);
                progressOffset = checked(progressOffset + TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(pieceN, nameof(Build)),
                    nameof(Build),
                    pieceN));
            }
            else if (OriginalNNodes.Length > 0 && (OriginalNNodes.Any(n => n == null) || ErrorThreshold != null))
            {
                // Auto-N or threshold-driven: construct via the int?[] overload.
                // Clone so each piece's ctor cannot mutate this Spline's stored array.
                int?[] pieceNNodes = (int?[])OriginalNNodes.Clone();
                piece = new ChebyshevApproximation(
                    Function!, NumDimensions, subDomain,
                    nNodes: pieceNNodes,
                    errorThreshold: ErrorThreshold,
                    maxN: MaxN,
                    maxDerivativeOrder: MaxDerivativeOrder,
                    additionalData: _additionalData,
                    nWorkers: _nWorkers, progress: pieceProgress);
                // Offset update happens after Build (actual N resolved then)
            }
            else
            {
                // Fixed-N: existing path
                piece = new ChebyshevApproximation(
                    Function!, NumDimensions, subDomain, NNodes,
                    maxDerivativeOrder: MaxDerivativeOrder,
                    additionalData: _additionalData,
                    nWorkers: _nWorkers, progress: pieceProgress);
                progressOffset = checked(progressOffset + TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(NNodes, nameof(Build)),
                    nameof(Build),
                    NNodes));
            }
            piece.Build(verbose: false);
            // For auto-N path, update offset after Build (N is now known)
            if (OriginalNNodes.Length > 0 && (OriginalNNodes.Any(n => n == null) || ErrorThreshold != null))
                progressOffset = checked(progressOffset + TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(piece.NNodes, nameof(Build)),
                    nameof(Build),
                    piece.NNodes));
            Pieces[flatIdx] = piece;

            if (verbose)
                Console.WriteLine(
                    $"  Piece {flatIdx + 1}/{totalPieces}: " +
                    $"domain [{string.Join(", ", subDomain.Select(d => $"[{d[0]}, {d[1]}]"))}]");

            flatIdx++;
        }

        sw.Stop();
        BuildTime = sw.Elapsed.TotalSeconds;
        Built = true;

        if (verbose)
            Console.WriteLine($"Build complete in {BuildTime:F3}s");
    }

    // ------------------------------------------------------------------
    // Piece routing
    // ------------------------------------------------------------------

    internal (int flatIdx, ChebyshevApproximation piece) FindPiece(double[] point)
    {
        int[] multiIdx = new int[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            if (Knots[d].Length == 0)
            {
                multiIdx[d] = 0;
            }
            else
            {
                // searchsorted with side='right': point at exact knot goes to right piece
                int idx = Array.BinarySearch(Knots[d], point[d]);
                if (idx >= 0)
                {
                    // Exact match — side='right' means we go one past
                    idx = idx + 1;
                }
                else
                {
                    // ~idx gives the insertion point (first element > point[d])
                    idx = ~idx;
                }
                // Clamp to valid range
                idx = Math.Min(idx, Shape[d] - 1);
                multiIdx[d] = idx;
            }
        }

        int flat = RavelMultiIndex(multiIdx, Shape);
        return (flat, Pieces[flat]!);
    }

    internal void CheckKnotBoundary(double[] point, int[] derivativeOrder)
    {
        bool anyDeriv = false;
        for (int d = 0; d < derivativeOrder.Length; d++)
        {
            if (derivativeOrder[d] > 0) { anyDeriv = true; break; }
        }
        if (!anyDeriv) return;

        for (int d = 0; d < NumDimensions; d++)
        {
            for (int k = 0; k < Knots[d].Length; k++)
            {
                if (Math.Abs(point[d] - Knots[d][k]) < 1e-14)
                    throw new ArgumentException(
                        $"Requested derivative is not defined at knot x[{d}]={Knots[d][k]}. " +
                        "The adjacent polynomial pieces may have different derivative values at this point.");
            }
        }
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate the spline approximation at a point.
    /// </summary>
    /// <param name="point">Evaluation point in the full domain.</param>
    /// <param name="derivativeOrder">Derivative order for each dimension (0 = function value).</param>
    /// <returns>Approximated function value or derivative.</returns>
    public double Eval(double[] point, int[] derivativeOrder)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before Eval().");
        EvaluationArguments.ValidatePoint(point, NumDimensions);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions);
        CheckKnotBoundary(point, derivativeOrder);
        var (_, piece) = FindPiece(point);
        return piece.VectorizedEval(point, derivativeOrder);
    }

    /// <summary>
    /// Evaluate multiple derivative orders at one point, sharing weights.
    /// </summary>
    /// <param name="point">Evaluation point in the full domain.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.</param>
    /// <returns>One result per derivative order.</returns>
    public double[] EvalMulti(double[] point, int[][] derivativeOrders)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before EvalMulti().");
        EvaluationArguments.ValidatePoint(point, NumDimensions);
        EvaluationArguments.ValidateDerivativeOrders(derivativeOrders, NumDimensions);
        foreach (var dord in derivativeOrders)
            CheckKnotBoundary(point, dord);
        var (_, piece) = FindPiece(point);
        return piece.VectorizedEvalMulti(point, derivativeOrders);
    }

    /// <summary>
    /// Evaluate at multiple points, grouping by piece for efficiency.
    /// </summary>
    /// <param name="points">Evaluation points (N x numDimensions).</param>
    /// <param name="derivativeOrder">Derivative order for each dimension.</param>
    /// <returns>Approximated values at each point.</returns>
    public double[] EvalBatch(double[][] points, int[] derivativeOrder)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before EvalBatch().");
        EvaluationArguments.ValidatePoints(points, NumDimensions);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions);

        int N = points.Length;
        double[] results = new double[N];

        // Compute piece index for each point
        int[] flatIndices = new int[N];
        for (int i = 0; i < N; i++)
        {
            CheckKnotBoundary(points[i], derivativeOrder);
            var (flatIdx, _) = FindPiece(points[i]);
            flatIndices[i] = flatIdx;
        }

        // Group by piece and batch-eval
        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < N; i++)
        {
            if (!groups.TryGetValue(flatIndices[i], out var list))
            {
                list = new List<int>();
                groups[flatIndices[i]] = list;
            }
            list.Add(i);
        }

        foreach (var kvp in groups)
        {
            var piece = Pieces[kvp.Key]!;
            var indices = kvp.Value;
            var subPoints = indices.Select(i => points[i]).ToArray();
            var subResults = piece.VectorizedEvalBatch(subPoints, derivativeOrder);
            for (int j = 0; j < indices.Count; j++)
                results[indices[j]] = subResults[j];
        }

        return results;
    }

    // ------------------------------------------------------------------
    // Error estimation
    // ------------------------------------------------------------------

    /// <summary>
    /// Estimate the supremum-norm interpolation error.
    /// Returns the maximum error estimate across all pieces.
    /// </summary>
    public double ErrorEstimate()
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before ErrorEstimate().");

        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;

        double maxError = 0.0;
        foreach (var piece in Pieces)
            maxError = Math.Max(maxError, piece!.ErrorEstimate());

        _cachedErrorEstimate = maxError;
        return maxError;
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// <summary>Total number of pieces (Cartesian product of per-dimension intervals).</summary>
    public int NumPieces
    {
        get
        {
            return TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(Shape, nameof(NumPieces)),
                nameof(NumPieces),
                Shape);
        }
    }

    /// <summary>Total number of function evaluations used during build.</summary>
    public int TotalBuildEvals
    {
        get
        {
            if (NestedNNodes != null)
            {
                long nestedTotal = 0;
                foreach (var multiIdx in NdIndex(Shape))
                {
                    var pieceNNodes = new int[NumDimensions];
                    for (int d = 0; d < NumDimensions; d++)
                        pieceNNodes[d] = NestedNNodes[d][multiIdx[d]];
                    nestedTotal = checked(nestedTotal +
                        TensorShape.CheckedProduct(pieceNNodes, nameof(TotalBuildEvals)));
                }
                return TensorShape.RequireArrayLength(nestedTotal, nameof(TotalBuildEvals));
            }

            if (NNodes.Any(n => n <= 0))
            {
                if (Pieces == null || Pieces.All(p => p == null)) return 0;
                long sum = 0;
                foreach (var piece in Pieces)
                    if (piece != null) sum = checked(sum + piece.GetNumEvaluationPoints());
                return TensorShape.RequireArrayLength(sum, nameof(TotalBuildEvals));
            }

            long perPiece = TensorShape.CheckedProduct(NNodes, nameof(TotalBuildEvals));
            long total = checked(NumPieces * perPiece);
            return TensorShape.RequireArrayLength(total, nameof(TotalBuildEvals));
        }
    }

    // ------------------------------------------------------------------
    // Static factory: WithSpecialPoints
    // ------------------------------------------------------------------

    /// <summary>
    /// Create a <see cref="ChebyshevSpline"/> with kinks declared via <paramref name="specialPoints"/>
    /// (a more user-friendly name than <c>knots</c> when the function has known non-smooth points).
    /// Functionally equivalent to passing the same values as knots to a regular constructor.
    /// </summary>
    /// <remarks>
    /// Python's <c>ChebyshevApproximation(special_points=...)</c> returns a
    /// <c>ChebyshevSpline</c> at construction time.  C# constructors cannot return a
    /// different type; this static factory is the C#-idiomatic equivalent.
    /// Exactly one of <paramref name="nNodesNested"/>, <paramref name="nNodes"/>, or
    /// <paramref name="errorThreshold"/> must be supplied.
    /// </remarks>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="specialPoints">Per-dim list of kink locations. Equivalent to knots;
    /// outer length must equal <paramref name="numDimensions"/>.</param>
    /// <param name="nNodesNested">Per-sub-interval node counts (per dim, per piece).
    /// Mutually exclusive with <paramref name="errorThreshold"/>.</param>
    /// <param name="nNodes">Flat per-dim node counts (shared across pieces).
    /// Mutually exclusive with <paramref name="nNodesNested"/> and <paramref name="errorThreshold"/>.</param>
    /// <param name="errorThreshold">Target error per piece.
    /// Mutually exclusive with <paramref name="nNodes"/>/<paramref name="nNodesNested"/>.</param>
    /// <param name="maxN">Cap on doubling-loop nodes per dimension (default 64).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <returns>An unbuilt <see cref="ChebyshevSpline"/> ready for <c>Build()</c>.</returns>
    public static ChebyshevSpline WithSpecialPoints(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        double[][] specialPoints,
        int[][]? nNodesNested = null,
        int[]? nNodes = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2)
    {
        if (specialPoints.Length != numDimensions)
            throw new ArgumentException(
                $"specialPoints must have {numDimensions} entries, got {specialPoints.Length}");

        // Validate using existing knots validator (sorted, strictly inside, no dupes).
        ValidateKnots(numDimensions, domain, specialPoints);

        int suppliedFormCount =
            (nNodesNested != null ? 1 : 0) +
            (nNodes != null ? 1 : 0) +
            (errorThreshold != null ? 1 : 0);

        if (suppliedFormCount == 0)
            throw new ArgumentException(
                "WithSpecialPoints requires exactly one of: nNodesNested, nNodes, or errorThreshold.");
        if (suppliedFormCount > 1)
            throw new ArgumentException(
                "WithSpecialPoints accepts only one of nNodesNested, nNodes, or errorThreshold (not multiple).");

        if (nNodesNested != null)
            return new ChebyshevSpline(function, numDimensions, domain,
                nNodesNested, specialPoints, maxDerivativeOrder);

        if (nNodes != null)
        {
            // Delegate to the flat int[] ctor (no errorThreshold, no maxN on that overload).
            int[] flatNodes = nNodes;
            return new ChebyshevSpline(function, numDimensions, domain,
                flatNodes, specialPoints, maxDerivativeOrder);
        }

        // errorThreshold path: every dim auto-N.
        return new ChebyshevSpline(function, numDimensions, domain,
            nNodes: (int?[]?)null, knots: specialPoints,
            errorThreshold: errorThreshold, maxN: maxN, maxDerivativeOrder: maxDerivativeOrder);
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// <summary>
    /// Save the built spline to a file.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    /// <param name="format">"json" (default) or "binary". Binary requires
    /// flat (non-nested) nNodes — throws NotSupportedException otherwise.</param>
    public void Save(string path, string format = "json")
    {
        if (!Built)
            throw new InvalidOperationException(
                "Cannot save an unbuilt spline. Call Build() first.");

        switch (format)
        {
            case "json":
                SaveJson(path);
                break;
            case "binary":
                EnsureBinarySerializable();
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
        var state = new SplineSerializationState
        {
            Type = "ChebyshevSpline",
            NumDimensions = NumDimensions,
            Domain = Domain,
            NNodes = NNodes,
            MaxDerivativeOrder = MaxDerivativeOrder,
            Knots = Knots,
            Shape = Shape,
            BuildTime = BuildTime,
            PieceStates = Pieces.Select(p =>
            {
                var ps = new PieceState
                {
                    NumDimensions = p!.NumDimensions,
                    Domain = p.Domain,
                    NNodes = p.NNodes,
                    MaxDerivativeOrder = p.MaxDerivativeOrder,
                    NodeArrays = p.NodeArrays,
                    TensorValues = p.TensorValues!,
                    Weights = p.Weights!,
                    DiffMatrices = p.DiffMatrices!.Select(ChebyshevApproximation.Flatten2D).ToArray(),
                    BuildTime = p.BuildTime,
                    NEvaluations = p.NEvaluations,
                };
                return ps;
            }).ToArray(),
            OriginalNNodes = OriginalNNodes.Length > 0 ? OriginalNNodes : null,
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            NestedNNodes = NestedNNodes,
            Version = "0.8.0",
            Descriptor = _descriptor,
            RegisteredDerivativeOrders = _registeredDerivativeOrders.Count > 0
                ? _registeredDerivativeOrders.Select(o => (int[])o.Clone()).ToArray()
                : null,
        };

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    private void EnsureBinarySerializable()
    {
        bool hasSharedPositiveNodes = NNodes.Length == NumDimensions && NNodes.All(n => n > 0);
        bool allPiecesShareNodes = Pieces.All(p => p != null && p.NNodes.SequenceEqual(NNodes));
        if (NestedNNodes != null || !hasSharedPositiveNodes || !allPiecesShareNodes)
            throw new NotSupportedException(
                "binary format requires shared positive n_nodes across pieces; " +
                "use format='json' for adaptive or nested-n_nodes splines");
    }

    private void SaveBinary(string path)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        Internal.PcbFormat.WriteHeader(w, Internal.PcbFormat.ClassTagSpline);
        var pieceTensors = Pieces.Select(p => p!.TensorValues!).ToArray();
        Internal.PcbFormat.WriteSplineBody(w, Domain, NNodes, Knots, pieceTensors);
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
    /// Load a previously saved spline from a file.
    /// Auto-detects binary (.pcb magic) vs JSON format.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>The restored spline.</returns>
    /// <exception cref="InvalidDataException">If the file contains a malformed ChebyshevSpline state.</exception>
    public static ChebyshevSpline Load(string path)
    {
        if (Internal.PcbFormat.IsBinary(path))
            return LoadBinary(path);
        return LoadJson(path);
    }

    private static ChebyshevSpline LoadBinary(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        var header = Internal.PcbFormat.ReadHeader(r);
        if (header.ClassTag != Internal.PcbFormat.ClassTagSpline)
            throw new InvalidDataException(
                $"binary file class_tag={header.ClassTag} is not ChebyshevSpline " +
                $"(tag {Internal.PcbFormat.ClassTagSpline}); " +
                $"call ChebyshevApproximation.Load instead if class_tag={Internal.PcbFormat.ClassTagApproximation}");

        var (domain, nNodes, knots, pieceTensors) = Internal.PcbFormat.ReadSplineBody(r);
        var spline = FromValues(pieceTensors, domain.Length, domain, nNodes, knots);
        spline._constructorType = "load";
        return spline;
    }

    private static ChebyshevSpline LoadJson(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SplineSerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize");

        if (state.Type != "ChebyshevSpline")
            throw new InvalidOperationException(
                $"Expected type ChebyshevSpline, got {state.Type}");

        ValidateSerializedState(state);

        var pieces = state.PieceStates.Select(ps =>
        {
            var piece = new ChebyshevApproximation
            {
                Function = null,
                NumDimensions = ps.NumDimensions,
                Domain = ps.Domain,
                NNodes = ps.NNodes,
                MaxDerivativeOrder = ps.MaxDerivativeOrder ?? 2,
                NodeArrays = ps.NodeArrays,
                TensorValues = ps.TensorValues,
                Weights = ps.Weights,
                BuildTime = ps.BuildTime,
                NEvaluations = ps.NEvaluations,
            };

            // Reconstruct diff matrices
            piece.DiffMatrices = new double[ps.NumDimensions][,];
            for (int d = 0; d < ps.NumDimensions; d++)
            {
                int n = ps.NNodes[d];
                piece.DiffMatrices[d] = ChebyshevApproximation.Unflatten2D(ps.DiffMatrices[d], n, n);
            }
            piece.PrecomputeTransposedDiffMatrices();
            return piece;
        }).ToArray();

        // Reconstruct intervals from knots
        var intervals = ComputeIntervals(state.NumDimensions, state.Domain, state.Knots);

        // v0.5.0 migration: OriginalNNodes / ErrorThreshold / MaxN / NestedNNodes may be absent in older files.
        int?[] originalNNodes = state.OriginalNNodes ?? Array.Empty<int?>();

        var spline = new ChebyshevSpline
        {
            Function = null,
            NumDimensions = state.NumDimensions,
            Domain = state.Domain,
            NNodes = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder ?? 2,
            Knots = state.Knots,
            Intervals = intervals,
            Shape = state.Shape,
            Pieces = pieces.Cast<ChebyshevApproximation?>().ToArray(),
            Built = true,
            BuildTime = state.BuildTime,
            OriginalNNodes = originalNNodes,
            ErrorThreshold = state.ErrorThreshold,
            MaxN = state.MaxN ?? 64,
            NestedNNodes = state.NestedNNodes,
            _cachedErrorEstimate = null,
        };
        // v0.8.0 migration: Descriptor may be absent in older files.
        spline._descriptor = state.Descriptor;
        // ConstructorType is intentionally NOT restored from state — Load always sets "load".
        spline._constructorType = "load";
        if (state.RegisteredDerivativeOrders != null)
        {
            foreach (var orders in state.RegisteredDerivativeOrders)
            {
                var key = new Internal.TupleKey(orders);
                int id = spline._registeredDerivativeOrders.Count;
                spline._registeredDerivativeOrders.Add((int[])orders.Clone());
                spline._derivativeIdRegistry[key] = id;
            }
        }
        return spline;
    }

    private static void ValidateSerializedState(SplineSerializationState state)
    {
        int d = state.NumDimensions;
        if (d <= 0)
            throw new InvalidDataException($"NumDimensions must be positive, got {d}.");

        ValidateDomain(state.Domain, d, nameof(SplineSerializationState.Domain));
        ValidateOriginalNNodes(state.OriginalNNodes, d);
        ValidateTopLevelNNodes(state.NNodes, d, state.OriginalNNodes, state.ErrorThreshold);
        ValidateKnotsForLoad(state.Knots, d, state.Domain);

        if (state.MaxDerivativeOrder is < 0)
            throw new InvalidDataException($"MaxDerivativeOrder must be non-negative, got {state.MaxDerivativeOrder}.");
        if (!double.IsFinite(state.BuildTime) || state.BuildTime < 0.0)
            throw new InvalidDataException($"BuildTime must be finite and non-negative, got {state.BuildTime}.");
        if (state.ErrorThreshold is { } threshold &&
            (!double.IsFinite(threshold) || threshold < 0.0))
            throw new InvalidDataException($"ErrorThreshold must be finite and non-negative, got {threshold}.");
        if (state.MaxN is <= 0)
            throw new InvalidDataException($"MaxN must be positive, got {state.MaxN}.");

        ValidateShape(state.Shape, state.Knots, d);
        ValidateNestedNNodes(state.NestedNNodes, state.Shape, d);
        ValidateDerivativeRegistry(state.RegisteredDerivativeOrders, d);

        int expectedPieces = CheckedArrayLengthForInvalidData(state.Shape, nameof(SplineSerializationState.PieceStates));
        if (state.PieceStates is null)
            throw new InvalidDataException("PieceStates must be present.");
        if (state.PieceStates.Length != expectedPieces)
            throw new InvalidDataException(
                $"PieceStates has length {state.PieceStates.Length}, expected {expectedPieces}.");

        var intervals = ComputeIntervals(d, state.Domain, state.Knots);
        for (int i = 0; i < state.PieceStates.Length; i++)
            ValidatePieceState(state.PieceStates[i], i, d, state.Shape, intervals);
    }

    private static void ValidatePieceState(
        PieceState? piece,
        int pieceIndex,
        int numDimensions,
        int[] shape,
        (double lo, double hi)[][] intervals)
    {
        if (piece is null)
            throw new InvalidDataException($"PieceStates[{pieceIndex}] must be present.");
        if (piece.NumDimensions != numDimensions)
            throw new InvalidDataException(
                $"PieceStates[{pieceIndex}].NumDimensions={piece.NumDimensions}, expected {numDimensions}.");
        if (piece.MaxDerivativeOrder is < 0)
            throw new InvalidDataException(
                $"PieceStates[{pieceIndex}].MaxDerivativeOrder must be non-negative, got {piece.MaxDerivativeOrder}.");
        if (!double.IsFinite(piece.BuildTime) || piece.BuildTime < 0.0)
            throw new InvalidDataException(
                $"PieceStates[{pieceIndex}].BuildTime must be finite and non-negative, got {piece.BuildTime}.");
        if (piece.NEvaluations < 0)
            throw new InvalidDataException(
                $"PieceStates[{pieceIndex}].NEvaluations must be non-negative, got {piece.NEvaluations}.");

        ValidateDomain(piece.Domain, numDimensions, $"PieceStates[{pieceIndex}].Domain");
        ValidatePositiveVector(piece.NNodes, numDimensions, $"PieceStates[{pieceIndex}].NNodes");
        ValidateApproxVectorArray(piece.NodeArrays, piece.NNodes, $"PieceStates[{pieceIndex}].NodeArrays");
        ValidateApproxVectorArray(piece.Weights, piece.NNodes, $"PieceStates[{pieceIndex}].Weights");
        ValidateDiffMatrices(piece.DiffMatrices, piece.NNodes, pieceIndex);

        int expectedTensorLength = CheckedArrayLengthForInvalidData(
            piece.NNodes,
            $"PieceStates[{pieceIndex}].TensorValues");
        ValidateFiniteVector(
            piece.TensorValues,
            expectedTensorLength,
            $"PieceStates[{pieceIndex}].TensorValues");

        int[] multiIndex = FlatToMultiIndex(pieceIndex, shape);
        for (int dim = 0; dim < numDimensions; dim++)
        {
            var expectedDomain = intervals[dim][multiIndex[dim]];
            if (piece.Domain[dim][0] != expectedDomain.lo || piece.Domain[dim][1] != expectedDomain.hi)
                throw new InvalidDataException(
                    $"PieceStates[{pieceIndex}].Domain[{dim}] does not match spline interval.");
        }
    }

    private static void ValidateShape(int[]? shape, double[][] knots, int numDimensions)
    {
        if (shape is null)
            throw new InvalidDataException("Shape must be present.");
        if (shape.Length != numDimensions)
            throw new InvalidDataException($"Shape has length {shape.Length}, expected {numDimensions}.");

        for (int dim = 0; dim < numDimensions; dim++)
        {
            int expected = knots[dim].Length + 1;
            if (shape[dim] != expected)
                throw new InvalidDataException($"Shape[{dim}]={shape[dim]}, expected {expected}.");
        }
    }

    private static void ValidateKnotsForLoad(double[][]? knots, int numDimensions, double[][] domain)
    {
        if (knots is null)
            throw new InvalidDataException("Knots must be present.");
        if (knots.Length != numDimensions)
            throw new InvalidDataException($"Knots has length {knots.Length}, expected {numDimensions}.");

        for (int dim = 0; dim < numDimensions; dim++)
        {
            double[] dimKnots = knots[dim]
                ?? throw new InvalidDataException($"Knots[{dim}] must be present.");
            double lo = domain[dim][0];
            double hi = domain[dim][1];
            double previous = double.NegativeInfinity;
            for (int i = 0; i < dimKnots.Length; i++)
            {
                double knot = dimKnots[i];
                if (!double.IsFinite(knot))
                    throw new InvalidDataException($"Knots[{dim}][{i}] must be finite.");
                if (!(lo < knot && knot < hi))
                    throw new InvalidDataException(
                        $"Knots[{dim}][{i}]={knot} must be strictly inside domain [{lo}, {hi}].");
                if (i > 0 && knot <= previous)
                    throw new InvalidDataException($"Knots[{dim}] must be strictly increasing.");
                previous = knot;
            }
        }
    }

    private static void ValidateDomain(double[][]? domain, int numDimensions, string name)
    {
        if (domain is null)
            throw new InvalidDataException($"{name} must be present.");
        if (domain.Length != numDimensions)
            throw new InvalidDataException($"{name} has length {domain.Length}, expected {numDimensions}.");

        for (int i = 0; i < numDimensions; i++)
        {
            double[] bounds = domain[i]
                ?? throw new InvalidDataException($"{name}[{i}] must be present.");
            if (bounds.Length != 2)
                throw new InvalidDataException($"{name}[{i}] must contain exactly two bounds.");
            if (!double.IsFinite(bounds[0]) || !double.IsFinite(bounds[1]))
                throw new InvalidDataException($"{name}[{i}] bounds must be finite.");
            if (bounds[0] >= bounds[1])
                throw new InvalidDataException($"{name}[{i}] lower bound must be less than upper bound.");
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

    private static void ValidateTopLevelNNodes(
        int[]? nNodes,
        int expectedLength,
        int?[]? originalNNodes,
        double? errorThreshold)
    {
        if (nNodes is null)
            throw new InvalidDataException($"{nameof(SplineSerializationState.NNodes)} must be present.");
        if (nNodes.Length != expectedLength)
            throw new InvalidDataException(
                $"{nameof(SplineSerializationState.NNodes)} has length {nNodes.Length}, expected {expectedLength}.");

        bool autoMode = errorThreshold != null &&
            originalNNodes != null &&
            originalNNodes.Length == expectedLength;

        for (int i = 0; i < nNodes.Length; i++)
        {
            if (nNodes[i] > 0) continue;
            if (nNodes[i] == 0 && autoMode && originalNNodes![i] is null) continue;
            throw new InvalidDataException(
                $"{nameof(SplineSerializationState.NNodes)}[{i}] must be positive, got {nNodes[i]}.");
        }
    }

    private static void ValidateFiniteVector(double[]? values, int expectedLength, string name)
    {
        if (values is null)
            throw new InvalidDataException($"{name} must be present.");
        if (values.Length != expectedLength)
            throw new InvalidDataException($"{name} has length {values.Length}, expected {expectedLength}.");

        for (int i = 0; i < values.Length; i++)
            if (!double.IsFinite(values[i]))
                throw new InvalidDataException($"{name}[{i}] must be finite.");
    }

    private static void ValidateApproxVectorArray(double[][]? arrays, int[] nNodes, string name)
    {
        if (arrays is null)
            throw new InvalidDataException($"{name} must be present.");
        if (arrays.Length != nNodes.Length)
            throw new InvalidDataException($"{name} has length {arrays.Length}, expected {nNodes.Length}.");

        for (int i = 0; i < arrays.Length; i++)
            ValidateFiniteVector(arrays[i], nNodes[i], $"{name}[{i}]");
    }

    private static void ValidateDiffMatrices(double[][]? matrices, int[] nNodes, int pieceIndex)
    {
        string name = $"PieceStates[{pieceIndex}].DiffMatrices";
        if (matrices is null)
            throw new InvalidDataException($"{name} must be present.");
        if (matrices.Length != nNodes.Length)
            throw new InvalidDataException($"{name} has length {matrices.Length}, expected {nNodes.Length}.");

        for (int i = 0; i < nNodes.Length; i++)
        {
            int expectedLength = CheckedArrayLengthForInvalidData(new[] { nNodes[i], nNodes[i] }, $"{name}[{i}]");
            ValidateFiniteVector(matrices[i], expectedLength, $"{name}[{i}]");
        }
    }

    private static void ValidateOriginalNNodes(int?[]? originalNNodes, int numDimensions)
    {
        if (originalNNodes is null) return;
        if (originalNNodes.Length != 0 && originalNNodes.Length != numDimensions)
            throw new InvalidDataException(
                $"OriginalNNodes has length {originalNNodes.Length}, expected 0 or {numDimensions}.");

        for (int i = 0; i < originalNNodes.Length; i++)
            if (originalNNodes[i] is <= 0)
                throw new InvalidDataException($"OriginalNNodes[{i}] must be positive or null.");
    }

    private static void ValidateNestedNNodes(int[][]? nestedNNodes, int[] shape, int numDimensions)
    {
        if (nestedNNodes is null) return;
        if (nestedNNodes.Length != numDimensions)
            throw new InvalidDataException($"NestedNNodes has length {nestedNNodes.Length}, expected {numDimensions}.");

        for (int dim = 0; dim < numDimensions; dim++)
            ValidatePositiveVector(nestedNNodes[dim], shape[dim], $"NestedNNodes[{dim}]");
    }

    private static void ValidateDerivativeRegistry(int[][]? registeredDerivativeOrders, int numDimensions)
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
                if (orders[j] < 0)
                    throw new InvalidDataException(
                        $"RegisteredDerivativeOrders[{i}][{j}] must be non-negative, got {orders[j]}.");
        }
    }

    private static int[] FlatToMultiIndex(int flatIndex, int[] shape)
    {
        var multi = new int[shape.Length];
        int rem = flatIndex;
        for (int dim = shape.Length - 1; dim >= 0; dim--)
        {
            multi[dim] = rem % shape[dim];
            rem /= shape[dim];
        }
        return multi;
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

    // ------------------------------------------------------------------
    // Nodes and FromValues
    // ------------------------------------------------------------------

    /// <summary>
    /// Generate Chebyshev nodes for every piece without evaluating any function.
    /// </summary>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Lower and upper bounds for each dimension.</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension per piece.</param>
    /// <param name="knots">Knot positions for each dimension (may be empty).</param>
    /// <returns>A SplineNodeInfo with per-piece node info.</returns>
    public static SplineNodeInfo Nodes(
        int numDimensions, double[][] domain, int[] nNodes, double[][] knots)
    {
        ValidateKnots(numDimensions, domain, knots);

        // Validate domain
        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d][0] >= domain[d][1])
                throw new ArgumentException(
                    $"domain[{d}]: lo={domain[d][0]} must be strictly less than hi={domain[d][1]}");
        }

        var intervals = ComputeIntervals(numDimensions, domain, knots);
        int[] shape = intervals.Select(iv => iv.Length).ToArray();

        var piecesInfo = new List<SplinePieceNodeInfo>();
        foreach (var multiIdx in NdIndex(shape))
        {
            double[][] subDomain = new double[numDimensions][];
            for (int d = 0; d < numDimensions; d++)
            {
                var iv = intervals[d][multiIdx[d]];
                subDomain[d] = new[] { iv.lo, iv.hi };
            }

            var pieceNodes = ChebyshevApproximation.Nodes(numDimensions, subDomain, nNodes);
            piecesInfo.Add(new SplinePieceNodeInfo
            {
                PieceIndex = (int[])multiIdx.Clone(),
                SubDomain = subDomain,
                NodesPerDim = pieceNodes.NodesPerDim,
                FullGrid = pieceNodes.FullGrid,
                Shape = pieceNodes.Shape,
            });
        }

        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(shape, nameof(FromValues)),
            nameof(FromValues),
            shape);

        return new SplineNodeInfo
        {
            Pieces = piecesInfo.ToArray(),
            NumPieces = totalPieces,
            PieceShape = (int[])shape.Clone(),
        };
    }

    /// <summary>
    /// Create a spline from pre-computed function values on each piece.
    /// </summary>
    /// <param name="pieceValues">Function values for each piece. Length must equal total pieces.</param>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Lower and upper bounds for each dimension.</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension per piece.</param>
    /// <param name="knots">Knot positions for each dimension.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order (default 2).</param>
    /// <returns>A fully built spline with Function=null.</returns>
    public static ChebyshevSpline FromValues(
        double[][] pieceValues,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        double[][] knots,
        int maxDerivativeOrder = 2)
    {
        ValidateKnots(numDimensions, domain, knots);

        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d][0] >= domain[d][1])
                throw new ArgumentException(
                    $"domain[{d}]: lo={domain[d][0]} must be strictly less than hi={domain[d][1]}");
        }

        var intervals = ComputeIntervals(numDimensions, domain, knots);
        int[] shape = intervals.Select(iv => iv.Length).ToArray();

        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(shape, nameof(FromValues)),
            nameof(FromValues),
            shape);

        if (pieceValues.Length != totalPieces)
            throw new ArgumentException(
                $"Expected {totalPieces} piece_values, got {pieceValues.Length}");

        // Validate per-piece shapes
        int expectedSize = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(nNodes, nameof(FromValues)),
            nameof(FromValues),
            nNodes);

        for (int i = 0; i < pieceValues.Length; i++)
        {
            if (pieceValues[i].Length != expectedSize)
                throw new ArgumentException(
                    $"piece_values[{i}] has length {pieceValues[i].Length}, expected {expectedSize}");
        }

        // Build each piece via ChebyshevApproximation.FromValues()
        var pieces = new ChebyshevApproximation?[totalPieces];
        int flatIdx = 0;
        foreach (var multiIdx in NdIndex(shape))
        {
            double[][] subDomain = new double[numDimensions][];
            for (int d = 0; d < numDimensions; d++)
            {
                var iv = intervals[d][multiIdx[d]];
                subDomain[d] = new[] { iv.lo, iv.hi };
            }

            pieces[flatIdx] = ChebyshevApproximation.FromValues(
                pieceValues[flatIdx], numDimensions, subDomain, nNodes,
                maxDerivativeOrder: maxDerivativeOrder);
            flatIdx++;
        }

        var spline = new ChebyshevSpline
        {
            Function = null,
            NumDimensions = numDimensions,
            Domain = domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])nNodes.Clone(),
            MaxDerivativeOrder = maxDerivativeOrder,
            Knots = knots.Select(k => (double[])k.Clone()).ToArray(),
            Intervals = intervals,
            Shape = shape,
            Pieces = pieces,
            Built = true,
            BuildTime = 0.0,
            _cachedErrorEstimate = null,
        };
        spline._constructorType = "from_values";
        return spline;
    }

    // ------------------------------------------------------------------
    // Internal factory for arithmetic operators
    // ------------------------------------------------------------------

    internal static ChebyshevSpline FromPieces(ChebyshevSpline source, ChebyshevApproximation?[] pieces)
    {
        return new ChebyshevSpline
        {
            Function = null,
            NumDimensions = source.NumDimensions,
            Domain = source.Domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])source.NNodes.Clone(),
            MaxDerivativeOrder = source.MaxDerivativeOrder,
            Knots = source.Knots.Select(k => (double[])k.Clone()).ToArray(),
            Intervals = source.Intervals.Select(iv => ((double, double)[])iv.Clone()).ToArray(),
            Shape = (int[])source.Shape.Clone(),
            Pieces = pieces,
            Built = true,
            BuildTime = 0.0,
            OriginalNNodes = (int?[])source.OriginalNNodes.Clone(),
            ErrorThreshold = source.ErrorThreshold,
            MaxN = source.MaxN,
            NestedNNodes = CloneNestedNNodes(source.NestedNNodes),
            _cachedErrorEstimate = null,
        };
    }

    // ------------------------------------------------------------------
    // Extrusion and slicing
    // ------------------------------------------------------------------

    /// <summary>
    /// Add new dimensions where the function is constant.
    /// </summary>
    /// <param name="extrudeParams">Tuples of (dimIndex, bounds, nNodes).</param>
    /// <returns>A new, higher-dimensional spline (already built).</returns>
    public ChebyshevSpline Extrude(params (int dimIndex, double[] bounds, int nNodes)[] extrudeParams)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var sorted = ExtrudeSlice.NormalizeExtrusionParams(extrudeParams, NumDimensions);

        var knots = Knots.Select(k => (double[])k.Clone()).ToList();
        var intervals = Intervals.Select(iv => (((double, double)[])iv.Clone())).ToList();
        var shape = Shape.ToList();
        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();
        var nestedNNodes = NestedNNodes?.Select(row => row.ToList()).ToList();
        List<int?>? originalNNodes = OriginalNNodes.Length == NumDimensions
            ? OriginalNNodes.ToList()
            : null;

        foreach (var (dimIdx, bounds, n) in sorted)
        {
            knots.Insert(dimIdx, Array.Empty<double>());
            intervals.Insert(dimIdx, new[] { (bounds[0], bounds[1]) });
            shape.Insert(dimIdx, 1);
            domain.Insert(dimIdx, (double[])bounds.Clone());
            nNodes.Insert(dimIdx, n);
            nestedNNodes?.Insert(dimIdx, new List<int> { n });
            originalNNodes?.Insert(dimIdx, n);
        }

        // Extrude each piece
        var pieces = new ChebyshevApproximation?[Pieces.Length];
        for (int i = 0; i < Pieces.Length; i++)
        {
            var p = Pieces[i]!;
            foreach (var (dimIdx, bounds, n) in sorted)
                p = p.Extrude((dimIdx, bounds, n));
            pieces[i] = p;
        }

        return new ChebyshevSpline
        {
            Function = null,
            NumDimensions = NumDimensions + sorted.Length,
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Knots = knots.ToArray(),
            Intervals = intervals.ToArray(),
            Shape = shape.ToArray(),
            Pieces = pieces,
            Built = true,
            BuildTime = 0.0,
            OriginalNNodes = originalNNodes?.ToArray() ?? Array.Empty<int?>(),
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            NestedNNodes = nestedNNodes?.Select(row => row.ToArray()).ToArray(),
            _cachedErrorEstimate = null,
        };
    }

    /// <summary>
    /// Fix one or more dimensions at given values, reducing dimensionality.
    /// </summary>
    /// <param name="sliceParams">Tuples of (dimIndex, value).</param>
    /// <returns>A new, lower-dimensional spline (already built).</returns>
    public ChebyshevSpline Slice(params (int dimIndex, double value)[] sliceParams)
    {
        if (!Built)
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

        var knots = Knots.Select(k => (double[])k.Clone()).ToList();
        var intervals = Intervals.Select(iv => ((double, double)[])iv.Clone()).ToList();
        var shape = Shape.ToList();
        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();
        var nestedNNodes = NestedNNodes?.Select(row => row.ToList()).ToList();
        List<int?>? originalNNodes = OriginalNNodes.Length == NumDimensions
            ? OriginalNNodes.ToList()
            : null;

        // Work with pieces as a multi-dimensional index structure
        // Use the shape to track which pieces survive
        var currentPieces = (ChebyshevApproximation?[])Pieces.Clone();
        var currentShape = Shape.ToList();

        foreach (var (dimIdx, value) in sorted) // descending order
        {
            // Find which interval contains the value along this dim
            double[] knotsD = knots[dimIdx];
            int intervalIdx;
            if (knotsD.Length == 0)
            {
                intervalIdx = 0;
            }
            else
            {
                int searchResult = Array.BinarySearch(knotsD, value);
                if (searchResult >= 0)
                    intervalIdx = searchResult + 1;
                else
                    intervalIdx = ~searchResult;
                intervalIdx = Math.Min(intervalIdx, currentShape[dimIdx] - 1);
            }

            // Select only pieces at this interval index along dimIdx
            // and slice each surviving piece
            var newShape = currentShape.ToList();
            newShape.RemoveAt(dimIdx);

            int newTotal = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(newShape, nameof(Slice)),
                nameof(Slice),
                newShape);
            var newPieces = new ChebyshevApproximation?[newTotal];

            if (newShape.Count > 0)
            {
                int newFlat = 0;
                foreach (var newMultiIdx in NdIndex(newShape.ToArray()))
                {
                    // Build the multi-index in the old shape by inserting intervalIdx at dimIdx
                    var oldMultiIdx = new int[currentShape.Count];
                    int nd = 0;
                    for (int d = 0; d < currentShape.Count; d++)
                    {
                        if (d == dimIdx)
                            oldMultiIdx[d] = intervalIdx;
                        else
                            oldMultiIdx[d] = newMultiIdx[nd++];
                    }

                    int oldFlat = RavelMultiIndex(oldMultiIdx, currentShape.ToArray());
                    newPieces[newFlat] = currentPieces[oldFlat]!.Slice((dimIdx, value));
                    newFlat++;
                }
            }
            else
            {
                // All dims being sliced — single piece survives
                var oldMultiIdx = new int[currentShape.Count];
                oldMultiIdx[dimIdx] = intervalIdx;
                int oldFlat = RavelMultiIndex(oldMultiIdx, currentShape.ToArray());
                newPieces[0] = currentPieces[oldFlat]!.Slice((dimIdx, value));
            }

            currentPieces = newPieces;
            currentShape.RemoveAt(dimIdx);

            knots.RemoveAt(dimIdx);
            intervals.RemoveAt(dimIdx);
            shape.RemoveAt(dimIdx);
            domain.RemoveAt(dimIdx);
            nNodes.RemoveAt(dimIdx);
            nestedNNodes?.RemoveAt(dimIdx);
            originalNNodes?.RemoveAt(dimIdx);
        }

        return new ChebyshevSpline
        {
            Function = null,
            NumDimensions = NumDimensions - sorted.Length,
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Knots = knots.ToArray(),
            Intervals = intervals.ToArray(),
            Shape = shape.ToArray(),
            Pieces = currentPieces,
            Built = true,
            BuildTime = 0.0,
            OriginalNNodes = originalNNodes?.ToArray() ?? Array.Empty<int?>(),
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            NestedNNodes = nestedNNodes?.Select(row => row.ToArray()).ToArray(),
            _cachedErrorEstimate = null,
        };
    }

    // ------------------------------------------------------------------
    // Calculus: integration, roots, optimization
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the spline over one or more dimensions.
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all.</param>
    /// <param name="bounds">Sub-interval bounds per dim. Null = full domain.</param>
    /// <returns>Scalar if all dims integrated, otherwise a lower-dimensional spline.</returns>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        if (!Built)
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

        // Full integration: sum piece integrals (with bounds)
        if (sortedDims.Length == NumDimensions)
        {
            double total = 0.0;
            foreach (var multiIdx in NdIndex(Shape))
            {
                int flat = RavelMultiIndex(multiIdx, Shape);
                var piece = Pieces[flat]!;

                var pieceBounds = new (double lo, double hi)?[NumDimensions];
                bool skip = false;

                for (int d = 0; d < NumDimensions; d++)
                {
                    var bd = perDimBounds[dimToIdx[d]];
                    if (bd == null)
                    {
                        pieceBounds[d] = null;
                    }
                    else
                    {
                        double pieceLo = Intervals[d][multiIdx[d]].lo;
                        double pieceHi = Intervals[d][multiIdx[d]].hi;
                        double overlapLo = Math.Max(bd.Value.lo, pieceLo);
                        double overlapHi = Math.Min(bd.Value.hi, pieceHi);
                        if (overlapLo >= overlapHi)
                        {
                            skip = true;
                            break;
                        }
                        if (Math.Abs(overlapLo - pieceLo) < 1e-14 && Math.Abs(overlapHi - pieceHi) < 1e-14)
                            pieceBounds[d] = null;
                        else
                            pieceBounds[d] = (overlapLo, overlapHi);
                    }
                }

                if (skip)
                    continue;

                bool allNull = true;
                foreach (var pb in pieceBounds)
                    if (pb != null) { allNull = false; break; }

                if (allNull)
                {
                    total += (double)piece.Integrate();
                }
                else
                {
                    // Build bounds array for piece.Integrate()
                    var pieceIntBounds = new (double lo, double hi)[NumDimensions];
                    for (int d = 0; d < NumDimensions; d++)
                    {
                        if (pieceBounds[d] == null)
                        {
                            // Full domain for this piece
                            pieceIntBounds[d] = (piece.Domain[d][0], piece.Domain[d][1]);
                        }
                        else
                        {
                            pieceIntBounds[d] = pieceBounds[d]!.Value;
                        }
                    }
                    total += (double)piece.Integrate(bounds: pieceIntBounds);
                }
            }
            return total;
        }

        // Partial integration: process dims in descending order
        var currentPieces = (ChebyshevApproximation?[])Pieces.Clone();
        var currentShape = Shape.ToList();
        var currentKnots = Knots.Select(k => (double[])k.Clone()).ToList();
        var currentIntervals = Intervals.Select(iv => ((double, double)[])iv.Clone()).ToList();
        var currentDomain = Domain.Select(d => (double[])d.Clone()).ToList();
        var currentNNodes = NNodes.ToList();
        var currentNestedNNodes = NestedNNodes?.Select(row => row.ToList()).ToList();
        List<int?>? currentOriginalNNodes = OriginalNNodes.Length == NumDimensions
            ? OriginalNNodes.ToList()
            : null;

        foreach (int d in sortedDims.OrderByDescending(x => x))
        {
            var bd = perDimBounds[dimToIdx[d]];

            // For each position in the remaining shape (excluding dim d),
            // sum the integrated pieces along dim d
            var newShape = currentShape.ToList();
            newShape.RemoveAt(d);

            int newTotal = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(newShape, nameof(Integrate)),
                nameof(Integrate),
                newShape);

            var newPieces = new ChebyshevApproximation?[newTotal];

            if (newShape.Count > 0)
            {
                int newFlat = 0;
                foreach (var newMultiIdx in NdIndex(newShape.ToArray()))
                {
                    var integrated = new List<ChebyshevApproximation>();
                    for (int pieceIdx = 0; pieceIdx < currentShape[d]; pieceIdx++)
                    {
                        // Build old multi index
                        var oldMultiIdx = new int[currentShape.Count];
                        int nd = 0;
                        for (int dd = 0; dd < currentShape.Count; dd++)
                        {
                            if (dd == d)
                                oldMultiIdx[dd] = pieceIdx;
                            else
                                oldMultiIdx[dd] = newMultiIdx[nd++];
                        }

                        int oldFlat = RavelMultiIndex(oldMultiIdx, currentShape.ToArray());
                        var piece = currentPieces[oldFlat]!;

                        if (bd == null)
                        {
                            integrated.Add((ChebyshevApproximation)piece.Integrate(dims: new[] { d }));
                        }
                        else
                        {
                            double pieceLo = currentIntervals[d][pieceIdx].Item1;
                            double pieceHi = currentIntervals[d][pieceIdx].Item2;
                            double overlapLo = Math.Max(bd.Value.lo, pieceLo);
                            double overlapHi = Math.Min(bd.Value.hi, pieceHi);
                            if (overlapLo >= overlapHi)
                                continue;
                            if (Math.Abs(overlapLo - pieceLo) < 1e-14 && Math.Abs(overlapHi - pieceHi) < 1e-14)
                                integrated.Add((ChebyshevApproximation)piece.Integrate(dims: new[] { d }));
                            else
                                integrated.Add((ChebyshevApproximation)piece.Integrate(
                                    dims: new[] { d }, bounds: new[] { (overlapLo, overlapHi) }));
                        }
                    }

                    if (integrated.Count == 0)
                    {
                        // Zero contribution
                        var oldMultiIdx = new int[currentShape.Count];
                        int nd = 0;
                        for (int dd = 0; dd < currentShape.Count; dd++)
                        {
                            if (dd == d) oldMultiIdx[dd] = 0;
                            else oldMultiIdx[dd] = newMultiIdx[nd++];
                        }
                        int oldFlat = RavelMultiIndex(oldMultiIdx, currentShape.ToArray());
                        integrated.Add((ChebyshevApproximation)currentPieces[oldFlat]!.Integrate(dims: new[] { d }) * 0.0);
                    }

                    // Sum integrated pieces
                    var result = integrated[0];
                    for (int i = 1; i < integrated.Count; i++)
                        result = result + integrated[i];

                    newPieces[newFlat] = result;
                    newFlat++;
                }
            }
            else
            {
                // Single resulting piece
                var integrated = new List<ChebyshevApproximation>();
                for (int pieceIdx = 0; pieceIdx < currentShape[d]; pieceIdx++)
                {
                    var piece = currentPieces[pieceIdx]!;
                    if (bd == null)
                    {
                        integrated.Add((ChebyshevApproximation)piece.Integrate(dims: new[] { d }));
                    }
                    else
                    {
                        double pieceLo = currentIntervals[d][pieceIdx].Item1;
                        double pieceHi = currentIntervals[d][pieceIdx].Item2;
                        double overlapLo = Math.Max(bd.Value.lo, pieceLo);
                        double overlapHi = Math.Min(bd.Value.hi, pieceHi);
                        if (overlapLo >= overlapHi)
                            continue;
                        if (Math.Abs(overlapLo - pieceLo) < 1e-14 && Math.Abs(overlapHi - pieceHi) < 1e-14)
                            integrated.Add((ChebyshevApproximation)piece.Integrate(dims: new[] { d }));
                        else
                            integrated.Add((ChebyshevApproximation)piece.Integrate(
                                dims: new[] { d }, bounds: new[] { (overlapLo, overlapHi) }));
                    }
                }

                if (integrated.Count == 0)
                    integrated.Add((ChebyshevApproximation)currentPieces[0]!.Integrate(dims: new[] { d }) * 0.0);

                var result = integrated[0];
                for (int i = 1; i < integrated.Count; i++)
                    result = result + integrated[i];
                newPieces[0] = result;
            }

            currentPieces = newPieces;
            currentShape.RemoveAt(d);
            currentKnots.RemoveAt(d);
            currentIntervals.RemoveAt(d);
            currentDomain.RemoveAt(d);
            currentNNodes.RemoveAt(d);
            currentNestedNNodes?.RemoveAt(d);
            currentOriginalNNodes?.RemoveAt(d);
        }

        // If 0D result, should not happen (handled in full-integration branch)
        if (currentShape.Count == 0)
            return (double)currentPieces[0]!.Integrate();

        return new ChebyshevSpline
        {
            Function = null,
            NumDimensions = NumDimensions - sortedDims.Length,
            Domain = currentDomain.ToArray(),
            NNodes = currentNNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Knots = currentKnots.ToArray(),
            Intervals = currentIntervals.ToArray(),
            Shape = currentShape.ToArray(),
            Pieces = currentPieces,
            Built = true,
            BuildTime = 0.0,
            OriginalNNodes = currentOriginalNNodes?.ToArray() ?? Array.Empty<int?>(),
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            NestedNNodes = currentNestedNNodes?.Select(row => row.ToArray()).ToArray(),
            _cachedErrorEstimate = null,
        };
    }

    /// <summary>
    /// Find all roots of the spline along a specified dimension.
    /// </summary>
    /// <param name="dim">Dimension along which to find roots.</param>
    /// <param name="fixedDims">For multi-D, dict of dim_index -&gt; value for all other dims.</param>
    /// <returns>Sorted array of root locations.</returns>
    public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(
            NumDimensions, dim, fixedDims, Domain);

        ChebyshevSpline sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        // Find roots in each piece
        var allRoots = new List<double>();
        foreach (var piece in sliced.Pieces)
        {
            double[] pieceRoots = Calculus.Roots1D(piece!.TensorValues!, piece.Domain[0]);
            allRoots.AddRange(pieceRoots);
        }

        if (allRoots.Count == 0)
            return Array.Empty<double>();

        allRoots.Sort();

        // Deduplicate near knot boundaries
        if (allRoots.Count > 1)
        {
            double domainScale = Math.Abs(Domain[targetDim][1] - Domain[targetDim][0]) + 1;
            var deduped = new List<double> { allRoots[0] };
            for (int i = 1; i < allRoots.Count; i++)
            {
                if (allRoots[i] - allRoots[i - 1] > 1e-10 * domainScale)
                    deduped.Add(allRoots[i]);
            }
            return deduped.ToArray();
        }

        return allRoots.ToArray();
    }

    /// <summary>
    /// Find the minimum value of the spline along a dimension.
    /// </summary>
    /// <param name="dim">Dimension along which to minimize.</param>
    /// <param name="fixedDims">For multi-D, dict of dim_index -&gt; value for all other dims.</param>
    /// <returns>Tuple of (value, location).</returns>
    public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(
            NumDimensions, dim, fixedDims, Domain);

        ChebyshevSpline sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        double bestVal = double.PositiveInfinity;
        double bestLoc = 0.0;
        foreach (var piece in sliced.Pieces)
        {
            var (val, loc) = Calculus.Optimize1D(
                piece!.TensorValues!, piece.NodeArrays[0], piece.Weights![0],
                piece.DiffMatrices![0], piece.Domain[0], "min");
            if (val < bestVal)
            {
                bestVal = val;
                bestLoc = loc;
            }
        }

        return (bestVal, bestLoc);
    }

    /// <summary>
    /// Find the maximum value of the spline along a dimension.
    /// </summary>
    /// <param name="dim">Dimension along which to maximize.</param>
    /// <param name="fixedDims">For multi-D, dict of dim_index -&gt; value for all other dims.</param>
    /// <returns>Tuple of (value, location).</returns>
    public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (targetDim, sliceParams) = Calculus.ValidateCalculusArgs(
            NumDimensions, dim, fixedDims, Domain);

        ChebyshevSpline sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;

        double bestVal = double.NegativeInfinity;
        double bestLoc = 0.0;
        foreach (var piece in sliced.Pieces)
        {
            var (val, loc) = Calculus.Optimize1D(
                piece!.TensorValues!, piece.NodeArrays[0], piece.Weights![0],
                piece.DiffMatrices![0], piece.Domain[0], "max");
            if (val > bestVal)
            {
                bestVal = val;
                bestLoc = loc;
            }
        }

        return (bestVal, bestLoc);
    }

    // ------------------------------------------------------------------
    // Arithmetic operators
    // ------------------------------------------------------------------

    internal void CheckSplineCompatible(ChebyshevSpline other)
    {
        // Check base compatibility for the first piece as proxy
        if (NumDimensions != other.NumDimensions)
            throw new ArgumentException($"Dimension mismatch: {NumDimensions} vs {other.NumDimensions}");
        if (!NNodes.SequenceEqual(other.NNodes))
            throw new ArgumentException(
                $"Node count mismatch: [{string.Join(", ", NNodes)}] vs [{string.Join(", ", other.NNodes)}]");
        for (int d = 0; d < NumDimensions; d++)
        {
            if (!Domain[d].SequenceEqual(other.Domain[d]))
                throw new ArgumentException($"Domain mismatch at dim {d}");
        }
        if (MaxDerivativeOrder != other.MaxDerivativeOrder)
            throw new ArgumentException(
                $"max_derivative_order mismatch: {MaxDerivativeOrder} vs {other.MaxDerivativeOrder}");

        if (!Built) throw new InvalidOperationException("Left operand is not built.");
        if (!other.Built) throw new InvalidOperationException("Right operand is not built.");

        // Check knot compatibility
        if (Knots.Length != other.Knots.Length)
            throw new ArgumentException("Knot dimension count mismatch");
        for (int d = 0; d < Knots.Length; d++)
        {
            if (!Knots[d].SequenceEqual(other.Knots[d]))
                throw new ArgumentException($"Knot mismatch at dimension {d}");
        }

        if (Pieces.Length != other.Pieces.Length)
            throw new ArgumentException(
                $"Piece count mismatch: {Pieces.Length} vs {other.Pieces.Length}");
        for (int i = 0; i < Pieces.Length; i++)
        {
            var leftNodes = Pieces[i]!.NNodes;
            var rightNodes = other.Pieces[i]!.NNodes;
            if (!leftNodes.SequenceEqual(rightNodes))
                throw new ArgumentException(
                    $"Piece {i} node count mismatch: " +
                    $"[{string.Join(", ", leftNodes)}] vs [{string.Join(", ", rightNodes)}]");
        }
    }

    /// <summary>Add two splines with the same grid and knots.</summary>
    public static ChebyshevSpline operator +(ChebyshevSpline a, ChebyshevSpline b)
    {
        a.CheckSplineCompatible(b);
        var pieces = new ChebyshevApproximation?[a.Pieces.Length];
        for (int i = 0; i < pieces.Length; i++)
        {
            double[] newValues = new double[a.Pieces[i]!.TensorValues!.Length];
            for (int j = 0; j < newValues.Length; j++)
                newValues[j] = a.Pieces[i]!.TensorValues![j] + b.Pieces[i]!.TensorValues![j];
            pieces[i] = ChebyshevApproximation.FromGrid(a.Pieces[i]!, newValues);
        }
        return FromPieces(a, pieces);
    }

    /// <summary>Subtract two splines with the same grid and knots.</summary>
    public static ChebyshevSpline operator -(ChebyshevSpline a, ChebyshevSpline b)
    {
        a.CheckSplineCompatible(b);
        var pieces = new ChebyshevApproximation?[a.Pieces.Length];
        for (int i = 0; i < pieces.Length; i++)
        {
            double[] newValues = new double[a.Pieces[i]!.TensorValues!.Length];
            for (int j = 0; j < newValues.Length; j++)
                newValues[j] = a.Pieces[i]!.TensorValues![j] - b.Pieces[i]!.TensorValues![j];
            pieces[i] = ChebyshevApproximation.FromGrid(a.Pieces[i]!, newValues);
        }
        return FromPieces(a, pieces);
    }

    /// <summary>Multiply spline by a scalar.</summary>
    public static ChebyshevSpline operator *(ChebyshevSpline a, double scalar)
    {
        if (!a.Built)
            throw new InvalidOperationException("Operand is not built. Call Build() first.");

        var pieces = new ChebyshevApproximation?[a.Pieces.Length];
        for (int i = 0; i < pieces.Length; i++)
        {
            double[] newValues = new double[a.Pieces[i]!.TensorValues!.Length];
            for (int j = 0; j < newValues.Length; j++)
                newValues[j] = a.Pieces[i]!.TensorValues![j] * scalar;
            pieces[i] = ChebyshevApproximation.FromGrid(a.Pieces[i]!, newValues);
        }
        return FromPieces(a, pieces);
    }

    /// <summary>Multiply scalar by spline.</summary>
    public static ChebyshevSpline operator *(double scalar, ChebyshevSpline a)
    {
        return a * scalar;
    }

    /// <summary>Divide spline by a scalar.</summary>
    public static ChebyshevSpline operator /(ChebyshevSpline a, double scalar)
    {
        return a * (1.0 / scalar);
    }

    /// <summary>Negate spline.</summary>
    public static ChebyshevSpline operator -(ChebyshevSpline a)
    {
        return a * -1.0;
    }

    // ------------------------------------------------------------------
    // Printing
    // ------------------------------------------------------------------

    /// <summary>Compact string representation.</summary>
    public string ToReprString()
    {
        return $"ChebyshevSpline(" +
            $"dims={NumDimensions}, " +
            $"pieces={NumPieces}, " +
            $"shape=({string.Join(", ", Shape)}), " +
            $"built={Built})";
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        string status = Built ? "built" : "not built";
        int totalEvals = TotalBuildEvals;
        int maxDisplay = 6;

        string nodesStr;
        if (NumDimensions > maxDisplay)
            nodesStr = "[" + string.Join(", ", NNodes.Take(maxDisplay)) + ", ...]";
        else
            nodesStr = "[" + string.Join(", ", NNodes) + "]";

        string knotsStr;
        if (NumDimensions > maxDisplay)
            knotsStr = "[" + string.Join(", ", Knots.Take(maxDisplay).Select(k => $"[{string.Join(", ", k)}]")) + ", ...]";
        else
            knotsStr = "[" + string.Join(", ", Knots.Select(k => $"[{string.Join(", ", k)}]")) + "]";

        string shapeStr = string.Join(" x ", Shape);

        string domainStr;
        if (NumDimensions > maxDisplay)
            domainStr = string.Join(" x ", Domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        else
            domainStr = string.Join(" x ", Domain.Select(d => $"[{d[0]}, {d[1]}]"));

        var sb = new StringBuilder();
        sb.AppendLine($"ChebyshevSpline ({NumDimensions}D, {status})");
        sb.AppendLine($"  Nodes:       {nodesStr} per piece");
        sb.AppendLine($"  Knots:       {knotsStr}");
        sb.AppendLine($"  Pieces:      {NumPieces} ({shapeStr})");

        if (Built)
            sb.AppendLine($"  Build:       {BuildTime:F3}s ({totalEvals:N0} function evals)");

        sb.AppendLine($"  Domain:      {domainStr}");

        if (Built)
            sb.Append($"  Error est:   {ErrorEstimate():E2}");
        else
            sb.Length -= Environment.NewLine.Length; // Remove trailing newline from last AppendLine

        return sb.ToString();
    }

    // ------------------------------------------------------------------
    // Utility: N-dimensional index iteration (C-order)
    // ------------------------------------------------------------------

    /// <summary>
    /// Iterate over all multi-indices for the given shape in C-order.
    /// Equivalent to Python's itertools.product(*[range(s) for s in shape]) or np.ndindex(*shape).
    /// </summary>
    internal static IEnumerable<int[]> NdIndex(int[] shape)
    {
        if (shape.Length == 0)
        {
            yield return Array.Empty<int>();
            yield break;
        }

        int total = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(shape, nameof(NdIndex)),
            nameof(NdIndex),
            shape);

        for (int flat = 0; flat < total; flat++)
        {
            int[] idx = new int[shape.Length];
            int rem = flat;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            yield return idx;
        }
    }

    /// <summary>
    /// Convert a multi-index to flat index (C-order/row-major).
    /// Equivalent to np.ravel_multi_index(multi_idx, shape).
    /// </summary>
    internal static int RavelMultiIndex(int[] multiIdx, int[] shape)
    {
        _ = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(shape, nameof(RavelMultiIndex)),
            nameof(RavelMultiIndex),
            shape);
        int flat = 0;
        int stride = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            flat += multiIdx[d] * stride;
            stride *= shape[d];
        }
        return flat;
    }

    // ------------------------------------------------------------------
    // Serialization types
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Phase 4 ergonomics — accessors
    // ------------------------------------------------------------------

    /// <summary>Set a free-form descriptor string for this spline.</summary>
    public void SetDescriptor(string descriptor) => _descriptor = descriptor;

    /// <summary>Get the descriptor previously set via <see cref="SetDescriptor"/>; null if unset.</summary>
    public string? GetDescriptor() => _descriptor;

    /// <summary>True if <see cref="Build"/>/<see cref="FromValues"/>/<see cref="Load"/> completed.</summary>
    public bool IsConstructionFinished() => Built;

    /// <summary>Returns one of: "function" (Build), "from_values" (FromValues factory), "load" (Load).</summary>
    public string GetConstructorType() => _constructorType;

    /// <summary>Per-dimension Chebyshev node counts actually used per piece.</summary>
    public int[] GetUsedNs() => (int[])NNodes.Clone();

    /// <summary>Maximum derivative order this spline supports.</summary>
    public int GetMaxDerivativeOrder() => MaxDerivativeOrder;

    /// <summary>
    /// Returns the user-supplied <c>additionalData</c> object passed to the constructor,
    /// or null if none was provided. Same value is threaded through every <c>f(point, data)</c>
    /// call during <see cref="Build"/>.
    /// </summary>
    public object? GetAdditionalData() => _additionalData;

    /// <summary>
    /// Total number of evaluation points across all spline pieces.
    /// </summary>
    /// <returns>The sum of GetNumEvaluationPoints() from each piece.</returns>
    public int GetNumEvaluationPoints()
    {
        long total = 0;
        if (Pieces == null) return 0;
        foreach (var piece in Pieces)
        {
            if (piece != null) total = checked(total + piece.GetNumEvaluationPoints());
        }
        return TensorShape.RequireArrayLength(total, nameof(GetNumEvaluationPoints));
    }

    /// <summary>
    /// Flat row-major array of all spline piece evaluation points, concatenated sequentially.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached.
    /// </summary>
    /// <returns>Double array of concatenated piece node coordinates, flattened in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return _evaluationPointsCache;

        int total = GetNumEvaluationPoints();
        int coordinateCount = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { total, NumDimensions }, nameof(GetEvaluationPoints)),
            nameof(GetEvaluationPoints),
            new[] { total, NumDimensions });
        var points = new double[coordinateCount];
        int offset = 0;

        foreach (var piece in Pieces!)
        {
            if (piece == null) continue;
            var piecePts = piece.GetEvaluationPoints();
            Array.Copy(piecePts, 0, points, offset, piecePts.Length);
            offset += piecePts.Length;
        }

        _evaluationPointsCache = points;
        return points;
    }

    /// <summary>
    /// Get the knots (special points) used for spline construction.
    /// </summary>
    /// <returns>Interior knots per dimension, or null if no interior knots were used.</returns>
    public double[][]? GetSpecialPoints()
    {
        if (Knots == null) return null;
        bool anyInterior = false;
        foreach (var k in Knots)
            if (k.Length > 0) { anyInterior = true; break; }
        return anyInterior ? Knots.Select(k => (double[])k.Clone()).ToArray() : null;
    }

    /// <summary>
    /// Compute variance-based sensitivity indices aggregated across spline pieces.
    /// Per-piece coefficients are computed under the Chebyshev measure on each piece's
    /// local domain; per-piece contributions are weighted by domain volume × variance,
    /// then normalized by global variance. For a single-piece spline, this reduces to
    /// the <see cref="ChebyshevApproximation.SobolIndices"/> case.
    /// </summary>
    /// <returns>A <see cref="SobolResult"/> with per-dim FirstOrder, TotalOrder, and Chebyshev-weighted global Variance.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    public SobolResult SobolIndices()
    {
        if (Pieces == null || Pieces.Length == 0 || Pieces.Any(p => p == null || p.TensorValues == null))
            throw new InvalidOperationException(
                "SobolIndices requires a built ChebyshevSpline. Call Build() first.");

        int nDim = NumDimensions;
        var globalFirstOrder = new double[nDim];
        var globalTotalOrder = new double[nDim];
        double globalVariance = 0.0;

        foreach (var piece in Pieces)
        {
            if (piece == null) continue;
            double vol = 1.0;
            for (int d = 0; d < nDim; d++)
            {
                double lo = piece.Domain[d][0], hi = piece.Domain[d][1];
                vol *= (hi - lo);
            }
            var coeffs = Internal.Sensitivity.ChebyshevCoefficientsND(piece.TensorValues!, piece.NNodes);
            var pieceResult = Internal.Sensitivity.ComputeSobolFromCoeffs(coeffs, piece.NNodes);
            globalVariance += vol * pieceResult.Variance;
            for (int d = 0; d < nDim; d++)
            {
                globalFirstOrder[d] += vol * pieceResult.FirstOrder[d] * pieceResult.Variance;
                globalTotalOrder[d] += vol * pieceResult.TotalOrder[d] * pieceResult.Variance;
            }
        }

        if (globalVariance == 0)
            return new SobolResult(new double[nDim], new double[nDim], 0);
        for (int d = 0; d < nDim; d++)
        {
            globalFirstOrder[d] /= globalVariance;
            globalTotalOrder[d] /= globalVariance;
        }
        return new SobolResult(globalFirstOrder, globalTotalOrder, globalVariance);
    }

    /// <summary>
    /// Populate this spline's tensor values from a precomputed flat array.
    /// Used after constructing with <c>deferBuild: true</c>. Bit-identical to
    /// the <c>FromValues</c> factory.
    /// Values are concatenated in flat piece-index C-order: piece 0 values first, then piece 1, etc.
    /// </summary>
    /// <param name="values">Flat array concatenating all pieces' values in piece-flat-index order.</param>
    /// <exception cref="ArgumentException">Thrown when values length does not match the expected total across all pieces.</exception>
    public void SetOriginalFunctionValues(double[] values)
    {
        int totalPieces = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(Shape, nameof(SetOriginalFunctionValues)),
            nameof(SetOriginalFunctionValues),
            Shape);

        var pieceSizes = new int[totalPieces];
        long totalExpected = 0;
        for (int p = 0; p < totalPieces; p++)
        {
            var (_, pieceNNodes) = ComputePieceDomainAndN(p);
            int n = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(pieceNNodes, nameof(SetOriginalFunctionValues)),
                nameof(SetOriginalFunctionValues),
                pieceNNodes);
            pieceSizes[p] = n;
            totalExpected = checked(totalExpected + n);
        }
        int requiredValues = TensorShape.RequireArrayLength(totalExpected, nameof(SetOriginalFunctionValues));

        if (values.Length != requiredValues)
            throw new ArgumentException(
                $"values has {values.Length} entries, expected {requiredValues} across all pieces");

        Pieces = new ChebyshevApproximation?[totalPieces];
        int offset = 0;
        for (int p = 0; p < totalPieces; p++)
        {
            int sz = pieceSizes[p];
            var pieceValues = new double[sz];
            Array.Copy(values, offset, pieceValues, 0, sz);
            offset += sz;
            var (pieceDomain, pieceNNodes) = ComputePieceDomainAndN(p);
            Pieces[p] = ChebyshevApproximation.FromValues(
                pieceValues,
                NumDimensions,
                pieceDomain,
                pieceNNodes,
                MaxDerivativeOrder);
        }

        Built = true;
        _evaluationPointsCache = null;
        _constructorType = "from_values";
    }

    private (double[][] pieceDomain, int[] pieceNNodes) ComputePieceDomainAndN(int flatPieceIdx)
    {
        // Decompose flat piece index into per-dim piece coords (C-order / row-major).
        var pieceCoords = new int[NumDimensions];
        int rem = flatPieceIdx;
        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            pieceCoords[d] = rem % Shape[d];
            rem /= Shape[d];
        }

        var pieceDomain = new double[NumDimensions][];
        var pieceNNodes = new int[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            var iv = Intervals[d][pieceCoords[d]];
            pieceDomain[d] = new[] { iv.lo, iv.hi };
            pieceNNodes[d] = NestedNNodes != null ? NestedNNodes[d][pieceCoords[d]] : NNodes[d];
        }
        return (pieceDomain, pieceNNodes);
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
        EvaluationArguments.ValidateDerivativeOrder(orders, NumDimensions, nameof(orders));
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
    /// Returns a deep copy of this spline. The source <see cref="Function"/>
    /// callable is NOT duplicated — clones cannot be rebuilt without re-supplying
    /// the function. All precomputed pieces and state are deep-copied.
    /// </summary>
    /// <returns>A fully independent <see cref="ChebyshevSpline"/> with <see cref="Function"/> set to null.</returns>
    public ChebyshevSpline Clone()
    {
        var copy = new ChebyshevSpline();
        copy.NumDimensions = NumDimensions;
        copy.Domain = Internal.CloneHelpers.DeepCopy(Domain)!;
        copy.NNodes = Internal.CloneHelpers.DeepCopy(NNodes)!;
        copy.Knots = Internal.CloneHelpers.DeepCopy(Knots)!;
        copy.Intervals = Internal.CloneHelpers.DeepCopyIntervals(Intervals)!;
        copy.Shape = Internal.CloneHelpers.DeepCopy(Shape)!;
        copy.MaxDerivativeOrder = MaxDerivativeOrder;
        copy.MaxN = MaxN;
        copy.ErrorThreshold = ErrorThreshold;
        copy.OriginalNNodes = Internal.CloneHelpers.DeepCopy(OriginalNNodes)!;
        copy.NestedNNodes = Internal.CloneHelpers.DeepCopy(NestedNNodes);
        copy.Built = Built;
        copy.BuildTime = BuildTime;
        copy._descriptor = _descriptor;
        copy._additionalData = _additionalData;
        copy._constructorType = "clone";
        copy._evaluationPointsCache = null;
        if (Pieces != null)
        {
            copy.Pieces = new ChebyshevApproximation?[Pieces.Length];
            for (int i = 0; i < Pieces.Length; i++)
                copy.Pieces[i] = Pieces[i]?.Clone();
        }
        foreach (var kv in _derivativeIdRegistry)
            copy._derivativeIdRegistry[kv.Key] = kv.Value;
        foreach (var orders in _registeredDerivativeOrders)
            copy._registeredDerivativeOrders.Add((int[])orders.Clone());
        return copy;
    }

    // ------------------------------------------------------------------
    // AutoKnots — curvature-spike knot detection (Phase 6 Task 8)
    // ------------------------------------------------------------------

    /// <summary>
    /// Auto-place knots at function kinks via a curvature-spike scan, then build the
    /// resulting <see cref="ChebyshevSpline"/>. Mirrors PyChebyshev <c>spline.py:2111</c>.
    /// </summary>
    /// <param name="function">f(point, additionalData) → double; must return finite at every scan point.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="numNodes">Per-piece node counts; same shape as the regular ctor.</param>
    /// <param name="maxOrderDerivative">Max derivative order. Default 2.</param>
    /// <param name="additionalData">Optional user data threaded through f calls.</param>
    /// <param name="descriptor">Optional free-form descriptor.</param>
    /// <param name="thresholdFactor">Spike threshold = thresholdFactor × mean(|d²f|). Default 5.0.</param>
    /// <param name="maxKnotsPerDim">Cap on knots per dimension. Default 5. Zero means no auto-knots.</param>
    /// <param name="nScanPoints">Number of scan points per dim. Default 200; must be at least 3.</param>
    /// <param name="nWorkers">See <see cref="ChebyshevSpline"/> ctor.</param>
    /// <param name="progress">See <see cref="ChebyshevSpline"/> ctor.</param>
    /// <param name="verbose">If true, print scan progress.</param>
    /// <returns>A built ChebyshevSpline with the discovered knots.</returns>
    /// <remarks>
    /// When <paramref name="nWorkers"/> is non-null, <paramref name="function"/> may be
    /// invoked concurrently from multiple threads. Functions that capture mutable state
    /// must use locks or external synchronization, or pass <c>nWorkers: null</c>.
    /// </remarks>
    public static ChebyshevSpline AutoKnots(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] numNodes,
        int maxOrderDerivative = 2,
        object? additionalData = null,
        string? descriptor = null,
        double thresholdFactor = 5.0,
        int maxKnotsPerDim = 5,
        int nScanPoints = 200,
        int? nWorkers = null,
        IProgress<int>? progress = null,
        bool verbose = false)
    {
        if (thresholdFactor <= 0)
            throw new ArgumentException("thresholdFactor must be > 0", nameof(thresholdFactor));
        if (maxKnotsPerDim < 0)
            throw new ArgumentException("maxKnotsPerDim must be >= 0", nameof(maxKnotsPerDim));
        if (nScanPoints < 3)
            throw new ArgumentException("nScanPoints must be at least 3 to compute a 2nd-derivative finite difference", nameof(nScanPoints));

        int? effectiveWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);

        // Scan each dim for curvature spikes; build per-dim knot arrays.
        var allKnots = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
        {
            if (maxKnotsPerDim == 0)
            {
                allKnots[d] = Array.Empty<double>();
                continue;
            }
            allKnots[d] = ScanForKnotsAlongDim(
                function, d, numDimensions, domain, additionalData,
                thresholdFactor, maxKnotsPerDim, nScanPoints,
                effectiveWorkers, progress);
        }

        // Construct the resulting spline.
        var sp = new ChebyshevSpline(function, numDimensions, domain, numNodes, allKnots,
            maxDerivativeOrder: maxOrderDerivative,
            additionalData: additionalData,
            nWorkers: nWorkers,
            progress: progress);
        sp.SetDescriptor(descriptor ?? string.Empty);
        sp.Build(verbose: verbose);
        return sp;
    }

    /// <summary>
    /// Scan one dimension for second-derivative spikes; cluster spikes; cap to
    /// maxKnotsPerDim. Returns the knot positions in that dimension's domain.
    /// </summary>
    private static double[] ScanForKnotsAlongDim(
        Func<double[], object?, double> function,
        int dim,
        int numDimensions,
        double[][] domain,
        object? additionalData,
        double thresholdFactor,
        int maxKnotsPerDim,
        int nScanPoints,
        int? effectiveWorkers,
        IProgress<int>? progress)
    {
        double lo = domain[dim][0], hi = domain[dim][1];
        // Build sample points: this dim varies, others fixed at midpoint.
        var samplePoints = new double[nScanPoints][];
        double dx = (hi - lo) / (nScanPoints - 1);
        for (int i = 0; i < nScanPoints; i++)
        {
            var pt = new double[numDimensions];
            for (int k = 0; k < numDimensions; k++)
                pt[k] = (k == dim) ? (lo + i * dx) : 0.5 * (domain[k][0] + domain[k][1]);
            samplePoints[i] = pt;
        }

        // Evaluate (parallelized if requested).
        double[] ys = Internal.ParallelBuild.EvaluateInParallel(
            function, samplePoints, additionalData, effectiveWorkers, progress);

        // Reject non-finite values.
        for (int i = 0; i < nScanPoints; i++)
            if (!double.IsFinite(ys[i]))
                throw new ArgumentException(
                    $"AutoKnots requires a finite-valued function over the entire domain " +
                    $"(non-finite at scan point {i} of dim {dim})");

        // 2nd-derivative finite difference; pad boundaries with 0.
        var d2 = new double[nScanPoints];
        double h2 = dx * dx;
        for (int i = 1; i < nScanPoints - 1; i++)
            d2[i] = (ys[i + 1] - 2.0 * ys[i] + ys[i - 1]) / h2;

        // Compute mean(|d2|) over interior.
        double sumAbs = 0;
        int interiorCount = 0;
        for (int i = 1; i < nScanPoints - 1; i++)
        {
            sumAbs += Math.Abs(d2[i]);
            interiorCount++;
        }
        double meanD2 = interiorCount > 0 ? sumAbs / interiorCount : 0.0;
        if (meanD2 == 0) return Array.Empty<double>();
        double threshold = thresholdFactor * meanD2;

        // Identify spike indices.
        var spikes = new List<int>();
        for (int i = 1; i < nScanPoints - 1; i++)
            if (Math.Abs(d2[i]) > threshold) spikes.Add(i);
        if (spikes.Count == 0) return Array.Empty<double>();

        // Cluster spikes within radius = max(1, nScanPoints / (maxKnotsPerDim * 4)).
        int clusterRadius = Math.Max(1, nScanPoints / Math.Max(1, maxKnotsPerDim * 4));
        var clusterPeaks = new List<int>();
        int j = 0;
        while (j < spikes.Count)
        {
            int peak = spikes[j];
            double peakAbs = Math.Abs(d2[peak]);
            int k = j + 1;
            while (k < spikes.Count && spikes[k] - peak <= clusterRadius)
            {
                if (Math.Abs(d2[spikes[k]]) > peakAbs)
                {
                    peak = spikes[k];
                    peakAbs = Math.Abs(d2[peak]);
                }
                k++;
            }
            clusterPeaks.Add(peak);
            j = k;
        }

        // Sort by |d²| desc; cap at maxKnotsPerDim.
        clusterPeaks.Sort((a, b) => Math.Abs(d2[b]).CompareTo(Math.Abs(d2[a])));
        if (clusterPeaks.Count > maxKnotsPerDim)
            clusterPeaks.RemoveRange(maxKnotsPerDim, clusterPeaks.Count - maxKnotsPerDim);

        // Sort by position ascending and convert to domain coordinates.
        clusterPeaks.Sort();
        var knots = clusterPeaks.Select(idx => lo + idx * dx).ToArray();
        return knots;
    }

    // ------------------------------------------------------------------
    // Serialization state
    // ------------------------------------------------------------------

    internal class SplineSerializationState
    {
        public string Type { get; set; } = "ChebyshevSpline";
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int? MaxDerivativeOrder { get; set; }
        public double[][] Knots { get; set; } = Array.Empty<double[]>();
        public int[] Shape { get; set; } = Array.Empty<int>();
        public double BuildTime { get; set; }
        public PieceState[] PieceStates { get; set; } = Array.Empty<PieceState>();
        public int?[]? OriginalNNodes { get; set; }
        public double? ErrorThreshold { get; set; }
        public int? MaxN { get; set; }
        public int[][]? NestedNNodes { get; set; }
        public string Version { get; set; } = "0.1.0";
        // v0.8.0 ergonomics fields (absent in pre-v0.8.0 JSON; null == not set)
        public string? Descriptor { get; set; }
        public string? ConstructorType { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
    }

    internal class PieceState
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
    }
}

/// <summary>Node information for a single piece of a spline.</summary>
public class SplinePieceNodeInfo
{
    /// <summary>Multi-index of this piece.</summary>
    public int[] PieceIndex { get; set; } = Array.Empty<int>();
    /// <summary>Sub-domain bounds for this piece.</summary>
    public double[][] SubDomain { get; set; } = Array.Empty<double[]>();
    /// <summary>Per-dimension node arrays.</summary>
    public double[][] NodesPerDim { get; set; } = Array.Empty<double[]>();
    /// <summary>Full Cartesian product grid.</summary>
    public double[][] FullGrid { get; set; } = Array.Empty<double[]>();
    /// <summary>Tensor shape.</summary>
    public int[] Shape { get; set; } = Array.Empty<int>();
}

/// <summary>Node information for all pieces of a spline.</summary>
public class SplineNodeInfo
{
    /// <summary>Per-piece node info.</summary>
    public SplinePieceNodeInfo[] Pieces { get; set; } = Array.Empty<SplinePieceNodeInfo>();
    /// <summary>Total number of pieces.</summary>
    public int NumPieces { get; set; }
    /// <summary>Per-dimension piece counts.</summary>
    public int[] PieceShape { get; set; } = Array.Empty<int>();
}
