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

    private double? _cachedErrorEstimate;

    /// <summary>
    /// Create a new ChebyshevSpline.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -&gt; double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension per piece.</param>
    /// <param name="knots">Interior knots for each dimension. Empty array for no knots.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        double[][] knots,
        int maxDerivativeOrder = 2)
    {
        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        NNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;

        // Validate and store knots
        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        // Compute per-dimension intervals
        Intervals = ComputeIntervals(numDimensions, domain, knots);

        // Shape: per-dimension piece counts
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        // Allocate flat piece storage
        int totalPieces = 1;
        foreach (int s in Shape) totalPieces *= s;
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }

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
        int perPieceEvals = 1;
        for (int d = 0; d < NumDimensions; d++)
            perPieceEvals *= NNodes[d];

        if (verbose)
            Console.WriteLine(
                $"Building {NumDimensions}D Chebyshev Spline " +
                $"({totalPieces} pieces, {totalPieces * perPieceEvals:N0} total evaluations)...");

        int flatIdx = 0;
        foreach (var multiIdx in NdIndex(Shape))
        {
            // Compute sub-domain for this piece
            double[][] subDomain = new double[NumDimensions][];
            for (int d = 0; d < NumDimensions; d++)
            {
                var iv = Intervals[d][multiIdx[d]];
                subDomain[d] = new[] { iv.lo, iv.hi };
            }

            var piece = new ChebyshevApproximation(
                Function, NumDimensions, subDomain, NNodes,
                maxDerivativeOrder: MaxDerivativeOrder);
            piece.Build(verbose: false);
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
            if (derivativeOrder[d] > 0)
            {
                for (int k = 0; k < Knots[d].Length; k++)
                {
                    if (Math.Abs(point[d] - Knots[d][k]) < 1e-14)
                        throw new ArgumentException(
                            $"Derivative w.r.t. dimension {d} is not defined " +
                            $"at knot x[{d}]={Knots[d][k]}. The left and right " +
                            $"derivatives may differ at this point.");
                }
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

        int N = points.Length;
        double[] results = new double[N];

        // Compute piece index for each point
        int[] flatIndices = new int[N];
        for (int i = 0; i < N; i++)
        {
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
            int total = 1;
            foreach (int s in Shape) total *= s;
            return total;
        }
    }

    /// <summary>Total number of function evaluations used during build.</summary>
    public int TotalBuildEvals
    {
        get
        {
            int perPiece = 1;
            for (int d = 0; d < NumDimensions; d++)
                perPiece *= NNodes[d];
            return NumPieces * perPiece;
        }
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// <summary>
    /// Save the built spline to a file using JSON serialization.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    public void Save(string path)
    {
        if (!Built)
            throw new InvalidOperationException(
                "Cannot save an unbuilt spline. Call Build() first.");

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
            Version = "0.1.0",
        };

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load a previously saved spline from a file.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>The restored spline.</returns>
    public static ChebyshevSpline Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SplineSerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize");

        if (state.Type != "ChebyshevSpline")
            throw new InvalidOperationException(
                $"Expected type ChebyshevSpline, got {state.Type}");

        var pieces = state.PieceStates.Select(ps =>
        {
            var piece = new ChebyshevApproximation
            {
                Function = null,
                NumDimensions = ps.NumDimensions,
                Domain = ps.Domain,
                NNodes = ps.NNodes,
                MaxDerivativeOrder = ps.MaxDerivativeOrder,
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

        return new ChebyshevSpline
        {
            Function = null,
            NumDimensions = state.NumDimensions,
            Domain = state.Domain,
            NNodes = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder,
            Knots = state.Knots,
            Intervals = intervals,
            Shape = state.Shape,
            Pieces = pieces.Cast<ChebyshevApproximation?>().ToArray(),
            Built = true,
            BuildTime = state.BuildTime,
            _cachedErrorEstimate = null,
        };
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

        int totalPieces = 1;
        foreach (int s in shape) totalPieces *= s;

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

        int totalPieces = 1;
        foreach (int s in shape) totalPieces *= s;

        if (pieceValues.Length != totalPieces)
            throw new ArgumentException(
                $"Expected {totalPieces} piece_values, got {pieceValues.Length}");

        // Validate per-piece shapes
        int expectedSize = 1;
        for (int d = 0; d < numDimensions; d++)
            expectedSize *= nNodes[d];

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

        return new ChebyshevSpline
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
            Intervals = source.Intervals,
            Shape = source.Shape,
            Pieces = pieces,
            Built = true,
            BuildTime = 0.0,
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

        foreach (var (dimIdx, bounds, n) in sorted)
        {
            knots.Insert(dimIdx, Array.Empty<double>());
            intervals.Insert(dimIdx, new[] { (bounds[0], bounds[1]) });
            shape.Insert(dimIdx, 1);
            domain.Insert(dimIdx, (double[])bounds.Clone());
            nNodes.Insert(dimIdx, n);
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

            int newTotal = 1;
            foreach (int s in newShape) newTotal *= s;
            if (newTotal == 0) newTotal = 1;
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

        foreach (int d in sortedDims.OrderByDescending(x => x))
        {
            var bd = perDimBounds[dimToIdx[d]];

            // For each position in the remaining shape (excluding dim d),
            // sum the integrated pieces along dim d
            var newShape = currentShape.ToList();
            newShape.RemoveAt(d);

            int newTotal = 1;
            foreach (int s in newShape) newTotal *= s;
            if (newTotal == 0) newTotal = 1;

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

        int total = 1;
        foreach (int s in shape) total *= s;

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

    internal class SplineSerializationState
    {
        public string Type { get; set; } = "ChebyshevSpline";
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int MaxDerivativeOrder { get; set; }
        public double[][] Knots { get; set; } = Array.Empty<double[]>();
        public int[] Shape { get; set; } = Array.Empty<int>();
        public double BuildTime { get; set; }
        public PieceState[] PieceStates { get; set; } = Array.Empty<PieceState>();
        public string Version { get; set; } = "0.1.0";
    }

    internal class PieceState
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
