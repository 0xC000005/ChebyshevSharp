using System.Diagnostics;
using System.Text;
using System.Text.Json;
using ChebyshevSharp.Internal;

namespace ChebyshevSharp;

/// <summary>
/// Chebyshev Sliding approximation for high-dimensional functions.
/// Decomposes f(x₁, …, xₙ) into a sum of low-dimensional Chebyshev
/// interpolants (slides) around a pivot point z:
///     f(x) ≈ f(z) + Σᵢ [sᵢ(x_groupᵢ) - f(z)]
/// where each slide sᵢ is a <see cref="ChebyshevApproximation"/> built on a
/// subset of dimensions with the remaining dimensions fixed at z.
/// </summary>
/// <remarks>
/// This trades accuracy for dramatically reduced build cost: instead of
/// evaluating f at n₁ × n₂ × … × nₐ grid points (exponential), the slider
/// evaluates at n₁ × n₂ + n₃ × n₄ + … (sum of products within each group).
/// Reference: Ruiz &amp; Zeron (2022), Ch. 7.
/// </remarks>
public class ChebyshevSlider
{
    private double[][] _domain = Array.Empty<double[]>();
    private int[] _nNodes = Array.Empty<int>();
    private int[][] _partition = Array.Empty<int[]>();
    private double[] _pivotPoint = Array.Empty<double>();

    /// <summary>The function to approximate. Null after load.</summary>
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

    /// <summary>Grouping of dimension indices into slides.</summary>
    public int[][] Partition
    {
        get => CloneHelpers.DeepCopy(_partition)!;
        internal set => _partition = value ?? Array.Empty<int[]>();
    }

    /// <summary>Reference point z around which slides are built.</summary>
    public double[] PivotPoint
    {
        get => CloneHelpers.DeepCopy(_pivotPoint)!;
        internal set => _pivotPoint = value ?? Array.Empty<double>();
    }

    internal double[][] DomainStorage
    {
        get => _domain;
        set => _domain = value ?? Array.Empty<double[]>();
    }

    internal int[] NNodesStorage
    {
        get => _nNodes;
        set => _nNodes = value ?? Array.Empty<int>();
    }

    internal int[][] PartitionStorage
    {
        get => _partition;
        set => _partition = value ?? Array.Empty<int[]>();
    }

    internal double[] PivotPointStorage
    {
        get => _pivotPoint;
        set => _pivotPoint = value ?? Array.Empty<double>();
    }

    /// <summary>Function value at the pivot point: f(z).</summary>
    public double PivotValue { get; internal set; }

    /// <summary>One ChebyshevApproximation per partition group.</summary>
    internal ChebyshevApproximation[] Slides { get; set; } = Array.Empty<ChebyshevApproximation>();

    /// <summary>Maps dimension index → slide index.</summary>
    internal Dictionary<int, int> DimToSlide { get; set; } = new();

    /// <summary>Whether Build() has been called.</summary>
    public bool Built { get; internal set; }

    /// <summary>Wall-clock time (seconds) for the most recent Build() call.</summary>
    public double BuildTime { get; internal set; }

    private double? _cachedErrorEstimate;
    private string? _descriptor;
    private string _constructorType = "function";
    private bool _isConstructionFinished;
    private object? _additionalData;
    private double[]? _evaluationPointsCache;
    private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
    private readonly List<int[]> _registeredDerivativeOrders = new();
    private int? _nWorkers;
    private IProgress<int>? _progress;

    /// <summary>
    /// Create a new ChebyshevSlider.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) → double.</param>
    /// <param name="numDimensions">Total number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension.</param>
    /// <param name="partition">Grouping of dimension indices into slides. Each dimension must appear exactly once.</param>
    /// <param name="pivotPoint">Reference point z around which slides are built.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    /// <param name="additionalData">Optional user data object threaded through every f(point, data) call during Build.</param>
    /// <param name="nWorkers">Number of parallel workers for function evaluation. null = sequential; -1 = <see cref="Environment.ProcessorCount"/>; positive = exact count.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative evaluation count across all slides.</param>
    /// <remarks>Thread safety: the user-supplied function must be thread-safe when <paramref name="nWorkers"/> is non-null.</remarks>
    public ChebyshevSlider(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int[][] partition,
        double[] pivotPoint,
        int maxDerivativeOrder = 2,
        object? additionalData = null,
        int? nWorkers = null,
        IProgress<int>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(function);
        ValidateConstructionArguments(numDimensions, domain, nNodes, partition, pivotPoint);

        Function = function;
        NumDimensions = numDimensions;
        _domain = domain.Select(d => (double[])d.Clone()).ToArray();
        _nNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;
        _partition = partition.Select(g => (int[])g.Clone()).ToArray();
        _pivotPoint = (double[])pivotPoint.Clone();

        // Build dim → slide mapping
        DimToSlide = BuildDimToSlide(_partition);
    }

    /// <summary>Internal parameterless constructor for factories.</summary>
    internal ChebyshevSlider() { }

    // ------------------------------------------------------------------
    // Validation helpers
    // ------------------------------------------------------------------

    private static void ValidateConstructionArguments(
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int[][] partition,
        double[] pivotPoint)
    {
        if (numDimensions <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(numDimensions),
                numDimensions,
                "numDimensions must be positive.");

        ArgumentNullException.ThrowIfNull(domain);
        ArgumentNullException.ThrowIfNull(nNodes);
        ArgumentNullException.ThrowIfNull(partition);
        ArgumentNullException.ThrowIfNull(pivotPoint);

        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}.",
                nameof(domain));
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}.",
                nameof(nNodes));
        if (pivotPoint.Length != numDimensions)
            throw new ArgumentException(
                $"pivotPoint has {pivotPoint.Length} entries but numDimensions={numDimensions}.",
                nameof(pivotPoint));

        for (int d = 0; d < numDimensions; d++)
        {
            if (domain[d] == null)
                throw new ArgumentException($"domain[{d}] must not be null.", nameof(domain));
            if (domain[d].Length != 2)
                throw new ArgumentException(
                    $"domain[{d}] must contain exactly two bounds.",
                    nameof(domain));

            double lo = domain[d][0];
            double hi = domain[d][1];
            if (!double.IsFinite(lo) || !double.IsFinite(hi) || lo >= hi)
                throw new ArgumentException(
                    $"domain[{d}] must contain finite ordered bounds lo < hi, got [{lo}, {hi}].",
                    nameof(domain));

            if (nNodes[d] <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(nNodes),
                    nNodes[d],
                    $"nNodes[{d}] must be positive.");

            double pivot = pivotPoint[d];
            if (!double.IsFinite(pivot))
                throw new ArgumentException(
                    $"pivotPoint[{d}] must be finite.",
                    nameof(pivotPoint));
            if (pivot < lo || pivot > hi)
                throw new ArgumentOutOfRangeException(
                    nameof(pivotPoint),
                    pivot,
                    $"pivotPoint[{d}] must be inside domain [{lo}, {hi}].");
        }

        ValidatePartition(partition, numDimensions);
    }

    internal static void ValidatePartition(int[][] partition, int numDimensions)
    {
        ArgumentNullException.ThrowIfNull(partition);

        var allDims = new List<int>();
        for (int groupIdx = 0; groupIdx < partition.Length; groupIdx++)
        {
            var group = partition[groupIdx];
            if (group == null)
                throw new ArgumentException(
                    $"partition[{groupIdx}] must not be null.",
                    nameof(partition));
            if (group.Length == 0)
                throw new ArgumentException(
                    $"partition[{groupIdx}] must not be empty.",
                    nameof(partition));
            allDims.AddRange(group);
        }
        allDims.Sort();

        var expected = Enumerable.Range(0, numDimensions).ToList();
        if (!allDims.SequenceEqual(expected))
        {
            throw new ArgumentException(
                $"Partition must cover all dimensions 0..{numDimensions - 1} exactly once. " +
                $"Got dimensions: [{string.Join(", ", allDims)}]");
        }
    }

    internal static Dictionary<int, int> BuildDimToSlide(int[][] partition)
    {
        var map = new Dictionary<int, int>();
        for (int slideIdx = 0; slideIdx < partition.Length; slideIdx++)
            foreach (int d in partition[slideIdx])
                map[d] = slideIdx;
        return map;
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// <summary>
    /// Build all slides by evaluating the function at slide-specific grids.
    /// For each slide, dimensions outside the group are fixed at pivot values.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    public void Build(bool verbose = true)
    {
        if (Function == null)
            throw new InvalidOperationException("Function is null. Cannot build.");

        var sw = Stopwatch.StartNew();
        _cachedErrorEstimate = null;

        // Evaluate pivot value
        double pivotValue = Function(_pivotPoint, _additionalData);
        ValidateFinitePivotValue(pivotValue);
        PivotValue = pivotValue;

        int totalEvals = TotalBuildEvals;
        long fullTensor = TensorShape.CheckedProduct(_nNodes, nameof(Build));

        if (verbose)
        {
            Console.WriteLine(
                $"Building {NumDimensions}D Chebyshev Slider " +
                $"({_partition.Length} slides, {totalEvals:N0} evaluations " +
                $"vs {fullTensor:N0} for full tensor)...");
        }

        Slides = new ChebyshevApproximation[_partition.Length];
        int progressOffset = 0;
        for (int slideIdx = 0; slideIdx < _partition.Length; slideIdx++)
        {
            var group = _partition[slideIdx];
            int slideDim = group.Length;
            var slideDomain = new double[slideDim][];
            var slideNNodes = new int[slideDim];
            for (int i = 0; i < slideDim; i++)
            {
                slideDomain[i] = (double[])_domain[group[i]].Clone();
                slideNNodes[i] = _nNodes[group[i]];
            }

            // Create closure that fixes non-group dims at pivot
            var grp = group;
            var pvt = _pivotPoint;
            var func = Function;
            int ndim = NumDimensions;
            Func<double[], object?, double> slideFunc = (subPoint, data) =>
            {
                var fullPoint = new double[ndim];
                Array.Copy(pvt, fullPoint, ndim);
                for (int i = 0; i < grp.Length; i++)
                    fullPoint[grp[i]] = subPoint[i];
                return func(fullPoint, data);
            };

            // Per-slide progress shim: offsets reported values by cumulative evals so far.
            int capturedOffset = progressOffset;
            IProgress<int>? slideProgress = _progress is null ? null
                : new Internal.OffsetProgress(_progress, capturedOffset);

            var slide = new ChebyshevApproximation(
                slideFunc, slideDim, slideDomain, slideNNodes,
                maxDerivativeOrder: MaxDerivativeOrder,
                additionalData: _additionalData,
                nWorkers: _nWorkers, progress: slideProgress);
            slide.Build(verbose: false);
            progressOffset = checked(progressOffset + TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(slideNNodes, nameof(Build)),
                nameof(Build),
                slideNNodes));
            Slides[slideIdx] = slide;

            if (verbose)
            {
                int slideEvals = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(slideNNodes, nameof(Build)),
                    nameof(Build),
                    slideNNodes);
                Console.WriteLine(
                    $"  Slide {slideIdx + 1}/{_partition.Length}: " +
                    $"dims [{string.Join(", ", group)}], {slideEvals:N0} evals");
            }
        }

        sw.Stop();
        BuildTime = sw.Elapsed.TotalSeconds;

        if (verbose)
            Console.WriteLine($"Build complete in {BuildTime:F3}s");

        Built = true;
        _isConstructionFinished = true;
    }

    private static void ValidateFinitePivotValue(double value)
    {
        if (!double.IsFinite(value))
            throw new ArgumentException(
                "function returned a non-finite value at the slider pivot point; " +
                "build cannot proceed with NaN/Infinity in PivotValue");
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate the slider approximation at a point.
    /// Uses Equation 7.5: f(x) ≈ f(z) + Σᵢ [sᵢ(x_groupᵢ) - f(z)].
    /// For derivatives, only the slide containing that dimension contributes.
    /// Cross-group mixed partials are exactly zero.
    /// </summary>
    /// <param name="point">Evaluation point inside the full declared domain.</param>
    /// <param name="derivativeOrder">Derivative order for each dimension (0 = function value).</param>
    /// <returns>Approximated function value or derivative.</returns>
    public double Eval(double[] point, int[] derivativeOrder)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before Eval().");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrder(derivativeOrder, NumDimensions);

        bool isDerivative = false;
        for (int i = 0; i < derivativeOrder.Length; i++)
        {
            if (derivativeOrder[i] > 0) { isDerivative = true; break; }
        }

        if (isDerivative)
        {
            // Find which slides have differentiated dimensions
            int activeSlide = -1;
            bool multipleActive = false;
            for (int d = 0; d < derivativeOrder.Length; d++)
            {
                if (derivativeOrder[d] > 0)
                {
                    int si = DimToSlide[d];
                    if (activeSlide == -1)
                        activeSlide = si;
                    else if (si != activeSlide)
                    {
                        multipleActive = true;
                        break;
                    }
                }
            }

            // Cross-group mixed partials are exactly zero
            if (multipleActive)
                return 0.0;

            // Single slide contributes
            var group = _partition[activeSlide];
            var subPoint = new double[group.Length];
            var subDeriv = new int[group.Length];
            for (int i = 0; i < group.Length; i++)
            {
                subPoint[i] = point[group[i]];
                subDeriv[i] = derivativeOrder[group[i]];
            }
            return Slides[activeSlide].VectorizedEval(subPoint, subDeriv);
        }
        else
        {
            // Eq 7.5: f(x) ≈ v + Σ [s_i(x_i) - v]
            double result = PivotValue;
            for (int slideIdx = 0; slideIdx < _partition.Length; slideIdx++)
            {
                var group = _partition[slideIdx];
                var subPoint = new double[group.Length];
                var subDeriv = new int[group.Length];
                for (int i = 0; i < group.Length; i++)
                {
                    subPoint[i] = point[group[i]];
                    // subDeriv[i] already 0
                }
                double slideVal = Slides[slideIdx].VectorizedEval(subPoint, subDeriv);
                result += slideVal - PivotValue;
            }
            return result;
        }
    }

    /// <summary>
    /// Evaluate slider at multiple derivative orders for the same point.
    /// </summary>
    /// <param name="point">Evaluation point inside the declared domain.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.</param>
    /// <returns>Results for each derivative order.</returns>
    public double[] EvalMulti(double[] point, int[][] derivativeOrders)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before EvalMulti().");
        EvaluationArguments.ValidatePointInDomain(point, NumDimensions, _domain);
        EvaluationArguments.ValidateDerivativeOrders(derivativeOrders, NumDimensions);
        var results = new double[derivativeOrders.Length];
        for (int i = 0; i < derivativeOrders.Length; i++)
            results[i] = Eval(point, derivativeOrders[i]);
        return results;
    }

    // ------------------------------------------------------------------
    // Error estimation
    // ------------------------------------------------------------------

    /// <summary>
    /// Estimate the sliding approximation error.
    /// Returns the sum of per-slide Chebyshev error estimates.
    /// Note: this captures per-slide interpolation error only; cross-group
    /// interaction error inherent to the sliding decomposition is not included.
    /// </summary>
    /// <returns>Estimated interpolation error (per-slide sum).</returns>
    public double ErrorEstimate()
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before ErrorEstimate().");
        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;
        double sum = 0;
        foreach (var slide in Slides)
            sum += slide.ErrorEstimate();
        _cachedErrorEstimate = sum;
        return sum;
    }

    // ------------------------------------------------------------------
    // Integration (Phase 5 — PyChebyshev v0.17)
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the slider approximation over one or more dimensions.
    /// Uses the closed-form decomposition of the sliding sum:
    ///   f(x) ≈ pv + Σ_i [s_i(x_{G_i}) - pv]
    /// Each slide's integral is computed via <see cref="ChebyshevApproximation.Integrate"/>.
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all (full integration → scalar).</param>
    /// <param name="bounds">Sub-interval bounds per dim (positional with sorted dims). Null = full domain.</param>
    /// <returns>A boxed <c>double</c> when every dim is integrated; otherwise a new <see cref="ChebyshevSlider"/> over surviving dims.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If <paramref name="dims"/> contains out-of-range indices, or <paramref name="bounds"/> are invalid. Duplicate <paramref name="dims"/> entries are silently deduplicated (matches <see cref="ChebyshevApproximation.Integrate"/>).</exception>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before Integrate().");

        // Normalize dims: null = all, sort + deduplicate, validate range.
        int[] sortedDims;
        if (dims == null)
            sortedDims = Enumerable.Range(0, NumDimensions).ToArray();
        else
            sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= NumDimensions)
                throw new ArgumentException(
                    $"dim {d} out-of-range [0, {NumDimensions - 1}]");
        }

        var perDimBounds = Internal.Calculus.NormalizeBounds(sortedDims, bounds, _domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        // Per-dim integration widths.
        var widths = new Dictionary<int, double>();
        var boundsForDim = new Dictionary<int, (double lo, double hi)?>();
        foreach (int d in sortedDims)
        {
            var bd = perDimBounds[dimToIdx[d]];
            double a = _domain[d][0], b = _domain[d][1];
            if (bd == null)
            {
                widths[d] = b - a;
                boundsForDim[d] = null;
            }
            else
            {
                widths[d] = bd.Value.hi - bd.Value.lo;
                boundsForDim[d] = bd;
            }
        }

        double volT = 1.0;
        foreach (int d in sortedDims) volT *= widths[d];

        // Per-slide classification.
        var slideKinds = new (string kind, int[] kept)[_partition.Length];
        for (int slideIdx = 0; slideIdx < _partition.Length; slideIdx++)
        {
            slideKinds[slideIdx] = Internal.Calculus.SliderPartitionIntersect(
                _partition[slideIdx], sortedDims);
        }

        // pv_new accumulator: starts as pv * vol_T (the first term of the sum).
        double pvNew = PivotValue * volT;

        // For each "full" slide: integrate over its full group with the
        // appropriate sub-interval bounds, then add contribution to pv_new.
        // Contribution = vol(T \ G_i) * (I_i - pv * vol(G_i ∩ T))
        // For "full" slides, vol(G_i ∩ T) is the product of widths over G_i.
        for (int slideIdx = 0; slideIdx < _partition.Length; slideIdx++)
        {
            var (kind, _) = slideKinds[slideIdx];
            if (kind != "full") continue;

            var slide = Slides[slideIdx];
            var group = _partition[slideIdx];

            // Local-dim list (always all dims of the slide) with corresponding bounds.
            int[] localDims = Enumerable.Range(0, group.Length).ToArray();
            var localBoundsList = new List<(double lo, double hi)>(group.Length);
            bool allFullDomain = true;
            for (int gi = 0; gi < group.Length; gi++)
            {
                var bd = boundsForDim[group[gi]];
                if (bd == null)
                {
                    // Use full slide-domain for this local dim
                    localBoundsList.Add((slide.DomainStorage[gi][0], slide.DomainStorage[gi][1]));
                }
                else
                {
                    localBoundsList.Add(bd.Value);
                    allFullDomain = false;
                }
            }

            double Ii;
            if (allFullDomain)
                Ii = (double)slide.Integrate(dims: localDims);
            else
                Ii = (double)slide.Integrate(dims: localDims, bounds: localBoundsList.ToArray());

            // vol(T \ G_i) — widths over dims in T but NOT in G_i.
            double volOutside = 1.0;
            var groupSet = new HashSet<int>(group);
            foreach (int d in sortedDims)
                if (!groupSet.Contains(d)) volOutside *= widths[d];

            // vol(G_i ∩ T) for "full" slides equals product of widths over G_i.
            double volGroup = 1.0;
            foreach (int d in group) volGroup *= widths[d];

            pvNew += volOutside * (Ii - PivotValue * volGroup);
        }

        // Full integration: every group classified "full", return scalar.
        if (sortedDims.Length == NumDimensions)
            return pvNew;

        // Partial integration: build new slider over surviving dims.
        // Surviving global dim indices, sorted.
        int[] survive = Enumerable.Range(0, NumDimensions)
            .Where(d => !dimToIdx.ContainsKey(d))
            .ToArray();
        // global -> new index map
        var oldToNew = new Dictionary<int, int>();
        for (int newIdx = 0; newIdx < survive.Length; newIdx++)
            oldToNew[survive[newIdx]] = newIdx;

        var newPartition = new List<int[]>();
        var newSlides = new List<ChebyshevApproximation>();

        for (int slideIdx = 0; slideIdx < _partition.Length; slideIdx++)
        {
            var (kind, kept) = slideKinds[slideIdx];
            if (kind == "full") continue; // absorbed into pv_new

            var group = _partition[slideIdx];
            var slide = Slides[slideIdx];

            ChebyshevApproximation newSlide;
            int[] newGroup;

            if (kind == "none")
            {
                // The slide passes through. Apply the partition-of-unity shift:
                //   new_tensor = vol_T * tensor + (pv_new - pv * vol_T)
                double shift = pvNew - PivotValue * volT;
                var tv = slide.TensorValuesStorage!;
                var newTensor = new double[tv.Length];
                for (int j = 0; j < tv.Length; j++)
                    newTensor[j] = volT * tv[j] + shift;
                newSlide = ChebyshevApproximation.FromGrid(slide, newTensor);
                newGroup = group.Select(d => oldToNew[d]).ToArray();
            }
            else
            {
                // "partial": integrate the group's intersection with T.
                // Build local indices (within slide) for dims to integrate.
                var localDimsList = new List<int>();
                var localBoundsList = new List<(double lo, double hi)>();
                bool sawAnyExplicitBounds = false;
                for (int localI = 0; localI < group.Length; localI++)
                {
                    int gd = group[localI];
                    if (dimToIdx.ContainsKey(gd))
                    {
                        localDimsList.Add(localI);
                        var bd = boundsForDim[gd];
                        if (bd == null)
                        {
                            // Local-dim full domain.
                            localBoundsList.Add(
                                (slide.DomainStorage[localI][0], slide.DomainStorage[localI][1]));
                        }
                        else
                        {
                            localBoundsList.Add(bd.Value);
                            sawAnyExplicitBounds = true;
                        }
                    }
                }

                ChebyshevApproximation reduced;
                if (!sawAnyExplicitBounds)
                    reduced = (ChebyshevApproximation)slide.Integrate(
                        dims: localDimsList.ToArray());
                else
                    reduced = (ChebyshevApproximation)slide.Integrate(
                        dims: localDimsList.ToArray(),
                        bounds: localBoundsList.ToArray());

                // vol(T \ G_i) — widths over dims in T but NOT in this group.
                double volOutside = 1.0;
                var groupSet = new HashSet<int>(group);
                foreach (int d in sortedDims)
                    if (!groupSet.Contains(d)) volOutside *= widths[d];

                // Apply unified rule:
                //   new_tensor = vol_outside * reduced.tensor + (pv_new - pv * vol_T)
                double shift = pvNew - PivotValue * volT;
                var rtv = reduced.TensorValuesStorage!;
                var newTensor = new double[rtv.Length];
                for (int j = 0; j < rtv.Length; j++)
                    newTensor[j] = volOutside * rtv[j] + shift;
                newSlide = ChebyshevApproximation.FromGrid(reduced, newTensor);
                newGroup = kept.Select(d => oldToNew[d]).ToArray();
            }

            newPartition.Add(newGroup);
            newSlides.Add(newSlide);
        }

        // Reconstruct slider metadata for surviving dims.
        // The decomposition is now:
        //   g(y) = pv_new + Σ_j [tilde_s_j(y_{G'_j}) - pv_new]
        // We constructed each tilde_s_j so that its tensor satisfies:
        //   tilde_s_j(y) = scale * source(y) + (pv_new - pv * vol_T)
        // for "none" (scale = vol_T) and "partial" (scale = vol_outside) slides.
        // Subtracting pv_new from tilde_s_j gives scale * source(y) - pv * vol_T,
        // the required contribution of the slide.
        var newDomain = survive.Select(d => (double[])_domain[d].Clone()).ToArray();
        var newNNodes = survive.Select(d => _nNodes[d]).ToArray();
        var newPivotPoint = survive.Select(d => _pivotPoint[d]).ToArray();
        var newPartitionArr = newPartition.ToArray();

        var result = new ChebyshevSlider();
        result.Function = null;
        result.NumDimensions = survive.Length;
        result.DomainStorage = newDomain;
        result.NNodesStorage = newNNodes;
        result.MaxDerivativeOrder = MaxDerivativeOrder;
        result.PartitionStorage = newPartitionArr;
        result.PivotPointStorage = newPivotPoint;
        result.PivotValue = pvNew;
        result.Slides = newSlides.ToArray();
        result.DimToSlide = BuildDimToSlide(newPartitionArr);
        result.Built = true;
        result.BuildTime = 0.0;
        // Inherit Phase 4 ergonomics fields per spec D7 (descriptor + additionalData
        // pass through; derivative-id registry is intentionally NOT copied).
        SliderInheritErgonomics(result);
        return result;
    }

    /// <summary>
    /// Copy descriptor, additionalData, _maxDerivativeOrder (already done via property),
    /// and _constructorType from this Slider to <paramref name="target"/>.
    /// The derivative-id registry is intentionally NOT copied — partial-integrate
    /// results have a different dim space (Python <c>slider.py:1130-1131</c>, spec D7).
    /// </summary>
    private void SliderInheritErgonomics(ChebyshevSlider target)
    {
        target._descriptor = _descriptor;
        target._additionalData = _additionalData;
        target._isConstructionFinished = true;
        target._constructorType = _constructorType;
    }

    /// <summary>Total number of function evaluations used during build.</summary>
    public int TotalBuildEvals
    {
        get
        {
            long total = 0;
            foreach (var group in _partition)
            {
                int[] slideShape = group.Select(d => _nNodes[d]).ToArray();
                int slideEvals = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(slideShape, nameof(TotalBuildEvals)),
                    nameof(TotalBuildEvals),
                    slideShape);
                total = checked(total + slideEvals);
            }
            return TensorShape.RequireArrayLength(total, nameof(TotalBuildEvals));
        }
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// <summary>
    /// Save the built slider to a JSON file.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    public void Save(string path)
    {
        if (!Built)
            throw new InvalidOperationException("Cannot save an unbuilt slider. Call Build() first.");

        var slideStates = new SlideState[Slides.Length];
        for (int i = 0; i < Slides.Length; i++)
        {
            var s = Slides[i];
            slideStates[i] = new SlideState
            {
                NumDimensions = s.NumDimensions,
                Domain = s.DomainStorage.Select(d => (double[])d.Clone()).ToArray(),
                NNodes = (int[])s.NNodesStorage.Clone(),
                MaxDerivativeOrder = s.MaxDerivativeOrder,
                NodeArrays = s.NodeArraysStorage.Select(a => (double[])a.Clone()).ToArray(),
                TensorValues = (double[])s.TensorValuesStorage!.Clone(),
                Weights = s.WeightsStorage!.Select(a => (double[])a.Clone()).ToArray(),
                DiffMatrices = s.DiffMatricesStorage!.Select(m => ChebyshevApproximation.Flatten2D(m)).ToArray(),
                DiffMatrixSizes = s.DiffMatricesStorage!.Select(m => new[] { m.GetLength(0), m.GetLength(1) }).ToArray(),
                BuildTime = s.BuildTime,
                NEvaluations = s.NEvaluations,
            };
        }

        var state = new SliderSerializationState
        {
            NumDimensions = NumDimensions,
            Domain = _domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])_nNodes.Clone(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Partition = _partition.Select(g => (int[])g.Clone()).ToArray(),
            PivotPoint = (double[])_pivotPoint.Clone(),
            PivotValue = PivotValue,
            BuildTime = BuildTime,
            Slides = slideStates,
            Descriptor = _descriptor,
            RegisteredDerivativeOrders = _registeredDerivativeOrders.Count > 0
                ? _registeredDerivativeOrders.Select(o => (int[])o.Clone()).ToArray()
                : null,
        };

        var options = new JsonSerializerOptions { WriteIndented = false };
        string json = JsonSerializer.Serialize(state, options);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load a previously saved slider from a JSON file.
    /// </summary>
    /// <param name="path">Path to the saved file.</param>
    /// <returns>A fully functional slider with Function=null.</returns>
    /// <exception cref="InvalidDataException">If the file contains a malformed ChebyshevSlider state.</exception>
    public static ChebyshevSlider Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SliderSerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize slider.");

        if (state.Type != "ChebyshevSlider")
            throw new InvalidOperationException(
                $"Expected type 'ChebyshevSlider', got '{state.Type}'");

        ValidateSerializedState(state);

        var slides = new ChebyshevApproximation[state.Slides.Length];
        for (int i = 0; i < state.Slides.Length; i++)
        {
            var ss = state.Slides[i];
            var diffMatrices = new double[ss.NumDimensions][,];
            for (int d = 0; d < ss.NumDimensions; d++)
            {
                int rows = ss.DiffMatrixSizes[d][0];
                int cols = ss.DiffMatrixSizes[d][1];
                diffMatrices[d] = ChebyshevApproximation.Unflatten2D(ss.DiffMatrices[d], rows, cols);
            }

            var slide = new ChebyshevApproximation
            {
                Function = null,
                NumDimensions = ss.NumDimensions,
                DomainStorage = ss.Domain,
                NNodesStorage = ss.NNodes,
                MaxDerivativeOrder = ss.MaxDerivativeOrder,
                NodeArraysStorage = ss.NodeArrays,
                TensorValuesStorage = ss.TensorValues,
                WeightsStorage = ss.Weights,
                DiffMatricesStorage = diffMatrices,
                BuildTime = ss.BuildTime,
                NEvaluations = ss.NEvaluations,
            };
            slide.PrecomputeTransposedDiffMatrices();
            slides[i] = slide;
        }

        var slider = new ChebyshevSlider
        {
            Function = null,
            NumDimensions = state.NumDimensions,
            DomainStorage = state.Domain,
            NNodesStorage = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder,
            PartitionStorage = state.Partition,
            PivotPointStorage = state.PivotPoint,
            PivotValue = state.PivotValue,
            Slides = slides,
            DimToSlide = BuildDimToSlide(state.Partition),
            Built = true,
            BuildTime = state.BuildTime,
        };
        // v0.8.0 migration: Descriptor may be absent in older files.
        slider._descriptor = state.Descriptor;
        // ConstructorType is intentionally NOT restored from state — Load always sets "load".
        slider._constructorType = "load";
        slider._isConstructionFinished = true;
        if (state.RegisteredDerivativeOrders != null)
        {
            foreach (var orders in state.RegisteredDerivativeOrders)
            {
                var key = new Internal.TupleKey(orders);
                int id = slider._registeredDerivativeOrders.Count;
                slider._registeredDerivativeOrders.Add((int[])orders.Clone());
                slider._derivativeIdRegistry[key] = id;
            }
        }
        return slider;
    }

    private static void ValidateSerializedState(SliderSerializationState state)
    {
        int d = state.NumDimensions;
        if (d <= 0)
            throw new InvalidDataException($"NumDimensions must be positive, got {d}.");

        ValidateDomain(state.Domain, d, nameof(SliderSerializationState.Domain));
        ValidatePositiveVector(state.NNodes, d, nameof(SliderSerializationState.NNodes));
        if (state.MaxDerivativeOrder < 0)
            throw new InvalidDataException($"MaxDerivativeOrder must be non-negative, got {state.MaxDerivativeOrder}.");
        if (!double.IsFinite(state.PivotValue))
            throw new InvalidDataException($"PivotValue must be finite, got {state.PivotValue}.");
        if (!double.IsFinite(state.BuildTime) || state.BuildTime < 0.0)
            throw new InvalidDataException($"BuildTime must be finite and non-negative, got {state.BuildTime}.");

        ValidateFiniteVector(state.PivotPoint, d, nameof(SliderSerializationState.PivotPoint));
        ValidatePartitionForLoad(state.Partition, d);
        ValidateDerivativeRegistry(state.RegisteredDerivativeOrders, d);

        if (state.Slides is null)
            throw new InvalidDataException("Slides must be present.");
        if (state.Slides.Length != state.Partition.Length)
            throw new InvalidDataException(
                $"Slides has length {state.Slides.Length}, expected Partition length {state.Partition.Length}.");

        for (int i = 0; i < state.Slides.Length; i++)
            ValidateSlideState(state.Slides[i], state.Partition[i], state, i);
    }

    private static void ValidateSlideState(
        SlideState? slide, int[] partitionGroup, SliderSerializationState parent, int slideIndex)
    {
        if (slide is null)
            throw new InvalidDataException($"Slides[{slideIndex}] must be present.");
        if (slide.NumDimensions <= 0)
            throw new InvalidDataException(
                $"Slides[{slideIndex}].NumDimensions must be positive, got {slide.NumDimensions}.");
        if (slide.NumDimensions != partitionGroup.Length)
            throw new InvalidDataException(
                $"Slides[{slideIndex}].NumDimensions={slide.NumDimensions} does not match " +
                $"Partition[{slideIndex}] length {partitionGroup.Length}.");
        if (slide.MaxDerivativeOrder < 0)
            throw new InvalidDataException(
                $"Slides[{slideIndex}].MaxDerivativeOrder must be non-negative, got {slide.MaxDerivativeOrder}.");
        if (!double.IsFinite(slide.BuildTime) || slide.BuildTime < 0.0)
            throw new InvalidDataException(
                $"Slides[{slideIndex}].BuildTime must be finite and non-negative, got {slide.BuildTime}.");
        if (slide.NEvaluations < 0)
            throw new InvalidDataException(
                $"Slides[{slideIndex}].NEvaluations must be non-negative, got {slide.NEvaluations}.");

        ValidateDomain(slide.Domain, slide.NumDimensions, $"Slides[{slideIndex}].Domain");
        ValidatePositiveVector(slide.NNodes, slide.NumDimensions, $"Slides[{slideIndex}].NNodes");
        ValidateApproxVectorArray(slide.NodeArrays, slide.NNodes, $"Slides[{slideIndex}].NodeArrays");
        ValidateApproxVectorArray(slide.Weights, slide.NNodes, $"Slides[{slideIndex}].Weights");
        ValidateDiffMatrices(slide.DiffMatrices, slide.DiffMatrixSizes, slide.NNodes, slideIndex);

        int expectedTensorLength = CheckedArrayLengthForInvalidData(slide.NNodes, $"Slides[{slideIndex}].TensorValues");
        ValidateFiniteVector(
            slide.TensorValues,
            expectedTensorLength,
            $"Slides[{slideIndex}].TensorValues");

        for (int localDim = 0; localDim < partitionGroup.Length; localDim++)
        {
            int parentDim = partitionGroup[localDim];
            if (slide.NNodes[localDim] != parent.NNodes[parentDim])
                throw new InvalidDataException(
                    $"Slides[{slideIndex}].NNodes[{localDim}]={slide.NNodes[localDim]} does not match " +
                    $"NNodes[{parentDim}]={parent.NNodes[parentDim]}.");

            double slideLo = slide.Domain[localDim][0];
            double slideHi = slide.Domain[localDim][1];
            double parentLo = parent.Domain[parentDim][0];
            double parentHi = parent.Domain[parentDim][1];
            if (slideLo != parentLo || slideHi != parentHi)
                throw new InvalidDataException(
                    $"Slides[{slideIndex}].Domain[{localDim}] does not match Domain[{parentDim}].");
        }
    }

    private static void ValidatePartitionForLoad(int[][]? partition, int numDimensions)
    {
        if (partition is null)
            throw new InvalidDataException("Partition must be present.");

        var allDims = new List<int>();
        for (int i = 0; i < partition.Length; i++)
        {
            int[] group = partition[i]
                ?? throw new InvalidDataException($"Partition[{i}] must be present.");
            allDims.AddRange(group);
        }

        allDims.Sort();
        var expected = Enumerable.Range(0, numDimensions).ToArray();
        if (!allDims.SequenceEqual(expected))
        {
            throw new InvalidDataException(
                $"Partition must cover all dimensions 0..{numDimensions - 1} exactly once. " +
                $"Got dimensions: [{string.Join(", ", allDims)}]");
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

    private static void ValidateDiffMatrices(
        double[][]? matrices, int[][]? matrixSizes, int[] nNodes, int slideIndex)
    {
        string matrixName = $"Slides[{slideIndex}].DiffMatrices";
        string sizeName = $"Slides[{slideIndex}].DiffMatrixSizes";
        if (matrices is null)
            throw new InvalidDataException($"{matrixName} must be present.");
        if (matrixSizes is null)
            throw new InvalidDataException($"{sizeName} must be present.");
        if (matrices.Length != nNodes.Length)
            throw new InvalidDataException($"{matrixName} has length {matrices.Length}, expected {nNodes.Length}.");
        if (matrixSizes.Length != nNodes.Length)
            throw new InvalidDataException($"{sizeName} has length {matrixSizes.Length}, expected {nNodes.Length}.");

        for (int i = 0; i < nNodes.Length; i++)
        {
            int[] size = matrixSizes[i]
                ?? throw new InvalidDataException($"{sizeName}[{i}] must be present.");
            if (size.Length != 2)
                throw new InvalidDataException($"{sizeName}[{i}] must contain exactly two dimensions.");
            if (size[0] != nNodes[i] || size[1] != nNodes[i])
                throw new InvalidDataException(
                    $"{sizeName}[{i}] must equal [{nNodes[i]},{nNodes[i]}], got [{string.Join(",", size)}].");

            int expectedLength = CheckedArrayLengthForInvalidData(size, $"{matrixName}[{i}]");
            ValidateFiniteVector(matrices[i], expectedLength, $"{matrixName}[{i}]");
        }
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
    // Internal factory for arithmetic operators
    // ------------------------------------------------------------------

    /// <summary>
    /// Create a new slider sharing grid metadata from source with new slides and pivotValue.
    /// </summary>
    internal static ChebyshevSlider FromSlides(
        ChebyshevSlider source, ChebyshevApproximation[] slides, double pivotValue)
    {
        return new ChebyshevSlider
        {
            Function = null,
            NumDimensions = source.NumDimensions,
            DomainStorage = source._domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodesStorage = (int[])source._nNodes.Clone(),
            MaxDerivativeOrder = source.MaxDerivativeOrder,
            PartitionStorage = source._partition.Select(g => (int[])g.Clone()).ToArray(),
            PivotPointStorage = (double[])source._pivotPoint.Clone(),
            Slides = slides,
            PivotValue = pivotValue,
            DimToSlide = new Dictionary<int, int>(source.DimToSlide),
            Built = true,
            BuildTime = 0.0,
            _isConstructionFinished = true,
        };
    }

    // ------------------------------------------------------------------
    // Extrusion and slicing
    // ------------------------------------------------------------------

    /// <summary>
    /// Add new dimensions where the function is constant.
    /// Each new dimension becomes its own single-dim slide group with constant
    /// tensor values equal to PivotValue, so it contributes nothing to the
    /// sliding sum.
    /// </summary>
    /// <param name="extrudeParams">Tuples of (dimIndex, bounds, nNodes).</param>
    /// <returns>A new, higher-dimensional slider (already built).</returns>
    public ChebyshevSlider Extrude(params (int dimIndex, double[] bounds, int nNodes)[] extrudeParams)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first.");

        var sorted = ExtrudeSlice.NormalizeExtrusionParams(extrudeParams, NumDimensions);

        var domain = _domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = _nNodes.ToList();
        var pivotPoint = _pivotPoint.ToList();
        var partition = _partition.Select(g => g.ToList()).ToList();
        var slides = Slides.ToList();

        foreach (var (dimIdx, bounds, n) in sorted)
        {
            double lo = bounds[0], hi = bounds[1];

            // Remap partition indices >= dimIdx
            foreach (var group in partition)
            {
                for (int i = 0; i < group.Count; i++)
                    if (group[i] >= dimIdx)
                        group[i]++;
            }

            // Create new 1D constant slide
            var newTensor = new double[n];
            Array.Fill(newTensor, PivotValue);

            var newSlide = ChebyshevApproximation.FromValues(
                newTensor,
                numDimensions: 1,
                domain: new[] { new[] { lo, hi } },
                nNodes: new[] { n },
                maxDerivativeOrder: MaxDerivativeOrder);

            // Add new group and slide
            partition.Add(new List<int> { dimIdx });
            slides.Add(newSlide);

            // Insert into domain/nNodes/pivotPoint
            domain.Insert(dimIdx, new[] { lo, hi });
            nNodes.Insert(dimIdx, n);
            pivotPoint.Insert(dimIdx, 0.5 * (lo + hi));
        }

        int newNdim = NumDimensions + sorted.Length;
        var newPartition = partition.Select(g => g.ToArray()).ToArray();

        return new ChebyshevSlider
        {
            Function = null,
            NumDimensions = newNdim,
            DomainStorage = domain.ToArray(),
            NNodesStorage = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            PartitionStorage = newPartition,
            PivotPointStorage = pivotPoint.ToArray(),
            Slides = slides.ToArray(),
            PivotValue = PivotValue,
            DimToSlide = BuildDimToSlide(newPartition),
            Built = true,
            _isConstructionFinished = true,
        };
    }

    /// <summary>
    /// Fix one or more dimensions at given values, reducing dimensionality.
    /// </summary>
    /// <param name="sliceParams">Tuples of (dimIndex, value).</param>
    /// <returns>A new, lower-dimensional slider (already built).</returns>
    public ChebyshevSlider Slice(params (int dimIndex, double value)[] sliceParams)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first.");

        var sorted = ExtrudeSlice.NormalizeSlicingParams(sliceParams, NumDimensions);

        // Validate values within domain
        foreach (var (dimIdx, value) in sorted)
        {
            double lo = _domain[dimIdx][0], hi = _domain[dimIdx][1];
            if (value < lo || value > hi)
                throw new ArgumentException(
                    $"Slice value {value} for dim {dimIdx} is outside domain [{lo}, {hi}]");
        }

        var domain = _domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = _nNodes.ToList();
        var pivotPoint = _pivotPoint.ToList();
        var partition = _partition.Select(g => g.ToList()).ToList();
        var slides = Slides.ToList();
        double pivotValue = PivotValue;

        foreach (var (dimIdx, value) in sorted) // descending order
        {
            // Find which slide group contains dimIdx
            int slideIdx = -1;
            int localDimIdx = -1;
            for (int si = 0; si < partition.Count; si++)
            {
                int idx = partition[si].IndexOf(dimIdx);
                if (idx >= 0)
                {
                    slideIdx = si;
                    localDimIdx = idx;
                    break;
                }
            }

            if (partition[slideIdx].Count > 1)
            {
                // Case 1: Multi-dim group — slice the ChebyshevApproximation
                slides[slideIdx] = slides[slideIdx].Slice((localDimIdx, value));
                partition[slideIdx].Remove(dimIdx);
            }
            else
            {
                // Case 2: Single-dim group — evaluate and absorb
                double sVal = slides[slideIdx].VectorizedEval(new[] { value }, new[] { 0 });
                double delta = sVal - pivotValue;

                // Add delta to each remaining slide's tensor_values
                for (int i = 0; i < slides.Count; i++)
                {
                    if (i != slideIdx)
                    {
                        var tv = slides[i].TensorValuesStorage!;
                        var newTv = new double[tv.Length];
                        for (int j = 0; j < tv.Length; j++)
                            newTv[j] = tv[j] + delta;
                        slides[i] = ChebyshevApproximation.FromGrid(slides[i], newTv);
                    }
                }

                pivotValue = sVal;

                // Remove group and slide
                partition.RemoveAt(slideIdx);
                slides.RemoveAt(slideIdx);
            }

            // Remap all partition indices > dimIdx down by 1
            foreach (var group in partition)
            {
                for (int i = 0; i < group.Count; i++)
                    if (group[i] > dimIdx)
                        group[i]--;
            }

            domain.RemoveAt(dimIdx);
            nNodes.RemoveAt(dimIdx);
            pivotPoint.RemoveAt(dimIdx);
        }

        int newNdim = NumDimensions - sorted.Length;
        var newPartition = partition.Select(g => g.ToArray()).ToArray();

        return new ChebyshevSlider
        {
            Function = null,
            NumDimensions = newNdim,
            DomainStorage = domain.ToArray(),
            NNodesStorage = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            PartitionStorage = newPartition,
            PivotPointStorage = pivotPoint.ToArray(),
            Slides = slides.ToArray(),
            PivotValue = pivotValue,
            DimToSlide = BuildDimToSlide(newPartition),
            Built = true,
            _isConstructionFinished = true,
        };
    }

    // ------------------------------------------------------------------
    // Calculus: roots / minimize / maximize (via 1-D projection)
    // ------------------------------------------------------------------

    /// <summary>
    /// Build a 1-D ChebyshevApproximation from this 1-D Slider by evaluating at
    /// Chebyshev Type-I nodes. Used by Roots to delegate to the existing 1-D
    /// calculus primitives on ChebyshevApproximation.
    /// </summary>
    /// <remarks>
    /// Precondition: this Slider must be 1-D (NumDimensions == 1). Call Slice()
    /// to reduce a multi-D Slider to 1-D before calling this helper.
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1138-1176</c>.
    /// </remarks>
    private ChebyshevApproximation To1DChebyshev()
    {
        if (NumDimensions != 1)
            throw new InvalidOperationException(
                $"To1DChebyshev requires a 1-D slider, got {NumDimensions}-D");

        int n = _nNodes[0];
        double a = _domain[0][0];
        double b = _domain[0][1];
        double[] chebNodes = Internal.BarycentricKernel.MakeNodesForDim(a, b, n);

        var zeroOrder = new int[] { 0 };
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = Eval(new[] { chebNodes[i] }, zeroOrder);

        return ChebyshevApproximation.FromValues(
            values,
            numDimensions: 1,
            domain: new[] { new[] { a, b } },
            nNodes: new[] { n });
    }

    /// <summary>
    /// Find all real roots of the slider along a specified dimension.
    /// Reduces to a 1-D problem by slicing all other dimensions to their
    /// fixed values, then delegates to <see cref="ChebyshevApproximation.Roots"/>.
    /// </summary>
    /// <param name="dim">Target dimension. For 1-D sliders, defaults to 0.</param>
    /// <param name="fixedDims">For multi-D sliders, <c>{dim_index: value}</c>
    /// for all dimensions except <paramref name="dim"/>.</param>
    /// <returns>Sorted real root locations in the physical domain. Empty if no roots.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If <paramref name="dim"/> or <paramref name="fixedDims"/> validation fails.</exception>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1178-1224</c>.
    /// </remarks>
    public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (validatedDim, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);

        var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Roots();
    }

    /// <summary>
    /// Find the minimum value of the slider along a specified dimension.
    /// Reduces to a 1-D problem by slicing all other dimensions to their fixed
    /// values, then delegates to <see cref="ChebyshevApproximation.Minimize"/>.
    /// </summary>
    /// <param name="dim">Target dimension. For 1-D sliders, defaults to 0.</param>
    /// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
    /// dims except <paramref name="dim"/>.</param>
    /// <returns>Tuple <c>(value, location)</c> where value is the minimum and
    /// location is its coordinate in the target dimension.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If validation fails.</exception>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1226-1264</c>.
    /// </remarks>
    public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (validatedDim, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);

        var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Minimize();
    }

    /// <summary>
    /// Find the maximum value of the slider along a specified dimension.
    /// See <see cref="Minimize"/> for parameter details.
    /// </summary>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1266-1283</c>.
    /// </remarks>
    public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() first");

        var (validatedDim, sliceParams) =
            Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, _domain);

        var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
        var cheb1D = sliced.To1DChebyshev();
        return cheb1D.Maximize();
    }

    // ------------------------------------------------------------------
    // Arithmetic operators
    // ------------------------------------------------------------------

    internal void CheckSliderCompatible(ChebyshevSlider other)
    {
        if (GetType() != other.GetType())
            throw new InvalidOperationException(
                $"Cannot combine {GetType().Name} with {other.GetType().Name}");
        if (!Built || !other.Built)
            throw new InvalidOperationException("Both sliders must be built.");
        if (NumDimensions != other.NumDimensions)
            throw new InvalidOperationException(
                $"Dimension mismatch: {NumDimensions} vs {other.NumDimensions}");
        if (!_nNodes.SequenceEqual(other._nNodes))
            throw new InvalidOperationException("Node count mismatch.");
        for (int d = 0; d < NumDimensions; d++)
        {
            if (Math.Abs(_domain[d][0] - other._domain[d][0]) > 1e-14 ||
                Math.Abs(_domain[d][1] - other._domain[d][1]) > 1e-14)
                throw new InvalidOperationException($"Domain mismatch at dimension {d}.");
        }
        if (MaxDerivativeOrder != other.MaxDerivativeOrder)
            throw new InvalidOperationException("MaxDerivativeOrder mismatch.");

        // Slider-specific checks
        if (_partition.Length != other._partition.Length)
            throw new ArgumentException(
                $"Partition mismatch: {FormatPartition(_partition)} vs {FormatPartition(other._partition)}");
        for (int i = 0; i < _partition.Length; i++)
        {
            if (!_partition[i].SequenceEqual(other._partition[i]))
                throw new ArgumentException(
                    $"Partition mismatch: {FormatPartition(_partition)} vs {FormatPartition(other._partition)}");
        }
        if (!_pivotPoint.SequenceEqual(other._pivotPoint))
            throw new ArgumentException(
                $"Pivot point mismatch: [{string.Join(", ", _pivotPoint)}] vs [{string.Join(", ", other._pivotPoint)}]");
    }

    private static string FormatPartition(int[][] partition)
    {
        return "[" + string.Join(", ", partition.Select(g => "[" + string.Join(", ", g) + "]")) + "]";
    }

    /// <summary>Pointwise addition of two sliders on the same grid.</summary>
    public static ChebyshevSlider operator +(ChebyshevSlider a, ChebyshevSlider b)
    {
        a.CheckSliderCompatible(b);
        var slides = new ChebyshevApproximation[a.Slides.Length];
        for (int i = 0; i < slides.Length; i++)
        {
            var tvA = a.Slides[i].TensorValuesStorage!;
            var tvB = b.Slides[i].TensorValuesStorage!;
            var sum = new double[tvA.Length];
            for (int j = 0; j < tvA.Length; j++)
                sum[j] = tvA[j] + tvB[j];
            slides[i] = ChebyshevApproximation.FromGrid(a.Slides[i], sum);
        }
        return FromSlides(a, slides, a.PivotValue + b.PivotValue);
    }

    /// <summary>Pointwise subtraction of two sliders on the same grid.</summary>
    public static ChebyshevSlider operator -(ChebyshevSlider a, ChebyshevSlider b)
    {
        a.CheckSliderCompatible(b);
        var slides = new ChebyshevApproximation[a.Slides.Length];
        for (int i = 0; i < slides.Length; i++)
        {
            var tvA = a.Slides[i].TensorValuesStorage!;
            var tvB = b.Slides[i].TensorValuesStorage!;
            var diff = new double[tvA.Length];
            for (int j = 0; j < tvA.Length; j++)
                diff[j] = tvA[j] - tvB[j];
            slides[i] = ChebyshevApproximation.FromGrid(a.Slides[i], diff);
        }
        return FromSlides(a, slides, a.PivotValue - b.PivotValue);
    }

    /// <summary>Scalar multiplication.</summary>
    public static ChebyshevSlider operator *(ChebyshevSlider a, double scalar)
    {
        if (!a.Built)
            throw new InvalidOperationException("Operand is not built. Call Build() first.");

        var slides = new ChebyshevApproximation[a.Slides.Length];
        for (int i = 0; i < slides.Length; i++)
        {
            var tv = a.Slides[i].TensorValuesStorage!;
            var scaled = new double[tv.Length];
            for (int j = 0; j < tv.Length; j++)
                scaled[j] = tv[j] * scalar;
            slides[i] = ChebyshevApproximation.FromGrid(a.Slides[i], scaled);
        }
        return FromSlides(a, slides, a.PivotValue * scalar);
    }

    /// <summary>Scalar multiplication (scalar on left).</summary>
    public static ChebyshevSlider operator *(double scalar, ChebyshevSlider a) => a * scalar;

    /// <summary>Scalar division.</summary>
    public static ChebyshevSlider operator /(ChebyshevSlider a, double scalar) => a * (1.0 / scalar);

    /// <summary>Unary negation.</summary>
    public static ChebyshevSlider operator -(ChebyshevSlider a) => a * -1.0;

    // ------------------------------------------------------------------
    // Display
    // ------------------------------------------------------------------

    /// <summary>Compact repr string.</summary>
    public string ToReprString()
    {
        return $"ChebyshevSlider(" +
            $"dims={NumDimensions}, " +
            $"slides={_partition.Length}, " +
            $"partition={FormatPartition(_partition)}, " +
            $"built={Built})";
    }

    /// <summary>Multi-line display string.</summary>
    public override string ToString()
    {
        string status = Built ? "built" : "not built";
        int totalSlideEvals = TotalBuildEvals;
        long fullTensorEvals = TensorShape.CheckedProduct(_nNodes, nameof(ToString));

        const int maxDisplay = 6;

        // Nodes line
        string nodesStr;
        if (NumDimensions > maxDisplay)
        {
            nodesStr = "[" + string.Join(", ", _nNodes.Take(maxDisplay)) + ", ...]";
        }
        else
        {
            nodesStr = "[" + string.Join(", ", _nNodes) + "]";
        }

        // Domain line
        string domainStr;
        if (NumDimensions > maxDisplay)
        {
            domainStr = string.Join(" x ",
                _domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        }
        else
        {
            domainStr = string.Join(" x ",
                _domain.Select(d => $"[{d[0]}, {d[1]}]"));
        }

        // Pivot line
        string pivotStr;
        if (NumDimensions > maxDisplay)
        {
            pivotStr = "[" + string.Join(", ", _pivotPoint.Take(maxDisplay)) + ", ...]";
        }
        else
        {
            pivotStr = "[" + string.Join(", ", _pivotPoint) + "]";
        }

        // Partition line
        string partitionStr;
        if (_partition.Length > maxDisplay)
        {
            partitionStr = "[" +
                string.Join(", ", _partition.Take(maxDisplay).Select(g => "[" + string.Join(", ", g) + "]")) +
                ", ...]";
        }
        else
        {
            partitionStr = FormatPartition(_partition);
        }

        var lines = new List<string>
        {
            $"ChebyshevSlider ({NumDimensions}D, {_partition.Length} slides, {status})",
            $"  Partition: {partitionStr}",
            $"  Pivot:     {pivotStr}",
            $"  Nodes:     {nodesStr} ({totalSlideEvals:N0} vs {fullTensorEvals:N0} full tensor)",
            $"  Domain:    {domainStr}",
        };

        if (Built && Slides.Length > 0)
        {
            lines.Add($"  Error est: {ErrorEstimate():E2}");
            lines.Add("  Slides:");
            for (int i = 0; i < _partition.Length; i++)
            {
                var group = _partition[i];
                int slideEvals = TensorShape.RequireArrayLength(
                    TensorShape.CheckedProduct(group.Select(d => _nNodes[d]), nameof(ToString)),
                    nameof(ToString));
                lines.Add(
                    $"    [{i}] dims [{string.Join(", ", group)}]: " +
                    $"{slideEvals:N0} evals, " +
                    $"built in {Slides[i].BuildTime:F3}s");
            }
        }

        return string.Join("\n", lines);
    }

    // ------------------------------------------------------------------
    // Phase 4 ergonomics — accessors
    // ------------------------------------------------------------------

    /// <summary>Set a free-form descriptor string for this slider.</summary>
    public void SetDescriptor(string descriptor) => _descriptor = descriptor;

    /// <summary>Get the descriptor previously set via <see cref="SetDescriptor"/>; null if unset.</summary>
    public string? GetDescriptor() => _descriptor;

    /// <summary>True if <see cref="Build"/>/<see cref="Load"/> completed.</summary>
    public bool IsConstructionFinished() => _isConstructionFinished;

    /// <summary>Returns one of: "function" (Build), "load" (Load).</summary>
    public string GetConstructorType() => _constructorType;

    /// <summary>Per-dimension Chebyshev node counts actually used.</summary>
    public int[] GetUsedNs() => (int[])_nNodes.Clone();

    /// <summary>Maximum derivative order this slider supports.</summary>
    public int GetMaxDerivativeOrder() => MaxDerivativeOrder;

    /// <summary>
    /// Returns the user-supplied <c>additionalData</c> object passed to the constructor,
    /// or null if none was provided. Same value is threaded through every <c>f(point, data)</c>
    /// call during <see cref="Build"/>.
    /// </summary>
    public object? GetAdditionalData() => _additionalData;

    /// <summary>
    /// Total number of evaluation points across all slides.
    /// </summary>
    /// <returns>The sum of GetNumEvaluationPoints() from each slide.</returns>
    public int GetNumEvaluationPoints()
    {
        if (Slides == null) return 0;
        long total = 0;
        foreach (var slide in Slides) total = checked(total + slide.GetNumEvaluationPoints());
        return TensorShape.RequireArrayLength(total, nameof(GetNumEvaluationPoints));
    }

    /// <summary>
    /// Flat row-major array of all slider evaluation points, expanded to full ndim using PivotPoint.
    /// Each slide's local coordinates are mapped to full-ndim space via the Partition and PivotPoint.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached internally.
    /// </summary>
    /// <returns>A snapshot of full-ndim node coordinates, flattened in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return CloneHelpers.DeepCopy(_evaluationPointsCache)!;

        int total = GetNumEvaluationPoints();
        int coordinateCount = TensorShape.RequireArrayLength(
            TensorShape.CheckedProduct(new[] { total, NumDimensions }, nameof(GetEvaluationPoints)),
            nameof(GetEvaluationPoints),
            new[] { total, NumDimensions });
        var points = new double[coordinateCount];
        int offset = 0;

        for (int slideIdx = 0; slideIdx < Slides!.Length; slideIdx++)
        {
            var slide = Slides[slideIdx];
            var group = _partition[slideIdx];
            var slidePts = slide.GetEvaluationPoints();
            int slideNum = slide.GetNumEvaluationPoints();
            int gdim = group.Length;

            for (int p = 0; p < slideNum; p++)
            {
                for (int d = 0; d < NumDimensions; d++)
                    points[offset + p * NumDimensions + d] = _pivotPoint[d];
                for (int gi = 0; gi < gdim; gi++)
                    points[offset + p * NumDimensions + group[gi]] = slidePts[p * gdim + gi];
            }
            offset += slideNum * NumDimensions;
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
    /// Returns a deep copy of this slider. The source <see cref="Function"/>
    /// callable is NOT duplicated — clones cannot be rebuilt without re-supplying
    /// the function. All precomputed slides and state are deep-copied.
    /// </summary>
    /// <returns>A fully independent <see cref="ChebyshevSlider"/> with <see cref="Function"/> set to null.</returns>
    public ChebyshevSlider Clone()
    {
        var copy = new ChebyshevSlider();
        copy.NumDimensions = NumDimensions;
        copy.DomainStorage = Internal.CloneHelpers.DeepCopy(_domain)!;
        copy.NNodesStorage = Internal.CloneHelpers.DeepCopy(_nNodes)!;
        copy.PartitionStorage = Internal.CloneHelpers.DeepCopy(_partition)!;
        copy.PivotPointStorage = Internal.CloneHelpers.DeepCopy(_pivotPoint)!;
        copy.PivotValue = PivotValue;
        copy.MaxDerivativeOrder = MaxDerivativeOrder;
        copy.Built = Built;
        copy.BuildTime = BuildTime;
        copy._descriptor = _descriptor;
        copy._additionalData = _additionalData;
        copy._isConstructionFinished = _isConstructionFinished;
        copy._constructorType = "clone";
        copy._evaluationPointsCache = null;
        copy.DimToSlide = new System.Collections.Generic.Dictionary<int, int>(DimToSlide);
        if (Slides != null)
        {
            copy.Slides = new ChebyshevApproximation[Slides.Length];
            for (int i = 0; i < Slides.Length; i++)
                copy.Slides[i] = Slides[i].Clone();
        }
        foreach (var kv in _derivativeIdRegistry)
            copy._derivativeIdRegistry[kv.Key] = kv.Value;
        foreach (var orders in _registeredDerivativeOrders)
            copy._registeredDerivativeOrders.Add((int[])orders.Clone());
        return copy;
    }

    // ------------------------------------------------------------------
    // Serialization state classes
    // ------------------------------------------------------------------

    internal class SliderSerializationState
    {
        public string Type { get; set; } = "ChebyshevSlider";
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int MaxDerivativeOrder { get; set; } = 2;
        public int[][] Partition { get; set; } = Array.Empty<int[]>();
        public double[] PivotPoint { get; set; } = Array.Empty<double>();
        public double PivotValue { get; set; }
        public double BuildTime { get; set; }
        public SlideState[] Slides { get; set; } = Array.Empty<SlideState>();
        // v0.8.0 ergonomics fields (absent in pre-v0.8.0 JSON; null == not set)
        public string? Descriptor { get; set; }
        public string? ConstructorType { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
    }

    internal class SlideState
    {
        public int NumDimensions { get; set; }
        public double[][] Domain { get; set; } = Array.Empty<double[]>();
        public int[] NNodes { get; set; } = Array.Empty<int>();
        public int MaxDerivativeOrder { get; set; } = 2;
        public double[][] NodeArrays { get; set; } = Array.Empty<double[]>();
        public double[] TensorValues { get; set; } = Array.Empty<double>();
        public double[][] Weights { get; set; } = Array.Empty<double[]>();
        public double[][] DiffMatrices { get; set; } = Array.Empty<double[]>();
        public int[][] DiffMatrixSizes { get; set; } = Array.Empty<int[]>();
        public double BuildTime { get; set; }
        public int NEvaluations { get; set; }
    }
}
