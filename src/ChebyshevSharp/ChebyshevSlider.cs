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
    /// <summary>The function to approximate. Null after load.</summary>
    public Func<double[], object?, double>? Function { get; internal set; }

    /// <summary>Number of input dimensions.</summary>
    public int NumDimensions { get; internal set; }

    /// <summary>Domain bounds for each dimension, as list of [lo, hi].</summary>
    public double[][] Domain { get; internal set; } = Array.Empty<double[]>();

    /// <summary>Number of Chebyshev nodes per dimension.</summary>
    public int[] NNodes { get; internal set; } = Array.Empty<int>();

    /// <summary>Maximum supported derivative order.</summary>
    public int MaxDerivativeOrder { get; internal set; } = 2;

    /// <summary>Grouping of dimension indices into slides.</summary>
    public int[][] Partition { get; internal set; } = Array.Empty<int[]>();

    /// <summary>Reference point z around which slides are built.</summary>
    public double[] PivotPoint { get; internal set; } = Array.Empty<double>();

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
        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        NNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;
        _additionalData = additionalData;
        _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
        _progress = progress;
        Partition = partition.Select(g => (int[])g.Clone()).ToArray();
        PivotPoint = (double[])pivotPoint.Clone();

        // Validate partition covers all dims exactly once
        ValidatePartition(Partition, numDimensions);

        // Build dim → slide mapping
        DimToSlide = BuildDimToSlide(Partition);
    }

    /// <summary>Internal parameterless constructor for factories.</summary>
    internal ChebyshevSlider() { }

    // ------------------------------------------------------------------
    // Validation helpers
    // ------------------------------------------------------------------

    internal static void ValidatePartition(int[][] partition, int numDimensions)
    {
        var allDims = new List<int>();
        foreach (var group in partition)
            allDims.AddRange(group);
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
        PivotValue = Function(PivotPoint, _additionalData);

        int totalEvals = TotalBuildEvals;
        int fullTensor = 1;
        foreach (int n in NNodes) fullTensor *= n;

        if (verbose)
        {
            Console.WriteLine(
                $"Building {NumDimensions}D Chebyshev Slider " +
                $"({Partition.Length} slides, {totalEvals:N0} evaluations " +
                $"vs {fullTensor:N0} for full tensor)...");
        }

        Slides = new ChebyshevApproximation[Partition.Length];
        int progressOffset = 0;
        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            var group = Partition[slideIdx];
            int slideDim = group.Length;
            var slideDomain = new double[slideDim][];
            var slideNNodes = new int[slideDim];
            for (int i = 0; i < slideDim; i++)
            {
                slideDomain[i] = (double[])Domain[group[i]].Clone();
                slideNNodes[i] = NNodes[group[i]];
            }

            // Create closure that fixes non-group dims at pivot
            var grp = group;
            var pvt = PivotPoint;
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
            progressOffset += slideNNodes.Aggregate(1, (acc, n) => acc * n);
            Slides[slideIdx] = slide;

            if (verbose)
            {
                int slideEvals = 1;
                foreach (int n in slideNNodes) slideEvals *= n;
                Console.WriteLine(
                    $"  Slide {slideIdx + 1}/{Partition.Length}: " +
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

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// <summary>
    /// Evaluate the slider approximation at a point.
    /// Uses Equation 7.5: f(x) ≈ f(z) + Σᵢ [sᵢ(x_groupᵢ) - f(z)].
    /// For derivatives, only the slide containing that dimension contributes.
    /// Cross-group mixed partials are exactly zero.
    /// </summary>
    /// <param name="point">Evaluation point in the full n-dimensional space.</param>
    /// <param name="derivativeOrder">Derivative order for each dimension (0 = function value).</param>
    /// <returns>Approximated function value or derivative.</returns>
    public double Eval(double[] point, int[] derivativeOrder)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before Eval().");

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
            var group = Partition[activeSlide];
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
            for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
            {
                var group = Partition[slideIdx];
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
    /// <param name="point">Evaluation point.</param>
    /// <param name="derivativeOrders">Each inner array specifies derivative order per dimension.</param>
    /// <returns>Results for each derivative order.</returns>
    public double[] EvalMulti(double[] point, int[][] derivativeOrders)
    {
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

        var perDimBounds = Internal.Calculus.NormalizeBounds(sortedDims, bounds, Domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        // Per-dim integration widths.
        var widths = new Dictionary<int, double>();
        var boundsForDim = new Dictionary<int, (double lo, double hi)?>();
        foreach (int d in sortedDims)
        {
            var bd = perDimBounds[dimToIdx[d]];
            double a = Domain[d][0], b = Domain[d][1];
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
        var slideKinds = new (string kind, int[] kept)[Partition.Length];
        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            slideKinds[slideIdx] = Internal.Calculus.SliderPartitionIntersect(
                Partition[slideIdx], sortedDims);
        }

        // pv_new accumulator: starts as pv * vol_T (the first term of the sum).
        double pvNew = PivotValue * volT;

        // For each "full" slide: integrate over its full group with the
        // appropriate sub-interval bounds, then add contribution to pv_new.
        // Contribution = vol(T \ G_i) * (I_i - pv * vol(G_i ∩ T))
        // For "full" slides, vol(G_i ∩ T) is the product of widths over G_i.
        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            var (kind, _) = slideKinds[slideIdx];
            if (kind != "full") continue;

            var slide = Slides[slideIdx];
            var group = Partition[slideIdx];

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
                    localBoundsList.Add((slide.Domain[gi][0], slide.Domain[gi][1]));
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

        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            var (kind, kept) = slideKinds[slideIdx];
            if (kind == "full") continue; // absorbed into pv_new

            var group = Partition[slideIdx];
            var slide = Slides[slideIdx];

            ChebyshevApproximation newSlide;
            int[] newGroup;

            if (kind == "none")
            {
                // The slide passes through. Apply the partition-of-unity shift:
                //   new_tensor = vol_T * tensor + (pv_new - pv * vol_T)
                double shift = pvNew - PivotValue * volT;
                var tv = slide.TensorValues!;
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
                                (slide.Domain[localI][0], slide.Domain[localI][1]));
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
                var rtv = reduced.TensorValues!;
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
        var newDomain = survive.Select(d => (double[])Domain[d].Clone()).ToArray();
        var newNNodes = survive.Select(d => NNodes[d]).ToArray();
        var newPivotPoint = survive.Select(d => PivotPoint[d]).ToArray();
        var newPartitionArr = newPartition.ToArray();

        var result = new ChebyshevSlider();
        result.Function = null;
        result.NumDimensions = survive.Length;
        result.Domain = newDomain;
        result.NNodes = newNNodes;
        result.MaxDerivativeOrder = MaxDerivativeOrder;
        result.Partition = newPartitionArr;
        result.PivotPoint = newPivotPoint;
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
            int total = 0;
            foreach (var group in Partition)
            {
                int prod = 1;
                foreach (int d in group)
                    prod *= NNodes[d];
                total += prod;
            }
            return total;
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
                Domain = s.Domain.Select(d => (double[])d.Clone()).ToArray(),
                NNodes = (int[])s.NNodes.Clone(),
                MaxDerivativeOrder = s.MaxDerivativeOrder,
                NodeArrays = s.NodeArrays.Select(a => (double[])a.Clone()).ToArray(),
                TensorValues = (double[])s.TensorValues!.Clone(),
                Weights = s.Weights!.Select(a => (double[])a.Clone()).ToArray(),
                DiffMatrices = s.DiffMatrices!.Select(m => ChebyshevApproximation.Flatten2D(m)).ToArray(),
                DiffMatrixSizes = s.DiffMatrices!.Select(m => new[] { m.GetLength(0), m.GetLength(1) }).ToArray(),
                BuildTime = s.BuildTime,
                NEvaluations = s.NEvaluations,
            };
        }

        var state = new SliderSerializationState
        {
            NumDimensions = NumDimensions,
            Domain = Domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])NNodes.Clone(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Partition = Partition.Select(g => (int[])g.Clone()).ToArray(),
            PivotPoint = (double[])PivotPoint.Clone(),
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
    public static ChebyshevSlider Load(string path)
    {
        string json = File.ReadAllText(path);
        var state = JsonSerializer.Deserialize<SliderSerializationState>(json)
            ?? throw new InvalidOperationException("Failed to deserialize slider.");

        if (state.Type != "ChebyshevSlider")
            throw new InvalidOperationException(
                $"Expected type 'ChebyshevSlider', got '{state.Type}'");

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
                Domain = ss.Domain,
                NNodes = ss.NNodes,
                MaxDerivativeOrder = ss.MaxDerivativeOrder,
                NodeArrays = ss.NodeArrays,
                TensorValues = ss.TensorValues,
                Weights = ss.Weights,
                DiffMatrices = diffMatrices,
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
            Domain = state.Domain,
            NNodes = state.NNodes,
            MaxDerivativeOrder = state.MaxDerivativeOrder,
            Partition = state.Partition,
            PivotPoint = state.PivotPoint,
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
            Domain = source.Domain.Select(d => (double[])d.Clone()).ToArray(),
            NNodes = (int[])source.NNodes.Clone(),
            MaxDerivativeOrder = source.MaxDerivativeOrder,
            Partition = source.Partition.Select(g => (int[])g.Clone()).ToArray(),
            PivotPoint = (double[])source.PivotPoint.Clone(),
            Slides = slides,
            PivotValue = pivotValue,
            DimToSlide = new Dictionary<int, int>(source.DimToSlide),
            Built = true,
            BuildTime = 0.0,
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

        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();
        var pivotPoint = PivotPoint.ToList();
        var partition = Partition.Select(g => g.ToList()).ToList();
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
            var newNodes = BarycentricKernel.MakeNodesForDim(lo, hi, n);
            var newWeights = BarycentricKernel.ComputeBarycentricWeights(newNodes);
            var newDiffMat = BarycentricKernel.ComputeDifferentiationMatrix(newNodes, newWeights);
            var newTensor = new double[n];
            Array.Fill(newTensor, PivotValue);

            var newSlide = new ChebyshevApproximation
            {
                Function = null,
                NumDimensions = 1,
                Domain = new[] { new[] { lo, hi } },
                NNodes = new[] { n },
                MaxDerivativeOrder = MaxDerivativeOrder,
                NodeArrays = new[] { newNodes },
                Weights = new[] { newWeights },
                DiffMatrices = new[] { newDiffMat },
                TensorValues = newTensor,
                BuildTime = 0.0,
                NEvaluations = 0,
            };
            newSlide.PrecomputeTransposedDiffMatrices();

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
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Partition = newPartition,
            PivotPoint = pivotPoint.ToArray(),
            Slides = slides.ToArray(),
            PivotValue = PivotValue,
            DimToSlide = BuildDimToSlide(newPartition),
            Built = true,
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
            double lo = Domain[dimIdx][0], hi = Domain[dimIdx][1];
            if (value < lo || value > hi)
                throw new ArgumentException(
                    $"Slice value {value} for dim {dimIdx} is outside domain [{lo}, {hi}]");
        }

        var domain = Domain.Select(d => (double[])d.Clone()).ToList();
        var nNodes = NNodes.ToList();
        var pivotPoint = PivotPoint.ToList();
        var partition = Partition.Select(g => g.ToList()).ToList();
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
                        var tv = slides[i].TensorValues!;
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
            Domain = domain.ToArray(),
            NNodes = nNodes.ToArray(),
            MaxDerivativeOrder = MaxDerivativeOrder,
            Partition = newPartition,
            PivotPoint = pivotPoint.ToArray(),
            Slides = slides.ToArray(),
            PivotValue = pivotValue,
            DimToSlide = BuildDimToSlide(newPartition),
            Built = true,
        };
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
        if (!NNodes.SequenceEqual(other.NNodes))
            throw new InvalidOperationException("Node count mismatch.");
        for (int d = 0; d < NumDimensions; d++)
        {
            if (Math.Abs(Domain[d][0] - other.Domain[d][0]) > 1e-14 ||
                Math.Abs(Domain[d][1] - other.Domain[d][1]) > 1e-14)
                throw new InvalidOperationException($"Domain mismatch at dimension {d}.");
        }
        if (MaxDerivativeOrder != other.MaxDerivativeOrder)
            throw new InvalidOperationException("MaxDerivativeOrder mismatch.");

        // Slider-specific checks
        if (Partition.Length != other.Partition.Length)
            throw new ArgumentException(
                $"Partition mismatch: {FormatPartition(Partition)} vs {FormatPartition(other.Partition)}");
        for (int i = 0; i < Partition.Length; i++)
        {
            if (!Partition[i].SequenceEqual(other.Partition[i]))
                throw new ArgumentException(
                    $"Partition mismatch: {FormatPartition(Partition)} vs {FormatPartition(other.Partition)}");
        }
        if (!PivotPoint.SequenceEqual(other.PivotPoint))
            throw new ArgumentException(
                $"Pivot point mismatch: [{string.Join(", ", PivotPoint)}] vs [{string.Join(", ", other.PivotPoint)}]");
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
            var tvA = a.Slides[i].TensorValues!;
            var tvB = b.Slides[i].TensorValues!;
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
            var tvA = a.Slides[i].TensorValues!;
            var tvB = b.Slides[i].TensorValues!;
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
        var slides = new ChebyshevApproximation[a.Slides.Length];
        for (int i = 0; i < slides.Length; i++)
        {
            var tv = a.Slides[i].TensorValues!;
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
            $"slides={Partition.Length}, " +
            $"partition={FormatPartition(Partition)}, " +
            $"built={Built})";
    }

    /// <summary>Multi-line display string.</summary>
    public override string ToString()
    {
        string status = Built ? "built" : "not built";
        int totalSlideEvals = TotalBuildEvals;
        int fullTensorEvals = 1;
        foreach (int n in NNodes) fullTensorEvals *= n;

        const int maxDisplay = 6;

        // Nodes line
        string nodesStr;
        if (NumDimensions > maxDisplay)
        {
            nodesStr = "[" + string.Join(", ", NNodes.Take(maxDisplay)) + ", ...]";
        }
        else
        {
            nodesStr = "[" + string.Join(", ", NNodes) + "]";
        }

        // Domain line
        string domainStr;
        if (NumDimensions > maxDisplay)
        {
            domainStr = string.Join(" x ",
                Domain.Take(maxDisplay).Select(d => $"[{d[0]}, {d[1]}]")) + " x ...";
        }
        else
        {
            domainStr = string.Join(" x ",
                Domain.Select(d => $"[{d[0]}, {d[1]}]"));
        }

        // Pivot line
        string pivotStr;
        if (NumDimensions > maxDisplay)
        {
            pivotStr = "[" + string.Join(", ", PivotPoint.Take(maxDisplay)) + ", ...]";
        }
        else
        {
            pivotStr = "[" + string.Join(", ", PivotPoint) + "]";
        }

        // Partition line
        string partitionStr;
        if (Partition.Length > maxDisplay)
        {
            partitionStr = "[" +
                string.Join(", ", Partition.Take(maxDisplay).Select(g => "[" + string.Join(", ", g) + "]")) +
                ", ...]";
        }
        else
        {
            partitionStr = FormatPartition(Partition);
        }

        var lines = new List<string>
        {
            $"ChebyshevSlider ({NumDimensions}D, {Partition.Length} slides, {status})",
            $"  Partition: {partitionStr}",
            $"  Pivot:     {pivotStr}",
            $"  Nodes:     {nodesStr} ({totalSlideEvals:N0} vs {fullTensorEvals:N0} full tensor)",
            $"  Domain:    {domainStr}",
        };

        if (Built && Slides.Length > 0)
        {
            lines.Add($"  Error est: {ErrorEstimate():E2}");
            lines.Add("  Slides:");
            for (int i = 0; i < Partition.Length; i++)
            {
                var group = Partition[i];
                int slideEvals = 1;
                foreach (int d in group)
                    slideEvals *= NNodes[d];
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
    public int[] GetUsedNs() => (int[])NNodes.Clone();

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
        int total = 0;
        foreach (var slide in Slides) total += slide.GetNumEvaluationPoints();
        return total;
    }

    /// <summary>
    /// Flat row-major array of all slider evaluation points, expanded to full ndim using PivotPoint.
    /// Each slide's local coordinates are mapped to full-ndim space via the Partition and PivotPoint.
    /// Length is GetNumEvaluationPoints() * NumDimensions. Result is lazily built and cached.
    /// </summary>
    /// <returns>Double array of full-ndim node coordinates, flattened in row-major order.</returns>
    public double[] GetEvaluationPoints()
    {
        if (_evaluationPointsCache != null) return _evaluationPointsCache;

        int total = GetNumEvaluationPoints();
        var points = new double[total * NumDimensions];
        int offset = 0;

        for (int slideIdx = 0; slideIdx < Slides!.Length; slideIdx++)
        {
            var slide = Slides[slideIdx];
            var group = Partition[slideIdx];
            var slidePts = slide.GetEvaluationPoints();
            int slideNum = slide.GetNumEvaluationPoints();
            int gdim = group.Length;

            for (int p = 0; p < slideNum; p++)
            {
                for (int d = 0; d < NumDimensions; d++)
                    points[offset + p * NumDimensions + d] = PivotPoint[d];
                for (int gi = 0; gi < gdim; gi++)
                    points[offset + p * NumDimensions + group[gi]] = slidePts[p * gdim + gi];
            }
            offset += slideNum * NumDimensions;
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
        copy.Domain = Internal.CloneHelpers.DeepCopy(Domain)!;
        copy.NNodes = Internal.CloneHelpers.DeepCopy(NNodes)!;
        copy.Partition = Internal.CloneHelpers.DeepCopy(Partition)!;
        copy.PivotPoint = Internal.CloneHelpers.DeepCopy(PivotPoint)!;
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
