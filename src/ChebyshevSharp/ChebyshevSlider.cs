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
    public ChebyshevSlider(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int[][] partition,
        double[] pivotPoint,
        int maxDerivativeOrder = 2)
    {
        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        NNodes = (int[])nNodes.Clone();
        MaxDerivativeOrder = maxDerivativeOrder;
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
        PivotValue = Function(PivotPoint, null);

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

            var slide = new ChebyshevApproximation(
                slideFunc, slideDim, slideDomain, slideNNodes,
                maxDerivativeOrder: MaxDerivativeOrder);
            slide.Build(verbose: false);
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

        return new ChebyshevSlider
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
