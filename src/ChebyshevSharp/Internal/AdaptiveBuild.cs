namespace ChebyshevSharp.Internal;

/// <summary>
/// Doubling-loop driver for ChebyshevApproximation auto-N construction.
/// Iteratively grows the worst-contributing auto-dim until ErrorEstimate is
/// below ErrorThreshold or every auto-dim has hit MaxN.
/// </summary>
internal static class AdaptiveBuild
{
    internal static void ValidateErrorThreshold(double? errorThreshold)
    {
        if (errorThreshold is { } value && (!double.IsFinite(value) || value <= 0.0))
            throw new ArgumentException(
                "errorThreshold must be finite and > 0.",
                nameof(errorThreshold));
    }

    /// <summary>
    /// Run the doubling loop on an approximation that has at least one null
    /// entry in OriginalNNodes. On return, the approximation is fully built
    /// (TensorValues populated, NNodes resolved to ints, NEvaluations and
    /// BuildTime accumulated across all iterations). If MaxN is reached
    /// before the threshold is satisfied, BuildWarning is set.
    /// </summary>
    public static void RunDoublingLoop(ChebyshevApproximation approx, bool verbose)
    {
        if (approx.ErrorThreshold == null)
            throw new InvalidOperationException("RunDoublingLoop requires ErrorThreshold to be set.");
        if (!approx.OriginalNNodes.Any(n => n == null))
            throw new InvalidOperationException("RunDoublingLoop requires at least one null entry in OriginalNNodes.");

        double threshold = approx.ErrorThreshold.Value;
        int maxN = approx.MaxN;
        int numDim = approx.NumDimensions;

        // Resolve: ints stay; nulls start at 3
        var current = new int[numDim];
        for (int d = 0; d < numDim; d++)
            current[d] = approx.OriginalNNodes[d] ?? 3;

        // Auto-dim indices (where OriginalNNodes[d] == null)
        var autoDims = Enumerable.Range(0, numDim).Where(i => approx.OriginalNNodes[i] == null).ToArray();

        int totalEvals = 0;
        double totalBuildTime = 0.0;
        approx.BuildWarning = null;

        while (true)
        {
            // Apply current grid
            approx.NNodesStorage = (int[])current.Clone();
            approx.NodeArraysStorage = new double[numDim][];
            for (int d = 0; d < numDim; d++)
                approx.NodeArraysStorage[d] = BarycentricKernel.MakeNodesForDim(
                    approx.DomainStorage[d][0], approx.DomainStorage[d][1], current[d]);

            approx.BuildFixedGrid(verbose: false);
            totalEvals += approx.NEvaluations;
            totalBuildTime += approx.BuildTime;

            double[] perDim = approx.ErrorEstimatePerDim();
            double err = perDim.Sum();
            // Seed cache so a public ErrorEstimate() call after build hits cache.
            approx.SetCachedErrorEstimate(err);

            if (verbose)
                Console.WriteLine($"[auto-N] nNodes=[{string.Join(", ", current)}], error={err:e3}");

            if (err <= threshold)
            {
                var validationSw = System.Diagnostics.Stopwatch.StartNew();
                var validation = ValidationErrorPerAutoDim(approx, current, autoDims);
                validationSw.Stop();
                totalEvals += validation.Evaluations;
                totalBuildTime += validationSw.Elapsed.TotalSeconds;

                double[] validationPerDim = validation.PerDim;
                double validationErr = validationPerDim.Length == 0 ? 0.0 : validationPerDim.Max();
                double combinedErr = Math.Max(err, validationErr);
                approx.SetCachedErrorEstimate(combinedErr);

                if (verbose && validationPerDim.Length > 0)
                    Console.WriteLine($"[auto-N] validation error={validationErr:e3}");

                if (combinedErr <= threshold)
                    break;

                int validationWorstDim = PickWorstGrowableDim(validationPerDim, autoDims, current, maxN);
                System.Diagnostics.Debug.Assert(
                    validationWorstDim >= 0,
                    "Validation above threshold requires at least one growable auto dimension.");
                current[validationWorstDim] = Math.Min(2 * current[validationWorstDim], maxN);
                continue;
            }

            // Pick the worst auto-dim not at maxN. Tie: lowest index first.
            int worstDim = PickWorstGrowableDim(perDim, autoDims, current, maxN);

            if (worstDim < 0)
            {
                approx.BuildWarning =
                    $"maxN={maxN} reached on all auto dims before errorThreshold={threshold:e2} satisfied " +
                    $"(last error={err:e3}). Increase maxN or relax errorThreshold.";
                break;
            }

            current[worstDim] = Math.Min(2 * current[worstDim], maxN);
        }

        approx.NEvaluations = totalEvals;
        approx.BuildTime = totalBuildTime;
    }

    private static int PickWorstGrowableDim(double[] perDim, int[] autoDims, int[] current, int maxN)
    {
        int worstDim = -1;
        double worstErr = -1.0;
        foreach (int d in autoDims)
        {
            if (current[d] >= maxN) continue;
            if (perDim[d] > worstErr)
            {
                worstErr = perDim[d];
                worstDim = d;
            }
        }
        return worstDim;
    }

    private static (double[] PerDim, int Evaluations) ValidationErrorPerAutoDim(
        ChebyshevApproximation approx, int[] current, int[] autoDims)
    {
        var perDim = new double[approx.NumDimensions];
        var function = approx.Function!;
        var derivativeOrder = new int[approx.NumDimensions];
        int evaluations = 0;

        foreach (int dim in autoDims)
        {
            int probeN = current[dim] < approx.MaxN
                ? Math.Min(2 * current[dim], approx.MaxN)
                : current[dim];

            var validationShape = (int[])current.Clone();
            validationShape[dim] = probeN;
            int total = TensorShape.RequireArrayLength(
                TensorShape.CheckedProduct(validationShape, nameof(ValidationErrorPerAutoDim)),
                nameof(ValidationErrorPerAutoDim),
                validationShape);

            double[] probeNodes = BarycentricKernel.MakeNodesForDim(
                approx.DomainStorage[dim][0], approx.DomainStorage[dim][1], probeN);

            double maxErr = 0.0;
            for (int flat = 0; flat < total; flat++)
            {
                var point = new double[approx.NumDimensions];
                int rem = flat;
                for (int d = approx.NumDimensions - 1; d >= 0; d--)
                {
                    int idx = rem % validationShape[d];
                    rem /= validationShape[d];
                    point[d] = d == dim ? probeNodes[idx] : approx.NodeArraysStorage[d][idx];
                }

                double expected = function(point, approx.AdditionalData);
                evaluations++;
                double actual = approx.VectorizedEval(point, derivativeOrder);
                double diff = Math.Abs(expected - actual);
                if (diff > maxErr)
                    maxErr = diff;
            }

            perDim[dim] = maxErr;
        }

        return (perDim, evaluations);
    }
}
