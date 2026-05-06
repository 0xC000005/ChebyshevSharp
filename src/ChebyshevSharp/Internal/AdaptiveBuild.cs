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
            approx.NNodes = (int[])current.Clone();
            approx.NodeArrays = new double[numDim][];
            for (int d = 0; d < numDim; d++)
                approx.NodeArrays[d] = BarycentricKernel.MakeNodesForDim(
                    approx.Domain[d][0], approx.Domain[d][1], current[d]);

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
                break;

            // Pick the worst auto-dim not at maxN. Tie: lowest index first.
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
}
