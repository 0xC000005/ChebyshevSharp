namespace ChebyshevSharp.Internal;

/// <summary>
/// Wraps an <see cref="IProgress{T}"/> and adds a fixed offset to every reported value.
/// Used by Spline and Slider to accumulate progress across multiple pieces/slides.
/// </summary>
internal sealed class OffsetProgress : IProgress<int>
{
    private readonly IProgress<int> _inner;
    private readonly int _offset;

    internal OffsetProgress(IProgress<int> inner, int offset)
    {
        _inner = inner;
        _offset = offset;
    }

    public void Report(int value) => _inner.Report(_offset + value);
}

/// <summary>
/// Build-time helpers for parallel function evaluation and progress reporting.
/// Phase 6: <c>nWorkers</c> ctor kwarg + <see cref="System.IProgress{T}"/> wiring.
/// Mirrors PyChebyshev <c>_parallel.py</c> (v0.19.0).
/// </summary>
internal static class ParallelBuild
{
    /// <summary>
    /// Validate and normalize the user-supplied <c>nWorkers</c> ctor kwarg.
    /// </summary>
    /// <param name="nWorkers">Raw user value: null (sequential), -1 (auto), or positive int.</param>
    /// <returns>null for sequential; positive int for the effective worker count.</returns>
    /// <exception cref="ArgumentException">On 0 or value &lt; -1.</exception>
    internal static int? NormalizeNWorkers(int? nWorkers)
    {
        if (nWorkers is null) return null;
        if (nWorkers == 0 || nWorkers < -1)
            throw new ArgumentException(
                $"nWorkers={nWorkers} not allowed (use null for sequential, " +
                "-1 for ProcessorCount, or a positive int).",
                nameof(nWorkers));
        return nWorkers == -1 ? Environment.ProcessorCount : nWorkers;
    }

    /// <summary>
    /// Evaluate <paramref name="function"/> at every <paramref name="points"/> entry,
    /// optionally in parallel via <see cref="System.Threading.Tasks.Parallel.For(int, int, System.Action{int})"/>,
    /// optionally reporting cumulative count to <paramref name="progress"/> after each
    /// successful evaluation.
    /// </summary>
    /// <param name="function">Picklable-equivalent in Python; here, must be thread-safe when <paramref name="effectiveWorkers"/> is non-null.</param>
    /// <param name="points">Flat array of input points.</param>
    /// <param name="additionalData">User context threaded as the second arg of <paramref name="function"/>.</param>
    /// <param name="effectiveWorkers">Already normalized via <see cref="NormalizeNWorkers"/>; null = sequential.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative count 1..N.</param>
    /// <returns>Result array of length <c>points.Length</c>.</returns>
    internal static double[] EvaluateInParallel(
        Func<double[], object?, double> function,
        double[][] points,
        object? additionalData,
        int? effectiveWorkers,
        IProgress<int>? progress)
    {
        var results = new double[points.Length];
        if (effectiveWorkers is null or 1)
        {
            for (int i = 0; i < points.Length; i++)
            {
                results[i] = function(points[i], additionalData);
                progress?.Report(i + 1);
            }
            return results;
        }

        int done = 0;
        var opts = new System.Threading.Tasks.ParallelOptions
        {
            MaxDegreeOfParallelism = effectiveWorkers.Value,
        };
        System.Threading.Tasks.Parallel.For(0, points.Length, opts, i =>
        {
            results[i] = function(points[i], additionalData);
            int n = System.Threading.Interlocked.Increment(ref done);
            progress?.Report(n);
        });
        return results;
    }
}
