using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for Chebyshev calculus operations (integration, roots, optimization).
/// </summary>
internal static class Calculus
{
    /// <summary>
    /// Compute Fejér-1 quadrature weights for n Type I Chebyshev nodes.
    /// Returns weights in ascending node order on [-1, 1].
    /// Uses DCT-III via Waldvogel (2006).
    /// </summary>
    internal static double[] ComputeFejer1Weights(int n)
    {
        // Integration moments: I_k = int_{-1}^{1} T_k(x) dx
        double[] moments = new double[n];
        for (int k = 0; k < n; k += 2)
            moments[k] = 2.0 / (1.0 - (double)k * k);

        // DCT-III: y[j] = x[0] + 2*sum_{k>=1} x[k]*cos(pi*k*(2j+1)/(2n))
        double[] weightsDesc = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = moments[0];
            for (int k = 1; k < n; k++)
                sum += 2.0 * moments[k] * Math.Cos(Math.PI * k * (2 * j + 1) / (2.0 * n));
            weightsDesc[j] = sum / n;
        }

        // Reverse to ascending order
        double[] weights = new double[n];
        for (int i = 0; i < n; i++)
            weights[i] = weightsDesc[n - 1 - i];

        return weights;
    }

    /// <summary>
    /// Compute quadrature weights for a sub-interval [tLo, tHi] within [-1, 1].
    /// </summary>
    internal static double[] ComputeSubIntervalWeights(int n, double tLo, double tHi)
    {
        // Compute T_k(tLo) and T_k(tHi) via recurrence
        double[] tLoArr = new double[n + 1];
        double[] tHiArr = new double[n + 1];
        tLoArr[0] = 1.0; tLoArr[1] = tLo;
        tHiArr[0] = 1.0; tHiArr[1] = tHi;
        for (int k = 2; k <= n; k++)
        {
            tLoArr[k] = 2.0 * tLo * tLoArr[k - 1] - tLoArr[k - 2];
            tHiArr[k] = 2.0 * tHi * tHiArr[k - 1] - tHiArr[k - 2];
        }

        // Sub-interval moments
        double[] moments = new double[n];
        moments[0] = tHi - tLo;
        if (n > 1)
            moments[1] = (tHi * tHi - tLo * tLo) / 2.0;
        for (int k = 2; k < n; k++)
        {
            moments[k] = 0.5 * (
                (tHiArr[k + 1] - tLoArr[k + 1]) / (k + 1)
                - (tHiArr[k - 1] - tLoArr[k - 1]) / (k - 1)
            );
        }

        // DCT-III -> weights (descending), then reverse
        double[] weightsDesc = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = moments[0];
            for (int k = 1; k < n; k++)
                sum += 2.0 * moments[k] * Math.Cos(Math.PI * k * (2 * j + 1) / (2.0 * n));
            weightsDesc[j] = sum / n;
        }

        double[] weights = new double[n];
        for (int i = 0; i < n; i++)
            weights[i] = weightsDesc[n - 1 - i];

        return weights;
    }

    /// <summary>
    /// Normalize bounds parameter for integrate().
    /// </summary>
    internal static (double lo, double hi)?[] NormalizeBounds(
        int[] dims, (double lo, double hi)[]? bounds, double[][] domain)
    {
        if (bounds == null)
        {
            var result = new (double lo, double hi)?[dims.Length];
            return result; // All null = full domain
        }

        if (bounds.Length != dims.Length)
            throw new ArgumentException(
                $"bounds length {bounds.Length} != dims length {dims.Length}");

        var normalized = new (double lo, double hi)?[dims.Length];
        for (int i = 0; i < bounds.Length; i++)
        {
            var bd = bounds[i];
            if (bd.lo > bd.hi)
                throw new ArgumentException($"bounds lo={bd.lo} > hi={bd.hi} for dim {dims[i]}");

            int d = dims[i];
            double domLo = domain[d][0];
            double domHi = domain[d][1];
            if (bd.lo < domLo - 1e-14 || bd.hi > domHi + 1e-14)
                throw new ArgumentException(
                    $"bounds ({bd.lo}, {bd.hi}) outside domain [{domLo}, {domHi}] for dim {d}");

            double lo = Math.Max(bd.lo, domLo);
            double hi = Math.Min(bd.hi, domHi);
            normalized[i] = (lo, hi);
        }
        return normalized;
    }

    /// <summary>
    /// Find all real roots of a 1-D Chebyshev interpolant within its domain.
    /// Uses companion matrix eigenvalue method.
    /// </summary>
    internal static double[] Roots1D(double[] values, double[] domain)
    {
        double[] coeffs = BarycentricKernel.ChebyshevCoefficients1D(values);
        double[] rawRoots = ChebyshevRoots(coeffs);

        double tol = 1e-10;
        var realRoots = new List<double>();
        foreach (double root in rawRoots)
        {
            double t = root;
            if (-1.0 - tol <= t && t <= 1.0 + tol)
            {
                t = Math.Clamp(t, -1.0, 1.0);
                realRoots.Add(t);
            }
        }

        if (realRoots.Count == 0)
            return Array.Empty<double>();

        // Map from [-1, 1] to [a, b]
        double a = domain[0], b = domain[1];
        double[] physical = new double[realRoots.Count];
        for (int i = 0; i < realRoots.Count; i++)
            physical[i] = 0.5 * (a + b) + 0.5 * (b - a) * realRoots[i];

        Array.Sort(physical);

        // Deduplicate
        if (physical.Length > 1)
        {
            var deduped = new List<double> { physical[0] };
            for (int i = 1; i < physical.Length; i++)
            {
                if (Math.Abs(physical[i] - physical[i - 1]) > 1e-10 * (b - a + 1))
                    deduped.Add(physical[i]);
            }
            physical = deduped.ToArray();
        }

        return physical;
    }

    /// <summary>
    /// Find min or max of a 1-D Chebyshev interpolant.
    /// </summary>
    internal static (double value, double location) Optimize1D(
        double[] values, double[] nodes, double[] baryWeights,
        double[,] diffMatrix, double[] domain, string mode)
    {
        int n = values.Length;

        // Derivative values at nodes: D @ values
        double[] derivValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += diffMatrix[i, j] * values[j];
            derivValues[i] = sum;
        }

        // Critical points: roots of the derivative
        double[] critical = Roots1D(derivValues, domain);

        // Candidates: critical points + domain endpoints
        double a = domain[0], b = domain[1];
        var candidates = new List<double> { a };
        candidates.AddRange(critical);
        candidates.Add(b);

        // Evaluate original function at all candidates
        double[] vals = new double[candidates.Count];
        for (int i = 0; i < candidates.Count; i++)
            vals[i] = BarycentricKernel.BarycentricInterpolate(candidates[i], nodes, values, baryWeights);

        int idx;
        if (mode == "min")
        {
            idx = 0;
            for (int i = 1; i < vals.Length; i++)
                if (vals[i] < vals[idx]) idx = i;
        }
        else
        {
            idx = 0;
            for (int i = 1; i < vals.Length; i++)
                if (vals[i] > vals[idx]) idx = i;
        }

        return (vals[idx], candidates[idx]);
    }

    /// <summary>
    /// Validate arguments for roots/minimize/maximize.
    /// </summary>
    internal static (int dim, (int dimIndex, double value)[] sliceParams) ValidateCalculusArgs(
        int ndim, int? dim, Dictionary<int, double>? fixedDims, double[][] domain)
    {
        if (ndim == 1)
        {
            int d = dim ?? 0;
            if (d != 0)
                throw new ArgumentException($"dim must be 0 for 1-D interpolant, got {d}");
            if (fixedDims != null && fixedDims.Count > 0)
                throw new ArgumentException("fixed must be empty for 1-D interpolant");
            return (d, Array.Empty<(int, double)>());
        }

        if (dim == null)
            throw new ArgumentException("dim is required for multi-D interpolant");
        if (dim.Value < 0 || dim.Value >= ndim)
            throw new ArgumentException($"dim {dim.Value} out of range [0, {ndim - 1}]");

        fixedDims ??= new Dictionary<int, double>();
        var expected = new HashSet<int>(Enumerable.Range(0, ndim));
        expected.Remove(dim.Value);
        var provided = new HashSet<int>(fixedDims.Keys);
        if (!provided.SetEquals(expected))
        {
            var missing = new HashSet<int>(expected);
            missing.ExceptWith(provided);
            throw new ArgumentException($"fixed must specify all dims except {dim.Value}; missing {{{string.Join(", ", missing)}}}");
        }

        var sliceParams = new List<(int, double)>();
        foreach (var kvp in fixedDims)
        {
            double lo = domain[kvp.Key][0];
            double hi = domain[kvp.Key][1];
            if (kvp.Value < lo || kvp.Value > hi)
                throw new ArgumentException(
                    $"Fixed value {kvp.Value} for dim {kvp.Key} outside domain [{lo}, {hi}]");
            sliceParams.Add((kvp.Key, kvp.Value));
        }

        return (dim.Value, sliceParams.ToArray());
    }

    /// <summary>
    /// Find roots of a Chebyshev polynomial given its coefficients.
    /// Uses the colleague matrix eigenvalue method with MathNet.Numerics.
    /// Matches numpy.polynomial.chebyshev.chebroots / chebcompanion exactly.
    /// </summary>
    private static double[] ChebyshevRoots(double[] coeffs)
    {
        // Trim trailing coefficients that are negligible relative to the largest.
        // This prevents noise from differentiation creating spurious high-degree
        // terms that reduce root-finding precision.
        int n = coeffs.Length;
        double maxCoeff = 0;
        for (int i = 0; i < n; i++)
            maxCoeff = Math.Max(maxCoeff, Math.Abs(coeffs[i]));
        double trimThreshold = Math.Max(1e-15, 1e-13 * maxCoeff);
        while (n > 1 && Math.Abs(coeffs[n - 1]) < trimThreshold)
            n--;

        if (n <= 1)
            return Array.Empty<double>();

        if (n == 2)
        {
            // c0 + c1*T1(x) = 0 => c0 + c1*x = 0 => x = -c0/c1
            return new[] { -coeffs[0] / coeffs[1] };
        }

        // Build companion (colleague) matrix for Chebyshev polynomials.
        // Standard form: sub-diagonal [1, 0.5, ...], super-diagonal [0.5, ...],
        // last column correction -c[i]/(2*c[n]).
        int m = n - 1;
        double[,] data = new double[m, m];

        // Sub-diagonal
        data[1, 0] = 1.0;
        for (int i = 2; i < m; i++)
            data[i, i - 1] = 0.5;

        // Super-diagonal
        for (int i = 0; i < m - 1; i++)
            data[i, i + 1] = 0.5;

        // Last column correction
        double cn = coeffs[n - 1];
        for (int i = 0; i < m; i++)
            data[i, m - 1] -= coeffs[i] / (2.0 * cn);

        // Compute eigenvalues using MathNet.Numerics
        var matrix = DenseMatrix.OfArray(data);
        var evd = matrix.Evd();
        var complexEigenvalues = evd.EigenValues;

        // Keep only real eigenvalues (small imaginary part)
        const double imagTol = 1e-10;
        var realEigs = new List<double>();
        for (int i = 0; i < complexEigenvalues.Count; i++)
        {
            if (Math.Abs(complexEigenvalues[i].Imaginary) < imagTol)
                realEigs.Add(complexEigenvalues[i].Real);
        }

        return realEigs.ToArray();
    }
}
