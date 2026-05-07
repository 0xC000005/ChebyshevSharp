namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for Chebyshev arithmetic operators.
/// </summary>
internal static class Algebra
{
    internal static void ValidateFiniteScalar(double scalar, string paramName)
    {
        if (!double.IsFinite(scalar))
            throw new ArgumentOutOfRangeException(paramName, scalar, $"{paramName} must be finite.");
    }

    internal static void ValidateFiniteNonZeroDivisor(double scalar, string paramName)
    {
        ValidateFiniteScalar(scalar, paramName);
        if (scalar == 0.0)
            throw new DivideByZeroException("Cannot divide a Chebyshev interpolant by zero.");
    }

    /// <summary>
    /// Numerical equality test for two double arrays. Mirrors NumPy's
    /// <c>np.allclose(a, b, rtol, atol)</c>: <c>|a - b| &lt;= atol + rtol * |b|</c>
    /// elementwise.
    /// </summary>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/_algebra.py:46-49</c>.
    /// Used to tolerate sub-ULP floating-point drift between equivalent
    /// allocations (e.g., <c>tuple-of-tuples</c> vs <c>list-of-lists</c>
    /// in upstream Python).
    /// </remarks>
    internal static bool DoublesAllClose(double[] a, double[] b,
        double rtol = 1e-5, double atol = 1e-8)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(a[i] - b[i]);
            double bound = atol + rtol * Math.Abs(b[i]);
            if (diff > bound) return false;
        }
        return true;
    }

    /// <summary>
    /// Validate that two ChebyshevApproximation objects can be combined arithmetically.
    /// </summary>
    internal static void CheckCompatible(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException(
                $"Cannot combine {a.GetType().Name} with {b.GetType().Name}; " +
                "operands must be the same type.");

        if (a.TensorValuesStorage == null)
            throw new InvalidOperationException("Left operand is not built. Call Build() first.");
        if (b.TensorValuesStorage == null)
            throw new InvalidOperationException("Right operand is not built. Call Build() first.");

        if (a.NumDimensions != b.NumDimensions)
            throw new ArgumentException(
                $"Dimension mismatch: {a.NumDimensions} vs {b.NumDimensions}");

        int[] aNNodes = a.NNodesStorage;
        int[] bNNodes = b.NNodesStorage;
        if (!aNNodes.SequenceEqual(bNNodes))
            throw new ArgumentException(
                $"Node count mismatch: [{string.Join(", ", aNNodes)}] vs [{string.Join(", ", bNNodes)}]");

        // v0.21.1: numerical comparison on Domain[d] (was SequenceEqual = exact).
        // Tolerates sub-ULP drift between equivalent allocations.
        double[][] aDomain = a.DomainStorage;
        double[][] bDomain = b.DomainStorage;
        for (int d = 0; d < a.NumDimensions; d++)
        {
            if (!DoublesAllClose(aDomain[d], bDomain[d]))
                throw new ArgumentException(
                    $"Domain mismatch at dim {d}: " +
                    $"[{aDomain[d][0]}, {aDomain[d][1]}] vs [{bDomain[d][0]}, {bDomain[d][1]}]");
        }

        if (a.MaxDerivativeOrder != b.MaxDerivativeOrder)
            throw new ArgumentException(
                $"max_derivative_order mismatch: {a.MaxDerivativeOrder} vs {b.MaxDerivativeOrder}");
    }
}
