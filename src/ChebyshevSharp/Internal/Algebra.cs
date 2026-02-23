namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for Chebyshev arithmetic operators.
/// </summary>
internal static class Algebra
{
    /// <summary>
    /// Validate that two ChebyshevApproximation objects can be combined arithmetically.
    /// </summary>
    internal static void CheckCompatible(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException(
                $"Cannot combine {a.GetType().Name} with {b.GetType().Name}; " +
                "operands must be the same type.");

        if (a.TensorValues == null)
            throw new InvalidOperationException("Left operand is not built. Call Build() first.");
        if (b.TensorValues == null)
            throw new InvalidOperationException("Right operand is not built. Call Build() first.");

        if (a.NumDimensions != b.NumDimensions)
            throw new ArgumentException(
                $"Dimension mismatch: {a.NumDimensions} vs {b.NumDimensions}");

        if (!a.NNodes.SequenceEqual(b.NNodes))
            throw new ArgumentException(
                $"Node count mismatch: [{string.Join(", ", a.NNodes)}] vs [{string.Join(", ", b.NNodes)}]");

        for (int d = 0; d < a.NumDimensions; d++)
        {
            if (!a.Domain[d].SequenceEqual(b.Domain[d]))
                throw new ArgumentException(
                    $"Domain mismatch at dim {d}");
        }

        if (a.MaxDerivativeOrder != b.MaxDerivativeOrder)
            throw new ArgumentException(
                $"max_derivative_order mismatch: {a.MaxDerivativeOrder} vs {b.MaxDerivativeOrder}");
    }
}
