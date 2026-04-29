namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension Chebyshev node counts.
/// Implicit conversions to and from integer arrays let you pass either
/// form to constructors that accept <c>Ns</c>.
/// </summary>
/// <param name="Counts">Number of nodes per dimension.</param>
public sealed record Ns(int[] Counts)
{
    /// <summary>Implicit conversion from integer array to Ns.</summary>
    public static implicit operator Ns(int[] counts) => new(counts);

    /// <summary>Implicit conversion from Ns to integer array.</summary>
    public static implicit operator int[](Ns n) => n.Counts;
}
