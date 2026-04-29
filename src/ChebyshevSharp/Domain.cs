namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for a multi-dimensional rectangular domain.
/// Each entry is a 2-element array <c>[lo, hi]</c>.
/// Implicit conversions to and from jagged double arrays let you pass
/// either form to constructors that accept <c>Domain</c>.
/// </summary>
/// <param name="Bounds">Per-dimension <c>[lo, hi]</c> pairs.</param>
public sealed record Domain(double[][] Bounds)
{
    /// <summary>Implicit conversion from jagged double array to Domain.</summary>
    public static implicit operator Domain(double[][] bounds) => new(bounds);

    /// <summary>Implicit conversion from Domain to jagged double array.</summary>
    public static implicit operator double[][](Domain d) => d.Bounds;
}
