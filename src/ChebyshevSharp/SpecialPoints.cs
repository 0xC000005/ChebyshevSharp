namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension special points (e.g., kinks where a
/// piecewise spline must place a knot).
/// Implicit conversions to and from jagged double arrays let you pass
/// either form to constructors that accept <c>SpecialPoints</c>.
/// </summary>
/// <param name="Points">Per-dimension array of special-point coordinates.</param>
public sealed record SpecialPoints(double[][] Points)
{
    /// <summary>Implicit conversion from jagged double array to SpecialPoints.</summary>
    public static implicit operator SpecialPoints(double[][] points) => new(points);

    /// <summary>Implicit conversion from SpecialPoints to jagged double array.</summary>
    public static implicit operator double[][](SpecialPoints sp) => sp.Points;
}
