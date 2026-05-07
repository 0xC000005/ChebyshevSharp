namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension special points (e.g., kinks where a
/// piecewise spline must place a knot).
/// Implicit conversions to and from jagged double arrays let you pass
/// either form to constructors that accept <c>SpecialPoints</c>.
/// </summary>
public sealed record SpecialPoints
{
    private readonly double[][] _points = [];

    /// <summary>Create a special-point wrapper from per-dimension coordinates.</summary>
    /// <param name="Points">Per-dimension array of special-point coordinates.</param>
    public SpecialPoints(double[][] Points)
    {
        this.Points = Points;
    }

    /// <summary>Per-dimension array of special-point coordinates.</summary>
    public double[][] Points
    {
        get => Internal.CloneHelpers.DeepCopy(_points)!;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            _points = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>Deconstruct into the wrapped special points.</summary>
    public void Deconstruct(out double[][] Points) => Points = this.Points;

    /// <summary>Implicit conversion from jagged double array to SpecialPoints.</summary>
    public static implicit operator SpecialPoints(double[][] Points) => new(Points);

    /// <summary>Implicit conversion from SpecialPoints to jagged double array.</summary>
    public static implicit operator double[][](SpecialPoints sp)
    {
        ArgumentNullException.ThrowIfNull(sp);
        return sp.Points;
    }
}
