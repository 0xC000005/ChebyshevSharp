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
            ValidatePoints(value);
            _points = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    private static void ValidatePoints(double[][] points)
    {
        ArgumentNullException.ThrowIfNull(points, nameof(Points));
        for (int d = 0; d < points.Length; d++)
        {
            double[] row = points[d] ?? throw new ArgumentException(
                $"{nameof(Points)}[{d}] must not be null.",
                nameof(Points));
            for (int i = 0; i < row.Length; i++)
            {
                if (!double.IsFinite(row[i]))
                    throw new ArgumentException(
                        $"{nameof(Points)}[{d}][{i}] must be finite.",
                        nameof(Points));
            }
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
