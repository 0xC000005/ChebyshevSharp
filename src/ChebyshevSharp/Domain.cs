namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for a multi-dimensional rectangular domain.
/// Each entry is a 2-element array <c>[lo, hi]</c>.
/// Implicit conversions to and from jagged double arrays let you pass
/// either form to constructors that accept <c>Domain</c>.
/// </summary>
public sealed record Domain
{
    private readonly double[][] _bounds = [];

    /// <summary>Create a domain wrapper from per-dimension <c>[lo, hi]</c> pairs.</summary>
    /// <param name="Bounds">Per-dimension <c>[lo, hi]</c> pairs.</param>
    public Domain(double[][] Bounds)
    {
        this.Bounds = Bounds;
    }

    /// <summary>Per-dimension <c>[lo, hi]</c> pairs.</summary>
    public double[][] Bounds
    {
        get => Internal.CloneHelpers.DeepCopy(_bounds)!;
        init
        {
            ValidateBounds(value);
            _bounds = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    private static void ValidateBounds(double[][] bounds)
    {
        ArgumentNullException.ThrowIfNull(bounds, nameof(Bounds));
        for (int d = 0; d < bounds.Length; d++)
        {
            double[] row = bounds[d] ?? throw new ArgumentException(
                $"{nameof(Bounds)}[{d}] must not be null.",
                nameof(Bounds));
            if (row.Length != 2)
                throw new ArgumentException(
                    $"{nameof(Bounds)}[{d}] must contain exactly [lo, hi].",
                    nameof(Bounds));
            if (!double.IsFinite(row[0]) || !double.IsFinite(row[1]))
                throw new ArgumentException(
                    $"{nameof(Bounds)}[{d}] endpoints must be finite.",
                    nameof(Bounds));
            if (row[0] >= row[1])
                throw new ArgumentException(
                    $"{nameof(Bounds)}[{d}] must satisfy lo < hi.",
                    nameof(Bounds));
        }
    }

    /// <summary>Deconstruct into the wrapped bounds.</summary>
    public void Deconstruct(out double[][] Bounds) => Bounds = this.Bounds;

    /// <summary>Implicit conversion from jagged double array to Domain.</summary>
    public static implicit operator Domain(double[][] Bounds) => new(Bounds);

    /// <summary>Implicit conversion from Domain to jagged double array.</summary>
    public static implicit operator double[][](Domain d)
    {
        ArgumentNullException.ThrowIfNull(d);
        return d.Bounds;
    }
}
