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
            ArgumentNullException.ThrowIfNull(value);
            _bounds = Internal.CloneHelpers.DeepCopy(value)!;
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
