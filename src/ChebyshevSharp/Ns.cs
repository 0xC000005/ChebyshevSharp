namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension Chebyshev node counts.
/// Implicit conversions to and from integer arrays let you pass either
/// form to constructors that accept <c>Ns</c>.
/// </summary>
public sealed record Ns
{
    private readonly int[] _counts = [];

    /// <summary>Create a node-count wrapper from per-dimension counts.</summary>
    /// <param name="Counts">Number of nodes per dimension.</param>
    public Ns(int[] Counts)
    {
        this.Counts = Counts;
    }

    /// <summary>Number of nodes per dimension.</summary>
    public int[] Counts
    {
        get => Internal.CloneHelpers.DeepCopy(_counts)!;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            _counts = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>Deconstruct into the wrapped node counts.</summary>
    public void Deconstruct(out int[] Counts) => Counts = this.Counts;

    /// <summary>Implicit conversion from integer array to Ns.</summary>
    public static implicit operator Ns(int[] Counts) => new(Counts);

    /// <summary>Implicit conversion from Ns to integer array.</summary>
    public static implicit operator int[](Ns n)
    {
        ArgumentNullException.ThrowIfNull(n);
        return n.Counts;
    }
}
