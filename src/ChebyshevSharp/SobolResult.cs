namespace ChebyshevSharp;

/// <summary>
/// Result of <c>SobolIndices()</c> on Chebyshev interpolants: per-dimension
/// Sobol sensitivity indices computed from spectral Chebyshev coefficients
/// (no Monte Carlo).
/// </summary>
public sealed record SobolResult
{
    private readonly double[] _firstOrder = [];
    private readonly double[] _totalOrder = [];

    /// <summary>Create a Sobol result from first-order and total-order index arrays.</summary>
    /// <param name="FirstOrder">First-order index per dimension. Sums to ≤ 1; sums to 1 for purely additive functions.</param>
    /// <param name="TotalOrder">Total-order index per dimension. <c>FirstOrder[d] ≤ TotalOrder[d]</c> always.</param>
    /// <param name="Variance">Total spectral variance.</param>
    public SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance)
    {
        this.FirstOrder = FirstOrder;
        this.TotalOrder = TotalOrder;
        this.Variance = Variance;
    }

    /// <summary>First-order index per dimension. Sums to ≤ 1; sums to 1 for purely additive functions.</summary>
    public double[] FirstOrder
    {
        get => Internal.CloneHelpers.DeepCopy(_firstOrder)!;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            _firstOrder = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>Total-order index per dimension. <c>FirstOrder[d] ≤ TotalOrder[d]</c> always.</summary>
    public double[] TotalOrder
    {
        get => Internal.CloneHelpers.DeepCopy(_totalOrder)!;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            _totalOrder = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>
    /// Total spectral variance Σ_{α≠0} c_α² ‖T_α‖². When zero or at numerical
    /// noise level, the function is effectively constant and the indices are
    /// meaningless.
    /// </summary>
    public double Variance { get; init; }

    /// <summary>Deconstruct into first-order indices, total-order indices, and variance.</summary>
    public void Deconstruct(out double[] FirstOrder, out double[] TotalOrder, out double Variance)
    {
        FirstOrder = this.FirstOrder;
        TotalOrder = this.TotalOrder;
        Variance = this.Variance;
    }
}
