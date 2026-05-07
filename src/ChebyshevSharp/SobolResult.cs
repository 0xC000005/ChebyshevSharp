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
    private readonly double _variance;

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
            ValidateFiniteVector(value, nameof(FirstOrder));
            ValidateMatchingLength(value.Length, _totalOrder.Length, nameof(FirstOrder), nameof(TotalOrder));
            _firstOrder = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>Total-order index per dimension. <c>FirstOrder[d] ≤ TotalOrder[d]</c> always.</summary>
    public double[] TotalOrder
    {
        get => Internal.CloneHelpers.DeepCopy(_totalOrder)!;
        init
        {
            ValidateFiniteVector(value, nameof(TotalOrder));
            ValidateMatchingLength(_firstOrder.Length, value.Length, nameof(FirstOrder), nameof(TotalOrder));
            _totalOrder = Internal.CloneHelpers.DeepCopy(value)!;
        }
    }

    /// <summary>
    /// Total spectral variance Σ_{α≠0} c_α² ‖T_α‖². When zero or at numerical
    /// noise level, the function is effectively constant and the indices are
    /// meaningless.
    /// </summary>
    public double Variance
    {
        get => _variance;
        init
        {
            if (!double.IsFinite(value))
                throw new ArgumentException($"{nameof(Variance)} must be finite.", nameof(Variance));
            if (value < 0.0)
                throw new ArgumentOutOfRangeException(
                    nameof(Variance),
                    value,
                    $"{nameof(Variance)} must be non-negative.");
            _variance = value;
        }
    }

    private static void ValidateFiniteVector(double[] values, string paramName)
    {
        ArgumentNullException.ThrowIfNull(values, paramName);
        if (values.Length == 0)
            throw new ArgumentException($"{paramName} must not be empty.", paramName);
        for (int i = 0; i < values.Length; i++)
        {
            if (!double.IsFinite(values[i]))
                throw new ArgumentException($"{paramName}[{i}] must be finite.", paramName);
        }
    }

    private static void ValidateMatchingLength(
        int firstLength,
        int totalLength,
        string firstName,
        string totalName)
    {
        if (firstLength == 0 || totalLength == 0)
            return;
        if (firstLength != totalLength)
            throw new ArgumentException(
                $"{firstName} and {totalName} must have the same length.");
    }

    /// <summary>Deconstruct into first-order indices, total-order indices, and variance.</summary>
    public void Deconstruct(out double[] FirstOrder, out double[] TotalOrder, out double Variance)
    {
        FirstOrder = this.FirstOrder;
        TotalOrder = this.TotalOrder;
        Variance = this.Variance;
    }
}
