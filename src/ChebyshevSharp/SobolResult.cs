namespace ChebyshevSharp;

/// <summary>
/// Result of <c>SobolIndices()</c> on <see cref="ChebyshevApproximation"/> and
/// <see cref="ChebyshevSpline"/>: per-dimension Sobol sensitivity indices
/// computed from spectral Chebyshev coefficients (no Monte Carlo).
/// </summary>
/// <param name="FirstOrder">First-order index per dimension. Sums to ≤ 1; sums to 1 for purely additive functions.</param>
/// <param name="TotalOrder">Total-order index per dimension. <c>FirstOrder[d] ≤ TotalOrder[d]</c> always.</param>
/// <param name="Variance">
/// Total spectral variance Σ_{α≠0} c_α² ‖T_α‖². When zero, the function is
/// constant and the indices are meaningless — callers should branch on this.
/// </param>
public sealed record SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance);
