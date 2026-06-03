namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Discrete action candidate used by finite-horizon Bellman collocation.
/// </summary>
/// <param name="Index">Stable action index.</param>
/// <param name="Name">Human-readable action label.</param>
/// <param name="Value">Optional scalar action coordinate, such as a savings rate or portfolio weight.</param>
public sealed record BellmanAction(int Index, string Name, double Value = 0.0);
