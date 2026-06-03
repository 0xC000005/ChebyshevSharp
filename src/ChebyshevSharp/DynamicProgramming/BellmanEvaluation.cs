namespace ChebyshevSharp.DynamicProgramming;

/// <summary>
/// Value, selected action, and local derivatives returned by a solved Bellman model.
/// </summary>
/// <param name="Value">Maximized value at the requested state.</param>
/// <param name="Action">Action whose fitted action-value branch is maximal at the requested state.</param>
/// <param name="FirstDerivative">First derivative of the active action-value branch.</param>
/// <param name="SecondDerivative">Second derivative of the active action-value branch.</param>
public sealed record BellmanEvaluation(
    double Value,
    BellmanAction Action,
    double FirstDerivative,
    double SecondDerivative);
