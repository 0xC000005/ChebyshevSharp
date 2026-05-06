namespace ChebyshevSharp.Internal;

internal static class EvaluationArguments
{
    internal static void ValidatePoint(double[] point, int numDimensions, string paramName = "point")
    {
        ArgumentNullException.ThrowIfNull(point, paramName);
        if (point.Length != numDimensions)
            throw new ArgumentException(
                $"{paramName} length {point.Length} must equal numDimensions={numDimensions}",
                paramName);

        for (int d = 0; d < point.Length; d++)
        {
            if (!double.IsFinite(point[d]))
                throw new ArgumentException($"{paramName}[{d}] must be finite", paramName);
        }
    }

    internal static void ValidatePoints(double[][] points, int numDimensions, string paramName = "points")
    {
        ArgumentNullException.ThrowIfNull(points, paramName);
        for (int i = 0; i < points.Length; i++)
        {
            if (points[i] is null)
                throw new ArgumentException($"{paramName}[{i}] must not be null", paramName);
            ValidatePoint(points[i], numDimensions, $"{paramName}[{i}]");
        }
    }

    internal static void ValidateDerivativeOrder(
        int[] derivativeOrder,
        int numDimensions,
        string paramName = "derivativeOrder")
    {
        ArgumentNullException.ThrowIfNull(derivativeOrder, paramName);
        if (derivativeOrder.Length != numDimensions)
            throw new ArgumentException(
                $"{paramName} length {derivativeOrder.Length} must equal numDimensions={numDimensions}",
                paramName);

        for (int d = 0; d < derivativeOrder.Length; d++)
        {
            int order = derivativeOrder[d];
            if (order < 0)
                throw new ArgumentOutOfRangeException(
                    paramName,
                    order,
                    $"{paramName}[{d}]={order} must be non-negative");
        }
    }

    internal static void ValidateDerivativeOrders(
        int[][] derivativeOrders,
        int numDimensions,
        string paramName = "derivativeOrders")
    {
        ArgumentNullException.ThrowIfNull(derivativeOrders, paramName);
        for (int i = 0; i < derivativeOrders.Length; i++)
        {
            if (derivativeOrders[i] is null)
                throw new ArgumentException($"{paramName}[{i}] must not be null", paramName);
            ValidateDerivativeOrder(derivativeOrders[i], numDimensions, $"{paramName}[{i}]");
        }
    }
}
