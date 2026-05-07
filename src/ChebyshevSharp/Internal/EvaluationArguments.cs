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

    internal static void ValidatePointInDomain(
        double[] point,
        int numDimensions,
        double[][] domain,
        string paramName = "point")
    {
        ValidatePoint(point, numDimensions, paramName);
        ValidateDomain(domain, numDimensions, nameof(domain));

        for (int d = 0; d < point.Length; d++)
        {
            double lo = domain[d][0];
            double hi = domain[d][1];
            double value = point[d];
            if (value < lo || value > hi)
                throw new ArgumentOutOfRangeException(
                    paramName,
                    value,
                    $"{paramName}[{d}]={value} must be within domain[{d}]=[{lo}, {hi}]");
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

    internal static void ValidatePointsInDomain(
        double[][] points,
        int numDimensions,
        double[][] domain,
        string paramName = "points")
    {
        ArgumentNullException.ThrowIfNull(points, paramName);
        ValidateDomain(domain, numDimensions, nameof(domain));
        for (int i = 0; i < points.Length; i++)
        {
            if (points[i] is null)
                throw new ArgumentException($"{paramName}[{i}] must not be null", paramName);
            ValidatePointInDomain(points[i], numDimensions, domain, $"{paramName}[{i}]");
        }
    }

    internal static void ValidatePointBatch(double[,] points, int numDimensions, string paramName = "points")
    {
        ArgumentNullException.ThrowIfNull(points, paramName);
        if (points.GetLength(1) != numDimensions)
            throw new ArgumentException(
                $"{paramName} must have {numDimensions} columns; got {points.GetLength(1)}",
                paramName);

        int rows = points.GetLength(0);
        for (int i = 0; i < rows; i++)
        {
            for (int d = 0; d < numDimensions; d++)
            {
                if (!double.IsFinite(points[i, d]))
                    throw new ArgumentException($"{paramName}[{i},{d}] must be finite", paramName);
            }
        }
    }

    internal static void ValidatePointBatchInDomain(
        double[,] points,
        int numDimensions,
        double[][] domain,
        string paramName = "points")
    {
        ValidatePointBatch(points, numDimensions, paramName);
        ValidateDomain(domain, numDimensions, nameof(domain));

        int rows = points.GetLength(0);
        for (int i = 0; i < rows; i++)
        {
            for (int d = 0; d < numDimensions; d++)
            {
                double lo = domain[d][0];
                double hi = domain[d][1];
                double value = points[i, d];
                if (value < lo || value > hi)
                    throw new ArgumentOutOfRangeException(
                        paramName,
                        value,
                        $"{paramName}[{i},{d}]={value} must be within domain[{d}]=[{lo}, {hi}]");
            }
        }
    }

    internal static void ValidateDerivativeOrder(
        int[] derivativeOrder,
        int numDimensions,
        string paramName = "derivativeOrder",
        int? maxDerivativeOrder = null)
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
            if (maxDerivativeOrder is { } maxOrder && order > maxOrder)
                throw new ArgumentOutOfRangeException(
                    paramName,
                    order,
                    $"{paramName}[{d}]={order} is not supported; maximum derivative order is {maxOrder}");
        }
    }

    internal static void ValidateDerivativeOrders(
        int[][] derivativeOrders,
        int numDimensions,
        string paramName = "derivativeOrders",
        int? maxDerivativeOrder = null)
    {
        ArgumentNullException.ThrowIfNull(derivativeOrders, paramName);
        for (int i = 0; i < derivativeOrders.Length; i++)
        {
            if (derivativeOrders[i] is null)
                throw new ArgumentException($"{paramName}[{i}] must not be null", paramName);
            ValidateDerivativeOrder(
                derivativeOrders[i],
                numDimensions,
                $"{paramName}[{i}]",
                maxDerivativeOrder);
        }
    }

    private static void ValidateDomain(double[][] domain, int numDimensions, string paramName)
    {
        ArgumentNullException.ThrowIfNull(domain, paramName);
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"{paramName} length {domain.Length} must equal numDimensions={numDimensions}",
                paramName);

        for (int d = 0; d < domain.Length; d++)
        {
            if (domain[d] is null)
                throw new ArgumentException($"{paramName}[{d}] must not be null", paramName);
            if (domain[d].Length != 2)
                throw new ArgumentException($"{paramName}[{d}] must contain [lo, hi]", paramName);
        }
    }
}
