namespace ChebyshevSharp.Internal;

internal static class EvaluationArguments
{
    public static void ValidatePoint(double[] point, int numDimensions, string paramName = "point")
    {
        if (point is null)
            throw new ArgumentNullException(paramName);

        if (point.Length != numDimensions)
            throw new ArgumentException(
                $"{paramName} length {point.Length} must equal numDimensions={numDimensions}.",
                paramName);

        for (int d = 0; d < point.Length; d++)
        {
            if (!double.IsFinite(point[d]))
                throw new ArgumentException($"{paramName}[{d}] must be finite.", paramName);
        }
    }

    public static void ValidatePointBatch(double[][] points, int numDimensions, string paramName = "points")
    {
        if (points is null)
            throw new ArgumentNullException(paramName);

        for (int i = 0; i < points.Length; i++)
        {
            if (points[i] is null)
                throw new ArgumentException($"{paramName}[{i}] must not be null.", paramName);

            ValidatePoint(points[i], numDimensions, $"{paramName}[{i}]");
        }
    }

    public static void ValidatePointBatch(double[,] points, int numDimensions, string paramName = "points")
    {
        if (points is null)
            throw new ArgumentNullException(paramName);

        if (points.GetLength(1) != numDimensions)
            throw new ArgumentException(
                $"{paramName} must have {numDimensions} columns; got {points.GetLength(1)}.",
                paramName);

        int rows = points.GetLength(0);
        for (int i = 0; i < rows; i++)
        {
            for (int d = 0; d < numDimensions; d++)
            {
                if (!double.IsFinite(points[i, d]))
                    throw new ArgumentException($"{paramName}[{i},{d}] must be finite.", paramName);
            }
        }
    }
}
