using System.Collections.Generic;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Deep-copy primitives used by all four classes' <c>Clone()</c> methods.
/// </summary>
internal static class CloneHelpers
{
    public static double[]? DeepCopy(double[]? src) =>
        src == null ? null : (double[])src.Clone();

    public static double[][]? DeepCopy(double[][]? src)
    {
        if (src == null) return null;
        var result = new double[src.Length][];
        for (int i = 0; i < src.Length; i++)
            result[i] = (double[])src[i].Clone();
        return result;
    }

    public static double[,]? DeepCopy(double[,]? src) =>
        src == null ? null : (double[,])src.Clone();

    public static double[][,]? DeepCopy(double[][,]? src)
    {
        if (src == null) return null;
        var result = new double[src.Length][,];
        for (int i = 0; i < src.Length; i++)
            result[i] = (double[,])src[i].Clone();
        return result;
    }

    public static int[]? DeepCopy(int[]? src) =>
        src == null ? null : (int[])src.Clone();

    public static int[][]? DeepCopy(int[][]? src)
    {
        if (src == null) return null;
        var result = new int[src.Length][];
        for (int i = 0; i < src.Length; i++)
            result[i] = (int[])src[i].Clone();
        return result;
    }

    public static int?[]? DeepCopy(int?[]? src) =>
        src == null ? null : (int?[])src.Clone();

    public static (double lo, double hi)[][]? DeepCopyIntervals((double lo, double hi)[][]? src)
    {
        if (src == null) return null;
        var result = new (double, double)[src.Length][];
        for (int i = 0; i < src.Length; i++)
            result[i] = ((double, double)[])src[i].Clone();
        return result;
    }

    public static Dictionary<TupleKey, int> DeepCopy(Dictionary<TupleKey, int> src)
    {
        var result = new Dictionary<TupleKey, int>(src.Count);
        foreach (var kv in src) result[kv.Key] = kv.Value;
        return result;
    }

    public static List<int[]> DeepCopyOrders(List<int[]> src)
    {
        var result = new List<int[]>(src.Count);
        foreach (var orders in src) result.Add((int[])orders.Clone());
        return result;
    }
}
