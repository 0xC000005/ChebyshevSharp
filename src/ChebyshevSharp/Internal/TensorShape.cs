using System;
using System.Collections.Generic;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for tensor shape products and dense-array materialization guards.
/// </summary>
internal static class TensorShape
{
    public static long CheckedProduct(IEnumerable<int> shape, string name)
    {
        int[] dims = shape as int[] ?? shape.ToArray();
        long product = 1;
        try
        {
            checked
            {
                foreach (int dim in dims)
                {
                    if (dim <= 0)
                        throw new ArgumentException(
                            $"{name} dimensions must be positive, got [{string.Join(",", dims)}].",
                            nameof(shape));
                    product *= dim;
                }
            }
        }
        catch (OverflowException ex)
        {
            throw new OverflowException(
                $"{name} shape [{string.Join(",", dims)}] exceeds long.MaxValue.", ex);
        }

        return product;
    }

    public static long ProductAtMost(IEnumerable<int> shape, long cap)
    {
        if (cap < 0) throw new ArgumentOutOfRangeException(nameof(cap), "cap must be non-negative.");

        int[] dims = shape as int[] ?? shape.ToArray();
        long product = 1;
        foreach (int dim in dims)
        {
            if (dim <= 0)
                throw new ArgumentException(
                    $"Shape dimensions must be positive, got [{string.Join(",", dims)}].",
                    nameof(shape));
            if (product > cap / dim)
                return cap == long.MaxValue ? long.MaxValue : cap + 1;
            product *= dim;
        }
        return product;
    }

    public static long CheckedByteSize(long elementCount, int bytesPerElement, string operation)
    {
        if (elementCount < 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if (bytesPerElement <= 0) throw new ArgumentOutOfRangeException(nameof(bytesPerElement));

        try
        {
            checked
            {
                return elementCount * bytesPerElement;
            }
        }
        catch (OverflowException ex)
        {
            throw new OverflowException(
                $"{operation} would require {elementCount:N0} elements * {bytesPerElement} bytes, exceeding long.MaxValue.",
                ex);
        }
    }

    public static int RequireArrayLength(long elementCount, string operation, IEnumerable<int>? shape = null)
    {
        if (elementCount < 0) throw new ArgumentOutOfRangeException(nameof(elementCount));
        if (elementCount > int.MaxValue)
        {
            string shapeText = shape is null ? string.Empty : $" for shape [{string.Join(",", shape)}]";
            throw new OverflowException(
                $"{operation} requires materializing {elementCount:N0} elements{shapeText}, exceeding int.MaxValue.");
        }

        return (int)elementCount;
    }
}
