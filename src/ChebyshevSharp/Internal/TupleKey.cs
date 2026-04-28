using System;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Value-equality wrapper around an int array for use as a dictionary key.
/// </summary>
internal readonly struct TupleKey : IEquatable<TupleKey>
{
    private readonly int[] _values;

    public TupleKey(int[] values)
    {
        _values = (int[])values.Clone();
    }

    public int[] Values => (int[])_values.Clone();

    public bool Equals(TupleKey other)
    {
        if (_values.Length != other._values.Length) return false;
        for (int i = 0; i < _values.Length; i++)
            if (_values[i] != other._values[i]) return false;
        return true;
    }

    public override bool Equals(object? obj) => obj is TupleKey o && Equals(o);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int v in _values) hash.Add(v);
        return hash.ToHashCode();
    }

    public override string ToString() => "[" + string.Join(",", _values) + "]";
}
