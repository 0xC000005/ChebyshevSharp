// tests/ChebyshevSharp.Tests/TupleKeyTests.cs
using ChebyshevSharp.Internal;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TupleKeyTests
{
    [Fact]
    public void TupleKey_value_equality()
    {
        var a = new TupleKey(new[] { 1, 0, 2 });
        var b = new TupleKey(new[] { 1, 0, 2 });
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void TupleKey_distinct_for_different_values()
    {
        var a = new TupleKey(new[] { 1, 0 });
        var b = new TupleKey(new[] { 0, 1 });
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void TupleKey_Values_returns_defensive_copy()
    {
        var src = new[] { 1, 2, 3 };
        var key = new TupleKey(src);
        var values1 = key.Values;
        var values2 = key.Values;
        Assert.Equal(src, values1);
        Assert.False(ReferenceEquals(values1, values2),
            "Values should return a defensive copy, not the same array");
    }

    [Fact]
    public void TupleKey_distinct_for_different_lengths()
    {
        var a = new TupleKey(new[] { 1, 0 });
        var b = new TupleKey(new[] { 1, 0, 0 });
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void TupleKey_Equals_object_returns_false_for_non_TupleKey()
    {
        var key = new TupleKey(new[] { 1, 0 });
        Assert.False(key.Equals((object?)"not a tuple"));
        Assert.False(key.Equals((object?)null));
    }

    [Fact]
    public void TupleKey_Equals_object_returns_true_for_equal_TupleKey()
    {
        var a = new TupleKey(new[] { 1, 0 });
        object boxed = new TupleKey(new[] { 1, 0 });
        Assert.True(a.Equals(boxed));
    }

    [Fact]
    public void TupleKey_ToString_formats_as_bracketed_csv()
    {
        var key = new TupleKey(new[] { 1, 0, 2 });
        Assert.Equal("[1,0,2]", key.ToString());
    }
}
