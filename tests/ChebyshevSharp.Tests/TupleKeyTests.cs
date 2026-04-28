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
}
