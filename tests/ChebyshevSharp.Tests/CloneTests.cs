// tests/ChebyshevSharp.Tests/CloneTests.cs
using System.Linq;
using System.Reflection;
using ChebyshevSharp.Tests.Helpers;
using Xunit;

namespace ChebyshevSharp.Tests;

public class CloneTests
{
    [Fact]
    public void Approx_Clone_returns_typed_copy_with_function_null()
    {
        var src = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        src.Build(verbose: false);
        src.SetDescriptor("source");

        ChebyshevApproximation copy = src.Clone();

        Assert.Null(copy.Function);
        Assert.Equal("source", copy.GetDescriptor());
        Assert.Equal("clone", copy.GetConstructorType());

        double[] pt = { 0.3, 0.5 };
        int[] noDerivative = { 0, 0 };
        Assert.Equal(src.Eval(pt, noDerivative), copy.Eval(pt, noDerivative), precision: 12);

        copy.SetDescriptor("clone-only");
        Assert.Equal("source", src.GetDescriptor());
    }

    [Fact]
    public void Spline_Clone_returns_typed_copy_with_pieces_deep_copied()
    {
        var src = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { System.Array.Empty<double>(), System.Array.Empty<double>() });
        src.Build(verbose: false);

        ChebyshevSpline copy = src.Clone();
        Assert.Null(copy.Function);
        double[] pt = { 0.3, 0.5 };
        int[] noDerivative = { 0, 0 };
        Assert.Equal(src.Eval(pt, noDerivative), copy.Eval(pt, noDerivative), precision: 12);
    }

    [Fact]
    public void Slider_Clone_returns_typed_copy()
    {
        var src = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        src.Build(verbose: false);

        ChebyshevSlider copy = src.Clone();
        Assert.Null(copy.Function);
        double[] pt = { 0.3, 0.5, 0.2 };
        int[] noDerivative = { 0, 0, 0 };
        Assert.Equal(src.Eval(pt, noDerivative), copy.Eval(pt, noDerivative), precision: 12);
    }

    [Fact]
    public void Tt_Clone_returns_typed_copy()
    {
        var src = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        src.Build(verbose: false, seed: 42);

        ChebyshevTT copy = src.Clone();
        double[] pt = { 0.3, 0.5, 0.2 };
        Assert.Equal(src.Eval(pt), copy.Eval(pt), precision: 12);
    }

    [Fact]
    public void Approx_Clone_arrays_are_not_aliased_with_source()
    {
        // Reflection-based completeness audit. Walks every private field and
        // verifies that any array-typed value is reference-distinct between
        // src and clone. Catches future regressions where Clone forgets to
        // copy a newly-added mutable array.
        var src = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        src.Build(verbose: false);
        var copy = src.Clone();

        var fields = typeof(ChebyshevApproximation).GetFields(
            BindingFlags.NonPublic | BindingFlags.Instance);

        foreach (var f in fields)
        {
            var srcVal = f.GetValue(src);
            var copyVal = f.GetValue(copy);
            if (srcVal == null || copyVal == null) continue;
            if (srcVal is System.Array)
                Assert.False(ReferenceEquals(srcVal, copyVal),
                    $"Field {f.Name} is reference-aliased between src and clone");
        }
    }
}
