using Xunit;

namespace ChebyshevSharp.Tests;

public class SplinePublicStateOwnershipTests
{
    private static ChebyshevSpline BuildSpline()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0] * p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 7 },
            knots: new[] { new[] { 0.0 } });
        spline.Build(verbose: false);
        return spline;
    }

    [Fact]
    public void Domain_NNodes_and_Knots_properties_return_snapshots()
    {
        var spline = BuildSpline();

        double[][] domain = spline.Domain;
        int[] nNodes = spline.NNodes;
        double[][] knots = spline.Knots;
        domain[0][0] = 10.0;
        nNodes[0] = 999;
        knots[0][0] = 0.75;

        Assert.Equal(-1.0, spline.Domain[0][0]);
        Assert.Equal(new[] { 7 }, spline.NNodes);
        Assert.Equal(0.0, spline.Knots[0][0]);
    }

    [Fact]
    public void Mutating_public_snapshots_does_not_change_eval_contract()
    {
        var spline = BuildSpline();
        double valueBefore = spline.Eval(new[] { -0.5 }, new[] { 0 });

        double[][] domain = spline.Domain;
        double[][] knots = spline.Knots;
        domain[0][0] = 0.25;
        knots[0][0] = 0.75;

        Assert.Equal(valueBefore, spline.Eval(new[] { -0.5 }, new[] { 0 }), precision: 12);
        Assert.Throws<ArgumentException>(() => spline.Eval(new[] { 0.0 }, new[] { 1 }));
    }

    [Fact]
    public void Internal_storage_accessors_remain_live_while_public_properties_snapshot()
    {
        var spline = new ChebyshevSpline();
        var domain = new[] { new[] { -1.0, 1.0 } };
        var nNodes = new[] { 5 };
        var knots = new[] { new[] { 0.0 } };

        spline.Domain = domain;
        spline.NNodes = nNodes;
        spline.Knots = knots;

        Assert.Same(domain, spline.DomainStorage);
        Assert.Same(nNodes, spline.NNodesStorage);
        Assert.Same(knots, spline.KnotsStorage);
        Assert.NotSame(domain, spline.Domain);
        Assert.NotSame(nNodes, spline.NNodes);
        Assert.NotSame(knots, spline.Knots);

        var replacementDomain = new[] { new[] { 0.0, 2.0 } };
        var replacementNNodes = new[] { 7 };
        var replacementKnots = new[] { new[] { 1.0 } };

        spline.DomainStorage = replacementDomain;
        spline.NNodesStorage = replacementNNodes;
        spline.KnotsStorage = replacementKnots;

        Assert.Same(replacementDomain, spline.DomainStorage);
        Assert.Same(replacementNNodes, spline.NNodesStorage);
        Assert.Same(replacementKnots, spline.KnotsStorage);
    }

    [Fact]
    public void Internal_setters_accept_null_without_exposing_mutable_empty_state()
    {
        var spline = new ChebyshevSpline();

        spline.Domain = null!;
        spline.NNodes = null!;
        spline.Knots = null!;

        Assert.Empty(spline.Domain);
        Assert.Empty(spline.NNodes);
        Assert.Empty(spline.Knots);
    }

    [Fact]
    public void TotalBuildEvals_handles_unresolved_auto_n_placeholders()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: null,
            knots: new[] { Array.Empty<double>() },
            errorThreshold: 1e-3);

        Assert.Equal(0, spline.TotalBuildEvals);
    }

    [Fact]
    public void GetSpecialPoints_returns_snapshots_for_interior_knots()
    {
        var spline = new ChebyshevSpline();

        spline.KnotsStorage = null!;
        Assert.Null(spline.GetSpecialPoints());

        spline.KnotsStorage = new[] { Array.Empty<double>() };
        Assert.Null(spline.GetSpecialPoints());

        var knots = new[] { new[] { 0.0 } };
        spline.KnotsStorage = knots;
        double[][] specialPoints = spline.GetSpecialPoints()!;
        specialPoints[0][0] = 0.75;

        Assert.NotSame(knots, specialPoints);
        Assert.Equal(0.0, spline.KnotsStorage[0][0]);
    }

    [Fact]
    public void Incompatible_knot_storage_length_is_rejected_by_spline_arithmetic()
    {
        var left = BuildSpline();
        var right = BuildSpline();
        right.KnotsStorage = Array.Empty<double[]>();

        Assert.Throws<ArgumentException>(() => left + right);
    }

    [Fact]
    public void SobolIndices_requires_built_piece_state()
    {
        var spline = new ChebyshevSpline();

        Assert.Throws<InvalidOperationException>(() => spline.SobolIndices());
    }

    [Fact]
    public void SetOriginalFunctionValues_uses_flat_node_storage()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 3 },
            knots: new[] { new[] { 0.0 } },
            deferBuild: true);

        spline.SetOriginalFunctionValues(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        Assert.True(spline.Built);
        Assert.Equal(new[] { 3 }, spline.Pieces[0]!.NNodesStorage);
        Assert.Equal(new[] { 3 }, spline.Pieces[1]!.NNodesStorage);
    }

    [Fact]
    public void SetOriginalFunctionValues_uses_nested_node_storage()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 3, 4 } },
            knots: new[] { new[] { 0.0 } },
            deferBuild: true);

        spline.SetOriginalFunctionValues(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        Assert.True(spline.Built);
        Assert.Equal(new[] { 3 }, spline.Pieces[0]!.NNodesStorage);
        Assert.Equal(new[] { 4 }, spline.Pieces[1]!.NNodesStorage);
    }

    [Fact]
    public void Roots_handles_spline_without_knots()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 7 },
            knots: new[] { Array.Empty<double>() });
        spline.Build(verbose: false);

        double root = Assert.Single(spline.Roots());
        Assert.Equal(0.0, root, precision: 12);
    }

    [Fact]
    public void Roots_handles_empty_internal_knot_storage()
    {
        var spline = BuildSpline();
        spline.KnotsStorage = Array.Empty<double[]>();

        Exception? exception = Record.Exception(() => spline.Roots());

        Assert.Null(exception);
    }

    [Fact]
    public void Roots_scans_knoted_spline_without_mutating_knot_storage()
    {
        var spline = BuildSpline();

        _ = spline.Roots();

        Assert.Equal(0.0, spline.KnotsStorage[0][0]);
    }
}
