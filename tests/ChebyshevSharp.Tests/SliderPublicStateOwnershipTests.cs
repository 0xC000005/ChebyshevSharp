using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderPublicStateOwnershipTests
{
    private static ChebyshevSlider BuildSlider()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + 2.0 * p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -2.0, 2.0 } },
            nNodes: new[] { 5, 7 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    [Fact]
    public void Domain_NNodes_Partition_and_PivotPoint_properties_return_snapshots()
    {
        var slider = BuildSlider();

        double[][] domain = slider.Domain;
        int[] nNodes = slider.NNodes;
        int[][] partition = slider.Partition;
        double[] pivotPoint = slider.PivotPoint;
        domain[0][0] = 10.0;
        nNodes[0] = 999;
        partition[0][0] = 1;
        pivotPoint[0] = 0.75;

        Assert.Equal(-1.0, slider.Domain[0][0]);
        Assert.Equal(new[] { 5, 7 }, slider.NNodes);
        Assert.Equal(0, slider.Partition[0][0]);
        Assert.Equal(0.0, slider.PivotPoint[0]);
    }

    [Fact]
    public void Mutating_public_snapshots_does_not_change_eval_contract()
    {
        var slider = BuildSlider();
        double valueBefore = slider.Eval(new[] { -0.5, 0.25 }, new[] { 0, 0 });

        double[][] domain = slider.Domain;
        int[][] partition = slider.Partition;
        double[] pivotPoint = slider.PivotPoint;
        domain[0][0] = 0.5;
        partition[0][0] = 1;
        pivotPoint[1] = 1.5;

        double valueAfter = slider.Eval(new[] { -0.5, 0.25 }, new[] { 0, 0 });

        Assert.Equal(valueBefore, valueAfter, precision: 12);
    }

    [Fact]
    public void Mutating_public_pivot_snapshot_does_not_change_evaluation_points()
    {
        var slider = BuildSlider();

        double[] pivotPoint = slider.PivotPoint;
        pivotPoint[1] = 1.5;

        double[] points = slider.GetEvaluationPoints();

        Assert.Contains(0.0, points.Where((_, index) => index % slider.NumDimensions == 1));
        Assert.DoesNotContain(1.5, points.Where((_, index) => index % slider.NumDimensions == 1));
    }

    [Fact]
    public void Internal_storage_accessors_remain_live_while_public_properties_snapshot()
    {
        var slider = new ChebyshevSlider();
        var domain = new[] { new[] { -1.0, 1.0 } };
        var nNodes = new[] { 5 };
        var partition = new[] { new[] { 0 } };
        var pivotPoint = new[] { 0.0 };

        slider.Domain = domain;
        slider.NNodes = nNodes;
        slider.Partition = partition;
        slider.PivotPoint = pivotPoint;

        Assert.Same(domain, slider.DomainStorage);
        Assert.Same(nNodes, slider.NNodesStorage);
        Assert.Same(partition, slider.PartitionStorage);
        Assert.Same(pivotPoint, slider.PivotPointStorage);
        Assert.NotSame(domain, slider.Domain);
        Assert.NotSame(nNodes, slider.NNodes);
        Assert.NotSame(partition, slider.Partition);
        Assert.NotSame(pivotPoint, slider.PivotPoint);

        var replacementDomain = new[] { new[] { 0.0, 2.0 } };
        var replacementNNodes = new[] { 7 };
        var replacementPartition = new[] { new[] { 0 } };
        var replacementPivot = new[] { 1.0 };

        slider.DomainStorage = replacementDomain;
        slider.NNodesStorage = replacementNNodes;
        slider.PartitionStorage = replacementPartition;
        slider.PivotPointStorage = replacementPivot;

        Assert.Same(replacementDomain, slider.DomainStorage);
        Assert.Same(replacementNNodes, slider.NNodesStorage);
        Assert.Same(replacementPartition, slider.PartitionStorage);
        Assert.Same(replacementPivot, slider.PivotPointStorage);
    }

    [Fact]
    public void Internal_setters_accept_null_without_exposing_mutable_empty_state()
    {
        var slider = new ChebyshevSlider();

        slider.Domain = null!;
        slider.NNodes = null!;
        slider.Partition = null!;
        slider.PivotPoint = null!;

        Assert.Empty(slider.Domain);
        Assert.Empty(slider.NNodes);
        Assert.Empty(slider.Partition);
        Assert.Empty(slider.PivotPoint);

        slider.DomainStorage = null!;
        slider.NNodesStorage = null!;
        slider.PartitionStorage = null!;
        slider.PivotPointStorage = null!;

        Assert.Empty(slider.DomainStorage);
        Assert.Empty(slider.NNodesStorage);
        Assert.Empty(slider.PartitionStorage);
        Assert.Empty(slider.PivotPointStorage);
    }

    [Fact]
    public void Slider_arithmetic_rejects_same_length_different_partition_storage()
    {
        var left = BuildSlider();
        var right = new ChebyshevSlider(
            (p, _) => p[0] + 2.0 * p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -2.0, 2.0 } },
            nNodes: new[] { 5, 7 },
            partition: new[] { new[] { 1 }, new[] { 0 } },
            pivotPoint: new[] { 0.0, 0.0 });
        right.Build(verbose: false);

        Assert.Throws<ArgumentException>(() => { var _ = left + right; });
    }
}
