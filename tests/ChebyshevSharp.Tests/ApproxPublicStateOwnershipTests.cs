using System.IO;
using Xunit;

namespace ChebyshevSharp.Tests;

public class ApproxPublicStateOwnershipTests
{
    private static ChebyshevApproximation BuildApproximation()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] * p[0] + 2.0 * p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 7, 7 });
        approx.Build(verbose: false);
        return approx;
    }

    [Fact]
    public void Domain_and_NNodes_properties_return_snapshots()
    {
        var approx = BuildApproximation();

        double[][] domain = approx.Domain;
        int[] nNodes = approx.NNodes;
        domain[0][0] = 10.0;
        nNodes[0] = 999;

        Assert.Equal(-1.0, approx.Domain[0][0]);
        Assert.Equal(new[] { 7, 7 }, approx.NNodes);
        Assert.Throws<ArgumentOutOfRangeException>(() => approx.Eval(new[] { 2.0, 0.0 }));
    }

    [Fact]
    public void NodeArrays_and_TensorValues_properties_return_snapshots()
    {
        var approx = BuildApproximation();
        double before = approx.Eval(new[] { 0.25, -0.5 });

        double[][] nodes = approx.NodeArrays;
        double[] tensor = approx.TensorValues!;
        nodes[0][0] = 123.0;
        tensor[0] = 123.0;

        double after = approx.Eval(new[] { 0.25, -0.5 });
        Assert.Equal(before, after, precision: 12);
        Assert.NotEqual(123.0, approx.NodeArrays[0][0]);
        Assert.NotEqual(123.0, approx.TensorValues![0]);
    }

    [Fact]
    public void Weight_and_diff_matrix_properties_return_snapshots()
    {
        var approx = BuildApproximation();
        double valueBefore = approx.Eval(new[] { 0.25, -0.5 });
        double derivativeBefore = approx.Eval(new[] { 0.25, -0.5 }, new[] { 1, 0 });

        double[][] weights = approx.Weights!;
        double[][,] diffMatrices = approx.DiffMatrices!;
        weights[0][0] = 123.0;
        diffMatrices[0][0, 0] = 123.0;

        double valueAfter = approx.Eval(new[] { 0.25, -0.5 });
        double derivativeAfter = approx.Eval(new[] { 0.25, -0.5 }, new[] { 1, 0 });
        Assert.Equal(valueBefore, valueAfter, precision: 12);
        Assert.Equal(derivativeBefore, derivativeAfter, precision: 12);
        Assert.NotEqual(123.0, approx.Weights![0][0]);
        Assert.NotEqual(123.0, approx.DiffMatrices![0][0, 0]);
    }

    [Fact]
    public void Internal_storage_accessors_remain_live_while_public_properties_snapshot()
    {
        var approx = new ChebyshevApproximation();
        var domain = new[] { new[] { -1.0, 1.0 } };
        var nNodes = new[] { 3 };
        var nodes = new[] { new[] { -0.5, 0.0, 0.5 } };
        var tensor = new[] { 1.0, 2.0, 3.0 };
        var weights = new[] { new[] { -1.0, 2.0, -1.0 } };
        var diffMatrices = new[] { new double[,] { { 0.0, 1.0 }, { -1.0, 0.0 } } };
        var diffMatricesFlat = new[] { new[] { 0.0, -1.0, 1.0, 0.0 } };

        approx.Domain = domain;
        approx.NNodes = nNodes;
        approx.NodeArrays = nodes;
        approx.TensorValues = tensor;
        approx.Weights = weights;
        approx.DiffMatrices = diffMatrices;
        approx.DiffMatricesTFlat = diffMatricesFlat;

        Assert.Same(domain, approx.DomainStorage);
        Assert.Same(nNodes, approx.NNodesStorage);
        Assert.Same(nodes, approx.NodeArraysStorage);
        Assert.Same(tensor, approx.TensorValuesStorage);
        Assert.Same(weights, approx.WeightsStorage);
        Assert.Same(diffMatrices, approx.DiffMatricesStorage);
        Assert.Same(diffMatricesFlat, approx.DiffMatricesTFlat);
        Assert.NotSame(domain, approx.Domain);
        Assert.NotSame(nNodes, approx.NNodes);
        Assert.NotSame(nodes, approx.NodeArrays);
        Assert.NotSame(tensor, approx.TensorValues);
        Assert.NotSame(weights, approx.Weights);
        Assert.NotSame(diffMatrices, approx.DiffMatrices);

        var replacementDomain = new[] { new[] { 0.0, 2.0 } };
        var replacementNNodes = new[] { 5 };
        var replacementNodes = new[] { new[] { -1.0, -0.5, 0.0, 0.5, 1.0 } };
        var replacementTensor = new[] { 1.0 };
        var replacementWeights = new[] { new[] { 1.0 } };
        var replacementDiffMatrices = new[] { new double[,] { { 1.0 } } };

        approx.DomainStorage = replacementDomain;
        approx.NNodesStorage = replacementNNodes;
        approx.NodeArraysStorage = replacementNodes;
        approx.TensorValuesStorage = replacementTensor;
        approx.WeightsStorage = replacementWeights;
        approx.DiffMatricesStorage = replacementDiffMatrices;

        Assert.Same(replacementDomain, approx.DomainStorage);
        Assert.Same(replacementNNodes, approx.NNodesStorage);
        Assert.Same(replacementNodes, approx.NodeArraysStorage);
        Assert.Same(replacementTensor, approx.TensorValuesStorage);
        Assert.Same(replacementWeights, approx.WeightsStorage);
        Assert.Same(replacementDiffMatrices, approx.DiffMatricesStorage);
    }

    [Fact]
    public void Internal_setters_accept_null_without_exposing_mutable_empty_state()
    {
        var approx = new ChebyshevApproximation();

        approx.Domain = null!;
        approx.NNodes = null!;
        approx.NodeArrays = null!;
        approx.TensorValues = null;
        approx.Weights = null;
        approx.DiffMatrices = null;
        approx.DiffMatricesTFlat = null;

        Assert.Empty(approx.Domain);
        Assert.Empty(approx.NNodes);
        Assert.Empty(approx.NodeArrays);
        Assert.Null(approx.TensorValues);
        Assert.Null(approx.Weights);
        Assert.Null(approx.DiffMatrices);
        Assert.Null(approx.DiffMatricesTFlat);

        approx.PrecomputeTransposedDiffMatrices();

        Assert.Null(approx.DiffMatricesTFlat);
    }

    [Fact]
    public void Nullable_nnodes_constructor_with_explicit_counts_initializes_private_storage()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -2.0, 2.0 }, new[] { 0.0, 1.0 } },
            nNodes: new int?[] { 4, 5 });

        Assert.Equal(new[] { 4, 5 }, approx.NNodes);
        Assert.Equal(2, approx.NodeArrays.Length);
        Assert.Equal(4, approx.NodeArrays[0].Length);
        Assert.Equal(5, approx.NodeArrays[1].Length);
        Assert.Same(approx.NNodesStorage, approx.NNodesStorage);
        Assert.Same(approx.NodeArraysStorage, approx.NodeArraysStorage);
    }

    [Fact]
    public void VectorizedEvalBatch_requires_built_interpolant()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 3 },
            deferBuild: true);

        Assert.Throws<InvalidOperationException>(
            () => approx.VectorizedEvalBatch(new[] { new[] { 0.0 } }, new[] { 0 }));
    }

    [Fact]
    public void Build_failure_during_adaptive_rebuild_preserves_previous_state()
    {
        bool failDuringRebuild = false;
        int rebuildCalls = 0;
        var approx = new ChebyshevApproximation(
            (p, _) =>
            {
                if (!failDuringRebuild)
                    return p[0];

                rebuildCalls++;
                if (rebuildCalls <= 3)
                    return 100.0 + p[0];

                throw new InvalidOperationException("validation failed");
            },
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: null,
            errorThreshold: 1e-6);
        approx.Build(verbose: false);
        double valueBefore = approx.Eval(new[] { 0.25 }, new[] { 0 });
        int[] nNodesBefore = approx.NNodes;

        failDuringRebuild = true;
        Assert.Throws<InvalidOperationException>(() => approx.Build(verbose: false));

        Assert.True(approx.IsConstructionFinished());
        Assert.Equal(nNodesBefore, approx.NNodes);
        Assert.Equal(valueBefore, approx.Eval(new[] { 0.25 }, new[] { 0 }), precision: 12);
    }

    [Fact]
    public void Adaptive_build_rejects_non_finite_validation_values()
    {
        int calls = 0;
        var approx = new ChebyshevApproximation(
            (p, _) =>
            {
                calls++;
                return calls <= 3 ? p[0] : double.NaN;
            },
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: null,
            errorThreshold: 1e-6);

        var ex = Assert.Throws<ArgumentException>(() => approx.Build(verbose: false));

        Assert.Contains("non-finite", ex.Message);
        Assert.False(approx.IsConstructionFinished());
        Assert.Null(approx.TensorValues);
    }

    [Fact]
    public void GetSpecialPoints_returns_snapshots_for_loaded_metadata()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, MinimalApproxJsonWithSpecialPoints());
            var approx = ChebyshevApproximation.Load(path);

            double[][] specialPoints = approx.GetSpecialPoints()!;
            specialPoints[0][0] = 0.75;

            double[][] specialPointsAgain = approx.GetSpecialPoints()!;
            Assert.NotSame(specialPoints, specialPointsAgain);
            Assert.Equal(0.0, specialPointsAgain[0][0]);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static string MinimalApproxJsonWithSpecialPoints() =>
        """
        {
          "NumDimensions": 1,
          "Domain": [[-1.0, 1.0]],
          "NNodes": [2],
          "MaxDerivativeOrder": 2,
          "NodeArrays": [[-0.5, 0.5]],
          "TensorValues": [1.0, 2.0],
          "Weights": [[-1.0, 1.0]],
          "DiffMatrices": [[0.0, 0.0, 0.0, 0.0]],
          "SpecialPoints": [[0.0]]
        }
        """;
}
