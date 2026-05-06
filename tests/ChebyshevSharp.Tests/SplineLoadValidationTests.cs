using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SplineLoadValidationTests
{
    public static IEnumerable<object[]> MalformedStates
    {
        get
        {
            yield return Case("zero dimensions", s => s.NumDimensions = 0);
            yield return Case("null domain", s => s.Domain = null);
            yield return Case("short domain", s => s.Domain = System.Array.Empty<double[]>());
            yield return Case("null domain row", s => s.Domain = new double[][] { null! });
            yield return Case("bad domain row length", s => s.Domain = new[] { new[] { -1.0 } });
            yield return Case("reversed domain", s => s.Domain![0] = new[] { 1.0, -1.0 });
            yield return Case("null node counts", s => s.NNodes = null);
            yield return Case("short node counts", s => s.NNodes = System.Array.Empty<int>());
            yield return Case("non-positive node count", s => s.NNodes![0] = 0);
            yield return Case("null knots", s => s.Knots = null);
            yield return Case("short knots", s => s.Knots = System.Array.Empty<double[]>());
            yield return Case("null knot row", s => s.Knots = new double[][] { null! });
            yield return Case("knot outside domain", s => s.Knots = new[] { new[] { 2.0 } });
            yield return Case("unsorted knots", MakeUnsortedKnots);
            yield return Case("null shape", s => s.Shape = null);
            yield return Case("short shape", s => s.Shape = System.Array.Empty<int>());
            yield return Case("null piece states", s => s.PieceStates = null);
            yield return Case("wrong shape", s => s.Shape = new[] { 2 });
            yield return Case("short piece states", s => s.PieceStates = System.Array.Empty<PieceState>());
            yield return Case("null piece state", s => s.PieceStates![0] = null!);
            yield return Case("negative max derivative order", s => s.MaxDerivativeOrder = -1);
            yield return Case("negative build time", s => s.BuildTime = -1.0);
            yield return Case("negative error threshold", s => s.ErrorThreshold = -1.0);
            yield return Case("non-positive max n", s => s.MaxN = 0);
            yield return Case("wrong original node count length", s => s.OriginalNNodes = new int?[] { 2, 2 });
            yield return Case("non-positive original node count", s => s.OriginalNNodes = new int?[] { 0 });
            yield return Case("wrong nested node count length", s => s.NestedNNodes = new[] { new[] { 2 }, new[] { 2 } });
            yield return Case("null nested node row", s => s.NestedNNodes = new int[][] { null! });
            yield return Case("short nested node row", s => s.NestedNNodes = new[] { System.Array.Empty<int>() });
            yield return Case("non-positive nested node count", s => s.NestedNNodes = new[] { new[] { 0 } });
            yield return Case("null derivative order row", s => s.RegisteredDerivativeOrders = new int[][] { null! });
            yield return Case("short derivative order row", s => s.RegisteredDerivativeOrders = new[] { System.Array.Empty<int>() });
            yield return Case("negative derivative order", s => s.RegisteredDerivativeOrders = new[] { new[] { -1 } });
            yield return Case("piece dimension mismatch", s => s.PieceStates![0].NumDimensions = 2);
            yield return Case("negative piece max derivative order", s => s.PieceStates![0].MaxDerivativeOrder = -1);
            yield return Case("negative piece build time", s => s.PieceStates![0].BuildTime = -1.0);
            yield return Case("negative piece eval count", s => s.PieceStates![0].NEvaluations = -1);
            yield return Case("null piece domain", s => s.PieceStates![0].Domain = null);
            yield return Case("null piece node counts", s => s.PieceStates![0].NNodes = null);
            yield return Case("null piece node arrays", s => s.PieceStates![0].NodeArrays = null);
            yield return Case("short piece node array", s => s.PieceStates![0].NodeArrays![0] = new[] { -1.0 });
            yield return Case("null piece weights", s => s.PieceStates![0].Weights = null);
            yield return Case("short piece weights", s => s.PieceStates![0].Weights![0] = new[] { -0.5 });
            yield return Case("wrong piece tensor length", s => s.PieceStates![0].TensorValues = new[] { 1.0 });
            yield return Case("null piece tensor values", s => s.PieceStates![0].TensorValues = null);
            yield return Case("null piece diff matrices", s => s.PieceStates![0].DiffMatrices = null);
            yield return Case("wrong diff matrix length", s => s.PieceStates![0].DiffMatrices![0] = new[] { 1.0, 2.0 });
            yield return Case("reversed piece domain", s => s.PieceStates![0].Domain![0] = new[] { 1.0, -1.0 });
            yield return Case("piece interval mismatch", s => s.PieceStates![0].Domain![0] = new[] { -0.5, 1.0 });
            yield return Case("oversized diff matrix shape", MakeOversizedDiffMatrixShape);
        }
    }

    [Theory]
    [MemberData(nameof(MalformedStates))]
    public void Load_WithMalformedState_ThrowsInvalidData(string name, string json)
    {
        Assert.NotEmpty(name);
        string path = WriteTempJson(json);
        try
        {
            InvalidDataException ex = Assert.Throws<InvalidDataException>(() => ChebyshevSpline.Load(path));
            Assert.NotEmpty(ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static object[] Case(string name, Action<State> mutate)
    {
        State state = ValidState();
        mutate(state);
        return new object[] { name, JsonSerializer.Serialize(state) };
    }

    private static State ValidState() => new()
    {
        Type = "ChebyshevSpline",
        NumDimensions = 1,
        Domain = new[] { new[] { -1.0, 1.0 } },
        NNodes = new[] { 2 },
        MaxDerivativeOrder = 2,
        Knots = new[] { System.Array.Empty<double>() },
        Shape = new[] { 1 },
        BuildTime = 0.0,
        PieceStates = new[] { Piece(-1.0, 1.0) },
        MaxN = 64,
    };

    private static PieceState Piece(double lo, double hi) => new()
    {
        NumDimensions = 1,
        Domain = new[] { new[] { lo, hi } },
        NNodes = new[] { 2 },
        MaxDerivativeOrder = 2,
        NodeArrays = new[] { new[] { lo, hi } },
        TensorValues = new[] { lo, hi },
        Weights = new[] { new[] { -0.5, 0.5 } },
        DiffMatrices = new[] { new[] { -0.5, 0.5, -0.5, 0.5 } },
        BuildTime = 0.0,
        NEvaluations = 2,
    };

    private static void MakeUnsortedKnots(State state)
    {
        state.Knots = new[] { new[] { 0.5, -0.5 } };
        state.Shape = new[] { 3 };
        state.PieceStates = new[]
        {
            Piece(-1.0, 0.5),
            Piece(0.5, -0.5),
            Piece(-0.5, 1.0),
        };
    }

    private static void MakeOversizedDiffMatrixShape(State state)
    {
        double[] many = new double[50_000];
        state.NNodes![0] = 50_000;
        state.PieceStates![0] = new PieceState
        {
            NumDimensions = 1,
            Domain = new[] { new[] { -1.0, 1.0 } },
            NNodes = new[] { 50_000 },
            MaxDerivativeOrder = 2,
            NodeArrays = new[] { many },
            TensorValues = many,
            Weights = new[] { many },
            DiffMatrices = new[] { System.Array.Empty<double>() },
            BuildTime = 0.0,
            NEvaluations = 50_000,
        };
    }

    private static string WriteTempJson(string json)
    {
        string path = Path.GetTempFileName();
        File.WriteAllText(path, json);
        return path;
    }

    private sealed class State
    {
        public string Type { get; set; } = "ChebyshevSpline";
        public int NumDimensions { get; set; }
        public double[][]? Domain { get; set; }
        public int[]? NNodes { get; set; }
        public int? MaxDerivativeOrder { get; set; }
        public double[][]? Knots { get; set; }
        public int[]? Shape { get; set; }
        public double BuildTime { get; set; }
        public PieceState[]? PieceStates { get; set; }
        public int?[]? OriginalNNodes { get; set; }
        public double? ErrorThreshold { get; set; }
        public int? MaxN { get; set; }
        public int[][]? NestedNNodes { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
    }

    private sealed class PieceState
    {
        public int NumDimensions { get; set; }
        public double[][]? Domain { get; set; }
        public int[]? NNodes { get; set; }
        public int? MaxDerivativeOrder { get; set; }
        public double[][]? NodeArrays { get; set; }
        public double[]? TensorValues { get; set; }
        public double[][]? Weights { get; set; }
        public double[][]? DiffMatrices { get; set; }
        public double BuildTime { get; set; }
        public int NEvaluations { get; set; }
    }
}
