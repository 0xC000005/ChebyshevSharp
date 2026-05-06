using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderLoadValidationTests
{
    public static IEnumerable<object[]> MalformedStates
    {
        get
        {
            yield return Case("zero dimensions", s => s.NumDimensions = 0);
            yield return Case("null domain", s => s.Domain = null);
            yield return Case("short domain", s => s.Domain = new[] { new[] { -1.0, 1.0 } });
            yield return Case("null domain row", s => s.Domain = new double[][] { new[] { -1.0, 1.0 }, null! });
            yield return Case("bad domain row length", s => s.Domain = new[] { new[] { -1.0 }, new[] { -2.0, 2.0 } });
            yield return Case("reversed domain", s => s.Domain![0] = new[] { 1.0, -1.0 });
            yield return Case("null node counts", s => s.NNodes = null);
            yield return Case("short node counts", s => s.NNodes = new[] { 2 });
            yield return Case("non-positive node count", s => s.NNodes![1] = 0);
            yield return Case("negative max derivative order", s => s.MaxDerivativeOrder = -1);
            yield return Case("null pivot point", s => s.PivotPoint = null);
            yield return Case("short pivot point", s => s.PivotPoint = new[] { 0.0 });
            yield return Case("negative build time", s => s.BuildTime = -1.0);
            yield return Case("null partition", s => s.Partition = null);
            yield return Case("null partition group", s => s.Partition = new int[][] { new[] { 0 }, null! });
            yield return Case("null slides", s => s.Slides = null);
            yield return Case("short slides", s => s.Slides = new[] { s.Slides![0] });
            yield return Case("null slide", s => s.Slides![0] = null!);
            yield return Case("duplicate partition dimension", s => s.Partition = new[] { new[] { 0 }, new[] { 0 } });
            yield return Case("out-of-range partition dimension", s => s.Partition = new[] { new[] { 0 }, new[] { 2 } });
            yield return Case("null derivative order row", s => s.RegisteredDerivativeOrders = new int[][] { null! });
            yield return Case("short derivative order row", s => s.RegisteredDerivativeOrders = new[] { new[] { 0 } });
            yield return Case("negative derivative order", s => s.RegisteredDerivativeOrders = new[] { new[] { 0, -1 } });
            yield return Case("zero slide dimensions", s => s.Slides![0].NumDimensions = 0);
            yield return Case("slide dimension partition mismatch", s => s.Slides![0].NumDimensions = 2);
            yield return Case("negative slide max derivative order", s => s.Slides![0].MaxDerivativeOrder = -1);
            yield return Case("negative slide build time", s => s.Slides![0].BuildTime = -1.0);
            yield return Case("negative slide eval count", s => s.Slides![0].NEvaluations = -1);
            yield return Case("null slide domain", s => s.Slides![0].Domain = null);
            yield return Case("null slide node counts", s => s.Slides![0].NNodes = null);
            yield return Case("null slide node arrays", s => s.Slides![0].NodeArrays = null);
            yield return Case("short slide node array", s => s.Slides![0].NodeArrays![0] = new[] { -1.0 });
            yield return Case("null slide weights", s => s.Slides![0].Weights = null);
            yield return Case("short slide weights", s => s.Slides![0].Weights![0] = new[] { -0.5 });
            yield return Case("wrong slide tensor length", s => s.Slides![0].TensorValues = new[] { 1.0 });
            yield return Case("null slide tensor values", s => s.Slides![0].TensorValues = null);
            yield return Case("null diff matrix sizes", s => s.Slides![0].DiffMatrixSizes = null);
            yield return Case("bad diff matrix size row", s => s.Slides![0].DiffMatrixSizes = new[] { new[] { 2 } });
            yield return Case("wrong diff matrix size", s => s.Slides![0].DiffMatrixSizes = new[] { new[] { 2, 3 } });
            yield return Case("null slide diff matrices", s => s.Slides![0].DiffMatrices = null);
            yield return Case("wrong diff matrix length", s => s.Slides![0].DiffMatrices![0] = new[] { 1.0, 2.0 });
            yield return Case("slide domain mismatch", s => s.Slides![0].Domain![0] = new[] { -0.5, 1.0 });
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
            InvalidDataException ex = Assert.Throws<InvalidDataException>(() => ChebyshevSlider.Load(path));
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
        Type = "ChebyshevSlider",
        NumDimensions = 2,
        Domain = new[] { new[] { -1.0, 1.0 }, new[] { -2.0, 2.0 } },
        NNodes = new[] { 2, 2 },
        MaxDerivativeOrder = 2,
        Partition = new[] { new[] { 0 }, new[] { 1 } },
        PivotPoint = new[] { 0.0, 0.0 },
        PivotValue = 0.0,
        BuildTime = 0.0,
        Slides = new[]
        {
            Slide(1, -1.0, 1.0),
            Slide(1, -2.0, 2.0),
        },
    };

    private static SlideState Slide(int n, double lo, double hi) => new()
    {
        NumDimensions = 1,
        Domain = new[] { new[] { lo, hi } },
        NNodes = new[] { n + 1 },
        MaxDerivativeOrder = 2,
        NodeArrays = new[] { new[] { lo, hi } },
        TensorValues = new[] { lo, hi },
        Weights = new[] { new[] { -0.5, 0.5 } },
        DiffMatrices = new[] { new[] { -0.5, 0.5, -0.5, 0.5 } },
        DiffMatrixSizes = new[] { new[] { 2, 2 } },
        BuildTime = 0.0,
        NEvaluations = 2,
    };

    private static void MakeOversizedDiffMatrixShape(State state)
    {
        double[] many = new double[50_000];
        state.NNodes![0] = 50_000;
        state.Slides![0] = new SlideState
        {
            NumDimensions = 1,
            Domain = new[] { new[] { -1.0, 1.0 } },
            NNodes = new[] { 50_000 },
            MaxDerivativeOrder = 2,
            NodeArrays = new[] { many },
            TensorValues = many,
            Weights = new[] { many },
            DiffMatrices = new[] { System.Array.Empty<double>() },
            DiffMatrixSizes = new[] { new[] { 50_000, 50_000 } },
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
        public string Type { get; set; } = "ChebyshevSlider";
        public int NumDimensions { get; set; }
        public double[][]? Domain { get; set; }
        public int[]? NNodes { get; set; }
        public int MaxDerivativeOrder { get; set; }
        public int[][]? Partition { get; set; }
        public double[]? PivotPoint { get; set; }
        public double PivotValue { get; set; }
        public double BuildTime { get; set; }
        public SlideState[]? Slides { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
    }

    private sealed class SlideState
    {
        public int NumDimensions { get; set; }
        public double[][]? Domain { get; set; }
        public int[]? NNodes { get; set; }
        public int MaxDerivativeOrder { get; set; }
        public double[][]? NodeArrays { get; set; }
        public double[]? TensorValues { get; set; }
        public double[][]? Weights { get; set; }
        public double[][]? DiffMatrices { get; set; }
        public int[][]? DiffMatrixSizes { get; set; }
        public double BuildTime { get; set; }
        public int NEvaluations { get; set; }
    }
}
