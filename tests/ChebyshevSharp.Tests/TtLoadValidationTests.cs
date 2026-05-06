using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtLoadValidationTests
{
    public static IEnumerable<object[]> MalformedStates
    {
        get
        {
            yield return Case("zero dimensions", s => s.NumDimensions = 0);
            yield return Case("null domain", s => s.Domain = null);
            yield return Case("short domain", s => s.Domain = new[] { new[] { -1.0, 1.0 } });
            yield return Case("null domain row", s => s.Domain = new double[][] { new[] { -1.0, 1.0 }, null! });
            yield return Case("bad domain row length", s => s.Domain = new[] { new[] { -1.0 }, new[] { -1.0, 1.0 } });
            yield return Case("reversed domain", s => s.Domain![0] = new[] { 1.0, -1.0 });
            yield return Case("null node counts", s => s.NNodes = null);
            yield return Case("short node counts", s => s.NNodes = new[] { 2 });
            yield return Case("non-positive node count", s => s.NNodes![1] = 0);
            yield return Case("non-positive max rank", s => s.MaxRank = 0);
            yield return Case("negative tolerance", s => s.Tolerance = -1e-6);
            yield return Case("negative max sweeps", s => s.MaxSweeps = -1);
            yield return Case("negative build time", s => s.BuildTime = -1.0);
            yield return Case("negative build eval count", s => s.TotalBuildEvals = -1);
            yield return Case("negative max derivative order", s => s.MaxDerivativeOrder = -1);
            yield return Case("null ranks", s => s.TtRanks = null);
            yield return Case("short ranks", s => s.TtRanks = new[] { 1, 1 });
            yield return Case("non-positive rank", s => s.TtRanks![1] = 0);
            yield return Case("bad rank endpoints", s => s.TtRanks![0] = 2);
            yield return Case("short dim order", s => s.DimOrder = new[] { 0 });
            yield return Case("duplicate dim order", s => s.DimOrder = new[] { 0, 0 });
            yield return Case("out-of-range dim order", s => s.DimOrder = new[] { 0, 2 });
            yield return Case("null derivative order row", s => s.RegisteredDerivativeOrders = new int[][] { null! });
            yield return Case("short derivative order row", s => s.RegisteredDerivativeOrders = new[] { new[] { 0 } });
            yield return Case("negative derivative order", s => s.RegisteredDerivativeOrders = new[] { new[] { 0, -1 } });
            yield return Case("null cores", s => s.Cores = null);
            yield return Case("short cores", s => s.Cores = new[] { s.Cores![0] });
            yield return Case("null core row", s => s.Cores![1] = null!);
            yield return Case("non-positive core dimension", s => s.Cores![0].NNodes = 0);
            yield return Case("core rank mismatch", s => s.Cores![1].RLeft = 3);
            yield return Case("core node mismatch", s => s.Cores![0].NNodes = 3);
            yield return Case("null core data", s => s.Cores![0].Data = null);
            yield return Case("wrong core data length", s => s.Cores![0].Data = new[] { 1.0, 2.0 });
            yield return Case("oversized core shape", MakeOversizedCoreShape);
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
            InvalidDataException ex = Assert.Throws<InvalidDataException>(() => ChebyshevTT.Load(path));
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
        Version = "0.0.0-test",
        NumDimensions = 2,
        Domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
        NNodes = new[] { 2, 2 },
        MaxRank = 3,
        Tolerance = 1e-6,
        MaxSweeps = 10,
        TtRanks = new[] { 1, 2, 1 },
        BuildTime = 0.0,
        TotalBuildEvals = 4,
        Cores = new[]
        {
            new Core { RLeft = 1, NNodes = 2, RRight = 2, Data = new[] { 1.0, 0.0, 0.0, 1.0 } },
            new Core { RLeft = 2, NNodes = 2, RRight = 1, Data = new[] { 1.0, 0.0, 0.0, 1.0 } },
        },
        JsonVersion = 2,
        DimOrder = new[] { 0, 1 },
        MaxDerivativeOrder = 2,
    };

    private static void MakeOversizedCoreShape(State state)
    {
        state.NNodes = new[] { 50_000, 50_000 };
        state.TtRanks = new[] { 1, 50_000, 1 };
        state.Cores = new[]
        {
            new Core { RLeft = 1, NNodes = 50_000, RRight = 50_000, Data = System.Array.Empty<double>() },
            new Core { RLeft = 50_000, NNodes = 50_000, RRight = 1, Data = System.Array.Empty<double>() },
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
        public string? Version { get; set; }
        public int NumDimensions { get; set; }
        public double[][]? Domain { get; set; }
        public int[]? NNodes { get; set; }
        public int MaxRank { get; set; }
        public double Tolerance { get; set; }
        public int MaxSweeps { get; set; }
        public int[]? TtRanks { get; set; }
        public double BuildTime { get; set; }
        public int TotalBuildEvals { get; set; }
        public Core[]? Cores { get; set; }
        public int? JsonVersion { get; set; }
        public int[]? DimOrder { get; set; }
        public int? MaxDerivativeOrder { get; set; }
        public int[][]? RegisteredDerivativeOrders { get; set; }
    }

    private sealed class Core
    {
        public int RLeft { get; set; }
        public int NNodes { get; set; }
        public int RRight { get; set; }
        public double[]? Data { get; set; }
    }
}
