// tests/ChebyshevSharp.Tests/RegistryRoundTripTests.cs
using System.IO;
using Xunit;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Regression tests verifying that the derivative-id registry survives a
/// Save/Load round-trip for ChebyshevApproximation and ChebyshevTT.
/// </summary>
public class RegistryRoundTripTests
{
    [Fact]
    public void Approx_DerivativeIdRegistry_survives_Save_and_Load()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        int id = approx.GetDerivativeId(new[] { 1, 0 });
        string tmp = Path.GetTempFileName();
        try
        {
            approx.Save(tmp);
            var loaded = ChebyshevApproximation.Load(tmp);
            // Same orders should resolve to same id (same registry state).
            Assert.Equal(id, loaded.GetDerivativeId(new[] { 1, 0 }));
            // Eval-by-id should still work.
            double byId = loaded.Eval(new[] { 0.3, 0.5 }, id);
            double byOrders = loaded.Eval(new[] { 0.3, 0.5 }, new[] { 1, 0 });
            Assert.Equal(byOrders, byId, precision: 12);
        }
        finally { File.Delete(tmp); }
    }

    [Fact]
    public void Tt_DerivativeIdRegistry_survives_Save_and_Load()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        tt.Build(verbose: false, seed: 42);
        int id = tt.GetDerivativeId(new[] { 1, 0, 0 });
        string tmp = Path.GetTempFileName();
        try
        {
            tt.Save(tmp);
            var loaded = ChebyshevTT.Load(tmp);
            Assert.Equal(id, loaded.GetDerivativeId(new[] { 1, 0, 0 }));
            // Eval-by-id should still work.
            double byId = loaded.Eval(new[] { 0.3, 0.5, 0.2 }, id);
            double byOrders = loaded.EvalMulti(new[] { 0.3, 0.5, 0.2 }, new[] { new[] { 1, 0, 0 } })[0];
            Assert.Equal(byOrders, byId, precision: 8);
        }
        finally { File.Delete(tmp); }
    }
}
