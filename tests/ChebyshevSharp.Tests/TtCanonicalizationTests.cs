using System;
using System.Linq;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_tensor_train.py classes
// TestOrthogonalization + TestInnerProduct (PyChebyshev v0.13.0).
// Tests added incrementally across Phase 2 Tasks 3 and 4.
public class TtCanonicalizationTests
{
}

public class OrthLeftRightTests
{
    private static ChebyshevTT MakeTt3D()
    {
        // f(x,y,z) = sin(x)*cos(y) + 0.3*z^2 — same fixture as Python test_orth_left/right.
        var tt = new ChebyshevTT(
            point => Math.Sin(point[0]) * Math.Cos(point[1]) + 0.3 * point[2] * point[2],
            3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11, 11 },
            maxRank: 6,
            tolerance: 1e-6);
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    private static double[,] CoreToMatrixLeft(double[] data, int rL, int n, int rR)
    {
        // Unfold (rL, n, rR) → (rL*n, rR) row-major
        var M = new double[rL * n, rR];
        for (int i = 0; i < rL; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < rR; k++)
                    M[i * n + j, k] = data[i * n * rR + j * rR + k];
        return M;
    }

    private static double[,] CoreToMatrixRight(double[] data, int rL, int n, int rR)
    {
        // Unfold (rL, n, rR) → (rL, n*rR) row-major
        var M = new double[rL, n * rR];
        for (int i = 0; i < rL; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < rR; k++)
                    M[i, j * rR + k] = data[i * n * rR + j * rR + k];
        return M;
    }

    [Fact]
    public void Test_orth_left_produces_left_orthogonal_cores()
    {
        var tt = MakeTt3D();
        tt.OrthLeft(position: 2);

        // Cores 0 and 1 must satisfy Q^T Q = I after the (rL*n, rR) unfolding.
        for (int k = 0; k < 2; k++)
        {
            var (rL, n, rR, data) = tt.GetCoreShape(k);
            double[,] Q = CoreToMatrixLeft(data, rL, n, rR);
            // Build gram = Q^T Q — should equal I_{rR x rR}
            for (int p = 0; p < rR; p++)
                for (int q = 0; q < rR; q++)
                {
                    double s = 0;
                    for (int row = 0; row < rL * n; row++)
                        s += Q[row, p] * Q[row, q];
                    double target = (p == q) ? 1.0 : 0.0;
                    Assert.True(Math.Abs(s - target) < 1e-10,
                        $"core {k} not left-orthogonal: gram[{p},{q}]={s}");
                }
        }
    }

    [Fact]
    public void Test_orth_right_produces_right_orthogonal_cores()
    {
        var tt = MakeTt3D();
        tt.OrthRight(position: 0);

        // Cores 1 and 2 must satisfy Q Q^T = I after the (rL, n*rR) unfolding.
        for (int k = 1; k < 3; k++)
        {
            var (rL, n, rR, data) = tt.GetCoreShape(k);
            double[,] Q = CoreToMatrixRight(data, rL, n, rR);
            for (int p = 0; p < rL; p++)
                for (int q = 0; q < rL; q++)
                {
                    double s = 0;
                    for (int col = 0; col < n * rR; col++)
                        s += Q[p, col] * Q[q, col];
                    double target = (p == q) ? 1.0 : 0.0;
                    Assert.True(Math.Abs(s - target) < 1e-10,
                        $"core {k} not right-orthogonal: gram[{p},{q}]={s}");
                }
        }
    }

    [Fact]
    public void Test_orth_left_preserves_eval()
    {
        var tt = MakeTt3D();
        double[][] pts = new[]
        {
            new[] { 0.1, -0.2, 0.3 },
            new[] { 0.5, 0.5, -0.5 },
            new[] { -0.9, 0.1, 0.7 },
        };
        double[] before = pts.Select(p => tt.Eval(p)).ToArray();
        tt.OrthLeft(position: 2);
        double[] after = pts.Select(p => tt.Eval(p)).ToArray();
        for (int i = 0; i < pts.Length; i++)
            TestFixtures.AssertClose(before[i], after[i], atol: 1e-10);
    }

    [Fact]
    public void Test_orth_right_preserves_eval()
    {
        var tt = MakeTt3D();
        double[][] pts = new[]
        {
            new[] { 0.1, -0.2, 0.3 },
            new[] { 0.5, 0.5, -0.5 },
            new[] { -0.9, 0.1, 0.7 },
        };
        double[] before = pts.Select(p => tt.Eval(p)).ToArray();
        tt.OrthRight(position: 0);
        double[] after = pts.Select(p => tt.Eval(p)).ToArray();
        for (int i = 0; i < pts.Length; i++)
            TestFixtures.AssertClose(before[i], after[i], atol: 1e-10);
    }

    [Fact]
    public void Test_orth_left_position_zero_raises()
    {
        var tt = MakeTt3D();
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => tt.OrthLeft(position: 0));
        Assert.Contains("position", ex.Message);
    }

    [Fact]
    public void Test_orth_right_position_last_raises()
    {
        var tt = MakeTt3D();
        // d=3, last valid position is d-2=1. position=2 should raise.
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.OrthRight(position: 2));
    }

    [Fact]
    public void Test_orth_left_out_of_range_raises()
    {
        var tt = MakeTt3D();
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.OrthLeft(position: 5));
    }

    [Fact]
    public void Test_orth_left_on_unbuilt_raises()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 });
        Assert.Throws<InvalidOperationException>(() => tt.OrthLeft(position: 1));
    }
}

public class InnerProductTests
{
    [Fact]
    public void Test_inner_product_matches_explicit_contraction_2d()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8 };
        var ttA = new ChebyshevTT(p => Math.Sin(p[0]) + 0.5 * p[1], 2, domain, nNodes,
            tolerance: 1e-8, maxRank: 8);
        var ttB = new ChebyshevTT(p => Math.Cos(p[0]) * p[1], 2, domain, nNodes,
            tolerance: 1e-8, maxRank: 8);
        ttA.Build(verbose: false, seed: 1);
        ttB.Build(verbose: false, seed: 2);

        double ip = ttA.InnerProduct(ttB);

        // Reference: contract full coefficient tensors via dense reconstruction.
        double[] FullCoeffTensor(ChebyshevTT tt)
        {
            int n0 = tt.NNodes[0], n1 = tt.NNodes[1];
            var (rL0, _, rR0, d0) = tt.GetCoreShape(0);
            var (rL1, _, rR1, d1) = tt.GetCoreShape(1);
            // Core 0 has rL=1; Core 1 has rR=1.
            var dense = new double[n0 * n1];
            for (int i = 0; i < n0; i++)
                for (int j = 0; j < n1; j++)
                {
                    double s = 0;
                    for (int a = 0; a < rR0; a++)
                        s += d0[0 * n0 * rR0 + i * rR0 + a] * d1[a * n1 * rR1 + j * rR1 + 0];
                    dense[i * n1 + j] = s;
                }
            return dense;
        }

        double[] tA = FullCoeffTensor(ttA);
        double[] tB = FullCoeffTensor(ttB);
        double reference = 0;
        for (int i = 0; i < tA.Length; i++) reference += tA[i] * tB[i];
        TestFixtures.AssertClose(reference, ip, atol: 1e-10);
    }

    [Fact]
    public void Test_self_inner_product_is_squared_norm()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var tt = new ChebyshevTT(p => Math.Cos(p[0]) + p[1] * p[1], 2, domain,
            new[] { 10, 10 }, tolerance: 1e-8, maxRank: 8);
        tt.Build(verbose: false, seed: 0);
        double ip = tt.InnerProduct(tt);
        Assert.True(ip > 0, $"self-inner-product must be positive, got {ip}");
    }

    [Fact]
    public void Test_inner_product_raises_on_null_other()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentNullException>(() => tt.InnerProduct(null!));
    }

    [Fact]
    public void Test_inner_product_raises_on_domain_mismatch()
    {
        var ttA = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        var ttB = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -2.0, 2.0 }, new[] { -2.0, 2.0 } }, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => ttA.InnerProduct(ttB));
        Assert.Contains("domain", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_inner_product_raises_on_n_nodes_mismatch()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var ttA = new ChebyshevTT(p => p[0], 2, domain, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        var ttB = new ChebyshevTT(p => p[0], 2, domain, new[] { 7, 7 },
            tolerance: 1e-4, maxRank: 3);
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => ttA.InnerProduct(ttB));
        Assert.Contains("nNodes", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_inner_product_raises_on_unbuilt_self()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var ttA = new ChebyshevTT(p => p[0], 2, domain, new[] { 5, 5 });
        var ttB = new ChebyshevTT(p => p[0], 2, domain, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        ttB.Build(verbose: false);
        Assert.Throws<InvalidOperationException>(() => ttA.InnerProduct(ttB));
    }

    [Fact]
    public void Test_inner_product_raises_on_unbuilt_other()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var ttA = new ChebyshevTT(p => p[0], 2, domain, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        var ttB = new ChebyshevTT(p => p[0], 2, domain, new[] { 5, 5 });
        ttA.Build(verbose: false);
        Assert.Throws<InvalidOperationException>(() => ttA.InnerProduct(ttB));
    }

    [Fact]
    public void Test_inner_product_raises_on_num_dim_mismatch()
    {
        var tt2D = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 },
            tolerance: 1e-4, maxRank: 3);
        var tt3D = new ChebyshevTT(p => p[0], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5, 5 }, tolerance: 1e-4, maxRank: 3);
        tt2D.Build(verbose: false);
        tt3D.Build(verbose: false);
        var ex = Assert.Throws<ArgumentException>(() => tt2D.InnerProduct(tt3D));
        Assert.Contains("numDimensions", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
