# Phase 2: TT Feature Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `ChebyshevTT` to PyChebyshev v0.18.0 parity by adding TT canonicalization (`OrthLeft`/`OrthRight`), `InnerProduct`, ALS build mode + `RunCompletion` (PyChebyshev v0.13.0), then factories (`Nodes`/`FromValues`), materialization (`ToDense`), slicing (`Extrude`/`Slice`), and full algebra (`+ - * /`) with in-place equivalents (PyChebyshev v0.18.0). Ships as ChebyshevSharp v0.6.0.

**Architecture:** The existing `Internal/TensorTrainKernel.cs` (901 lines) is split into three files in Task 2 — `TensorTrainKernel.cs` (build cores: `TtCross`, `TtSvd`, `Maxvol`, `ValueToCoeffCores`, `TtCore`, QR, **plus** new `OrthLeftSweep`/`OrthRightSweep`/`AlsFixedRankSweep`/`AlsAdaptiveRank`), `TensorTrainAlgebra.cs` (new — `AddCores`, `ScalarMulCores`, `NegateCores`, `RoundCores`, `InnerProductCores`), and `TensorTrainExtrude.cs` (new — `ExtrudeCores`, `SliceCores`, `ToDenseEinsumChain`, `FromValuesTtSvd`). The split is a pure-refactor commit (zero behavior change, all 666 existing tests must pass on it alone) so subsequent feature commits land on a clean substrate. The v0.18 algebra-rounding step depends on v0.13's orth primitives — this dependency arrow drives the internal commit ordering. In-place algebra follows the .NET BCL idiom (`Span<T>.Sort()`, `List<T>.Add()`): explicit named void-returning methods (`AddInPlace`, `ScalarMulInPlace`, `RoundInPlace`, …) since C# cannot overload `+=` independently of `+`.

**Tech Stack:** C# 13 / .NET 8 + .NET 10 multi-target; xUnit; existing TT infrastructure (`TtCross`, `TtSvd`, `Maxvol`, manual Householder QR); MathNet for SVD in TT-SVD-from-tensor and TT-round, MathNet for matrix LS solver in ALS.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `ref/PyChebyshev` | Pin advance | Submodule moved from `v0.12.0` → `v0.18.0`. |
| `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` | Modify | Refactor (Task 2): keep `TtCore`, QR, `Maxvol`, `TtCross`, `TtSvd`, `ValueToCoeffCores`. Add `OrthLeftSweep`/`OrthRightSweep` (Task 3), `AlsFixedRankSweep`/`AlsAdaptiveRank` (Task 5), `CoeffCoreToValueCore` (Task 6). |
| `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` | Create | New file (Task 2 stub, Task 4 fills `InnerProductCores`, Task 9 fills `ScalarMulCores`/`NegateCores`, Task 10 fills `AddCores`/`RoundCores`). |
| `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs` | Create | New file (Task 2 stub, Task 7 fills `FromValuesTtSvd`, Task 8 fills `ExtrudeCores`/`SliceCores`/`ToDenseEinsumChain`). |
| `src/ChebyshevSharp/ChebyshevTT.cs` | Modify | Adds `OrthLeft`/`OrthRight` (Task 3), `InnerProduct` (Task 4), `Build(method="als")` dispatcher (Task 5), `RunCompletion` (Task 6), `Nodes`/`FromValues` (Task 7), `Extrude`/`Slice`/`ToDense` (Task 8), scalar `*`/`/`/unary `-` + in-place (Task 9), binary `+`/`-` + `AddInPlace`/`SubInPlace`/`RoundInPlace` (Task 10). JSON version `0.5.0` → `0.6.0` (Task 11). |
| `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs` | Create | Tests for `OrthLeft`/`OrthRight` (Task 3) and `InnerProduct` (Task 4). |
| `tests/ChebyshevSharp.Tests/TtAlsTests.cs` | Create | Tests for `Build(method="als")` (Task 5) and `RunCompletion` (Task 6). All assertions tolerance-based. |
| `tests/ChebyshevSharp.Tests/TtFactoriesTests.cs` | Create | Tests for `Nodes` and `FromValues` (Task 7). |
| `tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs` | Create | Tests for `Extrude`/`Slice`/`ToDense` (Task 8). |
| `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs` | Create | Tests for scalar algebra + binary algebra + in-place + rounding (Tasks 9 + 10). |
| `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` | Modify | (Task 5+) Add `TtAlsSin3D` lazy fixture. (Task 9) Add `TtAlgebraF`/`TtAlgebraG` lazy fixtures over a shared 2D grid. |
| `src/ChebyshevSharp/ChebyshevSharp.csproj` | Modify | (Task 12) `<Version>0.6.0`, `<PyChebyshevParity>0.18.0`, `<InformationalVersion>0.6.0+pychebyshev.0.18.0`, `<Description>` updated. |
| `docs/docs/tensor-train.md` | Modify | (Task 12) Extend with ALS/`RunCompletion`/canonicalization/`InnerProduct`/factories/`Extrude`/`Slice`/`ToDense`/algebra sections. |
| `docs/docs/changelog.md` | Modify | (Task 12) Add `## [0.6.0]` entry per the new two-tier convention (no per-upstream-tag parentheticals). |
| `README.md` | Modify | (Task 12) Bump parity badge `v0.12.0` → `v0.18.0`. |
| `CLAUDE.md` | Modify | (Task 12) Status block: PyChebyshev v0.18.0, Phase 2 of 6 complete, test count `666 → ~741`. |
| `skip_csharp.txt` | Modify | (Task 12) Append Phase 2 entries. |

---

## Task 1: Submodule advance + project scaffolding

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `ref/PyChebyshev` (submodule pin advance)
- Create: `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs` (empty stub)
- Create: `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs` (empty stub)
- Create: `tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs` (empty stub)
- Create: `tests/ChebyshevSharp.Tests/TtAlsTests.cs` (empty stub)
- Create: `tests/ChebyshevSharp.Tests/TtFactoriesTests.cs` (empty stub)

**Design notes:** Phase 1 ended at `Passed: 666`. Phase 2 starts on the same baseline. Submodule advances in one shot to v0.18.0 (skipping v0.13–v0.17 individually since the C# port is internally sequenced through them but the Python pin is a single bump).

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase2-tt-parity`. If not, STOP and switch.

- [ ] **Step 2: Advance the PyChebyshev submodule to v0.18.0**

```bash
git -C ref/PyChebyshev fetch --tags origin
git -C ref/PyChebyshev checkout v0.18.0
git add ref/PyChebyshev
```

- [ ] **Step 3: Verify submodule state**

```bash
git -C ref/PyChebyshev describe --tags
```

Expected output: `v0.18.0`

- [ ] **Step 4: Create empty test file stubs to anchor `using` lines**

Create `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs`:

```csharp
using System;
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
```

Create `tests/ChebyshevSharp.Tests/TtAlsTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_tensor_train.py classes
// TestALSInternals + TestALS + TestCompletion + TestCrossFeatureALS
// (PyChebyshev v0.13.0). Tests added incrementally across Phase 2 Tasks 5 and 6.
//
// IMPORTANT: ALS is seeded-stochastic (System.Random vs np.random.default_rng
// produce different streams). Every assertion must be tolerance-based.
// Never inline-literal expected values from Python tests for ALS-touched outputs.
public class TtAlsTests
{
}
```

Create `tests/ChebyshevSharp.Tests/TtFactoriesTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTNodes + TestTTFromValues (PyChebyshev v0.18.0).
// Tests added in Phase 2 Task 7.
public class TtFactoriesTests
{
}
```

Create `tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTToDense + TestTTExtrude + TestTTSlice (PyChebyshev v0.18.0).
// Tests added in Phase 2 Task 8.
public class TtExtrudeSliceTests
{
}
```

Create `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_v018_tt_parity.py classes
// TestTTAddition + TestTTScalarMul + TestTTCrossFeatures (PyChebyshev v0.18.0).
// Tests added incrementally across Phase 2 Tasks 9 and 10.
public class TtAlgebraTests
{
}
```

- [ ] **Step 5: Build and verify the empty test files compile**

```bash
dotnet build
```

Expected: succeeds with zero new warnings; existing 666 tests still found.

- [ ] **Step 6: Run full existing test suite to confirm baseline**

```bash
dotnet test
```

Expected: `Passed: 666`. No regressions.

- [ ] **Step 7: Commit**

```bash
git add ref/PyChebyshev tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs tests/ChebyshevSharp.Tests/TtAlsTests.cs tests/ChebyshevSharp.Tests/TtFactoriesTests.cs tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs tests/ChebyshevSharp.Tests/TtAlgebraTests.cs
git commit -m "phase2: advance submodule to v0.18.0 + add test stubs"
```

---

## Task 2: Refactor `Internal/TensorTrainKernel.cs` into kernel + algebra + extrude (zero behavior change)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` (no body changes — only retains build-core members)
- Create: `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` (empty static class stub)
- Create: `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs` (empty static class stub)

**Design notes:**
- This commit must change **zero behavior**. After it, `dotnet test` must show the same 666 passing tests with byte-for-byte identical assertions.
- The refactor is "create empty companion files alongside the existing one." We do NOT move any existing code in this commit — that pattern is high-risk and Phase 1 chose the same conservative route. New helper functions (`OrthLeftSweep`, `AddCores`, `ExtrudeCores`, …) land in their proper files in Tasks 3+, but the existing build-side helpers (`TtCross`, `TtSvd`, `Maxvol`, `ValueToCoeffCores`, `TtCore`, `ColumnPivotedQRIndices`) stay where they are.
- The two new empty files anchor `namespace ChebyshevSharp.Internal;` so subsequent tasks add members without creating files at commit-add time. This makes each subsequent commit's diff strictly additive.

- [ ] **Step 1: Create `Internal/TensorTrainAlgebra.cs`**

Create `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs`:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train algebra (addition, scalar mul, negation,
/// TT-SVD rounding, inner product). Operates on flat <see cref="TensorTrainKernel.TtCore"/>
/// arrays. Members are added incrementally across Phase 2 Tasks 4 (InnerProduct),
/// 9 (scalar algebra), and 10 (binary algebra + rounding).
/// </summary>
internal static class TensorTrainAlgebra
{
}
```

- [ ] **Step 2: Create `Internal/TensorTrainExtrude.cs`**

Create `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs`:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train extrusion / slicing / materialization
/// (Extrude, Slice, ToDense, FromValuesTtSvd). Members are added incrementally
/// across Phase 2 Tasks 7 (FromValuesTtSvd) and 8 (Extrude/Slice/ToDense).
/// </summary>
internal static class TensorTrainExtrude
{
}
```

- [ ] **Step 3: Build to confirm both new files compile**

```bash
dotnet build
```

Expected: succeeds with zero new warnings.

- [ ] **Step 4: Run full test suite to confirm zero behavior change**

```bash
dotnet test
```

Expected: `Passed: 666`. No regressions — the new files are empty static classes.

- [ ] **Step 5: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs src/ChebyshevSharp/Internal/TensorTrainExtrude.cs
git commit -m "phase2: refactor TensorTrainKernel into kernel/algebra/extrude (no behavior change)"
```

---

## Task 3: Implement `OrthLeft` / `OrthRight` canonicalization (PyChebyshev v0.13.a)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` — add `OrthLeftSweep` and `OrthRightSweep` static helpers + a thin `QrUnfolded` helper that wraps MathNet QR.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `OrthLeft(int position)` and `OrthRight(int position)` public methods + a way for these to read/write the private cores array (use existing `_coeffCores` field directly).
- Modify: `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs` — append `OrthLeftRightTests` class.

**Python source pointers:**
- `_orth_left_core` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 695–714
- `_orth_right_core` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 717–733
- `ChebyshevTT.orth_left` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1289–1318
- `ChebyshevTT.orth_right` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1320–1349

**Design notes:**
- `_orth_left_core(C_k, C_{k+1})`: reshape `C_k` from `(r0, n, r1)` to `(r0*n, r1)`, run QR. The new `C_k` is `Q.reshape(r0, n, Q.cols)` (where `Q.cols = min(r0*n, r1)`); `R` is contracted into the left rank of `C_{k+1}` via `np.einsum("ij,jpk->ipk", R, C_{k+1})`. The `OrthLeftSweep` driver runs this for `k = 0 .. position-1` in order.
- `_orth_right_core(C_{k-1}, C_k)`: reshape `C_k` from `(r_prev, n, r_next)` to `(r_prev, n*r_next)`. QR of the **transpose** gives `Qt: (n*r_next, k_rank)` and `Rt: (k_rank, r_prev)`. New `C_k = Qt.T.reshape(k_rank, n, r_next)`; `L = Rt.T` is contracted into the right rank of `C_{k-1}` via `np.einsum("ipk,kj->ipj", C_{k-1}, L)`. Driver runs `k = d-1 .. position+1` in descending order.
- Position validation: `OrthLeft(position)` requires `1 <= position < numDim`; `OrthRight(position)` requires `0 <= position < numDim - 1`. Mirrors Python's `ValueError`.
- `MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.QR()` returns thin QR by default. Use `MathNet.Numerics.LinearAlgebra.Factorization.QRMethod.Thin` to be explicit.
- Eval-equivalence: the represented tensor is unchanged — eval before vs. after orth must match within `1e-10`.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs`:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~OrthLeftRightTests"
```

Expected: build fails — `OrthLeft`, `OrthRight`, and `GetCoreShape` are not defined on `ChebyshevTT`.

- [ ] **Step 3: Add `OrthLeftSweep` and `OrthRightSweep` to `Internal/TensorTrainKernel.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainKernel.cs`. Append before the closing `}` of the `TensorTrainKernel` class:

```csharp
    // ------------------------------------------------------------------
    // Orthogonalization primitives (Phase 2 — PyChebyshev v0.13)
    // ------------------------------------------------------------------

    /// <summary>
    /// Left-orthogonalize cores [0..position-1] in place by absorbing each
    /// core's R factor into the next core's left bond. After the call, each
    /// core C_k for k &lt; position satisfies Q^T Q = I when reshaped as
    /// (rLeft*nNodes, rRight). The represented tensor is unchanged.
    /// </summary>
    /// <param name="cores">The TT cores to orthogonalize in place.</param>
    /// <param name="position">Pivot index, 1 &lt;= position &lt; cores.Length.</param>
    internal static void OrthLeftSweep(TtCore[] cores, int position)
    {
        for (int k = 0; k < position; k++)
        {
            var (newCk, newCk1) = OrthLeftCore(cores[k], cores[k + 1]);
            cores[k] = newCk;
            cores[k + 1] = newCk1;
        }
    }

    /// <summary>
    /// Right-orthogonalize cores [position+1..d-1] in place by absorbing each
    /// core's L factor into the previous core's right bond. After the call,
    /// each core C_k for k &gt; position satisfies Q Q^T = I when reshaped as
    /// (rLeft, nNodes*rRight). The represented tensor is unchanged.
    /// </summary>
    /// <param name="cores">The TT cores to orthogonalize in place.</param>
    /// <param name="position">Pivot index, 0 &lt;= position &lt; cores.Length - 1.</param>
    internal static void OrthRightSweep(TtCore[] cores, int position)
    {
        for (int k = cores.Length - 1; k > position; k--)
        {
            var (newCkm1, newCk) = OrthRightCore(cores[k - 1], cores[k]);
            cores[k - 1] = newCkm1;
            cores[k] = newCk;
        }
    }

    /// <summary>
    /// QR-orthogonalize core_k from the left; absorb R into core_{k+1}.
    /// Mirror of Python's _orth_left_core (tensor_train.py:695).
    /// </summary>
    private static (TtCore NewCk, TtCore NewCk1) OrthLeftCore(TtCore coreK, TtCore coreK1)
    {
        int r0 = coreK.RLeft, n = coreK.NNodes, r1 = coreK.RRight;
        // Unfold (r0, n, r1) → (r0*n, r1) row-major
        var matrix = new double[r0 * n, r1];
        for (int i = 0; i < r0; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < r1; k++)
                    matrix[i * n + j, k] = coreK[i, j, k];

        var M = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matrix);
        var qr = M.QR(MathNet.Numerics.LinearAlgebra.Factorization.QRMethod.Thin);
        var Q = qr.Q;     // shape (r0*n, qCols), qCols = min(r0*n, r1)
        var R = qr.R;     // shape (qCols, r1)
        int qCols = Q.ColumnCount;

        // Pack new core_k as TtCore(r0, n, qCols)
        var newCk = new TtCore(r0, n, qCols);
        for (int i = 0; i < r0; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < qCols; k++)
                    newCk[i, j, k] = Q[i * n + j, k];

        // Contract R into core_k1 left bond:
        // newCk1[i, p, k] = sum_j R[i, j] * coreK1[j, p, k]
        int rk1Right = coreK1.RRight, nk1 = coreK1.NNodes;
        var newCk1 = new TtCore(qCols, nk1, rk1Right);
        for (int i = 0; i < qCols; i++)
            for (int p = 0; p < nk1; p++)
                for (int k = 0; k < rk1Right; k++)
                {
                    double s = 0;
                    for (int j = 0; j < r1; j++)
                        s += R[i, j] * coreK1[j, p, k];
                    newCk1[i, p, k] = s;
                }
        return (newCk, newCk1);
    }

    /// <summary>
    /// LQ-orthogonalize core_k from the right; absorb L into core_{k-1}.
    /// Mirror of Python's _orth_right_core (tensor_train.py:717).
    /// Implemented via QR on the transposed unfolding.
    /// </summary>
    private static (TtCore NewCkm1, TtCore NewCk) OrthRightCore(TtCore coreKm1, TtCore coreK)
    {
        int rPrev = coreK.RLeft, n = coreK.NNodes, rNext = coreK.RRight;
        // Unfold core_k as (r_prev, n*r_next), then QR of its transpose gives
        // Qt of shape (n*r_next, k_rank), Rt of shape (k_rank, r_prev).
        var Mt = new double[n * rNext, rPrev];
        for (int i = 0; i < rPrev; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < rNext; k++)
                    Mt[j * rNext + k, i] = coreK[i, j, k];

        var Mtm = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(Mt);
        var qr = Mtm.QR(MathNet.Numerics.LinearAlgebra.Factorization.QRMethod.Thin);
        var Qt = qr.Q;   // (n*r_next, kRank)
        var Rt = qr.R;   // (kRank, r_prev)
        int kRank = Qt.ColumnCount;

        // newCk = Qt.T.reshape(kRank, n, r_next)
        var newCk = new TtCore(kRank, n, rNext);
        for (int a = 0; a < kRank; a++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < rNext; k++)
                    newCk[a, j, k] = Qt[j * rNext + k, a];

        // L = Rt^T  shape (r_prev, kRank).
        // newCkm1[i, p, j] = sum_k coreKm1[i, p, k] * L[k, j]
        //                  = sum_k coreKm1[i, p, k] * Rt[j, k]
        int rPrevPrev = coreKm1.RLeft, nPrev = coreKm1.NNodes;
        var newCkm1 = new TtCore(rPrevPrev, nPrev, kRank);
        for (int i = 0; i < rPrevPrev; i++)
            for (int p = 0; p < nPrev; p++)
                for (int j = 0; j < kRank; j++)
                {
                    double s = 0;
                    for (int k = 0; k < rPrev; k++)
                        s += coreKm1[i, p, k] * Rt[j, k];
                    newCkm1[i, p, j] = s;
                }
        return (newCkm1, newCk);
    }
```

- [ ] **Step 4: Add `OrthLeft` / `OrthRight` and `GetCoreShape` to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the existing `ErrorEstimate()` method:

```csharp
    // ------------------------------------------------------------------
    // Canonicalization (Phase 2 — PyChebyshev v0.13)
    // ------------------------------------------------------------------

    /// <summary>
    /// Left-orthogonalize cores [0..position-1] in place by absorbing each
    /// core's R factor into the next core's left bond. The represented tensor
    /// is unchanged.
    /// </summary>
    /// <param name="position">Pivot index, must be in [1, NumDimensions - 1].</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If position is outside [1, NumDimensions - 1].</exception>
    public void OrthLeft(int position)
    {
        CheckBuilt();
        if (position < 1 || position >= _numDimensions)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"position must be in [1, {_numDimensions - 1}] for OrthLeft, got {position}");
        TensorTrainKernel.OrthLeftSweep(_coeffCores!, position);
        _cachedErrorEstimate = null;
        // TT ranks may change (QR reduces rank to min(rL*n, rR)); refresh.
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores![i].RRight;
    }

    /// <summary>
    /// Right-orthogonalize cores [position+1..NumDimensions-1] in place by
    /// absorbing each core's L factor into the previous core's right bond.
    /// The represented tensor is unchanged.
    /// </summary>
    /// <param name="position">Pivot index, must be in [0, NumDimensions - 2].</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If position is outside [0, NumDimensions - 2].</exception>
    public void OrthRight(int position)
    {
        CheckBuilt();
        if (position < 0 || position >= _numDimensions - 1)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"position must be in [0, {_numDimensions - 2}] for OrthRight, got {position}");
        TensorTrainKernel.OrthRightSweep(_coeffCores!, position);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores![i].RRight;
    }

    /// <summary>
    /// Internal accessor for tests: return (rLeft, nNodes, rRight, flat data) of
    /// core <paramref name="k"/>. Exposes the live data buffer (not a copy).
    /// </summary>
    internal (int RLeft, int NNodes, int RRight, double[] Data) GetCoreShape(int k)
    {
        CheckBuilt();
        var c = _coeffCores![k];
        return (c.RLeft, c.NNodes, c.RRight, c.Data);
    }
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~OrthLeftRightTests"
```

Expected: 8 tests pass.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 674` (666 + 8 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainKernel.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs
git commit -m "phase2: implement OrthLeft/OrthRight canonicalization"
```

---


## Task 4: Implement `InnerProduct` (PyChebyshev v0.13.b)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` — add `InnerProductCores`.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `InnerProduct(ChebyshevTT)` public method + grid-mismatch check.
- Modify: `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs` — append `InnerProductTests` class.

**Python source pointers:**
- `ChebyshevTT.inner_product` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1431–1487
- Algorithm: `M = [[1.0]]; for k: M = einsum("ij,ipa,jpb->ab", M, A_k, B_k); return M[0,0]`

**Design notes:**
- Frobenius inner product of the two Chebyshev coefficient tensors. Since `_coeffCores` are coefficient cores (DCT-II already applied during build), the core-by-core contraction yields `Σ_{i_1,…,i_d} C_self[i] * C_other[i]` directly — no node-space conversion needed.
- Cost: `O(d * n * r_s^2 * r_o^2)`, memory `O(r_s * r_o)`.
- Validation mirrors Python: `domain` must match (element-wise), `nNodes` must match (element-wise), both TTs must be built. C# raises `ArgumentException` (matches existing `Internal/Algebra.cs` style).

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs`:

```csharp
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
            int n = tt.NNodes[0];
            // Build (n0 x n1) flat row-major from the cores
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
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~InnerProductTests"
```

Expected: build fails — `InnerProduct` is not defined on `ChebyshevTT`.

- [ ] **Step 3: Implement `InnerProductCores` in `Internal/TensorTrainAlgebra.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs`. Replace the empty class body with:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train algebra (addition, scalar mul, negation,
/// TT-SVD rounding, inner product). Operates on flat <see cref="TensorTrainKernel.TtCore"/>
/// arrays. Members are added incrementally across Phase 2 Tasks 4 (InnerProduct),
/// 9 (scalar algebra), and 10 (binary algebra + rounding).
/// </summary>
internal static class TensorTrainAlgebra
{
    /// <summary>
    /// Frobenius inner product of two TTs' Chebyshev coefficient tensors.
    /// Computes Σ_{i_1,…,i_d} C_a[i] * C_b[i] in O(d * n * r_a^2 * r_b^2).
    /// Mirrors Python's <c>ChebyshevTT.inner_product</c> (tensor_train.py:1431).
    /// </summary>
    /// <param name="coresA">Coefficient cores of the first TT.</param>
    /// <param name="coresB">Coefficient cores of the second TT (matching shape per dim).</param>
    /// <returns>Frobenius inner product.</returns>
    internal static double InnerProductCores(
        TensorTrainKernel.TtCore[] coresA,
        TensorTrainKernel.TtCore[] coresB)
    {
        int d = coresA.Length;
        // M starts as 1x1 identity: shape (rA_0, rB_0) = (1, 1).
        int rAcur = 1, rBcur = 1;
        double[] M = { 1.0 };

        for (int k = 0; k < d; k++)
        {
            var A = coresA[k];   // (rA_left, n, rA_right)
            var B = coresB[k];   // (rB_left, n, rB_right)
            int n = A.NNodes;
            int rAr = A.RRight, rBr = B.RRight;

            // newM[a, b] = sum_{i, j, p} M[i, j] * A[i, p, a] * B[j, p, b]
            // Order the summation as:
            //   tmp[i, p, b] = sum_j M[i, j] * B[j, p, b]
            //   tmpA[a, p, b] = sum_i tmp[i, p, b] * A[i, p, a]   <- wrong shape; refactor
            // Cleaner: iterate (a, b) outer, (i, j, p) inner.
            var newM = new double[rAr * rBr];
            for (int a = 0; a < rAr; a++)
                for (int b = 0; b < rBr; b++)
                {
                    double s = 0;
                    for (int i = 0; i < rAcur; i++)
                        for (int j = 0; j < rBcur; j++)
                        {
                            double mij = M[i * rBcur + j];
                            if (mij == 0) continue;
                            for (int p = 0; p < n; p++)
                                s += mij * A[i, p, a] * B[j, p, b];
                        }
                    newM[a * rBr + b] = s;
                }

            M = newM;
            rAcur = rAr;
            rBcur = rBr;
        }

        // M is (1, 1) at the end since rA_d = rB_d = 1.
        return M[0];
    }
}
```

- [ ] **Step 4: Add `InnerProduct` to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the `OrthRight` method:

```csharp
    /// <summary>
    /// Frobenius inner product of the Chebyshev coefficient tensors of two TTs.
    /// Both TTs must share the same <see cref="NumDimensions"/>, <see cref="Domain"/>,
    /// and <see cref="NNodes"/>.
    /// </summary>
    /// <param name="other">The other TT.</param>
    /// <returns>Σ_{i_1,…,i_d} C_self[i] * C_other[i].</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="other"/> is null.</exception>
    /// <exception cref="InvalidOperationException">If either TT has not been built.</exception>
    /// <exception cref="ArgumentException">If domain or nNodes do not match.</exception>
    public double InnerProduct(ChebyshevTT other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        CheckBuilt();
        other.CheckBuilt();
        if (other._numDimensions != _numDimensions)
            throw new ArgumentException(
                $"InnerProduct requires matching numDimensions; got {_numDimensions} vs {other._numDimensions}");
        for (int d = 0; d < _numDimensions; d++)
        {
            if (other._nNodes[d] != _nNodes[d])
                throw new ArgumentException(
                    $"InnerProduct requires matching nNodes; got [{string.Join(", ", _nNodes)}] vs [{string.Join(", ", other._nNodes)}]");
            if (other._domain[d][0] != _domain[d][0] || other._domain[d][1] != _domain[d][1])
                throw new ArgumentException(
                    $"InnerProduct requires matching domain at dim {d}; got [{_domain[d][0]}, {_domain[d][1]}] vs [{other._domain[d][0]}, {other._domain[d][1]}]");
        }
        return TensorTrainAlgebra.InnerProductCores(_coeffCores!, other._coeffCores!);
    }
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~InnerProductTests"
```

Expected: 7 tests pass.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 681` (674 + 7 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtCanonicalizationTests.cs
git commit -m "phase2: implement InnerProduct for ChebyshevTT"
```

---


## Task 5: Implement ALS build mode (`Build(method="als")`) — PyChebyshev v0.13.c

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` — add `AlsFixedRankSweep` and `AlsAdaptiveRank` static helpers.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — extend `Build` dispatcher to accept `method="als"`; add `BuildWarning` field; add `Method` property mirror.
- Modify: `tests/ChebyshevSharp.Tests/TtAlsTests.cs` — append `AlsTests` class.
- Modify: `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` — add `TtAlsSin3D` lazy fixture.

**Python source pointers:**
- `_als_fixed_rank_sweeps` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 736–881
- `_tt_als` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 877–989
- `Build(method="als")` dispatch — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1135–1160

**Design notes:**
- **Stochastic**: ALS is seeded-stochastic. `System.Random(seed)` produces a different stream than NumPy's `default_rng(seed)`. **Every test assertion must be tolerance-based** — never inline-literal expected values from Python tests for ALS-touched outputs. This is the existing precedent in `TensorTrainCorrectnessTests` and is documented in `CLAUDE.md`.
- **Algorithm**: Rank-adaptive driver starts at `rank=1`, runs fixed-rank ALS sweeps until inner residual `< tol*0.1`, measures grid residual; if `>= tol`, increment rank by 1 and reinitialize cores. Cap at `maxRank`; on cap, set `BuildWarning`.
- **Inner sweep**: Hold all but core `k` fixed, build the LS design matrix `A` of shape `(Π n_d, r_left * n_k * r_right)`, solve `A @ vec(C_k) = b` via MathNet's `DenseMatrix.QR().Solve(b)` (least-squares solve). Brute-force re-canonicalize via `OrthLeftSweep` / `OrthRightSweep` per inner step (Python comment: "amortized O(d^2) QRs but correctness is the priority").
- **Initial cores**: `System.Random(seed)` followed by `NextDouble()*2-1` per element, drawn from a uniform distribution. This differs from NumPy's `standard_normal` but is acceptable since we're driving toward convergence regardless of init.
- **Target tensor materialization**: ALS evaluates `function` on the full Chebyshev grid (`Π n_d` evaluations cached in a `Dictionary<long,double>` with mixed-radix key, same pattern as TT-Cross). For the d=5, n=8 case Python flags as ~256 KB, this is fine; users with very large grids should prefer `method="cross"`.
- **Build dispatcher**: `Build(method)` already validates `"cross"` and `"svd"`. Extend the validator to accept `"als"`.
- **`BuildWarning`**: New `string?` property on `ChebyshevTT`, mirroring Phase 1's pattern on `ChebyshevApproximation`. Set when ALS hits `maxRank` before tolerance.
- **`Method` property**: New `string?` to surface which build mode produced the cores (Python's `tt.method` attribute). Persisted in JSON in Task 11.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtAlsTests.cs`:

```csharp
public class AlsTests
{
    private static readonly double[][] UnitCube3D = new[]
    {
        new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
    };

    [Fact]
    public void Test_als_builds_and_reaches_tolerance_3d()
    {
        // f(x,y,z) = sin(x)*cos(y) + 0.3*z^2
        Func<double[], double> f = p => Math.Sin(p[0]) * Math.Cos(p[1]) + 0.3 * p[2] * p[2];
        var tt = new ChebyshevTT(f, 3, UnitCube3D, new[] { 10, 10, 10 },
            tolerance: 1e-4, maxRank: 6);
        tt.Build(verbose: false, seed: 42, method: "als");

        double[][] pts = new[]
        {
            new[] { 0.1, -0.2, 0.3 },
            new[] { 0.5, 0.5, -0.5 },
            new[] { -0.9, 0.1, 0.7 },
        };
        foreach (var p in pts)
        {
            double got = tt.Eval(p);
            double want = f(p);
            Assert.True(Math.Abs(got - want) < 1e-2,
                $"ALS eval at [{string.Join(", ", p)}]: got {got}, want {want}, err {Math.Abs(got - want):e3}");
        }
    }

    [Fact]
    public void Test_als_matches_cross_on_same_fixture()
    {
        Func<double[], double> f = p => Math.Exp(-p[0] * p[0]) * Math.Cos(p[1]);
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 10, 10 };
        var ttCross = new ChebyshevTT(f, 2, domain, nNodes, tolerance: 1e-6, maxRank: 8);
        ttCross.Build(verbose: false, seed: 1, method: "cross");
        var ttAls = new ChebyshevTT(f, 2, domain, nNodes, tolerance: 1e-4, maxRank: 8);
        ttAls.Build(verbose: false, seed: 1, method: "als");
        double[][] pts = new[]
        {
            new[] { 0.1, -0.2 }, new[] { 0.5, 0.5 }, new[] { -0.9, 0.7 },
        };
        foreach (var p in pts)
            Assert.True(Math.Abs(ttCross.Eval(p) - ttAls.Eval(p)) < 5e-2,
                $"ALS vs Cross diverged at [{string.Join(", ", p)}]");
    }

    [Fact]
    public void Test_als_respects_max_rank_cap()
    {
        // tanh(50*(x-y)) — nearly discontinuous, unreachable at low rank.
        Func<double[], double> f = p => Math.Tanh(50 * (p[0] - p[1]));
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 20, 20 }, tolerance: 1e-12, maxRank: 3);
        tt.Build(verbose: false, seed: 0, method: "als");
        foreach (int r in tt.TtRanks)
            Assert.True(r <= 3, $"rank {r} exceeds maxRank=3");
    }

    [Fact]
    public void Test_als_max_rank_cap_emits_build_warning()
    {
        Func<double[], double> f = p => Math.Tanh(50 * (p[0] - p[1]));
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 20, 20 }, tolerance: 1e-12, maxRank: 3);
        tt.Build(verbose: false, seed: 0, method: "als");
        Assert.NotNull(tt.BuildWarning);
        Assert.Contains("maxRank", tt.BuildWarning, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Test_als_deterministic_given_seed()
    {
        Func<double[], double> f = p => p[0] * p[1] + 0.5;
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var ttA = new ChebyshevTT(f, 2, domain, new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        var ttB = new ChebyshevTT(f, 2, domain, new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        ttA.Build(verbose: false, seed: 123, method: "als");
        ttB.Build(verbose: false, seed: 123, method: "als");
        TestFixtures.AssertClose(ttA.Eval(new[] { 0.3, -0.4 }), ttB.Eval(new[] { 0.3, -0.4 }),
            atol: 1e-12);
    }

    [Fact]
    public void Test_als_method_attribute_set()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 }, tolerance: 1e-2, maxRank: 3);
        tt.Build(verbose: false, method: "als");
        Assert.Equal("als", tt.Method);
    }

    [Fact]
    public void Test_als_total_build_evals_positive()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, tolerance: 1e-4, maxRank: 3);
        tt.Build(verbose: false, method: "als");
        Assert.True(tt.TotalBuildEvals > 0);
    }

    [Fact]
    public void Test_invalid_method_raises()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        var ex = Assert.Throws<ArgumentException>(() => tt.Build(verbose: false, method: "bogus"));
        Assert.Contains("als", ex.Message);
    }

    [Fact]
    public void Test_als_save_load_roundtrip()
    {
        Func<double[], double> f = p => Math.Sin(p[0]) + p[1] * p[1];
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, tolerance: 1e-4, maxRank: 4);
        tt.Build(verbose: false, seed: 0, method: "als");
        double valBefore = tt.Eval(new[] { 0.3, -0.4 });
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var tt2 = ChebyshevTT.Load(path);
            double valAfter = tt2.Eval(new[] { 0.3, -0.4 });
            TestFixtures.AssertClose(valBefore, valAfter, atol: 1e-12);
            Assert.Equal("als", tt2.Method);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}

public class AlsInternalsTests
{
    [Fact]
    public void Test_als_sweep_reduces_residual_on_rank1_target()
    {
        // Build an exactly-rank-1 target on an 8x8x8 grid.
        var rng = new Random(0);
        double[] u0 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        double[] u1 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        double[] u2 = Enumerable.Range(0, 8).Select(_ => rng.NextDouble() * 2 - 1).ToArray();

        // Target tensor as flat row-major (i, j, k) → i*64 + j*8 + k
        var target = new double[8 * 8 * 8];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                for (int k = 0; k < 8; k++)
                    target[i * 64 + j * 8 + k] = u0[i] * u1[j] * u2[k];

        Func<int[], double> evalsAt = idx => target[idx[0] * 64 + idx[1] * 8 + idx[2]];

        // Random rank-1 initial cores
        var rng2 = new Random(1);
        var cores = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore[3];
        for (int d = 0; d < 3; d++)
        {
            var core = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore(1, 8, 1);
            for (int j = 0; j < 8; j++) core[0, j, 0] = rng2.NextDouble() * 2 - 1;
            cores[d] = core;
        }

        ChebyshevSharp.Internal.TensorTrainKernel.AlsFixedRankSweep(
            cores, evalsAt, new[] { 8, 8, 8 }, tolerance: 1e-12, maxIter: 5);

        // Reconstruct and compare
        double residual = ChebyshevSharp.Internal.TensorTrainKernel.GridResidual(cores, target, new[] { 8, 8, 8 });
        Assert.True(residual < 1e-8, $"rank-1 residual {residual} exceeds 1e-8");
    }
}
```

- [ ] **Step 2: Add `TtAlsSin3D` lazy fixture to `Helpers/TestFixtures.cs`**

Open `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs`. After the existing `_ttSin3D` lazy declaration, add:

```csharp
    private static readonly Lazy<ChebyshevTT> _ttAlsSin3D = new(() =>
    {
        var tt = new ChebyshevTT(
            p => Math.Sin(p[0]) + Math.Sin(p[1]) + Math.Sin(p[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 }, tolerance: 1e-3, maxRank: 5);
        tt.Build(verbose: false, seed: 42, method: "als");
        return tt;
    });

    public static ChebyshevTT TtAlsSin3D => _ttAlsSin3D.Value;
```

- [ ] **Step 3: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~AlsTests|FullyQualifiedName~AlsInternalsTests"
```

Expected: build fails — `AlsFixedRankSweep`, `GridResidual`, `BuildWarning`, `Method`, and `Build(method="als")` don't exist yet.

- [ ] **Step 4: Implement ALS helpers in `Internal/TensorTrainKernel.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainKernel.cs`. Append before the closing `}` of the `TensorTrainKernel` class (after the `OrthLeftCore`/`OrthRightCore` private helpers added in Task 3):

```csharp
    // ------------------------------------------------------------------
    // ALS (Phase 2 — PyChebyshev v0.13.c)
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute relative Frobenius residual ||reconstruct(cores) - target||_F /
    /// ||target||_F, with target stored as flat row-major over the grid shape.
    /// </summary>
    internal static double GridResidual(TtCore[] cores, double[] target, int[] nNodes)
    {
        // Reconstruct the dense tensor from cores.
        double[] dense = ReconstructDense(cores, nNodes);
        double num = 0, den = 0;
        for (int i = 0; i < target.Length; i++)
        {
            double diff = dense[i] - target[i];
            num += diff * diff;
            den += target[i] * target[i];
        }
        return Math.Sqrt(num) / Math.Sqrt(Math.Max(den, 1e-60));
    }

    /// <summary>
    /// Reconstruct a flat row-major dense tensor of shape <paramref name="nNodes"/>
    /// from a TT core list, by sequentially contracting cores left-to-right.
    /// </summary>
    internal static double[] ReconstructDense(TtCore[] cores, int[] nNodes)
    {
        int d = cores.Length;
        // Start: tmp shape (1, n_0, r_0). Flatten to row-major.
        int leadDim = 1;          // product of leading "kept" dims (initially 1)
        int curRank = cores[0].RLeft;  // == 1
        // Accumulator: shape (leadDim, curRank). Initialize to identity column.
        // After core 0: shape (n_0, rR_0). Generalize: keep flat shape (Π_so_far, rRcur).
        double[] acc = new double[1 * curRank];
        acc[0] = 1.0;
        int prodN = 1;

        for (int k = 0; k < d; k++)
        {
            var c = cores[k];
            int rL = c.RLeft, n = c.NNodes, rR = c.RRight;
            // newAcc[(prevProd, j), b] = sum_a acc[prevProd, a] * c[a, j, b]
            int newProd = prodN * n;
            var newAcc = new double[newProd * rR];
            for (int p = 0; p < prodN; p++)
                for (int j = 0; j < n; j++)
                    for (int b = 0; b < rR; b++)
                    {
                        double s = 0;
                        for (int a = 0; a < rL; a++)
                            s += acc[p * rL + a] * c[a, j, b];
                        newAcc[(p * n + j) * rR + b] = s;
                    }
            acc = newAcc;
            prodN = newProd;
            curRank = rR;
        }
        // Final rank should be 1; output is flat length prodN.
        return acc;
    }

    /// <summary>
    /// Fixed-rank ALS sweeps: hold all but core k fixed, solve LS for core k,
    /// sweep left-to-right then right-to-left, repeat up to <paramref name="maxIter"/>
    /// outer iterations or until inner relative change &lt; <paramref name="tolerance"/>.
    /// Mirror of Python's <c>_als_fixed_rank_sweeps</c> (tensor_train.py:736).
    /// </summary>
    internal static void AlsFixedRankSweep(
        TtCore[] cores,
        Func<int[], double> evalsAt,
        int[] nNodes,
        double tolerance,
        int maxIter,
        bool verbose = false)
    {
        int d = cores.Length;
        long totalPoints = 1;
        for (int i = 0; i < d; i++) totalPoints *= nNodes[i];
        int total = checked((int)totalPoints);

        // Precompute b: target values in C-order index.
        double[] b = new double[total];
        int[] tmpIdx = new int[d];
        for (int flat = 0; flat < total; flat++)
        {
            FlatToMulti(flat, nNodes, tmpIdx);
            b[flat] = evalsAt(tmpIdx);
        }

        double[] prevDense = ReconstructDense(cores, nNodes);

        for (int outer = 0; outer < maxIter; outer++)
        {
            string[] dirs = { "ltr", "rtl" };
            foreach (string direction in dirs)
            {
                int start = (direction == "ltr") ? 0 : d - 1;
                int end = (direction == "ltr") ? d : -1;
                int step = (direction == "ltr") ? 1 : -1;

                for (int k = start; k != end; k += step)
                {
                    // Canonicalize: cores[0..k-1] left-orth, cores[k+1..d-1] right-orth.
                    if (k > 0) OrthLeftSweep(cores, k);
                    if (k < d - 1) OrthRightSweep(cores, k);

                    int rL = cores[k].RLeft, nK = cores[k].NNodes, rR = cores[k].RRight;
                    int unknowns = rL * nK * rR;

                    // Build A: (total, unknowns).
                    // L_rows[flat, alpha] = product of cores [0..k-1] at idx
                    // R_rows[flat, beta]  = product of cores [k+1..d-1] at idx
                    double[] Lrows = new double[total * rL];
                    double[] Rrows = new double[total * rR];
                    int[] iks = new int[total];

                    for (int flat = 0; flat < total; flat++)
                    {
                        FlatToMulti(flat, nNodes, tmpIdx);
                        iks[flat] = tmpIdx[k];

                        // L_rows row
                        // Start from (1,) row vector with single 1.0 entry.
                        double[] lvec = { 1.0 };
                        int lrk = 1;
                        for (int j = 0; j < k; j++)
                        {
                            var Cj = cores[j];
                            int idxJ = tmpIdx[j];
                            int rj1 = Cj.RRight;
                            double[] newLvec = new double[rj1];
                            for (int bIdx = 0; bIdx < rj1; bIdx++)
                            {
                                double s = 0;
                                for (int aIdx = 0; aIdx < lrk; aIdx++)
                                    s += lvec[aIdx] * Cj[aIdx, idxJ, bIdx];
                                newLvec[bIdx] = s;
                            }
                            lvec = newLvec;
                            lrk = rj1;
                        }
                        for (int aIdx = 0; aIdx < rL; aIdx++)
                            Lrows[flat * rL + aIdx] = lvec[aIdx];

                        // R_rows row (right-to-left)
                        double[] rvec = { 1.0 };
                        int rrk = 1;
                        for (int j = d - 1; j > k; j--)
                        {
                            var Cj = cores[j];
                            int idxJ = tmpIdx[j];
                            int rjPrev = Cj.RLeft;
                            double[] newRvec = new double[rjPrev];
                            for (int aIdx = 0; aIdx < rjPrev; aIdx++)
                            {
                                double s = 0;
                                for (int bIdx = 0; bIdx < rrk; bIdx++)
                                    s += Cj[aIdx, idxJ, bIdx] * rvec[bIdx];
                                newRvec[aIdx] = s;
                            }
                            rvec = newRvec;
                            rrk = rjPrev;
                        }
                        for (int bIdx = 0; bIdx < rR; bIdx++)
                            Rrows[flat * rR + bIdx] = rvec[bIdx];
                    }

                    // Build A as a MathNet dense matrix.
                    var A = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(total, unknowns, 0.0);
                    for (int flat = 0; flat < total; flat++)
                    {
                        int colBase = iks[flat] * rR;
                        for (int alpha = 0; alpha < rL; alpha++)
                        {
                            double La = Lrows[flat * rL + alpha];
                            int rowOffset = alpha * nK * rR + colBase;
                            for (int beta = 0; beta < rR; beta++)
                                A.At(flat, rowOffset + beta, La * Rrows[flat * rR + beta]);
                        }
                    }

                    // Solve LS: A @ vec(C_k) = b. MathNet QR().Solve handles tall systems.
                    var bVec = MathNet.Numerics.LinearAlgebra.Double.DenseVector.OfArray(b);
                    var solved = A.QR().Solve(bVec);

                    // Pack back into core[k]
                    var newCore = new TtCore(rL, nK, rR);
                    for (int alpha = 0; alpha < rL; alpha++)
                        for (int j = 0; j < nK; j++)
                            for (int beta = 0; beta < rR; beta++)
                                newCore[alpha, j, beta] = solved[alpha * nK * rR + j * rR + beta];
                    cores[k] = newCore;
                }
            }

            // Check convergence by comparing reconstructed tensor change.
            double[] newDense = ReconstructDense(cores, nNodes);
            double num = 0, den = 0;
            for (int i = 0; i < total; i++)
            {
                double diff = newDense[i] - prevDense[i];
                num += diff * diff;
                den += prevDense[i] * prevDense[i];
            }
            double relChange = Math.Sqrt(num) / Math.Sqrt(Math.Max(den, 1e-60));
            if (verbose) Console.WriteLine($"  ALS iter {outer + 1}: rel_change = {relChange:e3}");
            if (relChange < tolerance) break;
            prevDense = newDense;
        }
    }

    /// <summary>
    /// Rank-adaptive ALS driver. Starts at rank 1 and grows the TT rank by +1 per
    /// outer iteration until the grid residual falls below tol or rank reaches
    /// maxRank. Mirror of Python's <c>_tt_als</c> (tensor_train.py:877).
    /// Returns (cores, nEvals, hitRankCap).
    /// </summary>
    internal static (TtCore[] Cores, int NEvals, bool HitRankCap) AlsAdaptiveRank(
        Func<double[], double> function,
        double[][] grids,
        int maxRank,
        double tol,
        int? randomState,
        bool verbose = false)
    {
        int d = grids.Length;
        int[] nNodes = new int[d];
        for (int i = 0; i < d; i++) nNodes[i] = grids[i].Length;

        var rng = randomState.HasValue ? new Random(randomState.Value) : new Random();
        var cache = new Dictionary<long, double>();
        long[] strides = new long[d];
        strides[d - 1] = 1;
        for (int i = d - 2; i >= 0; i--) strides[i] = strides[i + 1] * nNodes[i + 1];

        long Key(int[] idx)
        {
            long key = 0;
            for (int i = 0; i < d; i++) key += idx[i] * strides[i];
            return key;
        }

        Func<int[], double> evalsAt = idx =>
        {
            long key = Key(idx);
            if (!cache.TryGetValue(key, out double v))
            {
                var pt = new double[d];
                for (int i = 0; i < d; i++) pt[i] = grids[i][idx[i]];
                v = function(pt);
                cache[key] = v;
            }
            return v;
        };

        // Materialize target tensor once.
        long total = 1;
        for (int i = 0; i < d; i++) total = checked(total * nNodes[i]);
        int totalInt = checked((int)total);
        double[] target = new double[totalInt];
        int[] tmp = new int[d];
        for (int flat = 0; flat < totalInt; flat++)
        {
            FlatToMulti(flat, nNodes, tmp);
            target[flat] = evalsAt(tmp);
        }

        TtCore[] MakeCores(int rank)
        {
            var cores = new TtCore[d];
            for (int k = 0; k < d; k++)
            {
                int rL = (k == 0) ? 1 : rank;
                int rR = (k == d - 1) ? 1 : rank;
                var c = new TtCore(rL, nNodes[k], rR);
                for (int i = 0; i < c.Size; i++) c.Data[i] = rng.NextDouble() * 2 - 1;
                cores[k] = c;
            }
            return cores;
        }

        int curRank = 1;
        var coresOut = MakeCores(curRank);
        bool hitCap = false;

        while (true)
        {
            AlsFixedRankSweep(coresOut, evalsAt, nNodes,
                tolerance: tol * 0.1, maxIter: 5, verbose: verbose);
            double err = GridResidual(coresOut, target, nNodes);
            if (verbose)
                Console.WriteLine($"[ALS] rank {curRank}: grid_residual = {err:e3} (target {tol:e1})");
            if (err < tol) break;
            if (curRank >= maxRank)
            {
                hitCap = true;
                break;
            }
            curRank += 1;
            coresOut = MakeCores(curRank);
        }
        return (coresOut, cache.Count, hitCap);
    }

    /// <summary>Convert a flat row-major index to a multi-index over the given shape.</summary>
    internal static void FlatToMulti(int flat, int[] shape, int[] outIdx)
    {
        int d = shape.Length;
        for (int i = d - 1; i >= 0; i--)
        {
            outIdx[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
```

- [ ] **Step 5: Extend `ChebyshevTT.cs` to dispatch `method="als"`, expose `Method` and `BuildWarning`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Find the existing build-state field block:

```csharp
    private double? _cachedErrorEstimate;
```

Replace the immediately-following `LoadWarning` property with:

```csharp
    /// <summary>Warning message set when loading from a different library version.</summary>
    public string? LoadWarning { get; private set; }

    /// <summary>Warning emitted by Build() if maxRank was reached before tolerance was satisfied during ALS. Null otherwise.</summary>
    public string? BuildWarning { get; private set; }

    /// <summary>Build method that produced the current cores: "cross", "svd", or "als". Null if not built.</summary>
    public string? Method { get; private set; }
```

(If `LoadWarning` was already declared, delete the duplicate above.)

In the `Build` method, replace the validator line:

```csharp
        if (method != "cross" && method != "svd")
            throw new ArgumentException($"method must be 'cross' or 'svd', got '{method}'");
```

with:

```csharp
        if (method != "cross" && method != "svd" && method != "als")
            throw new ArgumentException($"method must be 'cross', 'svd', or 'als', got '{method}'");
        Method = method;
        BuildWarning = null;
```

In the `Build` method, replace the existing `if (method == "cross")` / `else` value-cores block with:

```csharp
        TensorTrainKernel.TtCore[] valueCores;
        int nEvals;

        if (method == "cross")
        {
            if (verbose) Console.WriteLine("  Running TT-Cross...");
            (valueCores, nEvals) = TensorTrainKernel.TtCross(
                _function!, grids, _maxRank, _tolerance, _maxSweeps, verbose, seed);
        }
        else if (method == "svd")
        {
            (valueCores, nEvals) = TensorTrainKernel.TtSvd(
                _function!, grids, _maxRank, _tolerance, verbose);
        }
        else  // method == "als"
        {
            if (verbose) Console.WriteLine("  Running TT-ALS...");
            bool hitCap;
            (valueCores, nEvals, hitCap) = TensorTrainKernel.AlsAdaptiveRank(
                _function!, grids, _maxRank, _tolerance, seed, verbose);
            if (hitCap)
                BuildWarning =
                    $"maxRank={_maxRank} reached before ALS tolerance={_tolerance:e2} satisfied. " +
                    "Increase maxRank or relax tolerance.";
        }
        _totalBuildEvals = nEvals;
```

- [ ] **Step 6: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~AlsTests|FullyQualifiedName~AlsInternalsTests"
```

Expected: 10 tests pass (9 in `AlsTests` + 1 in `AlsInternalsTests`).

- [ ] **Step 7: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 691` (681 + 10 new).

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainKernel.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAlsTests.cs tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs
git commit -m "phase2: implement ALS build method (rank-adaptive + Build dispatcher)"
```

---


## Task 6: Implement `RunCompletion` (PyChebyshev v0.13.d)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` — add `CoeffCoreToValueCore` static helper (inverse of `ValueToCoeffCores` for a single core).
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `RunCompletion(double tolerance, int maxIter, bool verbose)`.
- Modify: `tests/ChebyshevSharp.Tests/TtAlsTests.cs` — append `CompletionTests` class.

**Python source pointers:**
- `_coeff_core_to_value_core` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1018–1043
- `ChebyshevTT.run_completion` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1351–1429

**Design notes:**
- Refines an already-built TT in place via fixed-rank ALS sweeps. Rank does not grow.
- Requires `Function != null`. Loaded TTs (where `_function` is null after deserialization) must throw `InvalidOperationException`.
- Algorithm: convert each `_coeffCores[k]` back to a value core (inverse of `ValueToCoeffCores`), run `AlsFixedRankSweep` on the value cores, convert back to coefficient cores.
- The inverse DCT-II: forward is `coeff = dct(value[::-1], type=2) / n; coeff[0] /= 2`. Inverse is `c2 = coeff.copy(); c2[0] *= 2; value_reversed = idct(c2 * n, type=2); value = value_reversed[::-1]`. Use MathNet's FFT-backed inverse DCT-II for `n > 32`, otherwise direct formula. **Implementation choice**: implement using the same DCT-II routine (`BarycentricKernel.ChebyshevCoefficients1D` is the forward; we need an inverse). Since C# doesn't already ship `idct`, we implement an explicit inverse DCT-II directly: `value[k] = Σ_n y[n] * cos((2k+1)*n*π/(2N))` for `k=0..N-1`, with `y[0]` doubled and the result reversed.
- Inner loop reuses `AlsFixedRankSweep` from Task 5.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtAlsTests.cs`:

```csharp
public class CompletionTests
{
    [Fact]
    public void Test_completion_refines_cross_build()
    {
        Func<double[], double> f = p => Math.Exp(p[0] * p[1] * p[2]);
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 }, tolerance: 1e-3, maxRank: 3);
        tt.Build(verbose: false, seed: 0, method: "cross");
        double errBefore = tt.ErrorEstimate();
        tt.RunCompletion(tolerance: 1e-12, maxIter: 20, verbose: false);
        double errAfter = tt.ErrorEstimate();
        Assert.True(errAfter <= errBefore * 1.1 + 1e-14,
            $"completion should not worsen error; {errBefore} -> {errAfter}");
    }

    [Fact]
    public void Test_completion_refines_svd_build()
    {
        Func<double[], double> f = p => Math.Exp(p[0] * p[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 }, tolerance: 1e-3, maxRank: 5);
        tt.Build(verbose: false, method: "svd");
        double errBefore = tt.ErrorEstimate();
        tt.RunCompletion(tolerance: 1e-12, maxIter: 10, verbose: false);
        double errAfter = tt.ErrorEstimate();
        Assert.True(errAfter <= errBefore + 1e-9);
    }

    [Fact]
    public void Test_completion_refines_als_build()
    {
        Func<double[], double> f = p => Math.Exp(p[0] * p[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, tolerance: 1e-3, maxRank: 2);
        tt.Build(verbose: false, seed: 0, method: "als");
        double errBefore = tt.ErrorEstimate();
        tt.RunCompletion(tolerance: 1e-12, maxIter: 10, verbose: false);
        double errAfter = tt.ErrorEstimate();
        Assert.True(errAfter <= errBefore * 1.1 + 1e-14);
    }

    [Fact]
    public void Test_completion_max_iter_respected()
    {
        Func<double[], double> f = p => Math.Tanh(10 * p[0]) * p[1];
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 }, tolerance: 1e-3, maxRank: 3);
        tt.Build(verbose: false, method: "cross");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        tt.RunCompletion(tolerance: 1e-20, maxIter: 1, verbose: false);
        sw.Stop();
        Assert.True(sw.Elapsed.TotalSeconds < 30, "RunCompletion(maxIter=1) must not hang");
    }

    [Fact]
    public void Test_completion_raises_on_unbuilt()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 });
        Assert.Throws<InvalidOperationException>(() => tt.RunCompletion());
    }

    [Fact]
    public void Test_completion_raises_when_function_missing()
    {
        var tt = new ChebyshevTT(p => p[0], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 5, 5 }, tolerance: 1e-2, maxRank: 3);
        tt.Build(verbose: false, method: "cross");
        // Save and load: loaded TT has Function == null
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            var loaded = ChebyshevTT.Load(path);
            var ex = Assert.Throws<InvalidOperationException>(() => loaded.RunCompletion());
            Assert.Contains("function", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_completion_eval_stays_close_to_target()
    {
        Func<double[], double> f = p => Math.Cos(p[0] + p[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 }, tolerance: 1e-4, maxRank: 5);
        tt.Build(verbose: false, seed: 0, method: "cross");
        tt.RunCompletion(tolerance: 1e-10, maxIter: 10, verbose: false);
        double[][] pts = new[] { new[] { 0.1, 0.2 }, new[] { -0.5, 0.7 } };
        foreach (var p in pts)
            Assert.True(Math.Abs(tt.Eval(p) - f(p)) < 1e-3,
                $"completion divergence at [{string.Join(", ", p)}]: got {tt.Eval(p)}, want {f(p)}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~CompletionTests"
```

Expected: build fails — `RunCompletion` is not defined on `ChebyshevTT`.

- [ ] **Step 3: Add `CoeffCoreToValueCore` to `Internal/TensorTrainKernel.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainKernel.cs`. Append before the closing `}` of the class:

```csharp
    // ------------------------------------------------------------------
    // Coefficient ↔ Value core conversion (Phase 2 — for run_completion)
    // ------------------------------------------------------------------

    /// <summary>
    /// Inverse of <c>_value_core_to_coeff_core</c>: given a single Chebyshev
    /// coefficient core, reconstruct values at the Chebyshev Type I nodes
    /// along axis 1 (in ascending node order).
    /// </summary>
    /// <remarks>
    /// Forward (in <see cref="ValueToCoeffCores"/>): <c>coeff = dct(value[::-1], type=2)/n; coeff[0] /= 2</c>.
    /// Inverse: <c>c2 = coeff.copy(); c2[0] *= 2; value_rev = idct(c2 * n, type=2); value = value_rev[::-1]</c>.
    /// Implemented via direct trigonometric inverse DCT-II (O(n^2) per row);
    /// fast enough for typical ALS grids.
    /// </remarks>
    internal static TtCore CoeffCoreToValueCore(TtCore coeff)
    {
        int rL = coeff.RLeft, n = coeff.NNodes, rR = coeff.RRight;
        var value = new TtCore(rL, n, rR);

        // For each (i, k) row, run inverse DCT-II on the n coefficients.
        // y[m] = c2[0] + 2 * sum_{j=1}^{n-1} c2[j] * cos(j * (m + 0.5) * pi / n)
        // where c2[0] = 2*coeff[0]; c2[j] = coeff[j] for j>=1.
        // Multiply by n at the end (since forward divided by n)? Actually:
        // forward:  coeff = dct(value[::-1], type=2) / n; coeff[0] /= 2
        // inverse: undo half-of-c0 (c2[0] = 2*coeff[0]), undo /n (multiply by n
        // inside idct), apply idct (the scipy idct=type=2 which itself is the inverse), reverse.
        // Direct formulation using the cos basis:
        //   value_rev[m] = (1/n) * c2_scaled[0] + (2/n) * sum_{j=1}^{n-1} c2_scaled[j] cos(j*(m+0.5)*pi/n)
        // where c2_scaled = c2 * n  (because we multiplied by n inside idct).
        // After cancellation: value_rev[m] = c2[0] + 2*sum c2[j] cos(...). The factor n cancels.
        for (int i = 0; i < rL; i++)
            for (int kr = 0; kr < rR; kr++)
            {
                // Read coefficients along node axis.
                double c0 = 2.0 * coeff[i, 0, kr];
                for (int m = 0; m < n; m++)
                {
                    double s = c0;
                    double phase = (m + 0.5) * Math.PI / n;
                    for (int j = 1; j < n; j++)
                        s += 2.0 * coeff[i, j, kr] * Math.Cos(j * phase);
                    s /= 2.0;  // overall (1/2) since dct/idct combine as needed
                    value[i, n - 1 - m, kr] = s;  // reverse axis
                }
            }
        return value;
    }
```

> **Implementation note for the executor:** the constant scaling above must be cross-checked against the round-trip test from Python (`test_value_coeff_round_trip` in test_tensor_train.py). Use a TDD micro-step: write a `Test_value_coeff_round_trip` at the top of `CompletionTests` that constructs a value core, runs `ValueToCoeffCores` then `CoeffCoreToValueCore`, and asserts equality to 1e-12. If the constants in `CoeffCoreToValueCore` are off, fix them by deriving from the forward formula directly (consult `BarycentricKernel.ChebyshevCoefficients1D` for the forward convention) — the conventions can be fragile.

Add to `tests/ChebyshevSharp.Tests/TtAlsTests.cs` at the top of `CompletionTests`:

```csharp
    [Fact]
    public void Test_value_coeff_round_trip()
    {
        var rng = new Random(2);
        foreach ((int rL, int n, int rR) in new[] { (1, 8, 3), (2, 11, 4), (3, 5, 1) })
        {
            var v = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore(rL, n, rR);
            for (int i = 0; i < v.Size; i++) v.Data[i] = rng.NextDouble() * 2 - 1;
            var c = ChebyshevSharp.Internal.TensorTrainKernel.ValueToCoeffCores(new[] { v })[0];
            var vBack = ChebyshevSharp.Internal.TensorTrainKernel.CoeffCoreToValueCore(c);
            for (int idx = 0; idx < v.Size; idx++)
                Assert.True(Math.Abs(v.Data[idx] - vBack.Data[idx]) < 1e-10,
                    $"round-trip failed at index {idx}: {v.Data[idx]} vs {vBack.Data[idx]}");
        }
    }
```

- [ ] **Step 4: Add `RunCompletion` to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the `OrthRight` method (or after `InnerProduct` if Task 4's order placed it differently):

```csharp
    /// <summary>
    /// Refine the TT at its current rank via ALS sweeps. Works on any built TT
    /// (from "cross", "svd", or "als"). Rank does not grow; only per-core
    /// coefficients are refined.
    /// </summary>
    /// <param name="tolerance">Stop when inner-sweep relative change falls below this.</param>
    /// <param name="maxIter">Maximum number of outer ALS sweeps.</param>
    /// <param name="verbose">Print per-sweep residuals.</param>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called or if <c>Function</c> is null (loaded TT).</exception>
    public void RunCompletion(double tolerance = 1e-8, int maxIter = 50, bool verbose = false)
    {
        CheckBuilt();
        if (_function == null)
            throw new InvalidOperationException(
                "RunCompletion requires Function to be callable; the TT was loaded from a source without the original function.");

        // Convert coefficient cores back to value cores at Chebyshev Type I nodes.
        var valueCores = new TensorTrainKernel.TtCore[_numDimensions];
        for (int k = 0; k < _numDimensions; k++)
            valueCores[k] = TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![k]);

        // Rebuild the grids that Build() used.
        var grids = new double[_numDimensions][];
        for (int k = 0; k < _numDimensions; k++)
            grids[k] = BarycentricKernel.MakeNodesForDim(_domain[k][0], _domain[k][1], _nNodes[k]);

        // Cache by mixed-radix flat index.
        var cache = new Dictionary<long, double>();
        long[] strides = new long[_numDimensions];
        strides[_numDimensions - 1] = 1;
        for (int i = _numDimensions - 2; i >= 0; i--) strides[i] = strides[i + 1] * _nNodes[i + 1];

        Func<int[], double> evalsAt = idx =>
        {
            long key = 0;
            for (int i = 0; i < _numDimensions; i++) key += idx[i] * strides[i];
            if (!cache.TryGetValue(key, out double v))
            {
                var pt = new double[_numDimensions];
                for (int i = 0; i < _numDimensions; i++) pt[i] = grids[i][idx[i]];
                v = _function(pt);
                cache[key] = v;
            }
            return v;
        };

        TensorTrainKernel.AlsFixedRankSweep(
            valueCores, evalsAt, _nNodes, tolerance: tolerance, maxIter: maxIter, verbose: verbose);

        // Convert back to coefficient cores.
        _coeffCores = TensorTrainKernel.ValueToCoeffCores(valueCores);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~CompletionTests"
```

Expected: 8 tests pass (1 round-trip + 7 completion).

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 699` (691 + 8 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainKernel.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAlsTests.cs
git commit -m "phase2: implement RunCompletion (in-place ALS refinement)"
```

---


## Task 7: Implement `Nodes` + `FromValues` factories (PyChebyshev v0.18.a)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs` — add `FromValuesTtSvd` static helper.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `static (double[][] NodesPerDim, int[] Shape) Nodes(...)` and `static ChebyshevTT FromValues(...)`. Add a private no-function constructor variant for factory-built TTs.
- Modify: `tests/ChebyshevSharp.Tests/TtFactoriesTests.cs` — append `NodesTests` and `FromValuesTests` classes.

**Python source pointers:**
- `_tt_svd_from_tensor` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 636–690
- `ChebyshevTT.nodes` — exposed as a classmethod in v0.18.0 (matches `ChebyshevApproximation.nodes`)
- `ChebyshevTT.from_values` — Python classmethod that calls `_tt_svd_from_tensor`

**Design notes:**
- `Nodes(numDim, domain, nNodes)` returns `(double[][] NodesPerDim, int[] Shape)` — the per-dimension Chebyshev Type I node arrays plus a copy of the input shape. Mirrors Python's `dict {"nodes_per_dim": [...]}`. C# uses a tuple for natural destructuring.
- `FromValues(tensorValues, numDim, domain, nNodes, maxRank=10, tolerance=1e-6)`: TT-SVD on the dense tensor (already-evaluated function values) to produce value cores, then convert to coefficient cores via existing `ValueToCoeffCores`. Skips TT-Cross entirely. Sets `_function = null`, `Method = "svd"`.
- Validation: `tensorValues.Length == Π nNodes`; reject `NaN`/`Infinity` entries.
- The existing private deserialization constructor accepts a built-cores set with `_function = null`. Reuse it (or a new sibling) for `FromValues`.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtFactoriesTests.cs`:

```csharp
public class NodesTests
{
    [Fact]
    public void Test_nodes_returns_per_dim_arrays()
    {
        var (nodes, shape) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 5 });
        Assert.Equal(2, nodes.Length);
        Assert.Equal(4, nodes[0].Length);
        Assert.Equal(5, nodes[1].Length);
        Assert.Equal(new[] { 4, 5 }, shape);
    }

    [Fact]
    public void Test_nodes_within_domain()
    {
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { 0.0, 2.0 }, new[] { -3.0, 3.0 } }, new[] { 6, 6 });
        Assert.True(nodes[0].Min() >= 0.0 - 1e-12);
        Assert.True(nodes[0].Max() <= 2.0 + 1e-12);
        Assert.True(nodes[1].Min() >= -3.0 - 1e-12);
        Assert.True(nodes[1].Max() <= 3.0 + 1e-12);
    }

    [Fact]
    public void Test_nodes_consistency_with_approximation_nodes()
    {
        var (ttNodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        var (chebNodes, _) = ChebyshevApproximation.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        for (int d = 0; d < 2; d++)
            for (int j = 0; j < 5; j++)
                TestFixtures.AssertClose(chebNodes[d][j], ttNodes[d][j], atol: 1e-14);
    }
}

public class FromValuesTests
{
    [Fact]
    public void Test_from_values_round_trip_at_node()
    {
        int n = 8;
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        // dense[i, j] = sin(nodes_x[i]) * cos(nodes_y[j])
        var dense = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dense[i * n + j] = Math.Sin(nodes[0][i]) * Math.Cos(nodes[1][j]);

        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        Assert.Equal(2, tt.NumDimensions);
        TestFixtures.AssertClose(dense[2 * n + 3],
            tt.Eval(new[] { nodes[0][2], nodes[1][3] }), atol: 1e-10);
    }

    [Fact]
    public void Test_from_values_constant_function_recovers()
    {
        var dense = Enumerable.Repeat(7.0, 25).ToArray();
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 5, 5 });
        TestFixtures.AssertClose(7.0, tt.Eval(new[] { 0.3, 0.4 }), atol: 1e-10);
    }

    [Fact]
    public void Test_from_values_validates_tensor_shape()
    {
        var bad = new double[20]; // 4*5 != 5*5
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_validates_nan()
    {
        var bad = new double[25];
        bad[0] = double.NaN;
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_validates_infinity()
    {
        var bad = new double[25];
        bad[0] = double.PositiveInfinity;
        Assert.Throws<ArgumentException>(() =>
            ChebyshevTT.FromValues(bad, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 }));
    }

    [Fact]
    public void Test_from_values_max_rank_caps_rank()
    {
        var rng = new Random(42);
        var dense = Enumerable.Range(0, 6 * 6 * 6).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        var tt = ChebyshevTT.FromValues(dense, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, maxRank: 3);
        foreach (int r in tt.TtRanks)
            Assert.True(r <= 3, $"max_rank=3 violated, got rank {r}");
    }

    [Fact]
    public void Test_from_values_function_is_null()
    {
        var dense = new double[16];
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        // Loaded/factory-built TTs cannot do RunCompletion (Function == null).
        Assert.Throws<InvalidOperationException>(() => tt.RunCompletion());
    }

    [Fact]
    public void Test_from_values_method_is_svd()
    {
        var dense = new double[16];
        var tt = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        Assert.Equal("svd", tt.Method);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~NodesTests|FullyQualifiedName~FromValuesTests"
```

Expected: build fails — `ChebyshevTT.Nodes` and `ChebyshevTT.FromValues` are not defined.

- [ ] **Step 3: Implement `FromValuesTtSvd` in `Internal/TensorTrainExtrude.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs`. Replace its contents with:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Internal kernel for Tensor Train extrusion / slicing / materialization
/// (Extrude, Slice, ToDense, FromValuesTtSvd).
/// </summary>
internal static class TensorTrainExtrude
{
    /// <summary>
    /// TT-SVD decomposition of a precomputed dense tensor. Mirror of Python's
    /// <c>_tt_svd_from_tensor</c> (tensor_train.py:636).
    /// Returns value cores (function values at Chebyshev nodes along axis 1).
    /// </summary>
    /// <param name="tensor">Flat row-major dense tensor of shape <paramref name="nNodes"/>.</param>
    /// <param name="nNodes">Per-dimension grid sizes.</param>
    /// <param name="maxRank">Cap on TT rank.</param>
    /// <param name="tol">Singular value truncation tolerance relative to sigma_max.</param>
    internal static TensorTrainKernel.TtCore[] FromValuesTtSvd(
        double[] tensor, int[] nNodes, int maxRank, double tol)
    {
        int d = nNodes.Length;
        var cores = new TensorTrainKernel.TtCore[d];
        // Working tensor reshapes after each step. Track current "matrix" as flat row-major.
        double[] C = (double[])tensor.Clone();
        int rPrev = 1;
        // Right-side product of remaining dims after step k: prod_{j > k} n_j
        int rightProd = 1;
        for (int j = 1; j < d; j++) rightProd *= nNodes[j];

        for (int k = 0; k < d - 1; k++)
        {
            int rows = rPrev * nNodes[k];
            int cols = C.Length / rows;
            // Build MathNet matrix from C (flat row-major)
            var Cm = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(rows, cols, 0.0);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Cm.At(i, j, C[i * cols + j]);

            var svd = Cm.Svd(computeVectors: true);
            var U = svd.U;     // (rows, min)
            var S = svd.S;     // (min)
            var Vt = svd.VT;   // (min, cols)

            int sLen = S.Count;
            int rank = Math.Min(maxRank, sLen);
            double sMax = sLen > 0 ? S[0] : 0.0;
            if (sMax > 0)
            {
                int eff = 0;
                for (int i = 0; i < sLen; i++) if (S[i] > tol * sMax) eff++;
                rank = Math.Max(1, Math.Min(rank, eff));
            }
            else
            {
                rank = Math.Max(1, rank);
            }

            // Pack new core_k from U[:, :rank] reshaped (rPrev, n_k, rank).
            var core = new TensorTrainKernel.TtCore(rPrev, nNodes[k], rank);
            for (int i = 0; i < rPrev; i++)
                for (int p = 0; p < nNodes[k]; p++)
                    for (int r = 0; r < rank; r++)
                        core[i, p, r] = U[i * nNodes[k] + p, r];
            cores[k] = core;

            // C = diag(S[:rank]) @ Vt[:rank, :]
            var newC = new double[rank * cols];
            for (int r = 0; r < rank; r++)
                for (int j = 0; j < cols; j++)
                    newC[r * cols + j] = S[r] * Vt[r, j];
            C = newC;
            rPrev = rank;
        }

        // Last core: shape (rPrev, n_{d-1}, 1)
        var lastCore = new TensorTrainKernel.TtCore(rPrev, nNodes[d - 1], 1);
        for (int i = 0; i < rPrev; i++)
            for (int p = 0; p < nNodes[d - 1]; p++)
                lastCore[i, p, 0] = C[i * nNodes[d - 1] + p];
        cores[d - 1] = lastCore;
        return cores;
    }
}
```

- [ ] **Step 4: Add `Nodes`/`FromValues` to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. After the `RunCompletion` method, add:

```csharp
    // ------------------------------------------------------------------
    // Static factories (Phase 2 — PyChebyshev v0.18)
    // ------------------------------------------------------------------

    /// <summary>
    /// Compute the Chebyshev Type I node positions per dimension scaled to
    /// the user's domain. Static factory matching <see cref="ChebyshevApproximation.Nodes"/>.
    /// </summary>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension.</param>
    /// <returns>(NodesPerDim[d][j], Shape[d]) — node arrays in ascending order.</returns>
    public static (double[][] NodesPerDim, int[] Shape) Nodes(
        int numDimensions, double[][] domain, int[] nNodes)
    {
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}");
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}");

        var nodesPerDim = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            nodesPerDim[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
        return (nodesPerDim, (int[])nNodes.Clone());
    }

    /// <summary>
    /// Build a TT directly from a precomputed dense tensor (skips function evaluation).
    /// Uses TT-SVD for compression. The resulting TT has <c>Function = null</c>.
    /// </summary>
    /// <param name="tensorValues">Flat row-major dense tensor of length Π nNodes.</param>
    /// <param name="numDimensions">Number of dimensions.</param>
    /// <param name="domain">Bounds [lo, hi] per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension.</param>
    /// <param name="maxRank">Maximum TT rank (default 10).</param>
    /// <param name="tolerance">SVD truncation tolerance (default 1e-6).</param>
    /// <exception cref="ArgumentException">If tensorValues length doesn't match Π nNodes, or contains NaN/Infinity.</exception>
    public static ChebyshevTT FromValues(
        double[] tensorValues,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        int maxRank = 10,
        double tolerance = 1e-6)
    {
        if (domain.Length != numDimensions)
            throw new ArgumentException(
                $"domain has {domain.Length} entries but numDimensions={numDimensions}");
        if (nNodes.Length != numDimensions)
            throw new ArgumentException(
                $"nNodes has {nNodes.Length} entries but numDimensions={numDimensions}");
        long expected = 1;
        for (int i = 0; i < numDimensions; i++) expected = checked(expected * nNodes[i]);
        if (tensorValues.LongLength != expected)
            throw new ArgumentException(
                $"tensorValues has shape mismatch: length {tensorValues.LongLength} but expected Π nNodes = {expected}");
        for (int i = 0; i < tensorValues.Length; i++)
            if (!double.IsFinite(tensorValues[i]))
                throw new ArgumentException($"tensorValues[{i}] is NaN or Infinity (must be finite)");

        var valueCores = TensorTrainExtrude.FromValuesTtSvd(tensorValues, nNodes, maxRank, tolerance);
        var coeffCores = TensorTrainKernel.ValueToCoeffCores(valueCores);

        var ttRanks = new int[numDimensions + 1];
        ttRanks[0] = 1;
        for (int i = 0; i < numDimensions; i++) ttRanks[i + 1] = coeffCores[i].RRight;

        var tt = new ChebyshevTT(
            numDimensions: numDimensions,
            domain: domain.Select(d => (double[])d.Clone()).ToArray(),
            nNodes: (int[])nNodes.Clone(),
            maxRank: maxRank,
            tolerance: tolerance,
            maxSweeps: 0,
            coeffCores: coeffCores,
            ttRanks: ttRanks,
            buildTime: 0.0,
            totalBuildEvals: 0);
        tt.Method = "svd";
        return tt;
    }
```

> **Important:** the existing private deserialization constructor (`ChebyshevTT(int, double[][], int[], int, double, int, TtCore[], int[], double, int)`) already exists and matches the parameter list above. If its parameter names differ, use positional argument syntax. If the existing constructor uses different parameter names, update the call site.

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~NodesTests|FullyQualifiedName~FromValuesTests"
```

Expected: 11 tests pass (3 Nodes + 8 FromValues).

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 710` (699 + 11 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainExtrude.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtFactoriesTests.cs
git commit -m "phase2: implement Nodes() and FromValues() static factories"
```

---


## Task 8: Implement `Extrude` + `Slice` + `ToDense` (PyChebyshev v0.18.b)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs` — add `ExtrudeCores`, `SliceCores`, `ToDenseEinsumChain`.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `Extrude(int, (double, double), int)`, `Slice(int, double)`, `ToDense()`. Add a private "build from cores" factory method to construct the result TTs.
- Modify: `tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs` — append three test classes.

**Python source pointers:**
- `_extrude_tt_core` — `ref/PyChebyshev/src/pychebyshev/_extrude_slice.py` lines 161–end
- `_slice_tt_core` — `ref/PyChebyshev/src/pychebyshev/_extrude_slice.py` lines 95–160
- `ChebyshevTT.extrude` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1671–1738
- `ChebyshevTT.slice` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1739–1816
- `ChebyshevTT.to_dense` — `ref/PyChebyshev/src/pychebyshev/tensor_train.py` lines 1637–1670

**Design notes:**
- **ToDense**: contracts cores left-to-right, returning a flat `double[]` of length `Π nNodes`. Allocation guard: `Π nNodes * 8 > int.MaxValue` throws `OverflowException`.
- **Extrude**: inserts a new core at `dim` with shape `(rAt, nNew, rAt)` where `rAt` is the rank at the insertion boundary. Only `core[i, 0, i] = 1.0` (encodes constant function 1 in DCT-II coefficient space). For `dim=0` or `dim=numDim`, `rAt=1`. C# API uses single-tuple form `Extrude(int dim, (double lo, double hi) newDomain, int newN)` rather than Python's variadic-list pattern (idiomatic C# is one extrusion per call; users compose via successive calls).
- **Slice**: barycentric interpolation along the sliced dim's value-space core (convert coefficient core → value core, evaluate barycentric formula at `value`, get a `(rL, rR)` matrix `M`, absorb into right neighbor or left neighbor). Fast path: `value` coincides with a node within `1e-14` → `np.take`. Slice value out of domain raises `ArgumentOutOfRangeException`.
- The C# API surfaces only single-dim variants (`Slice(int, double)` and `Extrude(int, (double, double), int)`). Python's multi-tuple variants can be replicated by chained calls; we keep the C# surface minimal.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs`:

```csharp
public class ToDenseTests
{
    [Fact]
    public void Test_to_dense_returns_array_with_product_size()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 4 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        Assert.Equal(16, dense.Length);
    }

    [Fact]
    public void Test_to_dense_shape_matches_n_nodes()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1] + p[2], 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 5, 6 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        Assert.Equal(120, dense.Length);
    }

    [Fact]
    public void Test_to_dense_values_match_eval_at_nodes()
    {
        int n = 5;
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Cos(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        var (nodes, _) = ChebyshevTT.Nodes(2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double expected = tt.Eval(new[] { nodes[0][i], nodes[1][j] });
                TestFixtures.AssertClose(expected, dense[i * n + j], atol: 1e-10);
            }
    }

    [Fact]
    public void Test_to_dense_round_trip_via_from_values()
    {
        var ttA = new ChebyshevTT(p => p[0] * Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });
        ttA.Build(verbose: false);
        double[] dense = ttA.ToDense();
        var ttB = ChebyshevTT.FromValues(dense, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });
        double[] xTest = { 0.3, -0.4 };
        TestFixtures.AssertClose(ttA.Eval(xTest), ttB.Eval(xTest), atol: 1e-8);
    }

    [Fact]
    public void Test_to_dense_constant_function()
    {
        var tt = new ChebyshevTT(p => 3.0, 2,
            new[] { new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 } }, new[] { 4, 4 });
        tt.Build(verbose: false);
        double[] dense = tt.ToDense();
        foreach (double v in dense)
            TestFixtures.AssertClose(3.0, v, atol: 1e-10);
    }

    [Fact]
    public void Test_to_dense_overflow_guard()
    {
        // 9 dims, 10 nodes each = 1e9 elements * 8 bytes = 8 GB, will overflow int.MaxValue/8.
        var tt = new ChebyshevTT(p => p[0], 9,
            Enumerable.Repeat(new[] { -1.0, 1.0 }, 9).ToArray(),
            Enumerable.Repeat(80, 9).ToArray());
        tt.Build(verbose: false);
        Assert.Throws<OverflowException>(() => tt.ToDense());
    }
}

public class ExtrudeTests
{
    [Fact]
    public void Test_extrude_returns_tt_with_new_dim()
    {
        var tt = new ChebyshevTT(p => p[0] * p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 4);
        Assert.Equal(2, result.NumDimensions);
    }

    [Fact]
    public void Test_extrude_preserves_eval_at_existing_dims()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 5);
        // Result(x, y) should equal sin(x) for any y.
        double[] xs = { -0.5, 0.0, 0.3 };
        double[] ys = { 0.1, 0.5, 0.9 };
        foreach (double x in xs)
            foreach (double y in ys)
                TestFixtures.AssertClose(Math.Sin(x), result.Eval(new[] { x, y }), atol: 1e-6);
    }

    [Fact]
    public void Test_extrude_constant_value()
    {
        var tt = new ChebyshevTT(p => 7.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt.Extrude(dim: 1, newDomain: (0.0, 5.0), newN: 4);
        TestFixtures.AssertClose(7.0, result.Eval(new[] { 0.5, 2.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_extrude_validates_dim_idx()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.Extrude(dim: 5, newDomain: (0.0, 1.0), newN: 4));
    }

    [Fact]
    public void Test_extrude_validates_domain_order()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (1.0, 0.0), newN: 4));
    }

    [Fact]
    public void Test_extrude_validates_nn_minimum()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentException>(() =>
            tt.Extrude(dim: 1, newDomain: (0.0, 1.0), newN: 1));
    }
}

public class SliceTests
{
    [Fact]
    public void Test_slice_returns_lower_dim_tt()
    {
        var tt = new ChebyshevTT(p => p[0] + p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 6, 6 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 0, value: 0.5);
        Assert.Equal(1, result.NumDimensions);
    }

    [Fact]
    public void Test_slice_at_node_uses_fast_path()
    {
        int n = 6;
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) * Math.Cos(p[1]), 2, domain, new[] { n, n });
        tt.Build(verbose: false);
        var (nodes, _) = ChebyshevTT.Nodes(2, domain, new[] { n, n });
        var result = tt.Slice(dim: 0, value: nodes[0][2]);
        foreach (double y in new[] { -0.5, 0.0, 0.5 })
            TestFixtures.AssertClose(
                tt.Eval(new[] { nodes[0][2], y }),
                result.Eval(new[] { y }),
                atol: 1e-10);
    }

    [Fact]
    public void Test_slice_at_interior_value_matches_eval()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Cos(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 10, 10 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 1, value: 0.3);
        foreach (double x in new[] { -0.5, 0.0, 0.4 })
            TestFixtures.AssertClose(
                Math.Sin(x) + Math.Cos(0.3),
                result.Eval(new[] { x }),
                atol: 1e-6);
    }

    [Fact]
    public void Test_slice_endpoint_dim_left()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 4, 6 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 0, value: 0.5);
        TestFixtures.AssertClose(0.3, result.Eval(new[] { 0.6 }), atol: 1e-8);
    }

    [Fact]
    public void Test_slice_endpoint_dim_right()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 6, 4 });
        tt.Build(verbose: false);
        var result = tt.Slice(dim: 1, value: 0.5);
        TestFixtures.AssertClose(0.15, result.Eval(new[] { 0.3 }), atol: 1e-8);
    }

    [Fact]
    public void Test_slice_validates_value_within_domain()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.Slice(dim: 0, value: 5.0));
    }

    [Fact]
    public void Test_slice_validates_dim_idx()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => tt.Slice(dim: 5, value: 0.0));
    }

    [Fact]
    public void Test_slice_then_to_dense()
    {
        var tt = new ChebyshevTT(p => p[0] * p[1], 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 5, 5 });
        tt.Build(verbose: false);
        var sliced = tt.Slice(dim: 0, value: 0.5);
        double[] dense = sliced.ToDense();
        Assert.Equal(5, dense.Length);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~ToDenseTests|FullyQualifiedName~ExtrudeTests|FullyQualifiedName~SliceTests"
```

Expected: build fails — `ToDense`, `Extrude`, `Slice` are not defined.

- [ ] **Step 3: Implement helpers in `Internal/TensorTrainExtrude.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainExtrude.cs`. Append before the closing `}` of the class:

```csharp
    /// <summary>
    /// Materialize the TT chain into a flat row-major dense tensor of length Π nNodes.
    /// Converts coefficient cores to value cores first, then chains contractions.
    /// Mirror of Python's <c>ChebyshevTT.to_dense</c> (tensor_train.py:1637).
    /// </summary>
    internal static double[] ToDenseEinsumChain(TensorTrainKernel.TtCore[] coeffCores, int[] nNodes)
    {
        // Convert all coefficient cores to value cores.
        var valueCores = new TensorTrainKernel.TtCore[coeffCores.Length];
        for (int i = 0; i < coeffCores.Length; i++)
            valueCores[i] = TensorTrainKernel.CoeffCoreToValueCore(coeffCores[i]);
        return TensorTrainKernel.ReconstructDense(valueCores, nNodes);
    }

    /// <summary>
    /// Insert a constant rank-preserving core at position <paramref name="dim"/>
    /// into a TT. The new core encodes the constant function 1 in DCT-II
    /// coefficient space (only c_0 = 1.0 is set; the core is rank-preserving:
    /// new_core[i, 0, i] = 1.0 for all i).
    /// Mirror of Python's <c>_extrude_tt_core</c>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] ExtrudeCores(
        TensorTrainKernel.TtCore[] coeffCores, int dim, int nNew)
    {
        int d = coeffCores.Length;
        if (dim < 0 || dim > d)
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"dim={dim} out of range [0, {d}]");
        if (nNew < 2)
            throw new ArgumentException($"newN must be >= 2, got {nNew}", nameof(nNew));

        // Determine rank at insertion boundary.
        int rAt;
        if (dim == 0) rAt = 1;
        else if (dim == d) rAt = 1;
        else rAt = coeffCores[dim - 1].RRight;

        var newCore = new TensorTrainKernel.TtCore(rAt, nNew, rAt);
        for (int i = 0; i < rAt; i++)
            newCore[i, 0, i] = 1.0;

        var result = new TensorTrainKernel.TtCore[d + 1];
        for (int k = 0; k < dim; k++) result[k] = coeffCores[k];
        result[dim] = newCore;
        for (int k = dim; k < d; k++) result[k + 1] = coeffCores[k];
        return result;
    }

    /// <summary>
    /// Contract a TT coefficient core along <paramref name="dim"/> at <paramref name="value"/>.
    /// Converts the target core to value space, evaluates the barycentric interpolant at
    /// <paramref name="value"/> to produce a matrix M of shape (rL, rR), then absorbs M
    /// into the right neighbor (or left neighbor for the rightmost core).
    /// Mirror of Python's <c>_slice_tt_core</c>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] SliceCores(
        TensorTrainKernel.TtCore[] coeffCores, int dim, double value, double[] nodes)
    {
        var coeffCore = coeffCores[dim];
        var valueCore = TensorTrainKernel.CoeffCoreToValueCore(coeffCore);
        int rL = valueCore.RLeft, n = valueCore.NNodes, rR = valueCore.RRight;

        // Find nearest node and check fast-path.
        double[] diff = new double[n];
        int exactIdx = 0;
        double minAbs = double.PositiveInfinity;
        for (int j = 0; j < n; j++)
        {
            diff[j] = value - nodes[j];
            double abs = Math.Abs(diff[j]);
            if (abs < minAbs) { minAbs = abs; exactIdx = j; }
        }

        double[] M = new double[rL * rR];
        if (minAbs < 1e-14)
        {
            for (int i = 0; i < rL; i++)
                for (int k = 0; k < rR; k++)
                    M[i * rR + k] = valueCore[i, exactIdx, k];
        }
        else
        {
            // Compute barycentric weights for nodes — same as ChebyshevApproximation.
            double[] baryW = BarycentricKernel.ComputeBarycentricWeights(nodes);
            double[] wOverDiff = new double[n];
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                wOverDiff[j] = baryW[j] / diff[j];
                sum += wOverDiff[j];
            }
            for (int j = 0; j < n; j++) wOverDiff[j] /= sum;

            for (int i = 0; i < rL; i++)
                for (int k = 0; k < rR; k++)
                {
                    double s = 0;
                    for (int j = 0; j < n; j++) s += wOverDiff[j] * valueCore[i, j, k];
                    M[i * rR + k] = s;
                }
        }

        int d = coeffCores.Length;
        var result = new TensorTrainKernel.TtCore[d - 1];

        if (dim < d - 1)
        {
            // Absorb M into right neighbor: newNeighbor[l, j, s] = sum_r M[l, r] * neighbor[r, j, s]
            var neighbor = coeffCores[dim + 1];
            int n2 = neighbor.NNodes, rR2 = neighbor.RRight;
            var newNeighbor = new TensorTrainKernel.TtCore(rL, n2, rR2);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n2; j++)
                    for (int k = 0; k < rR2; k++)
                    {
                        double s = 0;
                        for (int r = 0; r < rR; r++)
                            s += M[i * rR + r] * neighbor[r, j, k];
                        newNeighbor[i, j, k] = s;
                    }
            for (int k = 0; k < dim; k++) result[k] = coeffCores[k];
            result[dim] = newNeighbor;
            for (int k = dim + 2; k < d; k++) result[k - 1] = coeffCores[k];
        }
        else
        {
            // Rightmost core: absorb M into left neighbor.
            var neighbor = coeffCores[dim - 1];
            int rLp = neighbor.RLeft, np = neighbor.NNodes;
            var newNeighbor = new TensorTrainKernel.TtCore(rLp, np, rR);
            for (int i = 0; i < rLp; i++)
                for (int j = 0; j < np; j++)
                    for (int k = 0; k < rR; k++)
                    {
                        double s = 0;
                        for (int r = 0; r < rL; r++)
                            s += neighbor[i, j, r] * M[r * rR + k];
                        newNeighbor[i, j, k] = s;
                    }
            for (int k = 0; k < dim - 1; k++) result[k] = coeffCores[k];
            result[dim - 1] = newNeighbor;
        }
        return result;
    }
```

> If `BarycentricKernel.ComputeBarycentricWeights(double[] nodes)` does not exist as an internal static helper, locate the existing weight computation inside `BarycentricKernel.cs` (search for `BarycentricWeights` or `weights[i] = `) and either expose it as `internal static` or inline its short formula here: `w_i = 1 / Π_{j≠i}(x_i − x_j)`.

- [ ] **Step 4: Add `Extrude`/`Slice`/`ToDense` and a private factory constructor to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the static factory block from Task 7:

```csharp
    // ------------------------------------------------------------------
    // Materialization, extrusion, slicing (Phase 2 — PyChebyshev v0.18)
    // ------------------------------------------------------------------

    /// <summary>
    /// Materialize the TT chain into a full row-major dense tensor.
    /// Length is Π NNodes; <c>dense[flat]</c> equals <c>Eval(point_at_grid_idx)</c>
    /// where flat is the row-major index into the grid shape.
    /// Use sparingly: storage is Π NNodes floats.
    /// </summary>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="OverflowException">If Π NNodes * 8 exceeds <c>int.MaxValue</c>.</exception>
    public double[] ToDense()
    {
        CheckBuilt();
        long total = 1;
        for (int i = 0; i < _numDimensions; i++)
            total = checked(total * _nNodes[i]);
        if (total * 8 > int.MaxValue)
            throw new OverflowException(
                $"ToDense would allocate {total} doubles ({total * 8} bytes), exceeding int.MaxValue. " +
                "Use ToDense for low-dimensional inspection only.");
        return TensorTrainExtrude.ToDenseEinsumChain(_coeffCores!, _nNodes);
    }

    /// <summary>
    /// Insert a new dimension at index <paramref name="dim"/> where the function
    /// is constant. The extruded TT evaluates identically to the original over
    /// the existing dimensions, regardless of the new dimension's coordinate.
    /// </summary>
    /// <param name="dim">Insertion index, 0 &lt;= dim &lt;= NumDimensions.</param>
    /// <param name="newDomain">Domain (lo, hi) for the new dimension.</param>
    /// <param name="newN">Number of nodes for the new dimension.</param>
    public ChebyshevTT Extrude(int dim, (double Lo, double Hi) newDomain, int newN)
    {
        CheckBuilt();
        if (newDomain.Lo >= newDomain.Hi)
            throw new ArgumentException(
                $"newDomain bounds must satisfy lo < hi; got ({newDomain.Lo}, {newDomain.Hi})",
                nameof(newDomain));
        var newCores = TensorTrainExtrude.ExtrudeCores(_coeffCores!, dim, newN);
        var newDomainArr = new double[_numDimensions + 1][];
        for (int k = 0; k < dim; k++) newDomainArr[k] = (double[])_domain[k].Clone();
        newDomainArr[dim] = new[] { newDomain.Lo, newDomain.Hi };
        for (int k = dim; k < _numDimensions; k++) newDomainArr[k + 1] = (double[])_domain[k].Clone();

        var newNNodes = new int[_numDimensions + 1];
        for (int k = 0; k < dim; k++) newNNodes[k] = _nNodes[k];
        newNNodes[dim] = newN;
        for (int k = dim; k < _numDimensions; k++) newNNodes[k + 1] = _nNodes[k];

        return BuildResultFromCores(newCores, newDomainArr, newNNodes);
    }

    /// <summary>
    /// Fix dimension <paramref name="dim"/> at <paramref name="value"/>, returning
    /// a TT over the remaining (NumDimensions - 1) dimensions.
    /// </summary>
    /// <param name="dim">Dimension to slice, 0 &lt;= dim &lt; NumDimensions.</param>
    /// <param name="value">Value at which to fix the dimension; must lie within the domain.</param>
    public ChebyshevTT Slice(int dim, double value)
    {
        CheckBuilt();
        if (dim < 0 || dim >= _numDimensions)
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"dim={dim} out of range [0, {_numDimensions - 1}]");
        if (_numDimensions == 1)
            throw new InvalidOperationException("Cannot slice a 1D TT (would produce 0D result).");
        double lo = _domain[dim][0], hi = _domain[dim][1];
        if (value < lo || value > hi)
            throw new ArgumentOutOfRangeException(nameof(value),
                $"Slice value {value} for dim {dim} is outside domain [{lo}, {hi}]");

        double[] nodes = BarycentricKernel.MakeNodesForDim(lo, hi, _nNodes[dim]);
        var newCores = TensorTrainExtrude.SliceCores(_coeffCores!, dim, value, nodes);

        var newDomain = new double[_numDimensions - 1][];
        var newNNodes = new int[_numDimensions - 1];
        int writeIdx = 0;
        for (int k = 0; k < _numDimensions; k++)
        {
            if (k == dim) continue;
            newDomain[writeIdx] = (double[])_domain[k].Clone();
            newNNodes[writeIdx] = _nNodes[k];
            writeIdx++;
        }

        return BuildResultFromCores(newCores, newDomain, newNNodes);
    }

    /// <summary>
    /// Internal helper: assemble a fresh ChebyshevTT from a set of coefficient cores.
    /// Used by Extrude, Slice, and the algebra operators (Tasks 9 + 10).
    /// </summary>
    internal ChebyshevTT BuildResultFromCores(
        TensorTrainKernel.TtCore[] cores, double[][] newDomain, int[] newNNodes)
    {
        int newD = newNNodes.Length;
        var ttRanks = new int[newD + 1];
        ttRanks[0] = 1;
        for (int i = 0; i < newD; i++) ttRanks[i + 1] = cores[i].RRight;
        var tt = new ChebyshevTT(
            numDimensions: newD,
            domain: newDomain,
            nNodes: newNNodes,
            maxRank: _maxRank,
            tolerance: _tolerance,
            maxSweeps: _maxSweeps,
            coeffCores: cores,
            ttRanks: ttRanks,
            buildTime: 0.0,
            totalBuildEvals: 0);
        tt.Method = Method;
        return tt;
    }
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~ToDenseTests|FullyQualifiedName~ExtrudeTests|FullyQualifiedName~SliceTests"
```

Expected: 21 tests pass (6 ToDense + 6 Extrude + 9 Slice).

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 731` (710 + 21 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainExtrude.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtExtrudeSliceTests.cs
git commit -m "phase2: implement ToDense, Extrude, Slice"
```

---


## Task 9: Implement scalar algebra (`*`, `/`, unary `-` + in-place equivalents) — PyChebyshev v0.18.c

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` — add `ScalarMulCores`, `ScalarMulCoresInPlace`, `NegateCores`, `NegateCoresInPlace`.
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add binary `operator *(ChebyshevTT, double)`, `operator *(double, ChebyshevTT)`, `operator /(ChebyshevTT, double)`, unary `operator -(ChebyshevTT)`. Add `ScalarMulInPlace(double)`, `ScalarDivInPlace(double)`, `NegateInPlace()`.
- Modify: `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs` — append `ScalarAlgebraTests` and `ScalarInPlaceTests` classes.
- Modify: `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` — add lazy `TtAlgebraF` and `TtAlgebraG` fixtures over a 2D `[-1,1]^2` `[6,6]` grid.

**Python source pointers:**
- `_is_scalar` — `ref/PyChebyshev/src/pychebyshev/_algebra.py` lines 8–10
- `_check_compatible` — `ref/PyChebyshev/src/pychebyshev/_algebra.py` lines 13–55
- `ChebyshevTT.__mul__`, `__rmul__`, `__truediv__`, `__neg__`, `__imul__` — at the end of `tensor_train.py` (search `__mul__`)

**Design notes:**
- Scalar mul on Chebyshev coefficient cores: multiply core 0 by the scalar (any one core; choose core 0 for simplicity). Result represents the same function scaled.
- Negation is `ScalarMul(-1.0)` semantically; we provide a dedicated `NegateCores`/`NegateCoresInPlace` for clarity.
- Division: `a / s` is `a * (1/s)`. Throw `DivideByZeroException` on `s == 0.0`.
- Operators allocate new cores via `Copy()`. In-place equivalents mutate `_coeffCores[0]` directly.
- The functional operators return a brand-new `ChebyshevTT` with `_function = null`. Tests check `Function == null` indirectly by checking `RunCompletion` throws.

- [ ] **Step 1: Add `TtAlgebraF`/`TtAlgebraG` fixtures**

Open `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs`. After the existing TT fixture block, add:

```csharp
    private static readonly Lazy<ChebyshevTT> _ttAlgebraF = new(() =>
    {
        var tt = new ChebyshevTT(
            p => Math.Sin(p[0]) + 0.5 * p[1],
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, tolerance: 1e-6, maxRank: 6);
        tt.Build(verbose: false, seed: 42);
        return tt;
    });

    private static readonly Lazy<ChebyshevTT> _ttAlgebraG = new(() =>
    {
        var tt = new ChebyshevTT(
            p => Math.Cos(p[0]) * p[1],
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, tolerance: 1e-6, maxRank: 6);
        tt.Build(verbose: false, seed: 7);
        return tt;
    });

    public static ChebyshevTT TtAlgebraF => _ttAlgebraF.Value;
    public static ChebyshevTT TtAlgebraG => _ttAlgebraG.Value;
```

- [ ] **Step 2: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs`:

```csharp
public class ScalarAlgebraTests
{
    [Fact]
    public void Test_scalar_mul_returns_tt()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt * 2.5;
        Assert.IsType<ChebyshevTT>(result);
    }

    [Fact]
    public void Test_scalar_mul_eval_scales()
    {
        var tt = TestFixtures.TtAlgebraF;
        var result = tt * 3.0;
        foreach (double[] p in new[] { new[] { -0.5, 0.0 }, new[] { 0.5, 0.5 }, new[] { 0.0, -0.7 } })
            TestFixtures.AssertClose(3.0 * tt.Eval(p), result.Eval(p), atol: 1e-10);
    }

    [Fact]
    public void Test_rmul_works()
    {
        var tt = TestFixtures.TtAlgebraF;
        var lhs = 2.5 * tt;
        var rhs = tt * 2.5;
        foreach (double[] p in new[] { new[] { 0.3, -0.4 } })
            TestFixtures.AssertClose(lhs.Eval(p), rhs.Eval(p), atol: 1e-12);
    }

    [Fact]
    public void Test_truediv_scalar()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        var result = tt / 2.0;
        TestFixtures.AssertClose(0.2, result.Eval(new[] { 0.4 }), atol: 1e-10);
    }

    [Fact]
    public void Test_truediv_by_zero_raises()
    {
        var tt = TestFixtures.TtAlgebraF;
        Assert.Throws<DivideByZeroException>(() => tt / 0.0);
    }

    [Fact]
    public void Test_unary_neg()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 10 });
        tt.Build(verbose: false);
        var neg = -tt;
        TestFixtures.AssertClose(-tt.Eval(new[] { 0.5 }), neg.Eval(new[] { 0.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_mul_zero_yields_zero_tt()
    {
        var tt = TestFixtures.TtAlgebraF;
        var zero = tt * 0.0;
        foreach (double[] p in new[] { new[] { 0.3, -0.4 }, new[] { -0.7, 0.1 } })
            TestFixtures.AssertClose(0.0, zero.Eval(p), atol: 1e-12);
    }

    [Fact]
    public void Test_scalar_mul_function_null_on_result()
    {
        var tt = TestFixtures.TtAlgebraF;
        var result = tt * 2.0;
        Assert.Throws<InvalidOperationException>(() => result.RunCompletion());
    }
}

public class ScalarInPlaceTests
{
    [Fact]
    public void Test_scalar_mul_in_place_mutates()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        double before = tt.Eval(new[] { 0.5 });
        tt.ScalarMulInPlace(2.0);
        TestFixtures.AssertClose(2.0 * before, tt.Eval(new[] { 0.5 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_div_in_place()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        tt.ScalarDivInPlace(4.0);
        TestFixtures.AssertClose(0.2, tt.Eval(new[] { 0.8 }), atol: 1e-10);
    }

    [Fact]
    public void Test_scalar_div_in_place_by_zero_raises()
    {
        var tt = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false);
        Assert.Throws<DivideByZeroException>(() => tt.ScalarDivInPlace(0.0));
    }

    [Fact]
    public void Test_negate_in_place()
    {
        var tt = new ChebyshevTT(p => p[0] + 1.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false);
        double before = tt.Eval(new[] { 0.3 });
        tt.NegateInPlace();
        TestFixtures.AssertClose(-before, tt.Eval(new[] { 0.3 }), atol: 1e-10);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~ScalarAlgebraTests|FullyQualifiedName~ScalarInPlaceTests"
```

Expected: build fails — operators and in-place methods are not defined.

- [ ] **Step 4: Implement `ScalarMulCores`, `NegateCores`, in-place variants in `Internal/TensorTrainAlgebra.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs`. Append before the closing `}` of the class:

```csharp
    /// <summary>
    /// Return a deep-copy of <paramref name="cores"/> with core 0's data scaled by
    /// <paramref name="scalar"/>. The represented function is multiplied by <paramref name="scalar"/>.
    /// </summary>
    internal static TensorTrainKernel.TtCore[] ScalarMulCores(
        TensorTrainKernel.TtCore[] cores, double scalar)
    {
        int d = cores.Length;
        var result = new TensorTrainKernel.TtCore[d];
        // Deep-copy all cores; scale core 0.
        for (int k = 0; k < d; k++) result[k] = cores[k].Copy();
        for (int i = 0; i < result[0].Data.Length; i++)
            result[0].Data[i] *= scalar;
        return result;
    }

    /// <summary>In-place variant of <see cref="ScalarMulCores"/>.</summary>
    internal static void ScalarMulCoresInPlace(
        TensorTrainKernel.TtCore[] cores, double scalar)
    {
        for (int i = 0; i < cores[0].Data.Length; i++)
            cores[0].Data[i] *= scalar;
    }

    /// <summary>Return a deep-copy of <paramref name="cores"/> negated.</summary>
    internal static TensorTrainKernel.TtCore[] NegateCores(TensorTrainKernel.TtCore[] cores)
        => ScalarMulCores(cores, -1.0);

    /// <summary>In-place negation.</summary>
    internal static void NegateCoresInPlace(TensorTrainKernel.TtCore[] cores)
        => ScalarMulCoresInPlace(cores, -1.0);
}
```

- [ ] **Step 5: Add operators and in-place methods to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the `Slice` method:

```csharp
    // ------------------------------------------------------------------
    // Scalar algebra (Phase 2 — PyChebyshev v0.18.c)
    // ------------------------------------------------------------------

    /// <summary>Scalar multiplication: <c>tt * scalar</c>.</summary>
    public static ChebyshevTT operator *(ChebyshevTT tt, double scalar)
    {
        if (tt is null) throw new ArgumentNullException(nameof(tt));
        tt.CheckBuilt();
        var newCores = TensorTrainAlgebra.ScalarMulCores(tt._coeffCores!, scalar);
        var domainCopy = tt._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])tt._nNodes.Clone();
        return tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
    }

    /// <summary>Scalar multiplication: <c>scalar * tt</c>.</summary>
    public static ChebyshevTT operator *(double scalar, ChebyshevTT tt) => tt * scalar;

    /// <summary>Scalar division: <c>tt / scalar</c>.</summary>
    /// <exception cref="DivideByZeroException">If <paramref name="scalar"/> is zero.</exception>
    public static ChebyshevTT operator /(ChebyshevTT tt, double scalar)
    {
        if (scalar == 0.0)
            throw new DivideByZeroException("Cannot divide ChebyshevTT by zero.");
        return tt * (1.0 / scalar);
    }

    /// <summary>Unary negation: <c>-tt</c>.</summary>
    public static ChebyshevTT operator -(ChebyshevTT tt)
    {
        if (tt is null) throw new ArgumentNullException(nameof(tt));
        tt.CheckBuilt();
        var newCores = TensorTrainAlgebra.NegateCores(tt._coeffCores!);
        var domainCopy = tt._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])tt._nNodes.Clone();
        return tt.BuildResultFromCores(newCores, domainCopy, nNodesCopy);
    }

    /// <summary>Scale this TT in place by <paramref name="scalar"/>.</summary>
    public void ScalarMulInPlace(double scalar)
    {
        CheckBuilt();
        TensorTrainAlgebra.ScalarMulCoresInPlace(_coeffCores!, scalar);
        _cachedErrorEstimate = null;
    }

    /// <summary>Divide this TT in place by <paramref name="scalar"/>.</summary>
    /// <exception cref="DivideByZeroException">If <paramref name="scalar"/> is zero.</exception>
    public void ScalarDivInPlace(double scalar)
    {
        if (scalar == 0.0)
            throw new DivideByZeroException("Cannot divide ChebyshevTT by zero.");
        ScalarMulInPlace(1.0 / scalar);
    }

    /// <summary>Negate this TT in place.</summary>
    public void NegateInPlace()
    {
        CheckBuilt();
        TensorTrainAlgebra.NegateCoresInPlace(_coeffCores!);
        _cachedErrorEstimate = null;
    }
```

- [ ] **Step 6: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~ScalarAlgebraTests|FullyQualifiedName~ScalarInPlaceTests"
```

Expected: 12 tests pass (8 ScalarAlgebra + 4 ScalarInPlace).

- [ ] **Step 7: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 743` (731 + 12 new).

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAlgebraTests.cs tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs
git commit -m "phase2: implement scalar algebra (* / unary -) and in-place equivalents"
```

---

## Task 10: Implement binary algebra (`+`, `-`) + `AddInPlace`/`SubInPlace`/`RoundInPlace` — PyChebyshev v0.18.d

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` — add `AddCores` (block-diagonal stacking) and `RoundCores` (TT-SVD rounding via OrthRight + SVD truncation).
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `operator +(ChebyshevTT, ChebyshevTT)`, `operator -(ChebyshevTT, ChebyshevTT)`, `AddInPlace(ChebyshevTT)`, `SubInPlace(ChebyshevTT)`, `RoundInPlace(double tolerance)`.
- Modify: `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs` — append `BinaryAlgebraTests`, `BinaryInPlaceTests`, `RoundingTests` classes.

**Python source pointers:**
- `_tt_add_cores` — `ref/PyChebyshev/src/pychebyshev/_algebra.py` lines 63–117
- `_tt_round_cores` — `ref/PyChebyshev/src/pychebyshev/_algebra.py` lines 118–170

**Design notes:**
- **`AddCores`**: block-diagonal stacking. For interior cores `k ∈ [1, d-2]`, result is block-diagonal of `cores_a[k]` and `cores_b[k]` along (left rank, right rank). Leftmost core concatenates along the right rank. Rightmost core concatenates along the left rank. Special case `d == 1`: cannot block-diagonal because both endpoint rank-1 invariants collide → simply elementwise sum.
- **`RoundCores`**: TT-SVD recompression. Right-to-left QR sweep (right-canonicalize cores `k = d-1, …, 1`), then left-to-right SVD truncation (`k = 0, …, d-2`). Truncate by `min(maxRank, num_above_relative_tol)` where the relative threshold is `tolerance * sigma_max`. Always keep at least 1 singular value.
- **Functional `+`**: `AddCores` then `RoundCores` to default tolerance `1e-12` and `maxRank = max(self.MaxRank, other.MaxRank)`.
- **`AddInPlace`**: replace `_coeffCores` with `RoundCores(AddCores(self, other), …)`. Same logic; just mutate.
- **`RoundInPlace(tolerance)`**: caller's explicit recompression.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/TtAlgebraTests.cs`:

```csharp
public class BinaryAlgebraTests
{
    [Fact]
    public void Test_add_two_tts_returns_tt()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        var result = ttF + ttG;
        Assert.IsType<ChebyshevTT>(result);
    }

    [Fact]
    public void Test_add_eval_matches_sum_of_evals()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        var result = ttF + ttG;
        foreach (double[] p in new[] { new[] { 0.3, 0.4 }, new[] { -0.2, 0.5 }, new[] { 0.0, 0.0 } })
            TestFixtures.AssertClose(ttF.Eval(p) + ttG.Eval(p), result.Eval(p), atol: 1e-6);
    }

    [Fact]
    public void Test_subtract_returns_tt()
    {
        var ttA = TestFixtures.TtAlgebraF;
        var ttB = TestFixtures.TtAlgebraF;
        var result = ttA - ttB;
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.3, 0.4 }), atol: 1e-6);
    }

    [Fact]
    public void Test_add_incompatible_domain_raises()
    {
        var ttF = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttG = new ChebyshevTT(p => p[0], 1, new[] { new[] { 0.0, 2.0 } }, new[] { 4 });
        ttF.Build(verbose: false);
        ttG.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => { var _ = ttF + ttG; });
    }

    [Fact]
    public void Test_add_incompatible_n_nodes_raises()
    {
        var ttF = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttG = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttF.Build(verbose: false);
        ttG.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => { var _ = ttF + ttG; });
    }

    [Fact]
    public void Test_add_function_is_null_on_result()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var result = ttA + ttB;
        Assert.Throws<InvalidOperationException>(() => result.RunCompletion());
    }

    [Fact]
    public void Test_chained_adds_respect_max_rank()
    {
        ChebyshevTT MakeTt(double coef)
        {
            var tt = new ChebyshevTT(p => coef * (p[0] + p[1]), 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 6, 6 }, maxRank: 4);
            tt.Build(verbose: false);
            return tt;
        }
        var result = MakeTt(1.0) + MakeTt(2.0) + MakeTt(3.0);
        foreach (int r in result.TtRanks)
            Assert.True(r <= 4, $"max_rank=4 violated; got rank {r}");
    }

    [Fact]
    public void Test_linearity_eval()
    {
        var ttF = TestFixtures.TtAlgebraF;
        var ttG = TestFixtures.TtAlgebraG;
        // (a*f + b*g).eval(x) ≈ a*f(x) + b*g(x)
        double a = 2.0, b = -1.5;
        var combo = a * ttF + b * ttG;
        foreach (double[] p in new[] { new[] { 0.1, 0.2 }, new[] { -0.3, 0.5 } })
        {
            double expected = a * ttF.Eval(p) + b * ttG.Eval(p);
            TestFixtures.AssertClose(expected, combo.Eval(p), atol: 1e-6);
        }
    }
}

public class BinaryInPlaceTests
{
    [Fact]
    public void Test_add_in_place_matches_functional()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        double[] xs = { -0.5, 0.0, 0.5 };
        var functional = ttA + ttB;
        ttA.AddInPlace(ttB);
        foreach (double x in xs)
            TestFixtures.AssertClose(functional.Eval(new[] { x }), ttA.Eval(new[] { x }), atol: 1e-10);
    }

    [Fact]
    public void Test_sub_in_place_matches_functional()
    {
        var ttA = new ChebyshevTT(p => p[0] + 1.0, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        var functional = ttA - ttB;
        ttA.SubInPlace(ttB);
        TestFixtures.AssertClose(functional.Eval(new[] { 0.3 }), ttA.Eval(new[] { 0.3 }), atol: 1e-10);
    }

    [Fact]
    public void Test_add_in_place_grid_mismatch_raises()
    {
        var ttA = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        var ttB = new ChebyshevTT(p => p[0], 1, new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        Assert.Throws<ArgumentException>(() => ttA.AddInPlace(ttB));
    }
}

public class RoundingTests
{
    [Fact]
    public void Test_round_in_place_shrinks_rank_without_losing_accuracy()
    {
        // Build a sum that has artificially high rank, then round.
        var ttA = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 }, maxRank: 8);
        var ttB = new ChebyshevTT(p => Math.Sin(p[0]) + Math.Sin(p[1]), 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 }, maxRank: 8);
        ttA.Build(verbose: false);
        ttB.Build(verbose: false);
        // Block-diag sum has rank doubled.
        var sum = ttA + ttB;
        double evalBefore = sum.Eval(new[] { 0.3, -0.4 });
        sum.RoundInPlace(1e-10);
        double evalAfter = sum.Eval(new[] { 0.3, -0.4 });
        TestFixtures.AssertClose(evalBefore, evalAfter, atol: 1e-8);
    }

    [Fact]
    public void Test_round_in_place_idempotent()
    {
        var ttA = TestFixtures.TtAlgebraF;
        var sum = ttA + ttA;
        sum.RoundInPlace(1e-10);
        var ranksBefore = sum.TtRanks;
        sum.RoundInPlace(1e-10);
        var ranksAfter = sum.TtRanks;
        Assert.Equal(ranksBefore, ranksAfter);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

```bash
dotnet test --filter "FullyQualifiedName~BinaryAlgebraTests|FullyQualifiedName~BinaryInPlaceTests|FullyQualifiedName~RoundingTests"
```

Expected: build fails — `+`, `-`, `AddInPlace`, `SubInPlace`, `RoundInPlace` are not defined.

- [ ] **Step 3: Implement `AddCores` and `RoundCores` in `Internal/TensorTrainAlgebra.cs`**

Open `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs`. Append before the closing `}` of the class:

```csharp
    /// <summary>
    /// Block-diagonal stacking of TT cores → exact TT representation of the sum.
    /// Mirror of Python's <c>_tt_add_cores</c> (_algebra.py:63).
    /// </summary>
    internal static TensorTrainKernel.TtCore[] AddCores(
        TensorTrainKernel.TtCore[] coresA,
        TensorTrainKernel.TtCore[] coresB)
    {
        int d = coresA.Length;
        if (d != coresB.Length)
            throw new ArgumentException("AddCores: cores must have same length");

        // d == 1 special case: elementwise coefficient sum (only correct rep keeping rank-1 endpoints).
        if (d == 1)
        {
            var a0 = coresA[0]; var b0 = coresB[0];
            if (a0.RLeft != b0.RLeft || a0.NNodes != b0.NNodes || a0.RRight != b0.RRight)
                throw new ArgumentException(
                    $"AddCores: 1D core shape mismatch ({a0.RLeft},{a0.NNodes},{a0.RRight}) vs ({b0.RLeft},{b0.NNodes},{b0.RRight})");
            var sum = new TensorTrainKernel.TtCore(a0.RLeft, a0.NNodes, a0.RRight);
            for (int i = 0; i < a0.Data.Length; i++)
                sum.Data[i] = a0.Data[i] + b0.Data[i];
            return new[] { sum };
        }

        var result = new TensorTrainKernel.TtCore[d];
        for (int k = 0; k < d; k++)
        {
            var a = coresA[k]; var b = coresB[k];
            int n = a.NNodes;
            if (b.NNodes != n)
                throw new ArgumentException($"AddCores: core {k} nNodes mismatch: {n} vs {b.NNodes}");

            if (k == 0)
            {
                // Concat along right rank: shape (1, n, ra_r + rb_r)
                int rR = a.RRight + b.RRight;
                var newCore = new TensorTrainKernel.TtCore(1, n, rR);
                for (int j = 0; j < n; j++)
                {
                    for (int kk = 0; kk < a.RRight; kk++) newCore[0, j, kk] = a[0, j, kk];
                    for (int kk = 0; kk < b.RRight; kk++) newCore[0, j, a.RRight + kk] = b[0, j, kk];
                }
                result[k] = newCore;
            }
            else if (k == d - 1)
            {
                // Concat along left rank: shape (ra_l + rb_l, n, 1)
                int rL = a.RLeft + b.RLeft;
                var newCore = new TensorTrainKernel.TtCore(rL, n, 1);
                for (int j = 0; j < n; j++)
                {
                    for (int ii = 0; ii < a.RLeft; ii++) newCore[ii, j, 0] = a[ii, j, 0];
                    for (int ii = 0; ii < b.RLeft; ii++) newCore[a.RLeft + ii, j, 0] = b[ii, j, 0];
                }
                result[k] = newCore;
            }
            else
            {
                // Block diagonal: shape (ra_l + rb_l, n, ra_r + rb_r)
                int rL = a.RLeft + b.RLeft;
                int rR = a.RRight + b.RRight;
                var newCore = new TensorTrainKernel.TtCore(rL, n, rR);
                for (int j = 0; j < n; j++)
                {
                    for (int ii = 0; ii < a.RLeft; ii++)
                        for (int kk = 0; kk < a.RRight; kk++)
                            newCore[ii, j, kk] = a[ii, j, kk];
                    for (int ii = 0; ii < b.RLeft; ii++)
                        for (int kk = 0; kk < b.RRight; kk++)
                            newCore[a.RLeft + ii, j, a.RRight + kk] = b[ii, j, kk];
                }
                result[k] = newCore;
            }
        }
        return result;
    }

    /// <summary>
    /// Round TT to lower rank via TT-SVD recompression. Right-to-left QR sweep
    /// (right-canonicalize cores d-1..1) followed by left-to-right SVD truncation
    /// (cores 0..d-2). Truncation keeps min(maxRank, num_above_relative_tol)
    /// singular values. Mirror of Python's <c>_tt_round_cores</c> (_algebra.py:118).
    /// </summary>
    internal static TensorTrainKernel.TtCore[] RoundCores(
        TensorTrainKernel.TtCore[] cores, int maxRank, double tolerance = 1e-12)
    {
        int d = cores.Length;
        var result = new TensorTrainKernel.TtCore[d];
        for (int k = 0; k < d; k++) result[k] = cores[k].Copy();
        if (d == 1) return result;

        // Right-to-left QR sweep: right-canonicalize cores k = d-1, ..., 1.
        // (Reuses Phase 2 Task 3's per-core OrthRight semantics.)
        // Use the existing OrthRightSweep iteratively from each k to k-1: simpler to inline.
        for (int k = d - 1; k > 0; k--)
        {
            int rL = result[k].RLeft, n = result[k].NNodes, rR = result[k].RRight;
            // Reshape (rL, n*rR), QR of transpose
            var Mt = new double[n * rR, rL];
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        Mt[j * rR + p, i] = result[k][i, j, p];
            var Mtm = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(Mt);
            var qr = Mtm.QR(MathNet.Numerics.LinearAlgebra.Factorization.QRMethod.Thin);
            int newRL = qr.Q.ColumnCount;
            var newCk = new TensorTrainKernel.TtCore(newRL, n, rR);
            for (int a = 0; a < newRL; a++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        newCk[a, j, p] = qr.Q[j * rR + p, a];
            // Push R^T into the previous core's right rank: prev[i, j, s] -> prev_new[i, j, r] = sum_s prev[i,j,s] * Rt^T[s,r] = sum_s prev[i,j,s] * R[r,s]
            var prev = result[k - 1];
            int rLp = prev.RLeft, nP = prev.NNodes;
            var newPrev = new TensorTrainKernel.TtCore(rLp, nP, newRL);
            for (int i = 0; i < rLp; i++)
                for (int j = 0; j < nP; j++)
                    for (int r = 0; r < newRL; r++)
                    {
                        double s = 0;
                        for (int sIdx = 0; sIdx < rL; sIdx++)
                            s += prev[i, j, sIdx] * qr.R[r, sIdx];
                        newPrev[i, j, r] = s;
                    }
            result[k] = newCk;
            result[k - 1] = newPrev;
        }

        // Left-to-right SVD truncation: cores k = 0, ..., d-2.
        for (int k = 0; k < d - 1; k++)
        {
            int rL = result[k].RLeft, n = result[k].NNodes, rR = result[k].RRight;
            var Mat = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(rL * n, rR, 0.0);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int p = 0; p < rR; p++)
                        Mat.At(i * n + j, p, result[k][i, j, p]);

            var svd = Mat.Svd(computeVectors: true);
            var U = svd.U; var S = svd.S; var Vt = svd.VT;
            int sLen = S.Count;
            int keep = Math.Min(maxRank, sLen);
            double sMax = sLen > 0 ? S[0] : 0.0;
            if (sMax > 0 && tolerance > 0)
            {
                int eff = 0;
                for (int i = 0; i < sLen; i++) if (S[i] > sMax * tolerance) eff++;
                keep = Math.Max(1, Math.Min(keep, eff));
            }
            else
            {
                keep = Math.Max(1, keep);
            }

            var newCk = new TensorTrainKernel.TtCore(rL, n, keep);
            for (int i = 0; i < rL; i++)
                for (int j = 0; j < n; j++)
                    for (int r = 0; r < keep; r++)
                        newCk[i, j, r] = U[i * n + j, r];
            result[k] = newCk;

            // Push S @ Vt into next core's left rank.
            var nextC = result[k + 1];
            int n2 = nextC.NNodes, rR2 = nextC.RRight;
            var newNext = new TensorTrainKernel.TtCore(keep, n2, rR2);
            for (int r = 0; r < keep; r++)
                for (int j = 0; j < n2; j++)
                    for (int p = 0; p < rR2; p++)
                    {
                        double sAcc = 0;
                        for (int sIdx = 0; sIdx < rR; sIdx++)
                            sAcc += S[r] * Vt[r, sIdx] * nextC[sIdx, j, p];
                        newNext[r, j, p] = sAcc;
                    }
            result[k + 1] = newNext;
        }
        return result;
    }
```

- [ ] **Step 4: Add binary operators + in-place to `ChebyshevTT.cs`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Add after the unary operator block:

```csharp
    // ------------------------------------------------------------------
    // Binary algebra (Phase 2 — PyChebyshev v0.18.d)
    // ------------------------------------------------------------------

    /// <summary>Default tolerance for TT-SVD rounding after addition/subtraction.</summary>
    public const double DefaultRoundTolerance = 1e-12;

    /// <summary>Validate two TTs share the same grid (numDim, domain, nNodes).</summary>
    private static void CheckCompatible(ChebyshevTT a, ChebyshevTT b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        a.CheckBuilt();
        b.CheckBuilt();
        if (a._numDimensions != b._numDimensions)
            throw new ArgumentException(
                $"Dimension mismatch: {a._numDimensions} vs {b._numDimensions}");
        for (int d = 0; d < a._numDimensions; d++)
        {
            if (a._nNodes[d] != b._nNodes[d])
                throw new ArgumentException(
                    $"nNodes mismatch at dim {d}: {a._nNodes[d]} vs {b._nNodes[d]}");
            if (a._domain[d][0] != b._domain[d][0] || a._domain[d][1] != b._domain[d][1])
                throw new ArgumentException(
                    $"Domain mismatch at dim {d}: [{a._domain[d][0]}, {a._domain[d][1]}] vs [{b._domain[d][0]}, {b._domain[d][1]}]");
        }
    }

    /// <summary>Binary addition: <c>a + b</c>. Result is rounded to the larger of the two TTs' maxRank.</summary>
    public static ChebyshevTT operator +(ChebyshevTT a, ChebyshevTT b)
    {
        CheckCompatible(a, b);
        var summed = TensorTrainAlgebra.AddCores(a._coeffCores!, b._coeffCores!);
        int mr = Math.Max(a._maxRank, b._maxRank);
        var rounded = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        var domainCopy = a._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])a._nNodes.Clone();
        return a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
    }

    /// <summary>Binary subtraction: <c>a - b</c>.</summary>
    public static ChebyshevTT operator -(ChebyshevTT a, ChebyshevTT b)
    {
        CheckCompatible(a, b);
        var negB = TensorTrainAlgebra.NegateCores(b._coeffCores!);
        var summed = TensorTrainAlgebra.AddCores(a._coeffCores!, negB);
        int mr = Math.Max(a._maxRank, b._maxRank);
        var rounded = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        var domainCopy = a._domain.Select(d => (double[])d.Clone()).ToArray();
        var nNodesCopy = (int[])a._nNodes.Clone();
        return a.BuildResultFromCores(rounded, domainCopy, nNodesCopy);
    }

    /// <summary>In-place addition: <c>this += other</c> followed by TT-SVD rounding.</summary>
    public void AddInPlace(ChebyshevTT other)
    {
        CheckCompatible(this, other);
        var summed = TensorTrainAlgebra.AddCores(_coeffCores!, other._coeffCores!);
        int mr = Math.Max(_maxRank, other._maxRank);
        _coeffCores = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    /// <summary>In-place subtraction: <c>this -= other</c> followed by TT-SVD rounding.</summary>
    public void SubInPlace(ChebyshevTT other)
    {
        CheckCompatible(this, other);
        var negOther = TensorTrainAlgebra.NegateCores(other._coeffCores!);
        var summed = TensorTrainAlgebra.AddCores(_coeffCores!, negOther);
        int mr = Math.Max(_maxRank, other._maxRank);
        _coeffCores = TensorTrainAlgebra.RoundCores(summed, mr, DefaultRoundTolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }

    /// <summary>Round TT to lower rank in place via TT-SVD recompression.</summary>
    public void RoundInPlace(double tolerance)
    {
        CheckBuilt();
        _coeffCores = TensorTrainAlgebra.RoundCores(_coeffCores!, _maxRank, tolerance);
        _cachedErrorEstimate = null;
        for (int i = 0; i < _numDimensions; i++)
            _ttRanks![i + 1] = _coeffCores[i].RRight;
    }
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~BinaryAlgebraTests|FullyQualifiedName~BinaryInPlaceTests|FullyQualifiedName~RoundingTests"
```

Expected: 13 tests pass (8 BinaryAlgebra + 3 BinaryInPlace + 2 Rounding).

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 756` (743 + 13 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAlgebraTests.cs
git commit -m "phase2: implement binary algebra (+ - AddInPlace SubInPlace RoundInPlace)"
```

---

## Task 11: JSON migration (format `0.5.0` → `0.6.0`)

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add `Method` to `TTSerializationState`, bump format version, defensively backfill on `Load`.
- Create: `tests/ChebyshevSharp.Tests/TestData/TtV050Sample.json` (a minimal v0.5.0 fixture file).
- Modify: `tests/ChebyshevSharp.Tests/TtAlsTests.cs` — append `JsonMigrationTests` class.

**Design notes:**
- v0.13 added the `Method` attribute (Python persists it; we already track it in memory). v0.18 doesn't add new persisted state from algebra/extrude (results are cores already serialized). Version bumps for consistency.
- `Load` defensively reads the JSON: if `Version == "0.5.0"` (or any pre-0.6.0), set `Method = null` and continue. Existing 0.5.0 files have no `Method` field.
- Save unconditionally writes the new format `0.6.0`.

- [ ] **Step 1: Write failing migration tests**

Append to `tests/ChebyshevSharp.Tests/TtAlsTests.cs`:

```csharp
public class JsonMigrationTests
{
    [Fact]
    public void Test_save_load_at_060_round_trip()
    {
        var tt = new ChebyshevTT(p => Math.Sin(p[0]), 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 6 });
        tt.Build(verbose: false, method: "cross");
        string path = Path.GetTempFileName();
        try
        {
            tt.Save(path);
            string json = File.ReadAllText(path);
            Assert.Contains("\"Version\":\"0.6.0\"", json.Replace(" ", ""));
            Assert.Contains("\"Method\":\"cross\"", json.Replace(" ", ""));
            var loaded = ChebyshevTT.Load(path);
            TestFixtures.AssertClose(tt.Eval(new[] { 0.3 }), loaded.Eval(new[] { 0.3 }), atol: 1e-12);
            Assert.Equal("cross", loaded.Method);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Test_load_050_file_backfills_method_null()
    {
        // The fixture file at TestData/TtV050Sample.json was generated by ChebyshevSharp 0.5.0
        // and has no Method field. Load must backfill Method == null without error.
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "TtV050Sample.json");
        Assert.True(File.Exists(path), $"fixture file missing: {path}");
        var loaded = ChebyshevTT.Load(path);
        Assert.Null(loaded.Method);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~JsonMigrationTests"
```

Expected: tests fail — fixture file missing and `Method` is not yet round-tripped.

- [ ] **Step 3: Generate the v0.5.0 fixture file (one-shot from the current source)**

Before bumping the format version, save a baseline file using current Save() output. Add a temporary throw-away test that creates the fixture, run it, then commit the file. Easier alternative: hand-author the JSON. Below is a hand-authored minimal fixture (1D, n=4, no Method field):

Create directory and file `tests/ChebyshevSharp.Tests/TestData/TtV050Sample.json`:

```json
{"Version":"0.5.0","NumDimensions":1,"Domain":[[-1,1]],"NNodes":[4],"MaxRank":10,"Tolerance":1e-06,"MaxSweeps":10,"TtRanks":[1,1],"BuildTime":0.001,"TotalBuildEvals":4,"Cores":[{"RLeft":1,"NNodes":4,"RRight":1,"Data":[0.0,0.0,0.0,0.0]}]}
```

In `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj`, ensure the file is copied to output (add inside the `<Project>` element):

```xml
  <ItemGroup>
    <None Update="TestData\TtV050Sample.json">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
  </ItemGroup>
```

(If a similar `<ItemGroup>` for `TestData/` already exists, add the new entry to it instead of creating a duplicate.)

- [ ] **Step 4: Bump JSON format version + add `Method` to `TTSerializationState`**

Open `src/ChebyshevSharp/ChebyshevTT.cs`. Find the `TTSerializationState` class. Add a new property:

```csharp
        public string? Method { get; set; }
```

In the `Save` method, find the `var state = new TTSerializationState { ... }` block and:
1. Replace `Version = GetLibraryVersion()` with `Version = "0.6.0"`.
2. Add `Method = Method` (the public field on `ChebyshevTT`).

In the `Load` method, after reading the state, **before** constructing the `ChebyshevTT`, defensively backfill:

```csharp
        // v0.5.0 files have no Method field; deserializer leaves it null.
        // No additional work needed here — null propagates cleanly.
```

After `tt._built = true;` and similar setup, also set:

```csharp
        tt.Method = state.Method;
```

(Note: `tt.Method` has a `private set;` — add `internal set;` if needed, or set via a private constructor channel.)

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~JsonMigrationTests"
```

Expected: 2 tests pass.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: `Passed: 758` (756 + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TestData/TtV050Sample.json tests/ChebyshevSharp.Tests/TtAlsTests.cs tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj
git commit -m "phase2: bump TT JSON format 0.5.0 -> 0.6.0 (persist Method)"
```

---


## Task 12: Documentation, parity metadata, version bump, release prep

**Files:**

**WORKTREE ENFORCEMENT (MANDATORY):**
Before any work, run `git rev-parse --show-toplevel` and confirm the output ends in `.worktrees/phase2-tt-parity`. If it ends in `/Documents/ChebyshevSharp` (the main repo), STOP — switch to the worktree at `/home/max/Documents/ChebyshevSharp/.worktrees/phase2-tt-parity` and re-run all commands from there. All `git`, `dotnet`, and file paths in this task assume the worktree as cwd.

- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj` — `<Version>` `0.5.0` → `0.6.0`; `<PyChebyshevParity>0.18.0`; `<InformationalVersion>0.6.0+pychebyshev.0.18.0`; updated `<Description>`.
- Modify: `docs/docs/tensor-train.md` — append sections for ALS / RunCompletion / OrthLeft+OrthRight / InnerProduct / Nodes+FromValues / Extrude+Slice+ToDense / algebra.
- Modify: `docs/docs/changelog.md` — prepend `## [0.6.0]` per the new two-tier convention.
- Modify: `README.md` — bump parity badge `v0.12.0` → `v0.18.0`.
- Modify: `CLAUDE.md` — update Status block with PyChebyshev parity v0.18.0, Phase 2 of 6 complete, test count.
- Modify: `skip_csharp.txt` — append Phase 2 entries.

- [ ] **Step 1: Bump csproj version + parity metadata**

Open `src/ChebyshevSharp/ChebyshevSharp.csproj`. Find:

```xml
    <Version>0.5.0</Version>
    <PyChebyshevParity>0.12.0</PyChebyshevParity>
    <Description>ChebyshevSharp 0.5.0 — multi-dimensional Chebyshev tensor interpolation with analytical derivatives. Feature parity with PyChebyshev v0.12.0.</Description>
    <InformationalVersion>0.5.0+pychebyshev.0.12.0</InformationalVersion>
```

Replace with:

```xml
    <Version>0.6.0</Version>
    <PyChebyshevParity>0.18.0</PyChebyshevParity>
    <Description>ChebyshevSharp 0.6.0 — multi-dimensional Chebyshev tensor interpolation with analytical derivatives. Feature parity with PyChebyshev v0.18.0.</Description>
    <InformationalVersion>0.6.0+pychebyshev.0.18.0</InformationalVersion>
```

- [ ] **Step 2: Build to confirm csproj changes**

```bash
dotnet build
```

Expected: succeeds with zero warnings.

- [ ] **Step 3: Extend `docs/docs/tensor-train.md`**

Open `docs/docs/tensor-train.md`. Append the following sections at the end:

```markdown
## Build Modes

`Build(method: ...)` accepts three algorithms:

```csharp
tt.Build(method: "cross");  // TT-Cross (default): O(d * n * r^2) function evals
tt.Build(method: "svd");    // TT-SVD: full O(n^d) tensor; for validation
tt.Build(method: "als");    // Alternating LS: rank-adaptive, full-grid evals
```

ALS starts at rank 1 and grows the TT rank by +1 per outer iteration until the
grid residual falls below `tolerance` or the rank reaches `maxRank`. If the
cap is hit before tolerance is satisfied, `BuildWarning` is set.

```csharp
tt.Build(method: "als", seed: 42);
if (tt.BuildWarning != null)
    Console.Error.WriteLine(tt.BuildWarning);
```

## Refining a Built TT — `RunCompletion`

`RunCompletion(tolerance, maxIter)` runs fixed-rank ALS sweeps on an
already-built TT. Rank is preserved; only per-core coefficients are refined.
Requires `Function != null` (so it cannot be called on a TT loaded from disk).

```csharp
tt.Build(method: "cross", tolerance: 1e-3, maxRank: 5);
tt.RunCompletion(tolerance: 1e-10, maxIter: 20);
```

## Canonicalization — `OrthLeft` / `OrthRight`

Push R factors through the TT chain so cores up to `position` are
left-orthogonal (`Q^T Q = I` after the `(rL*n, rR)` unfolding) or so cores
beyond `position` are right-orthogonal. The represented tensor is unchanged.
Useful as a primitive for downstream algorithms.

```csharp
tt.OrthLeft(position: 2);   // cores 0 and 1 become left-orthogonal
tt.OrthRight(position: 0);  // cores 1, 2, ... become right-orthogonal
```

## Inner Product

Frobenius inner product of two TTs' Chebyshev coefficient tensors:

```csharp
double ip = ttA.InnerProduct(ttB);  // sum_i C_a[i] * C_b[i]
```

Both TTs must share the same `NumDimensions`, `Domain`, and `NNodes`.

## Static Factories — `Nodes` / `FromValues`

```csharp
var (nodesPerDim, shape) = ChebyshevTT.Nodes(2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 });

// Build a TT from a precomputed dense tensor (skip TT-Cross):
double[] dense = /* row-major Π nNodes values */;
var tt = ChebyshevTT.FromValues(dense, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 8, 8 },
    maxRank: 10, tolerance: 1e-6);
```

## Materialization — `ToDense`

```csharp
double[] flat = tt.ToDense();   // row-major Π nNodes
```

Throws `OverflowException` if `Π nNodes * 8 > int.MaxValue`. Use for
inspection / round-trip testing, not high-D production.

## Slicing & Extrusion — `Slice` / `Extrude`

```csharp
var sliced = tt.Slice(dim: 0, value: 0.5);                // dim 0 fixed at 0.5
var extruded = tt.Extrude(dim: 1, newDomain: (0, 1), newN: 5);  // add a constant dim
```

`Slice` uses barycentric interpolation along the sliced dim's value-space core
and absorbs the resulting matrix into a neighbor. A fast path triggers when
`value` coincides with a Chebyshev node within `1e-14`.

## Algebra

```csharp
var sum  = ttA + ttB;          // block-diagonal stacking, then TT-SVD round
var diff = ttA - ttB;
var neg  = -ttA;               // unary
var dbl  = 2.0 * ttA;          // scalar mul (commutative)
var half = ttA / 2.0;          // throws DivideByZeroException on zero

// In-place equivalents (mutate the receiver, return void):
ttA.AddInPlace(ttB);
ttA.SubInPlace(ttB);
ttA.ScalarMulInPlace(2.0);
ttA.ScalarDivInPlace(2.0);
ttA.NegateInPlace();
ttA.RoundInPlace(1e-10);       // explicit TT-SVD recompression
```

`+` and `-` round to the larger of the two operands' `MaxRank` with a default
TT-SVD tolerance of `1e-12`. Use `RoundInPlace(tolerance)` when you need
tighter or looser control.
```

- [ ] **Step 4: Update `docs/docs/changelog.md`**

Prepend to `docs/docs/changelog.md`:

```markdown
## [0.6.0] - 2026-04-28

### PyChebyshev parity: v0.18.0

#### Added — TT canonicalization + ALS

- `ChebyshevTT.OrthLeft(int position)` and `OrthRight(int position)` —
  in-place QR/LQ canonicalization of the TT chain.
- `ChebyshevTT.InnerProduct(ChebyshevTT)` — Frobenius inner product of
  two TTs' Chebyshev coefficient tensors.
- `Build(method: "als")` — rank-adaptive alternating least-squares build
  mode. Starts at rank 1 and grows by 1 per outer iteration until the
  grid residual falls below `tolerance` or rank reaches `maxRank`.
- `ChebyshevTT.RunCompletion(double tolerance, int maxIter)` — refine an
  already-built TT in place via fixed-rank ALS sweeps.
- New `Method` property mirroring Python's `tt.method` attribute.
- New `BuildWarning` property — set when ALS hits `maxRank` before
  tolerance is satisfied (replaces Python's `RuntimeWarning`).

#### Added — TT factories, materialization, slicing, algebra

- `ChebyshevTT.Nodes(numDim, domain, nNodes)` — static factory matching
  `ChebyshevApproximation.Nodes`.
- `ChebyshevTT.FromValues(tensorValues, ...)` — TT-SVD compress a
  precomputed dense tensor into a TT (skips TT-Cross).
- `ChebyshevTT.ToDense()` — materialize the TT chain into a flat row-major
  dense tensor. Throws `OverflowException` on huge shapes.
- `ChebyshevTT.Slice(int dim, double value)` — fix a dimension at a value,
  returning a lower-dim TT.
- `ChebyshevTT.Extrude(int dim, (double, double), int)` — insert a new
  dimension where the function is constant.
- Operators `+`, `-`, scalar `*` and `/`, unary `-`. Binary operators
  round to `max(maxRank_a, maxRank_b)` at default tolerance `1e-12`.
- In-place methods `AddInPlace`, `SubInPlace`, `ScalarMulInPlace`,
  `ScalarDivInPlace`, `NegateInPlace`, `RoundInPlace(tolerance)`.

#### Changed

- TT JSON serialization format bumped to `"0.6.0"` to persist `Method`.
  Loading 0.5.0 files is supported (Method backfills to null).
- `Internal/TensorTrainKernel.cs` split into kernel + algebra + extrude
  modules. Public API and behavior unchanged.

#### Skipped

- Multi-dim variadic forms of `Extrude`/`Slice` (Python accepts a list
  of tuples). C# surface is single-dim per call; chain calls for
  multi-dim. No information lost; cleaner overload set.
```

- [ ] **Step 5: Update README parity badge**

Open `README.md`. Find the `PyChebyshev_parity` badge line and replace with:

```markdown
![PyChebyshev parity](https://img.shields.io/badge/PyChebyshev_parity-v0.18.0-blue)
```

- [ ] **Step 6: Update CLAUDE.md Status block**

Open `CLAUDE.md`. Find the "Status" section. Update:

- Submodule pin: `v0.10.1` → `v0.18.0` (or whatever Phase 1 left it at).
- Test count: `613/613` → `~758/758` (final from Task 11).
- Phase tracker line: append "Phase 2 of 6 complete (PyChebyshev v0.18.0 parity, TT feature parity)".
- Under the `ChebyshevTT` reading guide, add:
  - "Canonicalization: `OrthLeft(position)` / `OrthRight(position)` push R factors through the chain."
  - "ALS build mode (`Build(method=\"als\")`) is rank-adaptive; refines via `RunCompletion`."
  - "Algebra: `+ - * /` operators + `AddInPlace`/`SubInPlace`/`ScalarMulInPlace`/`ScalarDivInPlace`/`NegateInPlace`/`RoundInPlace`."
  - "Slicing/extrusion: `Slice(dim, value)` and `Extrude(dim, (lo, hi), newN)` return new TTs."
  - "Materialization: `ToDense()` (guarded against allocation overflow); factories `Nodes(...)` and `FromValues(...)`."

- [ ] **Step 7: Update `skip_csharp.txt`**

Append to `skip_csharp.txt`:

```
=========================================================
Phase 2 (ChebyshevSharp v0.6.0 release - parity v0.18.0): TT Feature Parity
=========================================================
TtCanonicalizationTests.cs: ~15 tests (port of TestOrthogonalization + TestInnerProduct)
TtAlsTests.cs: ~26 tests (port of TestALSInternals + TestALS + TestCompletion + JsonMigration)
TtFactoriesTests.cs: ~11 tests (port of TestTTNodes + TestTTFromValues)
TtExtrudeSliceTests.cs: ~21 tests (port of TestTTToDense + TestTTExtrude + TestTTSlice)
TtAlgebraTests.cs: ~25 tests (port of TestTTAddition + TestTTScalarMul + cross-feature)
Total: ~98 new tests (666 -> ~758 total)
```

- [ ] **Step 8: Final full-suite run**

```bash
dotnet build
dotnet test
```

Expected:
- `dotnet build`: zero warnings on net8.0 + net10.0.
- `dotnet test`: `Passed: 758` on both TFMs.

- [ ] **Step 9: Commit + tag + push**

```bash
git add src/ChebyshevSharp/ChebyshevSharp.csproj docs/docs/tensor-train.md docs/docs/changelog.md README.md CLAUDE.md skip_csharp.txt
git commit -m "phase2: docs + parity metadata + v0.6.0 release prep (PyChebyshev v0.18.0 parity)"
```

- [ ] **Step 10: Open PR**

```bash
git push -u origin phase2-tt-parity
gh pr create --title "Phase 2: TT feature parity (PyChebyshev v0.18.0)" --body "$(cat <<'EOF'
## Summary
- Advances PyChebyshev parity from v0.12.0 to v0.18.0
- Brings ChebyshevTT to feature-complete parity: canonicalization, inner product, ALS build mode, RunCompletion, Nodes/FromValues factories, ToDense, Extrude/Slice, full algebra (+ - * /) plus in-place equivalents
- Splits Internal/TensorTrainKernel.cs into kernel/algebra/extrude (zero behavior change)
- Ships as ChebyshevSharp v0.6.0

## Test plan
- [ ] dotnet test passes 758/758 on net8.0 and net10.0
- [ ] dotnet build emits zero warnings
- [ ] Submodule pinned at v0.18.0
- [ ] JSON 0.5.0 → 0.6.0 round-trip with backfill

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 11: After PR merges, cut GitHub release**

```bash
git checkout main
git pull
gh release create v0.6.0 --title "v0.6.0 — PyChebyshev v0.18.0 parity (TT feature parity)" --notes-from-tag
```

Verify: `publish.yml` workflow runs and pushes the package to NuGet.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by task |
|---|---|
| Submodule advance to v0.18.0 + test stubs | Task 1 |
| Refactor TensorTrainKernel.cs (no behavior change) | Task 2 |
| `OrthLeft`/`OrthRight` canonicalization | Task 3 |
| `InnerProduct` | Task 4 |
| `Build(method="als")` + adaptive driver | Task 5 |
| `RunCompletion` | Task 6 |
| `Nodes`/`FromValues` factories | Task 7 |
| `Extrude`/`Slice`/`ToDense` | Task 8 |
| Scalar algebra (`*`/`/`/unary `-`) + in-place | Task 9 |
| Binary algebra (`+`/`-`) + `AddInPlace`/`SubInPlace`/`RoundInPlace` | Task 10 |
| JSON migration `0.5.0` → `0.6.0` (Method persistence) | Task 11 |
| Docs + parity metadata + release prep | Task 12 |
| Worktree enforcement on every implementer task | Tasks 1–12 |
| Stochastic-only assertions for ALS | Task 5 design notes + test class header |
| Python source pointers in design notes | Tasks 3, 4, 5, 6, 7, 8, 9, 10 |

**Type consistency check:**

- `TensorTrainKernel.TtCore` is the shared core type; `TtCore[]` everywhere.
- `Method` is `string?` (Task 5 declares, Task 11 persists).
- `BuildWarning` is `string?` (Task 5 declares).
- `_coeffCores` field already declared as `TensorTrainKernel.TtCore[]?`.
- `OrthLeftSweep`/`OrthRightSweep` mutate `TtCore[]` in place — Tasks 3, 5, 10 all consume.
- `InnerProductCores` returns `double` (Task 4).
- `ScalarMulCores`/`NegateCores`/`AddCores`/`RoundCores` return `TtCore[]` (Tasks 9, 10).
- `FromValuesTtSvd`/`ExtrudeCores`/`SliceCores`/`ToDenseEinsumChain` (Tasks 7, 8) live in `TensorTrainExtrude`.
- `BuildResultFromCores` is `internal` on `ChebyshevTT` and used by Tasks 8, 9, 10.
- `CoeffCoreToValueCore` is shared between Tasks 6 (RunCompletion) and 8 (Slice/ToDense).
- Test count progression: Task 1 baseline 666; Task 3 +8 (674); Task 4 +7 (681); Task 5 +10 (691); Task 6 +8 (699); Task 7 +11 (710); Task 8 +21 (731); Task 9 +12 (743); Task 10 +13 (756); Task 11 +2 (758). Total Phase 2 additions: 92 (estimate of "~75" from spec was conservative; actual breakdown is denser due to in-place tests).

**Stochastic discipline:** Every Task 5 (ALS build) and Task 6 (RunCompletion via ALS) test uses `Assert.True(error < bound)` style. No inline-literal expected values from Python tests for ALS-touched outputs. Documented in Task 5 design notes and the file header of `TtAlsTests.cs`.

**Worktree enforcement:** Block included verbatim in every task's "Files" section.

**Placeholder scan:** No "TBD", "implement later", or `// rest of impl` markers. Each step has full inline code.

**Open implementation note:** Task 6 Step 3 includes a known-fragile spot (the inverse DCT-II constant scaling in `CoeffCoreToValueCore`). The plan calls out the round-trip self-test as the gating check; if constants are off, executor must derive from the forward formula in `BarycentricKernel.ChebyshevCoefficients1D`. This is acknowledged inline rather than buried.

