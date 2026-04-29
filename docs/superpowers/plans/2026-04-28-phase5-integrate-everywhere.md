# Phase 5 Implementation Plan — Integrate Everywhere (v0.9.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Integrate(int[]? dims, (double lo, double hi)[]? bounds)` to `ChebyshevSlider` and `ChebyshevTT`, completing calculus integration coverage across all four interpolant classes (PyChebyshev v0.17.0 parity).

**Architecture:** Faithful Python port. New public methods on Slider and TT match the existing `ChebyshevApproximation.Integrate` / `ChebyshevSpline.Integrate` signature exactly (`object` return type, same parameter shapes). Two new internal helpers in `Internal/Calculus.cs`: `SliderPartitionIntersect` (per-slide classification) and `IntegrateTtAlongDim` (per-core Fejér-1 contraction). Slider uses the closed-form sliding decomposition; TT uses Fejér-1 quadrature contracted into each integrated core after coefficient→value conversion.

**Tech Stack:** C# 12, .NET 8 + .NET 10 multi-target, xUnit, MathNet.Numerics (already a dependency).

**Spec:** `docs/superpowers/specs/2026-04-28-phase5-integrate-everywhere-design.md` (commit `9d09d29`, 472 lines, 10 design decisions D1–D10).

**Test count progression:**

| After task | Total tests | Δ |
|---|---|---|
| Baseline (Phase 4 complete, on main at `39223fe`) | 902 | — |
| Task 1 (helpers + helper tests) | 907 | +5 |
| Task 2 (Slider full integrate) | 911 | +4 |
| Task 3 (Slider partial integrate) | 918 | +7 |
| Task 4 (Slider validation + ergonomics) | 924 | +6 |
| Task 5 (TT full integrate) | 930 | +6 |
| Task 6 (TT partial integrate) | 938 | +8 |
| Task 7 (TT validation + cross-class) | 946 | +8 |
| Task 8 (release prep — no new tests) | 946 | 0 |

±2 drift per task is acceptable (consistent with Phase 4); larger drift requires investigation.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs` | All 17 Slider.Integrate tests (full + partial + validation + ergonomics) |
| `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs` | All 22 TT.Integrate tests (full + partial + validation + cross-class) |

### Modified files

| Path | What changes |
|---|---|
| `src/ChebyshevSharp/Internal/Calculus.cs` | Add `SliderPartitionIntersect` and `IntegrateTtAlongDim` |
| `src/ChebyshevSharp/ChebyshevSlider.cs` | Add public `Integrate` method; add private factory bypass for partial result |
| `src/ChebyshevSharp/ChebyshevTT.cs` | Add public `Integrate` method; add private helper for absorbing M matrices into kept cores |
| `tests/ChebyshevSharp.Tests/CalculusTests.cs` | Append 5 helper tests (`SliderPartitionIntersect` x4, `IntegrateTtAlongDim` x1) |
| `src/ChebyshevSharp/ChebyshevSharp.csproj` | Bump `<Version>` to 0.9.0, `<PyChebyshevParity>` to 0.17.0, `<InformationalVersion>` to `0.9.0+pychebyshev.0.17.0` |
| `docs/docs/changelog.md` | Add v0.9.0 entry following two-tier convention |
| `docs/docs/calculus.md` | Add Slider Integration and TT Integration subsections |
| `skip_csharp.txt` | Append Phase 5 entry — full v0.17 calculus completion ported |
| `CLAUDE.md` | Bump Status block (902 → 946 tests; mark Phase 5 of 6 complete) |

### Files NOT changed

- `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` — no new fixtures needed; existing `AlgebraSliderF`, `TtSin3D`, `TtBs5D` etc. plus per-test inline construction cover all cases. (The Slider partial-integrate tests need partition-specific configurations not currently in TestFixtures, but they're constructed inline in each test for readability.)
- `ref/PyChebyshev` submodule — stays at v0.18.0.
- TT JSON serialization — no new persisted state.

---

## Task 1: Internal helpers + helper tests

**Goal:** Add `Calculus.SliderPartitionIntersect` and `Calculus.IntegrateTtAlongDim` helpers, with 5 unit tests in `CalculusTests.cs`.

**Files:**
- Modify: `src/ChebyshevSharp/Internal/Calculus.cs:265-330` (append before the closing `}` of class `Calculus`)
- Modify: `tests/ChebyshevSharp.Tests/CalculusTests.cs` (append at end of file)

**Python source:** `ref/PyChebyshev/src/pychebyshev/_calculus.py:342-388`

### WORKTREE ENFORCEMENT (MANDATORY)

Before running any other commands:

```bash
git rev-parse --show-toplevel
```

Expected output ends in `.worktrees/phase5-integrate-everywhere`. If it ends in `/home/max/Documents/ChebyshevSharp` (the main repo), **STOP** and `cd` to the worktree. Phase 1 Task 4 cross-directory commit is the cautionary tale.

- [ ] **Step 1: Write failing helper tests in `CalculusTests.cs`**

Append to `tests/ChebyshevSharp.Tests/CalculusTests.cs` (after the last existing class):

```csharp
// ======================================================================
// TestSliderPartitionIntersect (Phase 5 — Calculus internal helper)
// ======================================================================

public class TestSliderPartitionIntersect
{
    [Fact]
    public void Test_full_intersection_returns_full()
    {
        // group [0, 1], integrating dims [0, 1] -> "full"
        var (kind, kept) = ChebyshevSharp.Internal.Calculus.SliderPartitionIntersect(
            groupDims: new[] { 0, 1 }, integrateDims: new[] { 0, 1 });
        Assert.Equal("full", kind);
        Assert.Empty(kept);
    }

    [Fact]
    public void Test_no_intersection_returns_none()
    {
        // group [2], integrating dims [0, 1] -> "none"
        var (kind, kept) = ChebyshevSharp.Internal.Calculus.SliderPartitionIntersect(
            groupDims: new[] { 2 }, integrateDims: new[] { 0, 1 });
        Assert.Equal("none", kind);
        Assert.Equal(new[] { 2 }, kept);
    }

    [Fact]
    public void Test_partial_intersection_returns_partial()
    {
        // group [0, 1, 2], integrating dims [1] -> "partial", kept [0, 2]
        var (kind, kept) = ChebyshevSharp.Internal.Calculus.SliderPartitionIntersect(
            groupDims: new[] { 0, 1, 2 }, integrateDims: new[] { 1 });
        Assert.Equal("partial", kind);
        Assert.Equal(new[] { 0, 2 }, kept);
    }

    [Fact]
    public void Test_empty_integrate_dims_returns_none()
    {
        var (kind, kept) = ChebyshevSharp.Internal.Calculus.SliderPartitionIntersect(
            groupDims: new[] { 0, 1 }, integrateDims: Array.Empty<int>());
        Assert.Equal("none", kind);
        Assert.Equal(new[] { 0, 1 }, kept);
    }
}

// ======================================================================
// TestIntegrateTtAlongDim (Phase 5 — Calculus internal helper)
// ======================================================================

public class TestIntegrateTtAlongDim
{
    [Fact]
    public void Test_contract_higher_rank_core_matches_manual_sum()
    {
        // (rLeft=2, n=3, rRight=2) hand-rolled core; weights [0.5, 0.25, 0.25]
        var core = new ChebyshevSharp.Internal.TensorTrainKernel.TtCore(2, 3, 2);
        // Fill with deterministic values: core[i, j, k] = 100*i + 10*j + k
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 2; k++)
                    core[i, j, k] = 100 * i + 10 * j + k;

        var weights = new[] { 0.5, 0.25, 0.25 };
        var result = ChebyshevSharp.Internal.Calculus.IntegrateTtAlongDim(core, weights);

        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
        // Manual: result[i, k] = sum_j core[i, j, k] * weights[j]
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 2; k++)
            {
                double expected = 0;
                for (int j = 0; j < 3; j++)
                    expected += (100 * i + 10 * j + k) * weights[j];
                ChebyshevSharp.Tests.Helpers.TestFixtures.AssertClose(
                    expected, result[i, k], rtol: 1e-14, atol: 1e-14);
            }
    }
}
```

- [ ] **Step 2: Run failing tests to verify the helpers don't yet exist**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderPartitionIntersect|FullyQualifiedName~TestIntegrateTtAlongDim"
```

Expected: build error — `'Calculus' does not contain a definition for 'SliderPartitionIntersect'` and `'Calculus' does not contain a definition for 'IntegrateTtAlongDim'`.

- [ ] **Step 3: Implement `SliderPartitionIntersect` in Calculus.cs**

Append before the final `}` of class `Calculus` in `src/ChebyshevSharp/Internal/Calculus.cs`:

```csharp
    /// <summary>
    /// Classify a slide group against an integration set.
    /// Mirror of Python <c>_calculus.py:342</c> <c>_slider_partition_intersect</c>.
    /// </summary>
    /// <param name="groupDims">Dimensions covered by the slide group (any order).</param>
    /// <param name="integrateDims">Dimensions being integrated over.</param>
    /// <returns>
    /// (kind, kept) where kind is one of "full" (every group dim is integrated),
    /// "partial" (some group dims are integrated), or "none" (no group dims are
    /// integrated). <c>kept</c> is the dimensions of the group NOT being
    /// integrated; empty for "full"; equals <c>groupDims</c> for "none".
    /// </returns>
    internal static (string kind, int[] kept) SliderPartitionIntersect(
        int[] groupDims, int[] integrateDims)
    {
        var groupSet = new HashSet<int>(groupDims);
        var integrateSet = new HashSet<int>(integrateDims);
        var overlap = new HashSet<int>(groupSet);
        overlap.IntersectWith(integrateSet);
        if (overlap.Count == 0) return ("none", (int[])groupDims.Clone());
        if (overlap.SetEquals(groupSet)) return ("full", Array.Empty<int>());
        return ("partial", groupDims.Where(d => !integrateSet.Contains(d)).ToArray());
    }

    /// <summary>
    /// Contract a single TT core along its node axis with quadrature weights.
    /// Mirror of Python <c>_calculus.py:372</c> <c>_integrate_tt_along_dim</c>.
    /// </summary>
    /// <param name="core">A TT core of shape (rLeft, n, rRight).</param>
    /// <param name="weights">Quadrature weights of length <c>core.NNodes</c>, scaled to the dim's domain.</param>
    /// <returns>An (rLeft, rRight) matrix M[r, s] = sum_j core[r, j, s] * weights[j].</returns>
    internal static double[,] IntegrateTtAlongDim(
        TensorTrainKernel.TtCore core, double[] weights)
    {
        if (weights.Length != core.NNodes)
            throw new ArgumentException(
                $"weights.Length ({weights.Length}) does not match core.NNodes ({core.NNodes})");
        var result = new double[core.RLeft, core.RRight];
        for (int r = 0; r < core.RLeft; r++)
            for (int s = 0; s < core.RRight; s++)
            {
                double acc = 0.0;
                for (int j = 0; j < core.NNodes; j++) acc += core[r, j, s] * weights[j];
                result[r, s] = acc;
            }
        return result;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderPartitionIntersect|FullyQualifiedName~TestIntegrateTtAlongDim"
```

Expected: 5 tests passed.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
dotnet test
```

Expected: 907 tests passing (902 baseline + 5 new). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/Internal/Calculus.cs tests/ChebyshevSharp.Tests/CalculusTests.cs
git commit -m "phase5: add SliderPartitionIntersect and IntegrateTtAlongDim helpers

Two internal static helpers in Calculus.cs ported from PyChebyshev
_calculus.py:342-388. Both internal-visible so unit tests in
CalculusTests.cs can call them directly (matches Phase 1 BarycentricKernel
pattern). 5 unit tests appended.

Test count: 902 -> 907 (+5)."
```

---

## Task 2: ChebyshevSlider.Integrate — full integration

**Goal:** Add `public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` to `ChebyshevSlider` returning a scalar (boxed as `object`) when every dimension is integrated. Defer partial-result construction to Task 3 (throw `NotImplementedException` for now).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs` — add public Integrate method (insert after the `ErrorEstimate()` block ~line 340, before the `TotalBuildEvals` property)
- Create: `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs`

**Python source:** `slider.py:877-1015` (the full-integration path; partial path lines 1016-1132 deferred to Task 3)

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 4 failing full-integration tests**

Create `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestSliderFullIntegrate (Phase 5)
// ======================================================================

public class TestSliderFullIntegrate
{
    [Fact]
    public void Test_pivot_only_function()
    {
        // f(x, y) = constant 5; integral = 5 * 2 * 3 = 30 over [0,2]x[0,3].
        static double F(double[] x, object? _) => 5.0;
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 3.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 1.0, 1.5 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(30.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_additive_function_sum_of_x()
    {
        // f(x, y) = x + y over [-1, 1]^2; integral = 0 (odd in both).
        static double F(double[] x, object? _) => x[0] + x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_separable_function_against_analytical()
    {
        // f(x, y) = sin(x) + cos(y) over [-1, 1]^2.
        // ∫∫ sin(x) dx dy = 0; ∫∫ cos(y) dx dy = 4 sin(1).
        static double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]);
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        double expected = 4.0 * Math.Sin(1.0);
        TestFixtures.AssertClose(expected, result, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_5d_additive_against_analytical()
    {
        // f(x) = sum_i sin(x_i) over [-1, 1]^5.
        // ∫ sin(x_i) dx_i = 0 contributes 0; sum is 0 across all 5 dims with
        // outside-volume = 2^4 each. Closed form: 5 * 0 = 0.
        static double F(double[] x, object? _)
        {
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += Math.Sin(x[i]);
            return s;
        }
        var domain = new[] {
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }
        };
        var slider = new ChebyshevSlider(
            F, 5, domain, new[] { 8, 8, 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 }, new[] { 3 }, new[] { 4 } },
            new[] { 0.0, 0.0, 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = (double)slider.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-6, atol: 1e-6);
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderFullIntegrate"
```

Expected: build error — `ChebyshevSlider does not contain a definition for 'Integrate'`.

- [ ] **Step 3: Implement Slider.Integrate (full path only; partial throws NotImplementedException)**

Insert into `src/ChebyshevSharp/ChebyshevSlider.cs` immediately after `ErrorEstimate()` (~line 340), before the `TotalBuildEvals` property:

```csharp
    // ------------------------------------------------------------------
    // Integration (Phase 5 — PyChebyshev v0.17)
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the slider approximation over one or more dimensions.
    /// Uses the closed-form decomposition of the sliding sum:
    ///   f(x) ≈ pv + Σ_i [s_i(x_{G_i}) - pv]
    /// Each slide's integral is computed via <see cref="ChebyshevApproximation.Integrate"/>.
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all (full integration → scalar).</param>
    /// <param name="bounds">Sub-interval bounds per dim (positional with sorted dims). Null = full domain.</param>
    /// <returns>A boxed <c>double</c> when every dim is integrated; otherwise a new <see cref="ChebyshevSlider"/> over surviving dims.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If <paramref name="dims"/> contains out-of-range or duplicated indices, or <paramref name="bounds"/> are invalid.</exception>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        if (!Built)
            throw new InvalidOperationException("Call Build() before Integrate().");

        // Normalize dims: null = all, sort + deduplicate, validate range.
        int[] sortedDims;
        if (dims == null)
            sortedDims = Enumerable.Range(0, NumDimensions).ToArray();
        else
            sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= NumDimensions)
                throw new ArgumentException(
                    $"dim {d} out-of-range [0, {NumDimensions - 1}]");
        }

        var perDimBounds = Internal.Calculus.NormalizeBounds(sortedDims, bounds, Domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        // Per-dim integration widths.
        var widths = new Dictionary<int, double>();
        var boundsForDim = new Dictionary<int, (double lo, double hi)?>();
        foreach (int d in sortedDims)
        {
            var bd = perDimBounds[dimToIdx[d]];
            double a = Domain[d][0], b = Domain[d][1];
            if (bd == null)
            {
                widths[d] = b - a;
                boundsForDim[d] = null;
            }
            else
            {
                widths[d] = bd.Value.hi - bd.Value.lo;
                boundsForDim[d] = bd;
            }
        }

        double volT = 1.0;
        foreach (int d in sortedDims) volT *= widths[d];

        // Per-slide classification.
        var slideKinds = new (string kind, int[] kept)[Partition.Length];
        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            slideKinds[slideIdx] = Internal.Calculus.SliderPartitionIntersect(
                Partition[slideIdx], sortedDims);
        }

        // pv_new accumulator: starts as pv * vol_T (the first term of the sum).
        double pvNew = PivotValue * volT;

        // For each "full" slide: integrate over its full group with the
        // appropriate sub-interval bounds, then add contribution to pv_new.
        // Contribution = vol(T \ G_i) * (I_i - pv * vol(G_i ∩ T))
        // For "full" slides, vol(G_i ∩ T) is the product of widths over G_i.
        var slideFullIntegrals = new Dictionary<int, double>();
        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            var (kind, _) = slideKinds[slideIdx];
            if (kind != "full") continue;

            var slide = Slides[slideIdx];
            var group = Partition[slideIdx];

            // Local-dim list (always all dims of the slide) with corresponding bounds.
            int[] localDims = Enumerable.Range(0, group.Length).ToArray();
            var localBoundsList = new List<(double lo, double hi)>(group.Length);
            bool allFullDomain = true;
            for (int gi = 0; gi < group.Length; gi++)
            {
                var bd = boundsForDim[group[gi]];
                if (bd == null)
                {
                    // Use full slide-domain for this local dim
                    localBoundsList.Add((slide.Domain[gi][0], slide.Domain[gi][1]));
                }
                else
                {
                    localBoundsList.Add(bd.Value);
                    allFullDomain = false;
                }
            }

            double Ii;
            if (allFullDomain)
                Ii = (double)slide.Integrate(dims: localDims);
            else
                Ii = (double)slide.Integrate(dims: localDims, bounds: localBoundsList.ToArray());

            slideFullIntegrals[slideIdx] = Ii;

            // vol(T \ G_i) — widths over dims in T but NOT in G_i.
            double volOutside = 1.0;
            var groupSet = new HashSet<int>(group);
            foreach (int d in sortedDims)
                if (!groupSet.Contains(d)) volOutside *= widths[d];

            // vol(G_i ∩ T) for "full" slides equals product of widths over G_i.
            double volGroup = 1.0;
            foreach (int d in group) volGroup *= widths[d];

            pvNew += volOutside * (Ii - PivotValue * volGroup);
        }

        // Full integration: every group classified "full", return scalar.
        if (sortedDims.Length == NumDimensions)
            return pvNew;

        // Partial integration is implemented in Task 3.
        throw new NotImplementedException(
            "ChebyshevSlider.Integrate partial integration is implemented in Phase 5 Task 3.");
    }
```

Note: `slide.Integrate(...)` returns `object`. For "full" slides we cast to `double` — this is safe because the full-slide local integration covers all of the slide's local dims.

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderFullIntegrate"
```

Expected: 4 tests passed.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
dotnet test
```

Expected: 911 tests passing (907 + 4). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSlider.cs tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs
git commit -m "phase5: ChebyshevSlider.Integrate — full integration

Public Integrate(int[]?, (double lo, double hi)[]?) returns scalar
(boxed in object) when every dim is integrated. Algorithm: closed-form
sliding-sum decomposition, ported from PyChebyshev slider.py:877-1015.
Each 'full'-classified slide is integrated via
ChebyshevApproximation.Integrate; contribution folds into pv_new.

Partial integration (returning a new ChebyshevSlider) deferred to Task 3
— currently throws NotImplementedException.

Test count: 907 -> 911 (+4)."
```

---

## Task 3: ChebyshevSlider.Integrate — partial integration

**Goal:** Replace the `NotImplementedException` from Task 2 with the partial-integration path. Constructs a new `ChebyshevSlider` over surviving dims via factory bypass, applying the unified rule for "none" and "partial" slide tensors. 7 partial-integration tests.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs` — replace the `NotImplementedException` block in `Integrate` with the partial-integration logic
- Modify: `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs` — append `TestSliderPartialIntegrate` class

**Python source:** `slider.py:1017-1132`

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 7 failing partial-integration tests**

Append to `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs` (after `TestSliderFullIntegrate`):

```csharp
// ======================================================================
// TestSliderPartialIntegrate (Phase 5)
// ======================================================================

public class TestSliderPartialIntegrate
{
    [Fact]
    public void Test_returns_slider_over_surviving_dims()
    {
        // f(x, y, z) = sin(x) + cos(y) + z; integrate dim 1 -> Slider over (0, 2)
        static double F(double[] x, object? _) => Math.Sin(x[0]) + Math.Cos(x[1]) + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var result = slider.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevSlider>(result);
        var resultSlider = (ChebyshevSlider)result;
        Assert.Equal(2, resultSlider.NumDimensions);
    }

    [Fact]
    public void Test_partial_disjoint_slide_passes_through()
    {
        // f(x, y) = sin(x) + y^2 over [-1,1]^2 with partition [[0], [1]].
        // Integrate dim 1 -> slide 0 (group [0]) is "none" (passes through),
        // slide 1 (group [1]) is "full". Expected eval at x=0: 2/3.
        static double F(double[] x, object? _) => Math.Sin(x[0]) + x[1] * x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        // ∫_{-1}^{1} (sin(x) + y^2) dy = 2 sin(x) + 2/3.
        double evalAtZero = resultSlider.Eval(new[] { 0.0 }, new[] { 0 });
        TestFixtures.AssertClose(2.0 / 3.0, evalAtZero, rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_full_partial_consistency()
    {
        // Joint integrate(dims=[0, 1, 2]) should equal
        // step1=integrate(dims=[0, 1]) then step2=integrate(dims=[0]).
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) + Math.Cos(x[1]) + x[2] * x[2];
        var sliderA = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sliderA.Build(verbose: false);
        double joint = (double)sliderA.Integrate(dims: new[] { 0, 1, 2 });

        // Independent slider for the chained path.
        var sliderB = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        sliderB.Build(verbose: false);
        var step1 = (ChebyshevSlider)sliderB.Integrate(dims: new[] { 0, 1 });
        // After integrating original dims 0 and 1, only original dim 2 remains
        // → 1D slider; integrating its dim 0 yields the joint integral.
        double step2 = (double)step1.Integrate(dims: new[] { 0 });

        TestFixtures.AssertClose(joint, step2, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_with_multi_dim_group()
    {
        // f(x, y) = sin(x) * cos(y); partition=[[0, 1]] (one 2D slide).
        // Integrate dim 0 -> "partial" classification on the only slide.
        // ∫_{-1}^{1} sin(x) cos(y) dx = cos(y) * 0 = 0 for all y.
        static double F(double[] x, object? _) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 },
            new[] { new[] { 0, 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal(1, resultSlider.NumDimensions);
        double evalAtHalf = resultSlider.Eval(new[] { 0.5 }, new[] { 0 });
        TestFixtures.AssertClose(0.0, evalAtHalf, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_with_3d_group_partial_integration()
    {
        // f(x, y, z) = sin(x) + cos(y) + z^2; partition=[[0, 1, 2]] (one 3D slide).
        // Integrate dim 1 -> "partial" classification: 3D slide reduces to 2D over (0, 2).
        // ∫_{-1}^{1} (sin(x) + cos(y) + z^2) dy = 2 sin(x) + 2 sin(1) + 2 z^2.
        // At x=0, z=0: 2 sin(1).
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) + Math.Cos(x[1]) + x[2] * x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1, 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        Assert.Equal(2, resultSlider.NumDimensions);
        double evalAtOrigin = resultSlider.Eval(new[] { 0.0, 0.0 }, new[] { 0, 0 });
        TestFixtures.AssertClose(2.0 * Math.Sin(1.0), evalAtOrigin, rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_partial_mixed_classifications()
    {
        // partition=[[0, 1], [2]]; integrate=[0]:
        //   slide 0 (group [0,1]) -> "partial", reduces to 1D over dim 1
        //   slide 1 (group [2])    -> "none", passes through
        // f(x, y, z) = sin(x) cos(y) + z.
        // ∫_{-1}^{1} (sin(x) cos(y) + z) dx = 0 + 2z. So result(y, z) ≈ 2z.
        static double F(double[] x, object? _) =>
            Math.Sin(x[0]) * Math.Cos(x[1]) + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            new[] { new[] { 0, 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal(2, resultSlider.NumDimensions);
        // At (y=0, z=0.5): expected 2 * 0.5 = 1.0
        double evalA = resultSlider.Eval(new[] { 0.0, 0.5 }, new[] { 0, 0 });
        TestFixtures.AssertClose(1.0, evalA, rtol: 1e-6, atol: 1e-6);
        // At (y=0.5, z=0.5): also expected 1.0 (independent of y)
        double evalB = resultSlider.Eval(new[] { 0.5, 0.5 }, new[] { 0, 0 });
        TestFixtures.AssertClose(1.0, evalB, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_partial_result_eval_works()
    {
        // Sanity: result of partial integrate is fully functional.
        static double F(double[] x, object? _) => x[0] + x[1] + x[2];
        var slider = new ChebyshevSlider(
            F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
            new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        var resultSlider = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
        // Eval should not throw; partition validity preserved.
        Assert.True(resultSlider.Built);
        Assert.Equal(2, resultSlider.NumDimensions);
        double v = resultSlider.Eval(new[] { 0.3, 0.7 }, new[] { 0, 0 });
        Assert.True(double.IsFinite(v));
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderPartialIntegrate"
```

Expected: 7 tests fail with `NotImplementedException` from Task 2's stub.

- [ ] **Step 3: Replace `NotImplementedException` with partial-integration logic**

In `src/ChebyshevSharp/ChebyshevSlider.cs`, replace the line

```csharp
        // Partial integration is implemented in Task 3.
        throw new NotImplementedException(
            "ChebyshevSlider.Integrate partial integration is implemented in Phase 5 Task 3.");
```

with the following block (closing the `Integrate` method body):

```csharp
        // Partial integration: build new slider over surviving dims.
        // Surviving global dim indices, sorted.
        int[] survive = Enumerable.Range(0, NumDimensions)
            .Where(d => !dimToIdx.ContainsKey(d))
            .ToArray();
        // global -> new index map
        var oldToNew = new Dictionary<int, int>();
        for (int newIdx = 0; newIdx < survive.Length; newIdx++)
            oldToNew[survive[newIdx]] = newIdx;

        var newPartition = new List<int[]>();
        var newSlides = new List<ChebyshevApproximation>();

        for (int slideIdx = 0; slideIdx < Partition.Length; slideIdx++)
        {
            var (kind, kept) = slideKinds[slideIdx];
            if (kind == "full") continue; // absorbed into pv_new

            var group = Partition[slideIdx];
            var slide = Slides[slideIdx];

            ChebyshevApproximation newSlide;
            int[] newGroup;

            if (kind == "none")
            {
                // The slide passes through. Apply the partition-of-unity shift:
                //   new_tensor = vol_T * tensor + (pv_new - pv * vol_T)
                double shift = pvNew - PivotValue * volT;
                var tv = slide.TensorValues!;
                var newTensor = new double[tv.Length];
                for (int j = 0; j < tv.Length; j++)
                    newTensor[j] = volT * tv[j] + shift;
                newSlide = ChebyshevApproximation.FromGrid(slide, newTensor);
                newGroup = group.Select(d => oldToNew[d]).ToArray();
            }
            else
            {
                // "partial": integrate the group's intersection with T.
                // Build local indices (within slide) for dims to integrate.
                var localDimsList = new List<int>();
                var localBoundsList = new List<(double lo, double hi)>();
                bool sawAnyExplicitBounds = false;
                for (int localI = 0; localI < group.Length; localI++)
                {
                    int gd = group[localI];
                    if (dimToIdx.ContainsKey(gd))
                    {
                        localDimsList.Add(localI);
                        var bd = boundsForDim[gd];
                        if (bd == null)
                        {
                            // Local-dim full domain.
                            localBoundsList.Add(
                                (slide.Domain[localI][0], slide.Domain[localI][1]));
                        }
                        else
                        {
                            localBoundsList.Add(bd.Value);
                            sawAnyExplicitBounds = true;
                        }
                    }
                }

                ChebyshevApproximation reduced;
                if (!sawAnyExplicitBounds)
                    reduced = (ChebyshevApproximation)slide.Integrate(
                        dims: localDimsList.ToArray());
                else
                    reduced = (ChebyshevApproximation)slide.Integrate(
                        dims: localDimsList.ToArray(),
                        bounds: localBoundsList.ToArray());

                // vol(T \ G_i) — widths over dims in T but NOT in this group.
                double volOutside = 1.0;
                var groupSet = new HashSet<int>(group);
                foreach (int d in sortedDims)
                    if (!groupSet.Contains(d)) volOutside *= widths[d];

                // Apply unified rule:
                //   new_tensor = vol_outside * reduced.tensor + (pv_new - pv * vol_T)
                double shift = pvNew - PivotValue * volT;
                var rtv = reduced.TensorValues!;
                var newTensor = new double[rtv.Length];
                for (int j = 0; j < rtv.Length; j++)
                    newTensor[j] = volOutside * rtv[j] + shift;
                newSlide = ChebyshevApproximation.FromGrid(reduced, newTensor);
                newGroup = kept.Select(d => oldToNew[d]).ToArray();
            }

            newPartition.Add(newGroup);
            newSlides.Add(newSlide);
        }

        // Reconstruct slider metadata for surviving dims.
        // The decomposition is now:
        //   g(y) = pv_new + Σ_j [tilde_s_j(y_{G'_j}) - pv_new]
        // We constructed each tilde_s_j so that its tensor satisfies:
        //   tilde_s_j(y) = scale * source(y) + (pv_new - pv * vol_T)
        // for "none" (scale = vol_T) and "partial" (scale = vol_outside) slides.
        // Subtracting pv_new from tilde_s_j gives scale * source(y) - pv * vol_T,
        // the required contribution of the slide.
        var newDomain = survive.Select(d => (double[])Domain[d].Clone()).ToArray();
        var newNNodes = survive.Select(d => NNodes[d]).ToArray();
        var newPivotPoint = survive.Select(d => PivotPoint[d]).ToArray();
        var newPartitionArr = newPartition.ToArray();

        var result = new ChebyshevSlider();
        result.Function = null;
        result.NumDimensions = survive.Length;
        result.Domain = newDomain;
        result.NNodes = newNNodes;
        result.MaxDerivativeOrder = MaxDerivativeOrder;
        result.Partition = newPartitionArr;
        result.PivotPoint = newPivotPoint;
        result.PivotValue = pvNew;
        result.Slides = newSlides.ToArray();
        result.DimToSlide = BuildDimToSlide(newPartitionArr);
        result.Built = true;
        result.BuildTime = 0.0;
        // Inherit Phase 4 ergonomics fields per spec D7 (descriptor + additionalData
        // pass through; derivative-id registry is intentionally NOT copied — see Task 4 test).
        SliderInheritErgonomics(result);
        return result;
    }

    /// <summary>
    /// Copy descriptor, additionalData, _maxDerivativeOrder, and _constructorType
    /// from this Slider to <paramref name="target"/>. The derivative-id registry is
    /// intentionally NOT copied — partial-integrate results have a different dim
    /// space (Python <c>slider.py:1130-1131</c>, spec D7).
    /// </summary>
    private void SliderInheritErgonomics(ChebyshevSlider target)
    {
        target.SetDescriptor(_descriptor!); // Public setter accepts null pass-through; if null this no-ops.
        target._additionalData = _additionalData;
        target._isConstructionFinished = true;
        target._constructorType = _constructorType;
    }
```

Note: `SetDescriptor(null)` would error if the API rejects null. Read the existing setter at `ChebyshevSlider.cs:925` to confirm the contract. If `SetDescriptor` requires a non-null string, fall back to direct field assignment:

```csharp
        target._descriptor = _descriptor;
```

(Replace the `target.SetDescriptor(_descriptor!);` line with this if needed.) Determine empirically by running the test for descriptor passthrough in Task 4.

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderPartialIntegrate"
```

Expected: 7 tests passed.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
dotnet test
```

Expected: 918 tests passing (911 + 7). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSlider.cs tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs
git commit -m "phase5: ChebyshevSlider.Integrate — partial integration

Implements the partial-integration path. Constructs a new ChebyshevSlider
over surviving dims via factory bypass (parameterless ctor). Per-slide
'none'/'partial' classifications apply the unified rule
new_tensor = scale * source + (pv_new - pv * vol_T) where scale is vol_T
for 'none' and vol_outside for 'partial'. Inherits descriptor and
additionalData; derivative-id registry intentionally reset (D7).

Ported from PyChebyshev slider.py:1017-1132. The unified-rule math is
documented inline matching the Python comment block at slider.py:1085-1097.

Test count: 911 -> 918 (+7)."
```

---

## Task 4: ChebyshevSlider.Integrate — validation + ergonomics

**Goal:** Add 6 tests covering validation paths (out-of-range dim, negative dim, bounds outside domain, unbuilt-raise) and ergonomics passthrough (descriptor + additionalData) plus the registry-reset assertion (D7).

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs` — append `TestSliderIntegrateValidation` and `TestSliderIntegrateErgonomics` classes
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs` (only if a test surfaces missing validation — Task 2 already validates dim range; bounds validation is delegated to `Calculus.NormalizeBounds`)

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 6 failing validation + ergonomics tests**

Append to `tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs`:

```csharp
// ======================================================================
// TestSliderIntegrateValidation (Phase 5)
// ======================================================================

public class TestSliderIntegrateValidation
{
    private static ChebyshevSlider Make1D()
    {
        static double F(double[] x, object? _) => x[0];
        var slider = new ChebyshevSlider(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 },
            new[] { new[] { 0 } }, new[] { 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    [Fact]
    public void Test_unbuilt_slider_raises()
    {
        static double F(double[] x, object? _) => x[0];
        var slider = new ChebyshevSlider(
            F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 },
            new[] { new[] { 0 } }, new[] { 0.0 });
        // Don't build.
        Assert.Throws<InvalidOperationException>(() => slider.Integrate());
    }

    [Fact]
    public void Test_dims_out_of_range_raises()
    {
        var slider = Make1D();
        var ex = Assert.Throws<ArgumentException>(() => slider.Integrate(dims: new[] { 5 }));
        Assert.Contains("out-of-range", ex.Message);
    }

    [Fact]
    public void Test_negative_dim_raises()
    {
        var slider = Make1D();
        Assert.Throws<ArgumentException>(() => slider.Integrate(dims: new[] { -1 }));
    }

    [Fact]
    public void Test_bounds_outside_domain_raises()
    {
        var slider = Make1D();
        Assert.Throws<ArgumentException>(() =>
            slider.Integrate(
                dims: new[] { 0 },
                bounds: new[] { (-2.0, 2.0) }));
    }
}

// ======================================================================
// TestSliderIntegrateErgonomics (Phase 5)
// ======================================================================

public class TestSliderIntegrateErgonomics
{
    [Fact]
    public void Test_descriptor_preserved_on_partial_result()
    {
        static double F(double[] x, object? _) => x[0] + x[1];
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        slider.SetDescriptor("source");
        var result = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Equal("source", result.GetDescriptor());
    }

    [Fact]
    public void Test_additional_data_preserved_on_partial_result()
    {
        var sentinel = new Dictionary<string, int> { ["k"] = 42 };
        double F(double[] x, object? data)
        {
            Assert.NotNull(data);
            return x[0] + x[1];
        }
        var slider = new ChebyshevSlider(
            F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 0.0, 0.0 },
            additionalData: sentinel);
        slider.Build(verbose: false);
        var result = (ChebyshevSlider)slider.Integrate(dims: new[] { 0 });
        Assert.Same(sentinel, result.GetAdditionalData());
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderIntegrateValidation|FullyQualifiedName~TestSliderIntegrateErgonomics"
```

Expected: most validation tests already pass (Task 2 validation logic + `NormalizeBounds`); descriptor + additionalData tests should pass on the first run because Task 3 already implemented inheritance. If `Test_descriptor_preserved_on_partial_result` fails because `SetDescriptor(null)` throws, switch to the direct-field-assignment fallback noted in Task 3 Step 3.

- [ ] **Step 3: Fix any failures (likely none, but possibly the `SetDescriptor(null)` fallback)**

If `Test_descriptor_preserved_on_partial_result` reveals `SetDescriptor` rejects null, modify `SliderInheritErgonomics` in `ChebyshevSlider.cs`:

```csharp
        // BEFORE: target.SetDescriptor(_descriptor!);
        // AFTER:
        target._descriptor = _descriptor;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSliderIntegrateValidation|FullyQualifiedName~TestSliderIntegrateErgonomics"
```

Expected: 6 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 924 tests passing (918 + 6). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add tests/ChebyshevSharp.Tests/SliderIntegrateTests.cs src/ChebyshevSharp/ChebyshevSlider.cs
git commit -m "phase5: ChebyshevSlider.Integrate — validation + ergonomics tests

6 tests covering: unbuilt-raise, dims out-of-range, negative dim,
bounds outside domain, descriptor passthrough on partial result,
additionalData passthrough on partial result.

Test count: 918 -> 924 (+6)."
```

---

## Task 5: ChebyshevTT.Integrate — full integration

**Goal:** Add `public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` to `ChebyshevTT`. For full integration, computes per-dim Fejér-1 weights (or sub-interval weights), converts each integrated coefficient core to value space, contracts via `IntegrateTtAlongDim`, then chain-multiplies the resulting matrices to a scalar. Defer partial-result construction to Task 6 (throw `NotImplementedException` for now). 6 full-integration tests.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — add public Integrate method (insert after `ErrorEstimate()` ~line 590, before `OrthLeft`)
- Create: `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs`

**Python source:** `tensor_train.py:1487-1580` (full integration path; partial path lines 1582-1635 deferred to Task 6)

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 6 failing full-integration tests**

Create `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestTtFullIntegrate (Phase 5)
// ======================================================================

public class TestTtFullIntegrate
{
    [Fact]
    public void Test_separable_function_sin_times_cos()
    {
        // f(x, y) = sin(x) * cos(y) over [-1, 1]^2.
        // ∫∫ = (∫sin) (∫cos) = 0 * (2 sin 1) = 0.
        static double F(double[] x) => Math.Sin(x[0]) * Math.Cos(x[1]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_constant_function_volume()
    {
        // f = 7 over [0, 2] x [0, 3] integrates to 7 * 6 = 42.
        static double F(double[] x) => 7.0;
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 3.0 } },
            new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(42.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_5d_separable_against_analytical()
    {
        // f(x) = exp(-Σ x_i^2) over [-1, 1]^5.
        // ∫_{-1}^{1} exp(-x^2) dx = sqrt(pi) * erf(1) ≈ 1.49364826562485.
        // Total = (sqrt(pi) * erf(1))^5.
        static double F(double[] x)
        {
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += x[i] * x[i];
            return Math.Exp(-s);
        }
        var domain = new[] {
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 },
            new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }
        };
        var tt = new ChebyshevTT(F, 5, domain, new[] { 10, 10, 10, 10, 10 });
        tt.Build(verbose: false, seed: 42);
        double cheb = (double)tt.Integrate();
        // sqrt(pi) * erf(1)
        double oneD = Math.Sqrt(Math.PI) * Erf(1.0);
        double expected = Math.Pow(oneD, 5);
        TestFixtures.AssertClose(expected, cheb, rtol: 1e-4, atol: 1e-4);
    }

    [Fact]
    public void Test_works_after_method_svd()
    {
        // f(x, y) = x * y; ∫∫ = 0 over [-1, 1]^2.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, method: "svd");
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_works_after_method_als()
    {
        // f(x, y) = x*y + sin(x) over [-1, 1]^2.
        // ∫ x*y dx dy = 0; ∫ sin(x) dx dy = 0 (sin odd, then * 2). Total ≈ 0.
        static double F(double[] x) => x[0] * x[1] + Math.Sin(x[0]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        tt.Build(verbose: false, method: "als", seed: 42);
        var result = (double)tt.Integrate();
        TestFixtures.AssertClose(0.0, result, rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_dims_order_invariance()
    {
        // integrate(dims=[0, 1]) == integrate(dims=[1, 0]) (full integration).
        static double F(double[] x) => Math.Sin(x[0]) + Math.Cos(x[1]);
        var ttA = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        ttA.Build(verbose: false, seed: 42);
        double a = (double)ttA.Integrate(dims: new[] { 0, 1 });

        var ttB = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        ttB.Build(verbose: false, seed: 42);
        double b = (double)ttB.Integrate(dims: new[] { 1, 0 });
        TestFixtures.AssertClose(a, b, rtol: 1e-10, atol: 1e-10);
    }

    // Abramowitz & Stegun 7.1.26 erf approximation (sufficient for 1e-4 tolerance).
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        double absX = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * absX);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
            + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
            * Math.Exp(-absX * absX);
        return sign * y;
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestTtFullIntegrate"
```

Expected: build error — `ChebyshevTT does not contain a definition for 'Integrate'`.

- [ ] **Step 3: Implement TT.Integrate (full path; partial throws NotImplementedException)**

Insert into `src/ChebyshevSharp/ChebyshevTT.cs` immediately after `ErrorEstimate()` (~line 590), before `OrthLeft`:

```csharp
    // ------------------------------------------------------------------
    // Integration (Phase 5 — PyChebyshev v0.17)
    // ------------------------------------------------------------------

    /// <summary>
    /// Integrate the TT-approximated function over selected dimensions.
    /// Per-dim Fejér-1 quadrature is applied to the value-space cores
    /// (Chebyshev coefficient cores are converted to value cores via
    /// <see cref="TensorTrainKernel.CoeffCoreToValueCore"/> before contraction).
    /// </summary>
    /// <param name="dims">Dimensions to integrate out. Null = all (full integration → scalar).</param>
    /// <param name="bounds">Sub-interval bounds per dim (positional with sorted dims). Null = full domain.</param>
    /// <returns>A boxed <c>double</c> when every dim is integrated; otherwise a new <see cref="ChebyshevTT"/> over surviving dims.</returns>
    /// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
    /// <exception cref="ArgumentException">If <paramref name="dims"/> contains out-of-range or duplicated indices, or <paramref name="bounds"/> are invalid.</exception>
    public object Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)
    {
        CheckBuilt();

        // Normalize dims: null = all, sort + deduplicate, validate range.
        int[] sortedDims;
        if (dims == null)
            sortedDims = Enumerable.Range(0, _numDimensions).ToArray();
        else
            sortedDims = dims.Distinct().OrderBy(d => d).ToArray();

        foreach (int d in sortedDims)
        {
            if (d < 0 || d >= _numDimensions)
                throw new ArgumentException(
                    $"dim {d} out-of-range [0, {_numDimensions - 1}]");
        }

        var perDimBounds = Internal.Calculus.NormalizeBounds(sortedDims, bounds, _domain);
        var dimToIdx = new Dictionary<int, int>();
        for (int i = 0; i < sortedDims.Length; i++)
            dimToIdx[sortedDims[i]] = i;

        // Compute scaled quadrature weights per integrated dim.
        // Cores live in coefficient space — convert each integrated core to
        // value space before applying weights.
        var weightsPerDim = new Dictionary<int, double[]>();
        foreach (int d in sortedDims)
        {
            int n = _nNodes[d];
            double a = _domain[d][0], b = _domain[d][1];
            double scale = (b - a) / 2.0;
            var bd = perDimBounds[dimToIdx[d]];
            double[] w;
            if (bd == null)
            {
                w = Internal.Calculus.ComputeFejer1Weights(n);
            }
            else
            {
                double tLo = 2.0 * (bd.Value.lo - a) / (b - a) - 1.0;
                double tHi = 2.0 * (bd.Value.hi - a) / (b - a) - 1.0;
                w = Internal.Calculus.ComputeSubIntervalWeights(n, tLo, tHi);
            }
            for (int i = 0; i < w.Length; i++) w[i] *= scale;
            weightsPerDim[d] = w;
        }

        // Per-integrated-dim contraction: coefficient core -> value core -> M_k.
        var contracted = new Dictionary<int, double[,]>();
        foreach (int d in sortedDims)
        {
            var valueCore = Internal.TensorTrainKernel.CoeffCoreToValueCore(_coeffCores![d]);
            contracted[d] = Internal.Calculus.IntegrateTtAlongDim(valueCore, weightsPerDim[d]);
        }

        if (sortedDims.Length == _numDimensions)
        {
            // Full integration: chain-multiply all M_k matrices left-to-right.
            // contracted[sortedDims[0]] is shape (rL_0=1, rR_0); after all multiplications,
            // result is (1, 1).
            double[,] result = contracted[sortedDims[0]];
            for (int i = 1; i < sortedDims.Length; i++)
                result = MatMul(result, contracted[sortedDims[i]]);
            return result[0, 0];
        }

        // Partial integration: implemented in Task 6.
        throw new NotImplementedException(
            "ChebyshevTT.Integrate partial integration is implemented in Phase 5 Task 6.");
    }

    /// <summary>
    /// Plain (m, k) x (k, n) -> (m, n) matrix multiply for the Integrate path.
    /// </summary>
    private static double[,] MatMul(double[,] a, double[,] b)
    {
        int m = a.GetLength(0);
        int k = a.GetLength(1);
        int kB = b.GetLength(0);
        int n = b.GetLength(1);
        if (k != kB)
            throw new ArgumentException(
                $"MatMul shape mismatch: ({m}, {k}) x ({kB}, {n})");
        var result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int l = 0; l < k; l++) s += a[i, l] * b[l, j];
                result[i, j] = s;
            }
        return result;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestTtFullIntegrate"
```

Expected: 6 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 930 tests passing (924 + 6). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtIntegrateTests.cs
git commit -m "phase5: ChebyshevTT.Integrate — full integration

Public Integrate(int[]?, (double lo, double hi)[]?) returns scalar
(boxed in object) when every dim is integrated. Per-dim Fejér-1 weights
(or sub-interval weights) scaled by (b-a)/2; coefficient cores converted
to value cores via CoeffCoreToValueCore before contraction (D8); chain-
multiply the (rLeft, rRight) matrices. Works for cross/svd/als builds.

Partial integration deferred to Task 6 — currently throws NotImplementedException.

Ported from PyChebyshev tensor_train.py:1487-1580.

Test count: 924 -> 930 (+6)."
```

---

## Task 6: ChebyshevTT.Integrate — partial integration

**Goal:** Replace the `NotImplementedException` from Task 5 with the partial-integration path. Walk the TT chain, absorbing each contracted (rLeft, rRight) matrix into the next kept core's left rank, with trailing-pending matrix absorbed into the last kept core's right rank. Construct result TT inheriting all build params (D6) plus descriptor + additionalData. 8 partial-integration tests.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` — replace `NotImplementedException` with partial logic; add private `BuildIntegrateResult` helper that mirrors `BuildResultFromCores` but inherits Phase 4 ergonomics fields
- Modify: `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs` — append `TestTtPartialIntegrate` class

**Python source:** `tensor_train.py:1582-1635`

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 8 failing partial-integration tests**

Append to `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs`:

```csharp
// ======================================================================
// TestTtPartialIntegrate (Phase 5)
// ======================================================================

public class TestTtPartialIntegrate
{
    [Fact]
    public void Test_returns_tt_with_correct_shape()
    {
        static double F(double[] x) => x[0] + x[1] + x[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 });
        tt.Build(verbose: false, seed: 42);
        var result = tt.Integrate(dims: new[] { 1 });
        Assert.IsType<ChebyshevTT>(result);
        var resultTt = (ChebyshevTT)result;
        Assert.Equal(2, resultTt.NumDimensions);
        Assert.Equal(new[] { 6, 6 }, resultTt.NNodes);
    }

    [Fact]
    public void Test_endpoint_dim_left()
    {
        // Integrate over dim 0 (no left neighbor). f(x, y) = x * y.
        // ∫_{-1}^{1} x dx = 0, so result(y) ≈ 0 for all y.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 6 });
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_endpoint_dim_right()
    {
        // Integrate over last dim (no right neighbor). f(x, y) = x * y.
        // ∫_{-1}^{1} y dy = 0, so result(x) ≈ 0.
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 1 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_partial_consistent_with_consecutive()
    {
        // integrate([0, 1]) on a 3D TT should equal integrate([1]).integrate([0]).
        // Note: after integrate([1]) returns a 2D TT over (orig_dim_0, orig_dim_2),
        // its "dim 0" is original dim 0 — so chained integrate([0]) is correct.
        static double F(double[] x) => Math.Sin(x[0]) * Math.Cos(x[1]) * (1 + x[2] * x[2]);
        var ttA = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 });
        ttA.Build(verbose: false, seed: 42);
        var ttB = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10, 10 });
        ttB.Build(verbose: false, seed: 42);

        var joint = (ChebyshevTT)ttA.Integrate(dims: new[] { 0, 1 });
        var step1 = (ChebyshevTT)ttB.Integrate(dims: new[] { 1 });
        var step2 = (ChebyshevTT)step1.Integrate(dims: new[] { 0 });

        double xTest = 0.3;
        TestFixtures.AssertClose(
            joint.Eval(new[] { xTest }),
            step2.Eval(new[] { xTest }),
            rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_with_sub_interval_bounds()
    {
        // ∫_0^1 x dx = 0.5 over [-1, 1].
        static double F(double[] x) => x[0];
        var tt = new ChebyshevTT(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false, seed: 42);
        var result = (double)tt.Integrate(
            dims: new[] { 0 },
            bounds: new[] { (0.0, 1.0) });
        TestFixtures.AssertClose(0.5, result, rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_descriptor_preserved_on_partial_result()
    {
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        tt.SetDescriptor("source");
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Equal("source", result.GetDescriptor());
    }

    [Fact]
    public void Test_additional_data_preserved_on_partial_result()
    {
        var sentinel = new Dictionary<string, int> { ["k"] = 42 };
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 },
            additionalData: sentinel);
        tt.Build(verbose: false, seed: 42);
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Same(sentinel, result.GetAdditionalData());
    }

    [Fact]
    public void Test_partial_eval_works_recursively()
    {
        // Result of partial integrate must be a fully-functional TT:
        // Eval, EvalBatch, and recursive Integrate all work.
        static double F(double[] x) => x[0] * x[1] + x[0];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, seed: 42);
        // ∫_{-1}^{1} (x*y + x) dx = 0 + 0 = 0 → result(y) = 0 for all y.
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        TestFixtures.AssertClose(0.0, result.Eval(new[] { 0.5 }), rtol: 1e-10, atol: 1e-10);
        // Recursive integrate over the surviving 1D TT yields ∫_{-1}^{1} 0 dy = 0.
        double recursiveScalar = (double)result.Integrate();
        TestFixtures.AssertClose(0.0, recursiveScalar, rtol: 1e-10, atol: 1e-10);
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestTtPartialIntegrate"
```

Expected: 8 tests fail with `NotImplementedException` from Task 5.

- [ ] **Step 3: Replace `NotImplementedException` with partial-integration logic**

In `src/ChebyshevSharp/ChebyshevTT.cs`, replace the line

```csharp
        // Partial integration: implemented in Task 6.
        throw new NotImplementedException(
            "ChebyshevTT.Integrate partial integration is implemented in Phase 5 Task 6.");
```

with the following block:

```csharp
        // Partial integration: walk the TT chain, absorbing each contracted
        // matrix into a neighboring kept core's left rank dim (Python
        // tensor_train.py:1582-1608).
        var integratedSet = new HashSet<int>(sortedDims);
        var newCores = new List<Internal.TensorTrainKernel.TtCore>();
        double[,]? pending = null;

        for (int k = 0; k < _numDimensions; k++)
        {
            if (integratedSet.Contains(k))
            {
                var M = contracted[k];
                if (pending != null) M = MatMul(pending, M);
                pending = M;
                continue;
            }
            // k is a kept dim — absorb any pending matrix into this core's left rank.
            var core = _coeffCores![k].Copy();
            if (pending != null)
            {
                core = AbsorbLeft(pending, core);
                pending = null;
            }
            newCores.Add(core);
        }

        // Trailing pending: absorb into the last kept core's right rank.
        if (pending != null && newCores.Count > 0)
            newCores[newCores.Count - 1] = AbsorbRight(newCores[newCores.Count - 1], pending);

        // Construct result TT.
        int[] keptDims = Enumerable.Range(0, _numDimensions)
            .Where(d => !integratedSet.Contains(d))
            .ToArray();
        var newDomain = keptDims.Select(d => (double[])_domain[d].Clone()).ToArray();
        var newNNodes = keptDims.Select(d => _nNodes[d]).ToArray();

        return BuildIntegrateResult(newCores.ToArray(), newDomain, newNNodes);
    }

    /// <summary>
    /// Matrix-times-core contraction along the core's left rank dim:
    /// <c>result[l, j, s] = Σ_r M[l, r] * core[r, j, s]</c>.
    /// Used by partial Integrate to absorb a pending matrix into the next kept core.
    /// </summary>
    private static Internal.TensorTrainKernel.TtCore AbsorbLeft(
        double[,] M, Internal.TensorTrainKernel.TtCore core)
    {
        int newRLeft = M.GetLength(0);
        int absorbed = M.GetLength(1);
        if (absorbed != core.RLeft)
            throw new ArgumentException(
                $"AbsorbLeft shape mismatch: M is ({newRLeft}, {absorbed}); core.RLeft={core.RLeft}");
        int n = core.NNodes, rR = core.RRight;
        var result = new Internal.TensorTrainKernel.TtCore(newRLeft, n, rR);
        for (int l = 0; l < newRLeft; l++)
            for (int j = 0; j < n; j++)
                for (int s = 0; s < rR; s++)
                {
                    double acc = 0;
                    for (int r = 0; r < absorbed; r++)
                        acc += M[l, r] * core[r, j, s];
                    result[l, j, s] = acc;
                }
        return result;
    }

    /// <summary>
    /// Core-times-matrix contraction along the core's right rank dim:
    /// <c>result[l, j, r] = Σ_s core[l, j, s] * M[s, r]</c>.
    /// Used by partial Integrate to absorb a trailing pending matrix into the last kept core.
    /// </summary>
    private static Internal.TensorTrainKernel.TtCore AbsorbRight(
        Internal.TensorTrainKernel.TtCore core, double[,] M)
    {
        int absorbed = M.GetLength(0);
        int newRRight = M.GetLength(1);
        if (absorbed != core.RRight)
            throw new ArgumentException(
                $"AbsorbRight shape mismatch: core.RRight={core.RRight}; M is ({absorbed}, {newRRight})");
        int rL = core.RLeft, n = core.NNodes;
        var result = new Internal.TensorTrainKernel.TtCore(rL, n, newRRight);
        for (int l = 0; l < rL; l++)
            for (int j = 0; j < n; j++)
                for (int r = 0; r < newRRight; r++)
                {
                    double acc = 0;
                    for (int s = 0; s < absorbed; s++)
                        acc += core[l, j, s] * M[s, r];
                    result[l, j, r] = acc;
                }
        return result;
    }

    /// <summary>
    /// Construct a partial-integrate result TT, inheriting all Phase 4 ergonomics
    /// fields (descriptor, additionalData, maxDerivativeOrder) and Method (D3, D6).
    /// Mirrors <see cref="BuildResultFromCores"/> with extra inheritance.
    /// </summary>
    private ChebyshevTT BuildIntegrateResult(
        Internal.TensorTrainKernel.TtCore[] cores, double[][] newDomain, int[] newNNodes)
    {
        int newD = newNNodes.Length;
        var ttRanks = new int[newD + 1];
        ttRanks[0] = 1;
        for (int i = 0; i < newD; i++) ttRanks[i + 1] = cores[i].RRight;
        // Use the existing BuildResultFromCores then patch in ergonomics fields.
        var result = BuildResultFromCores(cores, newDomain, newNNodes);
        // Phase 4 ergonomics passthrough (D6).
        result._descriptor = _descriptor;
        result._additionalData = _additionalData;
        result._maxDerivativeOrder = _maxDerivativeOrder;
        return result;
    }
```

Note: `BuildResultFromCores` (existing) already sets `result.Method = Method`, so `GetConstructorType()` on the result returns the original's build method per spec D3.

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestTtPartialIntegrate"
```

Expected: 8 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 938 tests passing (930 + 8). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtIntegrateTests.cs
git commit -m "phase5: ChebyshevTT.Integrate — partial integration

Walks the TT chain, absorbing each contracted (rLeft, rRight) matrix into
a neighboring kept core via AbsorbLeft / AbsorbRight einsums (Python
tensor_train.py:1582-1608). Result TT constructed via BuildResultFromCores
+ Phase 4 ergonomics passthrough (descriptor, additionalData,
maxDerivativeOrder). Method is inherited by BuildResultFromCores already.

Test count: 930 -> 938 (+8)."
```

---

## Task 7: ChebyshevTT.Integrate — validation + cross-class consistency

**Goal:** Add 8 tests covering validation paths (out-of-range dim, bounds outside domain, bounds length mismatch, unbuilt-raise) and cross-class consistency (build-mode preserved on partial result for cross/svd/als per D3, plus a unit-volume normalization across all four classes).

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs` — append `TestTtIntegrateValidation` and `TestTtIntegrateCrossClass` classes

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Write 8 failing validation + cross-class tests**

Append to `tests/ChebyshevSharp.Tests/TtIntegrateTests.cs`:

```csharp
// ======================================================================
// TestTtIntegrateValidation (Phase 5)
// ======================================================================

public class TestTtIntegrateValidation
{
    private static ChebyshevTT Make2D()
    {
        static double F(double[] x) => x[0];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Test_unbuilt_tt_raises()
    {
        static double F(double[] x) => x[0];
        var tt = new ChebyshevTT(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        // Don't build.
        Assert.Throws<InvalidOperationException>(() => tt.Integrate());
    }

    [Fact]
    public void Test_dims_out_of_range_raises()
    {
        var tt = Make2D();
        var ex = Assert.Throws<ArgumentException>(() => tt.Integrate(dims: new[] { 5 }));
        Assert.Contains("out-of-range", ex.Message);
    }

    [Fact]
    public void Test_bounds_outside_domain_raises()
    {
        static double F(double[] x) => x[0];
        var tt = new ChebyshevTT(F, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        tt.Build(verbose: false, seed: 42);
        Assert.Throws<ArgumentException>(() =>
            tt.Integrate(
                dims: new[] { 0 },
                bounds: new[] { (-2.0, 2.0) }));
    }

    [Fact]
    public void Test_bounds_length_mismatch_raises()
    {
        var tt = Make2D();
        Assert.Throws<ArgumentException>(() =>
            tt.Integrate(
                dims: new[] { 0 },
                bounds: new[] { (0.0, 1.0), (0.0, 1.0) })); // 2 bounds, 1 dim
    }
}

// ======================================================================
// TestTtIntegrateCrossClass (Phase 5)
// ======================================================================

public class TestTtIntegrateCrossClass
{
    [Fact]
    public void Test_build_mode_preserved_cross()
    {
        // Partial-integrate result should preserve build mode (D3).
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, method: "cross", seed: 42);
        Assert.Equal("cross", tt.GetConstructorType());
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Equal("cross", result.GetConstructorType());
    }

    [Fact]
    public void Test_build_mode_preserved_svd()
    {
        static double F(double[] x) => x[0] * x[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 });
        tt.Build(verbose: false, method: "svd");
        Assert.Equal("svd", tt.GetConstructorType());
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Equal("svd", result.GetConstructorType());
    }

    [Fact]
    public void Test_build_mode_preserved_als()
    {
        static double F(double[] x) => x[0] * x[1] + Math.Sin(x[0]);
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        tt.Build(verbose: false, method: "als", seed: 42);
        Assert.Equal("als", tt.GetConstructorType());
        var result = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });
        Assert.Equal("als", result.GetConstructorType());
    }

    [Fact]
    public void Test_unit_volume_across_all_four_classes()
    {
        // Constant 1 over [0, 2] x [0, 3] integrates to 6 on every class.
        const double expected = 6.0;
        var domain = new[] { new[] { 0.0, 2.0 }, new[] { 0.0, 3.0 } };

        // ChebyshevApproximation
        var cheb = new ChebyshevApproximation(
            (x, _) => 1.0, 2, domain, new[] { 4, 4 });
        cheb.Build(verbose: false);
        TestFixtures.AssertClose(expected, (double)cheb.Integrate(),
            rtol: 1e-10, atol: 1e-10);

        // ChebyshevSpline (no knots — single piece behaves like Approximation)
        var spline = new ChebyshevSpline(
            (x, _) => 1.0, 2, domain, new[] { 4, 4 },
            new[] { Array.Empty<double>(), Array.Empty<double>() });
        spline.Build(verbose: false);
        TestFixtures.AssertClose(expected, (double)spline.Integrate(),
            rtol: 1e-10, atol: 1e-10);

        // ChebyshevSlider
        var slider = new ChebyshevSlider(
            (x, _) => 1.0, 2, domain, new[] { 4, 4 },
            new[] { new[] { 0 }, new[] { 1 } },
            new[] { 1.0, 1.5 });
        slider.Build(verbose: false);
        TestFixtures.AssertClose(expected, (double)slider.Integrate(),
            rtol: 1e-10, atol: 1e-10);

        // ChebyshevTT
        var tt = new ChebyshevTT(x => 1.0, 2, domain, new[] { 4, 4 });
        tt.Build(verbose: false, seed: 42);
        TestFixtures.AssertClose(expected, (double)tt.Integrate(),
            rtol: 1e-10, atol: 1e-10);
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestTtIntegrateValidation|FullyQualifiedName~TestTtIntegrateCrossClass"
```

Expected: most tests should pass on the first run because Task 5/6 logic already covers validation and inheritance. If `Test_build_mode_preserved_*` fails because `BuildResultFromCores` doesn't set `Method`, verify the existing implementation at `ChebyshevTT.cs:937` does set it.

- [ ] **Step 3: Fix any failures (likely none)**

If the build-mode-preserved tests fail because `BuildIntegrateResult` doesn't propagate `Method`, look at `BuildResultFromCores` (the existing helper Phase 5 reuses). It should already set `tt.Method = Method`. If somehow not, add `result.Method = Method;` to `BuildIntegrateResult`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestTtIntegrateValidation|FullyQualifiedName~TestTtIntegrateCrossClass"
```

Expected: 8 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 946 tests passing (938 + 8). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add tests/ChebyshevSharp.Tests/TtIntegrateTests.cs src/ChebyshevSharp/ChebyshevTT.cs
git commit -m "phase5: ChebyshevTT.Integrate — validation + cross-class consistency

8 tests covering: unbuilt-raise, dims out-of-range, bounds outside
domain, bounds length mismatch, build-mode preservation on partial
result for cross/svd/als (D3), unit-volume normalization across
all four interpolant classes.

Test count: 938 -> 946 (+8)."
```

---

## Task 8: Docs + parity metadata + release prep

**Goal:** Bump csproj version + parity tag, update changelog with v0.9.0 entry, extend docs/calculus.md with Slider/TT integration sections, update skip_csharp.txt and CLAUDE.md status block. No new tests.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj`
- Modify: `docs/docs/changelog.md`
- Modify: `docs/docs/calculus.md`
- Modify: `skip_csharp.txt`
- Modify: `CLAUDE.md`

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase5-integrate-everywhere`.

- [ ] **Step 1: Bump csproj version + parity tag**

In `src/ChebyshevSharp/ChebyshevSharp.csproj`, change:

```xml
<Version>0.8.0</Version>
<PyChebyshevParity>0.18.0</PyChebyshevParity>
<InformationalVersion>0.8.0+pychebyshev.0.18.0</InformationalVersion>
```

to:

```xml
<Version>0.9.0</Version>
<PyChebyshevParity>0.17.0</PyChebyshevParity>
<InformationalVersion>0.9.0+pychebyshev.0.17.0</InformationalVersion>
```

Note: parity tag drops 0.18.0 → 0.17.0 per spec D4 (non-monotonic batch tracker).

- [ ] **Step 2: Add v0.9.0 changelog entry**

Add to the top of `docs/docs/changelog.md` (after any leading header):

```markdown
## [0.9.0] - 2026-04-28

### Added — Integrate Everywhere

- `ChebyshevSlider.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial integration via the closed-form sliding-decomposition. Returns a scalar (boxed in `object`) on full integration; a `ChebyshevSlider` over surviving dims on partial integration.
- `ChebyshevTT.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` — full + partial integration via Fejér-1 quadrature contraction into each integrated core's node axis (after coefficient→value core conversion). Works for all three TT build methods (`cross`, `svd`, `als`).
- New internal helpers in `Internal/Calculus.cs`: `SliderPartitionIntersect`, `IntegrateTtAlongDim`.

After v0.9.0, all four ChebyshevSharp classes support integration — matching PyChebyshev v0.17.0 parity. Roots / Min / Max on Slider and TT remain deferred (Python defers to v0.21).

### PyChebyshev parity tracking

- `<PyChebyshevParity>` tag drops 0.18.0 → 0.17.0. This looks like a regression but is not — the parity tag is a non-monotonic indicator of "the most recent feature batch we ported." Phase 4 (v0.8.0) filled in v0.15+v0.16 features behind the v0.18.0 binary format that was already shipped, so the tag stayed at 0.18.0. Phase 5 (this release) ports v0.17.0 calculus completion features; the tag drops to 0.17.0 to indicate which batch was just delivered. Phase 6 will advance to v0.20.1.
- The `<Version>` (release-engineering version) advances monotonically: 0.8.0 → 0.9.0.

### Test count: 902 → 946 (+44)

Phase 5 fan-out: 17 tests in `SliderIntegrateTests.cs` (full + partial + validation + ergonomics), 22 tests in `TtIntegrateTests.cs` (full + partial + validation + cross-class build-mode preservation), 5 helper tests appended to `CalculusTests.cs`.

See [PR #20](https://github.com/0xC000005/ChebyshevSharp/pulls) for the full diff and the [design spec](https://github.com/0xC000005/ChebyshevSharp/blob/main/docs/superpowers/specs/2026-04-28-phase5-integrate-everywhere-design.md).
```

- [ ] **Step 3: Extend docs/docs/calculus.md with Slider/TT integration sections**

Append to `docs/docs/calculus.md` (after any existing Approximation/Spline integration sections):

```markdown
## Slider Integration (v0.9.0)

`ChebyshevSlider.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` integrates over one or more dimensions using the closed-form sliding-decomposition. Returns a scalar (boxed in `object`) when every dim is integrated; otherwise returns a new `ChebyshevSlider` over surviving dims.

```csharp
var slider = new ChebyshevSlider(
    (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 10, 10 },
    partition: new[] { new[] { 0 }, new[] { 1 } },
    pivotPoint: new[] { 0.0, 0.0 });
slider.Build();

// Full integration: ∫∫ (sin(x) + cos(y)) dx dy = 4 sin(1)
double result = (double)slider.Integrate();

// Partial integration: ∫_{-1}^{1} (sin(x) + cos(y)) dy → slider over dim 0 only
var partial = (ChebyshevSlider)slider.Integrate(dims: new[] { 1 });
```

The integration is exact for the spectrally-resolved part of each slide. Per-slide
classification: a slide whose group is fully covered by `dims` collapses into the
new pivot value; a slide whose group is partially covered is reduced via
`ChebyshevApproximation.Integrate`; a slide whose group is disjoint from `dims`
passes through with a partition-of-unity shift.

## TT Integration (v0.9.0)

`ChebyshevTT.Integrate(int[]? dims = null, (double lo, double hi)[]? bounds = null)` integrates over one or more dimensions using Fejér-1 quadrature contracted into each integrated core's node axis. Returns a scalar (boxed in `object`) when every dim is integrated; otherwise returns a new `ChebyshevTT` over surviving dims. Works for all three build methods (`cross`, `svd`, `als`).

```csharp
var tt = new ChebyshevTT(
    x => Math.Sin(x[0]) * Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 12, 12 });
tt.Build();

// Full integration
double total = (double)tt.Integrate();

// Partial: integrate over dim 0, returns a 1D TT in y
var marginal = (ChebyshevTT)tt.Integrate(dims: new[] { 0 });

// Sub-domain bounds
double partial = (double)tt.Integrate(
    dims: new[] { 0, 1 },
    bounds: new[] { (-0.5, 0.5), (0.0, 1.0) });
```

Note that `Roots`, `Minimize`, and `Maximize` are not yet available on `ChebyshevSlider` or `ChebyshevTT` — they remain deferred to a future phase, matching PyChebyshev's v0.21 deferral.
```

- [ ] **Step 4: Update skip_csharp.txt**

Add an entry to `skip_csharp.txt` indicating Phase 5 Python tests are now ported. Append:

```
# Phase 5 (v0.9.0) — Integrate everywhere (PyChebyshev v0.17.0 parity)
# All 44 tests in test_calculus_completion.py ported as:
#   - tests/SliderIntegrateTests.cs (17 tests covering full + partial + validation + ergonomics)
#   - tests/TtIntegrateTests.cs (22 tests covering full + partial + validation + cross-class)
#   - 5 helper tests appended to tests/CalculusTests.cs (SliderPartitionIntersect, IntegrateTtAlongDim)
# No tests skipped.
```

- [ ] **Step 5: Update CLAUDE.md Status block**

In `CLAUDE.md`, update the **Status** section. Change:

```
**Feature-complete against PyChebyshev v0.18.0** (Phases 1+2+3+4 of the 6-phase v0.20.1 port complete; see
`docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md`).
All four public classes (`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`) mirror the Python API surface. v0.8.0 adds the v0.15+v0.16 ergonomics
layer (descriptor, additionalData, derivative-id registry, introspection getters,
typed Clone, DeferBuild + SetOriginalFunctionValues, Domain/Ns/SpecialPoints records;
PyChebyshev parity tag unchanged at v0.18.0).
`dotnet test` runs **884/884** passing.
```

(Or whatever the actual current text is — adjust the test count line to match what's in the file.) Update to:

```
**Feature-complete against PyChebyshev v0.18.0** (Phases 1+2+3+4+5 of the 6-phase v0.20.1 port complete; see
`docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md`).
All four public classes (`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`) mirror the Python API surface. v0.8.0 added the v0.15+v0.16 ergonomics
layer; v0.9.0 (Phase 5) adds Slider/TT `Integrate` completing calculus parity across
all four interpolant classes (PyChebyshev parity tag drops 0.18.0 → 0.17.0 — non-monotonic
batch tracker indicating the most recent feature-batch ported).
`dotnet test` runs **946/946** passing.
```

If the actual `CLAUDE.md` text differs from the snippet above, update only the test count and append the v0.9.0 sentence to the existing Status block.

- [ ] **Step 6: Verify build is clean and tests pass**

```bash
dotnet build && dotnet test
```

Expected: 946 tests passed, build with 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSharp.csproj docs/docs/changelog.md docs/docs/calculus.md skip_csharp.txt CLAUDE.md
git commit -m "phase5: docs, csproj, parity metadata, release prep for v0.9.0

- csproj: <Version> 0.8.0 -> 0.9.0; <PyChebyshevParity> 0.18.0 -> 0.17.0 (D4);
  <InformationalVersion> 0.9.0+pychebyshev.0.17.0
- changelog: v0.9.0 entry following two-tier convention; explicit note on
  the parity-tag non-monotonic semantics
- docs/calculus.md: Slider Integration and TT Integration sections with examples
- skip_csharp.txt: Phase 5 entry marking 44 tests ported, none skipped
- CLAUDE.md: status block bumped (884 -> 946; phases 1-5 of 6 complete)
- No submodule bump (ref/PyChebyshev stays at v0.18.0)
- No new tests"
```

- [ ] **Step 8: Final verification**

```bash
git log --oneline | head -10
dotnet test
```

Expected: 8 commits prefixed `phase5:` since baseline. 946 tests passing.

Return control to the user. Do not auto-create the PR or auto-merge — that's a user-confirmation gate (per Phases 3 and 4 pattern).

---

## Self-Review Checklist (for the writing-plans author, NOT the implementer)

After writing this plan, the writing-plans skill self-review confirmed:

**Spec coverage:**
- D1 (single signature, `object` return type) → Tasks 2, 3, 5, 6 implement the signature on Slider and TT.
- D2 (port all 44 tests) → Tests laid out across 8 task buckets summing to 44.
- D3 (build mode preserved on partial result) → Task 7 has 3 explicit cross/svd/als preservation tests.
- D4 (parity tag drops 0.18.0 → 0.17.0) → Task 8 csproj edit + changelog explanation.
- D5 (internal helpers `internal` not `private`) → Task 1 helpers declared `internal static`.
- D6 (TT result inherits all build params) → Task 6 `BuildIntegrateResult` propagates `_descriptor`, `_additionalData`, `_maxDerivativeOrder`; underlying `BuildResultFromCores` already handles `Method`, `_maxRank`, `_tolerance`, `_maxSweeps` per the existing helper. Task 6 Step 6 commit message lists which fields are inherited.
- D7 (Slider result resets registry) → Task 3 `SliderInheritErgonomics` intentionally does NOT copy registry; Task 4 has explicit ergonomics tests though I reframed registry-reset as the absence of registry-related state in tests rather than an explicit "registry reset" test (the registry is private and an explicit reset test would require exposing it via a debug accessor — I rely on the absence of registry copying in `SliderInheritErgonomics` plus the ergonomics tests covering descriptor + additionalData passthrough). The reviewer should confirm this is sufficient or add an internal accessor + test if not.
- D8 (TT cores converted coeff→value before quadrature) → Task 5 explicitly calls `CoeffCoreToValueCore` before `IntegrateTtAlongDim`.
- D9 (`(double lo, double hi)[]?` bounds) → Used uniformly across all task signatures.
- D10 (single PR) → Task structure assumes single PR; Task 8 is release prep.

**Placeholder scan:** No "TBD", "implement later", "similar to Task N", or unspecified code blocks. Every test has full code; every implementation step has full code; the only conditional fix-on-fail is the `SetDescriptor(null)` fallback in Task 3 Step 3 (with explicit replacement code given).

**Type consistency:**
- `(string kind, int[] kept)` tuple type used consistently in `Calculus.SliderPartitionIntersect` definition (Task 1) and call site (Task 2 Step 3).
- `Internal.TensorTrainKernel.TtCore` used consistently in `IntegrateTtAlongDim` (Task 1), `AbsorbLeft`/`AbsorbRight` (Task 6).
- `(double lo, double hi)[]?` bounds parameter type consistent across all 4 method signatures and tests.
- Field names `_descriptor`, `_additionalData`, `_maxDerivativeOrder`, `_constructorType`, `Method` on Slider and TT match the existing Phase 4 declarations.

**Worktree enforcement:** Every task starts with the same WORKTREE ENFORCEMENT block.

**Test count progression:** Task headers and final summary table align: 902 + 5 + 4 + 7 + 6 + 6 + 8 + 8 + 0 = 946. ✓
