# Phase 6 Implementation Plan — Build Perf + Adaptive Refinement (v0.10.0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle PyChebyshev v0.19.0 + v0.20.0 + v0.20.1 into one cohesive ChebyshevSharp v0.10.0 release: build perf (`nWorkers`, `IProgress<int>`) on all four ctors; adaptive refinement (`AutoKnots`, `SobolIndices` returning a new `SobolResult` record, `WithAutoOrder`, `Reorder`, `DimOrder` on TT); full `_dimOrder` threading through every public TT method; JSON v1→v2 migration. This is the final phase of the v0.20.1 port.

**Architecture:** Two new internal helper files (`Internal/ParallelBuild.cs`, `Internal/Sensitivity.cs`) plus an extension to `Internal/TensorTrainAlgebra.cs` (the `TtSwapAdjacent` helper). One new public record (`SobolResult`). Ctor kwargs are appended at the end of each existing signature so v0.9.0 callers compile unchanged. TT gains a private `_dimOrder` field initialized to identity in entry-point ctors and factories; factory bypasses derive `_dimOrder` from source per the threading rules in §6.5 of the spec.

**Tech Stack:** C# 12, .NET 8 + .NET 10 multi-target, xUnit, MathNet.Numerics (already a dependency).

**Spec:** `docs/superpowers/specs/2026-04-29-phase6-perf-and-adaptive-design.md` (commit `9f9e4aa`, 606 lines, 10 design decisions D1–D10).

**Test count progression:**

| After task | Total tests | Δ |
|---|---|---|
| Baseline (Phase 5 complete, on main at `9f9e4aa`) | 950 | — |
| Task 1 (submodule bump + scaffolding) | 950 | 0 |
| Task 2 (ParallelBuild helper) | 953 | +3 |
| Task 3 (Approx ctor wiring) | 958 | +5 |
| Task 4 (Spline+Slider+TT ctor wiring) | 964 | +6 |
| Task 5 (nWorkers validation edge cases) | 967 | +3 |
| Task 6 (Sensitivity + Approx.SobolIndices) | 975 | +8 |
| Task 7 (Spline.SobolIndices aggregation) | 979 | +4 |
| Task 8 (Spline.AutoKnots) | 989 | +10 |
| Task 9 (TtSwapAdjacent + Reorder + DimOrder) | 997 | +8 |
| Task 10 (WithAutoOrder + JSON migration) | 1007 | +10 |
| Task 11 (_dimOrder threading across TT public methods) | 1017 | +10 |
| Task 12 (release prep — no new tests) | 1017 | 0 |

±2 drift per task is acceptable (consistent with Phases 4 and 5); larger drift requires investigation before proceeding.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `src/ChebyshevSharp/SobolResult.cs` | New public record `SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance)`. |
| `src/ChebyshevSharp/Internal/ParallelBuild.cs` | `NormalizeNWorkers(int? nWorkers)`, `EvaluateInParallel(...)` static helpers. |
| `src/ChebyshevSharp/Internal/Sensitivity.cs` | `ChebyshevCoefficientsND`, `ComputeSobolFromCoeffs` static helpers. |
| `tests/ChebyshevSharp.Tests/BuildPerfTests.cs` | 17 build-perf tests (parallel parity, progress counters, validation). |
| `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs` | 12 Sobol tests (Approx + Spline). |
| `tests/ChebyshevSharp.Tests/AutoKnotsTests.cs` | 10 AutoKnots tests. |
| `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs` | 10 WithAutoOrder + Reorder + DimOrder + JSON migration tests. |
| `tests/ChebyshevSharp.Tests/TtDimOrderTests.cs` | 10 _dimOrder threading tests across the TT public surface. |
| `tests/ChebyshevSharp.Tests/fixtures/v0.9.0_sin3d_tt.json` | Pre-Phase-6 TT save committed for v1→v2 migration test. |

### Modified files

| Path | What changes |
|---|---|
| `src/ChebyshevSharp/ChebyshevApproximation.cs` | Both ctors gain `nWorkers` + `progress` kwargs; `BuildFixedGrid` routes through `ParallelBuild.EvaluateInParallel`. New instance method `SobolIndices()`. |
| `src/ChebyshevSharp/ChebyshevSpline.cs` | All three ctors gain `nWorkers` + `progress` kwargs (passed to per-piece Approx ctors). New instance method `SobolIndices()`. New static factory `AutoKnots(...)`. |
| `src/ChebyshevSharp/ChebyshevSlider.cs` | Ctor gains `nWorkers` + `progress` kwargs (passed to per-slide Approx ctors). |
| `src/ChebyshevSharp/ChebyshevTT.cs` | Ctor gains `nWorkers` + `progress` kwargs (nWorkers ignored by design; progress fires per-sweep). Private field `_dimOrder` initialized to identity in two ctors. Public read-only `DimOrder` property. New static factory `WithAutoOrder(...)`. New instance method `Reorder(...)`. `_dimOrder` threaded through `Eval`, `EvalBatch`, `EvalMulti`, `Slice`, `Extrude`, `ToDense`, partial `Integrate`, unary algebra, binary algebra. JSON Save bumps to `"jsonVersion": 2` writing `"dimOrder"`; JSON Load backfills identity for v1 files. |
| `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` | Add `internal static TtCore[] TtSwapAdjacent(TtCore[] cores, int i, int maxRank, double tolerance = 1e-12)`. |
| `src/ChebyshevSharp/ChebyshevSharp.csproj` | Bump `<Version>` to 0.10.0; bump `<PyChebyshevParity>` to 0.20.1; update `<InformationalVersion>` to `0.10.0+pychebyshev.0.20.1`. |
| `ref/PyChebyshev` (submodule) | Bump v0.18.0 → v0.20.1 (15 commits). |
| `docs/docs/changelog.md` | v0.10.0 entry, two-tier convention. |
| `docs/docs/parallel-build.md` | New page documenting `nWorkers` + `IProgress<int>` with thread-safety contract examples. |
| `docs/docs/adaptive-refinement.md` | New page documenting `AutoKnots` + `SobolIndices` + `WithAutoOrder` with worked examples. |
| `docs/docs/toc.yml` | Add the two new pages. |
| `skip_csharp.txt` | Append Phase 6 entry — full v0.19+v0.20.0+v0.20.1 ported (modulo matplotlib plotting, Option C). |
| `CLAUDE.md` | Status block: 950 → 1017 tests; phases 1-6 of 6 complete (port complete); parity tag 0.18.0 → 0.20.1. |

### Files NOT changed

- Existing tests pass unchanged (identity `_dimOrder` is the default; new ctor kwargs are at the end of signatures with default values, so v0.9.0 call sites bind unchanged).
- `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` — no new fixtures needed; per-test inline construction covers all cases. (We do add helper assertions for permutation array equality in `TtDimOrderTests.cs` but they live in that test file, not in TestFixtures.)
- The Phase 5 `Integrate` API surface is unchanged; partial integrate gains `_dimOrder` threading internally in Task 11.

---

## Task 1: Submodule bump + scaffolding

**Goal:** Bump `ref/PyChebyshev` from v0.18.0 to v0.20.1; create empty stub files for `SobolResult`, `Internal/ParallelBuild`, `Internal/Sensitivity` so subsequent tasks have somewhere to land code; verify the bump introduces no unexpected upstream changes outside the v0.19/v0.20.0/v0.20.1 windows. No new tests in this task — `dotnet test` continues to report 950/950.

**Files:**
- Submodule: bump `ref/PyChebyshev` to v0.20.1
- Create: `src/ChebyshevSharp/SobolResult.cs` (placeholder record; populated in Task 6)
- Create: `src/ChebyshevSharp/Internal/ParallelBuild.cs` (empty internal static class shell; populated in Task 2)
- Create: `src/ChebyshevSharp/Internal/Sensitivity.cs` (empty internal static class shell; populated in Task 6)

**Python source:** N/A (this task only does scaffolding + submodule bookkeeping). The 15 commits across v0.18.0 → v0.20.1 are reviewed in Step 2 below.

### WORKTREE ENFORCEMENT (MANDATORY)

Before running any other commands:

```bash
git rev-parse --show-toplevel
```

Expected output ends in `.worktrees/phase6-perf-and-adaptive`. If it ends in `/home/max/Documents/ChebyshevSharp` (the main repo), **STOP** and `cd` to the worktree. Phase 1 Task 4 cross-directory commit is the cautionary tale.

- [ ] **Step 1: Bump submodule v0.18.0 → v0.20.1**

```bash
cd ref/PyChebyshev
git fetch --tags
git checkout v0.20.1
cd ../..
git add ref/PyChebyshev
```

Then verify:

```bash
cd ref/PyChebyshev && git describe --tags HEAD && cd ../..
```

Expected: `v0.20.1`.

- [ ] **Step 2: Diff-review the 15 intermediate commits**

```bash
cd ref/PyChebyshev
git log v0.18.0..v0.20.1 --oneline
```

Expected: 15 commits. Each commit's subject line should map cleanly to one of the v0.19, v0.20.0, or v0.20.1 changelog entries (build perf, plot helpers, sobol indices, with_auto_order, dim_order threading, _tt_swap_adjacent). If any commit subject does NOT correspond to one of these scopes (e.g. an unrelated bug fix or refactor), capture the subject and SHA in the next commit's message body so the spec-compliance reviewer can flag it.

```bash
cd ../..
```

- [ ] **Step 3: Create stub files**

`src/ChebyshevSharp/SobolResult.cs`:

```csharp
namespace ChebyshevSharp;

/// <summary>
/// Result of <see cref="ChebyshevApproximation.SobolIndices"/> and
/// <see cref="ChebyshevSpline.SobolIndices"/>: per-dimension Sobol sensitivity
/// indices computed from spectral Chebyshev coefficients (no Monte Carlo).
/// </summary>
/// <param name="FirstOrder">First-order index per dimension. Sums to ≤ 1; sums to 1 for purely additive functions.</param>
/// <param name="TotalOrder">Total-order index per dimension. <c>FirstOrder[d] ≤ TotalOrder[d]</c> always.</param>
/// <param name="Variance">
/// Total spectral variance Σ_{α≠0} c_α² ‖T_α‖². When zero, the function is
/// constant and the indices are meaningless — callers should branch on this.
/// </param>
public sealed record SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance);
```

`src/ChebyshevSharp/Internal/ParallelBuild.cs`:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Build-time helpers for parallel function evaluation and progress reporting.
/// Phase 6: <c>nWorkers</c> ctor kwarg + <see cref="IProgress{T}"/> wiring.
/// Mirrors PyChebyshev <c>_parallel.py</c> (v0.19.0).
/// </summary>
internal static class ParallelBuild
{
    // Implementations land in Task 2.
}
```

`src/ChebyshevSharp/Internal/Sensitivity.cs`:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Sobol sensitivity indices computed from Chebyshev spectral coefficients.
/// Mirrors PyChebyshev <c>_sensitivity.py</c> (v0.20.0).
/// </summary>
internal static class Sensitivity
{
    // Implementations land in Task 6.
}
```

- [ ] **Step 4: Verify build is clean and tests still pass**

```bash
dotnet build && dotnet test
```

Expected: 0 warnings, 950 tests passed.

- [ ] **Step 5: Commit**

```bash
git add ref/PyChebyshev src/ChebyshevSharp/SobolResult.cs src/ChebyshevSharp/Internal/ParallelBuild.cs src/ChebyshevSharp/Internal/Sensitivity.cs
git commit -m "phase6: bump submodule to v0.20.1; add stub files for Phase 6

- ref/PyChebyshev: v0.18.0 -> v0.20.1 (15 commits across v0.19, v0.20.0, v0.20.1)
- new SobolResult.cs (public record); populated in Task 6
- new Internal/ParallelBuild.cs (stub); populated in Task 2
- new Internal/Sensitivity.cs (stub); populated in Task 6
- no behavior change; tests still 950/950"
```

---

## Task 2: Internal/ParallelBuild.cs + helper unit tests

**Goal:** Implement `NormalizeNWorkers(int? nWorkers) → int?` and `EvaluateInParallel(...)` in `Internal/ParallelBuild.cs`. Add 3 unit tests in a new `BuildPerfTests.cs` test file.

**Files:**
- Modify: `src/ChebyshevSharp/Internal/ParallelBuild.cs` (replace stub body with implementations)
- Create: `tests/ChebyshevSharp.Tests/BuildPerfTests.cs`

**Python source:** `ref/PyChebyshev/src/pychebyshev/_parallel.py:1-78` — `_normalize_n_workers` (lines 8-32), `_evaluate_in_parallel` (lines 35-65), `_Worker` (lines 67-78). The C# port replaces process-pool with `Parallel.For` and surfaces a thread-safety contract instead of a picklability contract (D1).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write failing tests in `BuildPerfTests.cs`**

Create `tests/ChebyshevSharp.Tests/BuildPerfTests.cs`:

```csharp
using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestParallelBuildHelpers (Phase 6 Task 2)
// ======================================================================

public class TestParallelBuildHelpers
{
    [Fact]
    public void Test_normalize_zero_throws()
    {
        Assert.Throws<ArgumentException>(() => ParallelBuild.NormalizeNWorkers(0));
    }

    [Fact]
    public void Test_normalize_minus_one_resolves_to_processor_count()
    {
        int? resolved = ParallelBuild.NormalizeNWorkers(-1);
        Assert.Equal(Environment.ProcessorCount, resolved);
    }

    [Fact]
    public void Test_evaluate_in_parallel_matches_sequential()
    {
        // f(point, _) = point[0]^2 + point[1]^2 — pure, thread-safe.
        static double F(double[] p, object? _) => p[0] * p[0] + p[1] * p[1];

        var points = new double[][]
        {
            new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
            new[] { 1.0, 1.0 }, new[] { 2.0, 3.0 }, new[] { -1.0, 4.0 },
            new[] { 0.5, 0.5 }, new[] { 1.5, 2.5 },
        };

        double[] sequential = ParallelBuild.EvaluateInParallel(F, points, null, null, null);
        double[] parallel = ParallelBuild.EvaluateInParallel(F, points, null, 4, null);

        Assert.Equal(points.Length, parallel.Length);
        for (int i = 0; i < points.Length; i++)
            TestFixtures.AssertClose(sequential[i], parallel[i], rtol: 0, atol: 0);  // bit-exact for pure F
    }
}
```

- [ ] **Step 2: Run failing tests to verify the helpers don't yet exist**

```bash
dotnet test --filter "FullyQualifiedName~TestParallelBuildHelpers"
```

Expected: build error — `'ParallelBuild' does not contain a definition for 'NormalizeNWorkers'` and `'ParallelBuild' does not contain a definition for 'EvaluateInParallel'`.

- [ ] **Step 3: Implement helpers in `Internal/ParallelBuild.cs`**

Replace the stub body of `src/ChebyshevSharp/Internal/ParallelBuild.cs` with:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Build-time helpers for parallel function evaluation and progress reporting.
/// Phase 6: nWorkers ctor kwarg + IProgress&lt;int&gt; wiring.
/// Mirrors PyChebyshev _parallel.py (v0.19.0).
/// </summary>
internal static class ParallelBuild
{
    /// <summary>
    /// Validate and normalize the user-supplied <c>nWorkers</c> ctor kwarg.
    /// </summary>
    /// <param name="nWorkers">Raw user value: null (sequential), -1 (auto), or positive int.</param>
    /// <returns>null for sequential; positive int for the effective worker count.</returns>
    /// <exception cref="ArgumentException">On 0 or value &lt; -1.</exception>
    internal static int? NormalizeNWorkers(int? nWorkers)
    {
        if (nWorkers is null) return null;
        if (nWorkers == 0 || nWorkers < -1)
            throw new ArgumentException(
                $"nWorkers={nWorkers} not allowed (use null for sequential, " +
                "-1 for ProcessorCount, or a positive int).",
                nameof(nWorkers));
        return nWorkers == -1 ? Environment.ProcessorCount : nWorkers;
    }

    /// <summary>
    /// Evaluate <paramref name="function"/> at every <paramref name="points"/> entry,
    /// optionally in parallel via <see cref="System.Threading.Tasks.Parallel.For"/>,
    /// optionally reporting cumulative count to <paramref name="progress"/> after each
    /// successful evaluation.
    /// </summary>
    /// <param name="function">Picklable-equivalent in Python; here, must be thread-safe when <paramref name="effectiveWorkers"/> is non-null.</param>
    /// <param name="points">Flat array of input points.</param>
    /// <param name="additionalData">User context threaded as the second arg of <paramref name="function"/>.</param>
    /// <param name="effectiveWorkers">Already normalized via <see cref="NormalizeNWorkers"/>; null = sequential.</param>
    /// <param name="progress">Optional progress reporter; receives cumulative count 1..N.</param>
    /// <returns>Result array of length <c>points.Length</c>.</returns>
    internal static double[] EvaluateInParallel(
        Func<double[], object?, double> function,
        double[][] points,
        object? additionalData,
        int? effectiveWorkers,
        IProgress<int>? progress)
    {
        var results = new double[points.Length];
        if (effectiveWorkers is null or 1)
        {
            for (int i = 0; i < points.Length; i++)
            {
                results[i] = function(points[i], additionalData);
                progress?.Report(i + 1);
            }
            return results;
        }

        int done = 0;
        var opts = new System.Threading.Tasks.ParallelOptions
        {
            MaxDegreeOfParallelism = effectiveWorkers.Value,
        };
        System.Threading.Tasks.Parallel.For(0, points.Length, opts, i =>
        {
            results[i] = function(points[i], additionalData);
            int n = System.Threading.Interlocked.Increment(ref done);
            progress?.Report(n);
        });
        return results;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestParallelBuildHelpers"
```

Expected: 3 tests passed.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
dotnet test
```

Expected: 953 tests passing (950 baseline + 3 new). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/Internal/ParallelBuild.cs tests/ChebyshevSharp.Tests/BuildPerfTests.cs
git commit -m "phase6: add ParallelBuild helpers (NormalizeNWorkers, EvaluateInParallel)

Two internal static helpers in Internal/ParallelBuild.cs ported from
PyChebyshev _parallel.py:1-78. C# uses Parallel.For over thread pool
(true parallel via no GIL); contract shifts from picklable to thread-
safe (D1).

3 unit tests in new tests/BuildPerfTests.cs.

Test count: 950 -> 953 (+3)."
```

---

## Task 3: ChebyshevApproximation ctor wiring + 5 tests

**Goal:** Add `int? nWorkers = null` and `IProgress<int>? progress = null` kwargs at the end of both `ChebyshevApproximation` ctor signatures. Refactor `BuildFixedGrid` to use `ParallelBuild.EvaluateInParallel`. Document the thread-safety contract in XML `<remarks>`. 5 tests in `BuildPerfTests.cs`.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs:87-187` (both ctors), `:215-272` (`BuildFixedGrid`)
- Modify: `tests/ChebyshevSharp.Tests/BuildPerfTests.cs` (append new test class)

**Python source:** `ref/PyChebyshev/src/pychebyshev/barycentric.py:284-380` (ctor + n_workers wiring), `:660-690` (build path that dispatches to `_evaluate_in_parallel`).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 5 failing tests**

Append to `tests/ChebyshevSharp.Tests/BuildPerfTests.cs`:

```csharp
// ======================================================================
// TestApproxBuildPerf (Phase 6 Task 3)
// ======================================================================

public class TestApproxBuildPerf
{
    private static double F(double[] p, object? _) => Math.Sin(p[0]) * Math.Cos(p[1]);

    [Fact]
    public void Test_parallel_build_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 16, 16 };

        var seqApprox = new ChebyshevApproximation(F, 2, domain, nNodes);
        seqApprox.Build(verbose: false);

        var parApprox = new ChebyshevApproximation(F, 2, domain, nNodes, nWorkers: 4);
        parApprox.Build(verbose: false);

        Assert.Equal(seqApprox.TensorValues!.Length, parApprox.TensorValues!.Length);
        for (int i = 0; i < seqApprox.TensorValues.Length; i++)
            TestFixtures.AssertClose(seqApprox.TensorValues[i], parApprox.TensorValues[i],
                rtol: 0, atol: 0);  // bit-exact: pure deterministic F, identical points order
    }

    [Fact]
    public void Test_progress_count_matches_grid_size()
    {
        var counter = new ProgressCounter();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8 };

        var approx = new ChebyshevApproximation(F, 2, domain, nNodes, progress: counter);
        approx.Build(verbose: false);

        Assert.Equal(64, counter.LastValue);
        Assert.Equal(64, counter.CallCount);
    }

    [Fact]
    public void Test_progress_null_no_op()
    {
        var domain = new[] { new[] { -1.0, 1.0 } };
        var approx = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 }, progress: null);
        approx.Build(verbose: false);
        Assert.NotNull(approx.TensorValues);
    }

    [Fact]
    public void Test_nworkers_null_is_sequential()
    {
        var counter = new ProgressCounter();
        var domain = new[] { new[] { -1.0, 1.0 } };
        var approx = new ChebyshevApproximation(F, 1, domain, new[] { 5 },
            nWorkers: null, progress: counter);
        approx.Build(verbose: false);

        // Sequential path increments 1, 2, 3, 4, 5 in order — no race.
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, counter.AllValues);
    }

    [Fact]
    public void Test_thread_safety_smoke_with_lock_wrapped_counter()
    {
        // Function captures shared mutable state via lock — acts as a smoke check
        // that with a thread-safe wrapper, the parallel path produces identical
        // values to sequential.
        int sharedCounter = 0;
        object lockObj = new();
        double F2(double[] p, object? _)
        {
            lock (lockObj) { sharedCounter++; }
            return p[0] * p[0];
        }

        var domain = new[] { new[] { 0.0, 1.0 } };
        var approx = new ChebyshevApproximation(F2, 1, domain, new[] { 10 }, nWorkers: 4);
        approx.Build(verbose: false);
        Assert.Equal(10, sharedCounter);  // Each grid point evaluated exactly once.
    }

    /// <summary>Records every reported progress value for assertion.</summary>
    private sealed class ProgressCounter : IProgress<int>
    {
        private readonly object _lock = new();
        private readonly List<int> _values = new();
        public int LastValue { get; private set; }
        public int CallCount { get; private set; }
        public int[] AllValues
        {
            get { lock (_lock) return _values.ToArray(); }
        }

        public void Report(int value)
        {
            lock (_lock)
            {
                _values.Add(value);
                LastValue = value;
                CallCount++;
            }
        }
    }
}
```

- [ ] **Step 2: Run failing tests to verify the ctor kwargs don't yet exist**

```bash
dotnet test --filter "FullyQualifiedName~TestApproxBuildPerf"
```

Expected: build error — `ChebyshevApproximation(...)` does not have a `nWorkers:` named argument.

- [ ] **Step 3: Add kwargs to both ctors**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, find the first ctor at line 87 (the fixed-N ctor) and append the two kwargs:

```csharp
public ChebyshevApproximation(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxDerivativeOrder = 2,
    object? additionalData = null,
    bool deferBuild = false,
    int? nWorkers = null,
    IProgress<int>? progress = null)
{
    Function = function;
    NumDimensions = numDimensions;
    Domain = domain.Select(d => (double[])d.Clone()).ToArray();
    NNodes = (int[])nNodes.Clone();
    MaxDerivativeOrder = maxDerivativeOrder;
    _additionalData = additionalData;
    _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
    _progress = progress;

    if (!deferBuild)
    {
        NodeArrays = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
        {
            NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
        }
    }
    else
    {
        NodeArrays = Array.Empty<double[]>();
    }
}
```

Find the second ctor at line 132 (the auto-N / errorThreshold ctor) and append the same two kwargs at the end of its signature, plus assign them at the end of the body:

```csharp
public ChebyshevApproximation(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int?[]? nNodes = null,
    double? errorThreshold = null,
    int maxN = 64,
    int maxDerivativeOrder = 2,
    object? additionalData = null,
    int? nWorkers = null,
    IProgress<int>? progress = null)
{
    // ... existing body unchanged through the NNodes/NodeArrays assignment ...
    // Add these two lines just before the if/else on resolved.All(n => n != null):
    _nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
    _progress = progress;

    // ... rest of body unchanged ...
}
```

Add the two private fields near the top of the class (e.g. just below the existing `_additionalData` field):

```csharp
private int? _nWorkers;
private IProgress<int>? _progress;
```

Add an XML `<remarks>` block to BOTH ctors documenting the thread-safety contract:

```csharp
/// <remarks>
/// When <paramref name="nWorkers"/> is non-null, <paramref name="function"/> may be
/// invoked concurrently from multiple threads via <c>Parallel.For</c>. Functions that
/// capture mutable state must use locks or external synchronization, or pass
/// <c>nWorkers: null</c> (the default).
/// </remarks>
```

- [ ] **Step 4: Refactor `BuildFixedGrid` to route through ParallelBuild**

Replace the body of `BuildFixedGrid` in `src/ChebyshevSharp/ChebyshevApproximation.cs:215-272` with:

```csharp
internal void BuildFixedGrid(bool verbose = true)
{
    int total = 1;
    for (int d = 0; d < NumDimensions; d++)
        total *= NNodes[d];

    if (verbose)
        Console.WriteLine($"Building {NumDimensions}D Chebyshev approximation ({total:N0} evaluations)...");

    var sw = Stopwatch.StartNew();
    _cachedErrorEstimate = null;

    // Step 1: Materialize the full points array (C-order / ndindex), then
    // evaluate sequentially or in parallel via ParallelBuild.
    var points = new double[total][];
    int[] indices = new int[NumDimensions];
    for (int flat = 0; flat < total; flat++)
    {
        int rem = flat;
        for (int d = NumDimensions - 1; d >= 0; d--)
        {
            indices[d] = rem % NNodes[d];
            rem /= NNodes[d];
        }
        var pt = new double[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
            pt[d] = NodeArrays[d][indices[d]];
        points[flat] = pt;
    }
    TensorValues = Internal.ParallelBuild.EvaluateInParallel(
        Function!, points, _additionalData, _nWorkers, _progress);
    NEvaluations = total;

    // Step 2: Pre-compute barycentric weights
    Weights = new double[NumDimensions][];
    for (int d = 0; d < NumDimensions; d++)
        Weights[d] = BarycentricKernel.ComputeBarycentricWeights(NodeArrays[d]);

    // Step 3: Pre-compute differentiation matrices
    DiffMatrices = new double[NumDimensions][,];
    for (int d = 0; d < NumDimensions; d++)
        DiffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(NodeArrays[d], Weights[d]);

    // Step 4: Pre-transpose diff matrices for VectorizedEval
    PrecomputeTransposedDiffMatrices();

    sw.Stop();
    BuildTime = sw.Elapsed.TotalSeconds;

    if (verbose)
    {
        int totalWeights = Weights.Sum(w => w.Length);
        Console.WriteLine($"  Built in {BuildTime:F3}s ({totalWeights} weights, {totalWeights * 8} bytes)");
    }

    _isConstructionFinished = true;
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestApproxBuildPerf"
```

Expected: 5 tests passed.

- [ ] **Step 6: Run full suite to verify no regressions**

```bash
dotnet test
```

Expected: 958 tests passing (953 + 5). 0 failures, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/BuildPerfTests.cs
git commit -m "phase6: ChebyshevApproximation gains nWorkers + IProgress<int> kwargs

- Both ctors append nWorkers (int?) and progress (IProgress<int>?) kwargs
- BuildFixedGrid routes through ParallelBuild.EvaluateInParallel
- Thread-safety contract documented in XML <remarks> on both ctors
- Cumulative IProgress<int> count via Interlocked.Increment (D2)
- 5 tests in BuildPerfTests.cs

Test count: 953 -> 958 (+5)."
```

---

## Task 4: ChebyshevSpline + Slider + TT ctor wiring + 6 tests

**Goal:** Add the same `nWorkers` + `progress` kwargs to all three remaining classes' ctors and route per-piece / per-slide / per-sweep. TT accepts the kwargs but documents `nWorkers` as ignored (D10); progress fires per-sweep in TT-Cross.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs:86-247` (all three ctors)
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs:80-147` (ctor)
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs:113-140` (ctor) and `:182-255` (Build progress wiring)
- Modify: `src/ChebyshevSharp/Internal/TensorTrainKernel.cs` (`TtCross` gains `IProgress<int>? sweepProgress = null` parameter)
- Modify: `tests/ChebyshevSharp.Tests/BuildPerfTests.cs` (append new test class)

**Python source:** `spline.py:106-180` (Spline ctor n_workers init + per-piece pass-through), `slider.py:166-240` (Slider ctor + per-slide pass-through), `tensor_train.py:1088-1140` (TT ctor + progress=2 verbose mode wiring), `barycentric.py:660-690` (per-evaluation progress hook reference).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 6 failing tests**

Append to `tests/ChebyshevSharp.Tests/BuildPerfTests.cs`:

```csharp
// ======================================================================
// TestSplineBuildPerf, TestSliderBuildPerf, TestTtBuildPerf (Phase 6 Task 4)
// ======================================================================

public class TestSplineBuildPerf
{
    private static double F(double[] p, object? _) => Math.Sin(p[0]) * Math.Cos(p[1]);

    [Fact]
    public void Test_spline_parallel_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8 };
        var knots = new[] { new double[] { 0.0 }, new double[] { 0.0 } };

        var seq = new ChebyshevSpline(F, 2, domain, nNodes, knots);
        seq.Build(verbose: false);

        var par = new ChebyshevSpline(F, 2, domain, nNodes, knots, nWorkers: 4);
        par.Build(verbose: false);

        Assert.Equal(seq.Pieces.Length, par.Pieces.Length);
        for (int p = 0; p < seq.Pieces.Length; p++)
        {
            var sv = seq.Pieces[p]!.TensorValues!;
            var pv = par.Pieces[p]!.TensorValues!;
            Assert.Equal(sv.Length, pv.Length);
            for (int i = 0; i < sv.Length; i++)
                TestFixtures.AssertClose(sv[i], pv[i], rtol: 0, atol: 0);
        }
    }

    [Fact]
    public void Test_spline_progress_count_sums_across_pieces()
    {
        var counter = new CountingProgress();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 4, 4 };
        var knots = new[] { new double[] { 0.0 }, new double[] { 0.0 } };

        var spline = new ChebyshevSpline(F, 2, domain, nNodes, knots, progress: counter);
        spline.Build(verbose: false);

        // 4 pieces × 16 nodes each = 64 expected.
        Assert.Equal(64, counter.LastValue);
    }
}

public class TestSliderBuildPerf
{
    private static double F(double[] p, object? _) => p[0] + p[1] + p[2];

    [Fact]
    public void Test_slider_parallel_matches_sequential()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 6, 6, 6 };
        var partition = new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } };
        var pivot = new[] { 0.0, 0.0, 0.0 };

        var seq = new ChebyshevSlider(F, 3, domain, nNodes, partition, pivot);
        seq.Build(verbose: false);

        var par = new ChebyshevSlider(F, 3, domain, nNodes, partition, pivot, nWorkers: 4);
        par.Build(verbose: false);

        var pt = new[] { 0.3, -0.2, 0.7 };
        TestFixtures.AssertClose(
            seq.Eval(pt, new int[3]), par.Eval(pt, new int[3]), rtol: 1e-12, atol: 1e-12);
    }

    [Fact]
    public void Test_slider_progress_count_sums_across_slides()
    {
        var counter = new CountingProgress();
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 4, 4 };
        var partition = new[] { new[] { 0 }, new[] { 1 } };
        var pivot = new[] { 0.0, 0.0 };

        var slider = new ChebyshevSlider(F, 2, domain, nNodes, partition, pivot, progress: counter);
        slider.Build(verbose: false);

        // 2 slides × 4 nodes each = 8 evaluations expected.
        Assert.Equal(8, counter.LastValue);
    }
}

public class TestTtBuildPerf
{
    [Fact]
    public void Test_tt_progress_per_sweep()
    {
        var counter = new CountingProgress();
        static double F(double[] p) => Math.Sin(p[0]) + p[1] * p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, maxRank: 5, maxSweeps: 3,
            progress: counter);
        tt.Build(verbose: false, seed: 42);
        Assert.True(counter.CallCount >= 1);
    }

    [Fact]
    public void Test_tt_nworkers_ignored_does_not_break_build()
    {
        // TT does not parallelize TT-Cross (D10); nWorkers != null must be a no-op.
        static double F(double[] p) => p[0] * p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, maxSweeps: 3, nWorkers: 4);
        tt.Build(verbose: false, seed: 42);
        Assert.Equal(2, tt.NumDimensions);
        var pt = new[] { 0.3, 0.4 };
        double v = tt.Eval(pt);
        TestFixtures.AssertClose(0.12, v, rtol: 1e-3, atol: 1e-3);
    }
}

/// <summary>Shared progress counter; Interlocked-safe.</summary>
internal sealed class CountingProgress : IProgress<int>
{
    private int _last;
    private int _calls;
    public int LastValue => System.Threading.Volatile.Read(ref _last);
    public int CallCount => System.Threading.Volatile.Read(ref _calls);
    public void Report(int value)
    {
        System.Threading.Interlocked.Exchange(ref _last, value);
        System.Threading.Interlocked.Increment(ref _calls);
    }
}
```

(Note: `CountingProgress` is internal to the test project; it can be reused by Task 3's tests too, if we choose to consolidate. For Task 3 we used a per-test private nested class for clarity; this version is shared because Tasks 4 and beyond need it across multiple test classes.)

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestSplineBuildPerf|FullyQualifiedName~TestSliderBuildPerf|FullyQualifiedName~TestTtBuildPerf"
```

Expected: build error — `ChebyshevSpline`, `ChebyshevSlider`, `ChebyshevTT` ctors don't have `nWorkers:` named argument.

- [ ] **Step 3: Add kwargs to all three `ChebyshevSpline` ctors**

In `src/ChebyshevSharp/ChebyshevSpline.cs` (line 86, 136, 210), append to each ctor signature:

```csharp
        int? nWorkers = null,
        IProgress<int>? progress = null)
```

Add fields near the other private fields:

```csharp
private int? _nWorkers;
private IProgress<int>? _progress;
```

Inside each ctor body, before any work that may construct pieces:

```csharp
_nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
_progress = progress;
```

Locate the `Build()` method (and any internal piece-construction site — look for `new ChebyshevApproximation(...)` inside Spline). Add `nWorkers: _nWorkers, progress: _progress` to that ctor call so per-piece progress counts accumulate across pieces (the implementation of `Internal.ParallelBuild.EvaluateInParallel` increments a single global counter by virtue of the IProgress instance being shared across pieces).

Add the same `<remarks>` thread-safety XML doc block to all three ctors.

- [ ] **Step 4: Add kwargs to `ChebyshevSlider` ctor**

In `src/ChebyshevSharp/ChebyshevSlider.cs:80`, append `int? nWorkers = null, IProgress<int>? progress = null` at the end of the signature. Store on private fields:

```csharp
private int? _nWorkers;
private IProgress<int>? _progress;
```

In `Build()` body (line 147), find the per-slide `new ChebyshevApproximation(...)` ctor call and add `nWorkers: _nWorkers, progress: _progress`. Add `<remarks>` thread-safety doc.

- [ ] **Step 5: Add kwargs to `ChebyshevTT` ctor + per-sweep progress wiring**

In `src/ChebyshevSharp/ChebyshevTT.cs:113`, append:

```csharp
        int? nWorkers = null,
        IProgress<int>? progress = null)
```

Store on private fields:

```csharp
private int? _nWorkers;        // accepted for API symmetry; ignored (D10).
private IProgress<int>? _progress;
```

Normalize/validate in body:

```csharp
_nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
_progress = progress;
```

Document `nWorkers` is ignored:

```csharp
/// <param name="nWorkers">Accepted for API symmetry with the other classes but
/// ignored: TT-Cross is adaptive sampling, not pre-grid evaluation. Pass null.</param>
/// <param name="progress">Optional progress reporter; receives the cumulative
/// sweep count after each TT-Cross sweep.</param>
```

In `Build()` at line 213 (where `TensorTrainKernel.TtCross(...)` is called), add `_progress` as the final argument. The `TtCross` signature in `Internal/TensorTrainKernel.cs` gains a new optional final parameter `IProgress<int>? sweepProgress = null`. Inside `TtCross`'s sweep loop (a `for` over `maxSweeps`), at the end of each sweep iteration: `sweepProgress?.Report(sweep + 1);`. SVD and ALS modes do not get progress wiring (matches Python — only sweeps emit).

- [ ] **Step 6: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSplineBuildPerf|FullyQualifiedName~TestSliderBuildPerf|FullyQualifiedName~TestTtBuildPerf"
```

Expected: 6 tests passed.

- [ ] **Step 7: Run full suite**

```bash
dotnet test
```

Expected: 964 tests passing (958 + 6). 0 failures, 0 warnings.

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs src/ChebyshevSharp/ChebyshevSlider.cs src/ChebyshevSharp/ChebyshevTT.cs src/ChebyshevSharp/Internal/TensorTrainKernel.cs tests/ChebyshevSharp.Tests/BuildPerfTests.cs
git commit -m "phase6: Spline+Slider+TT gain nWorkers + IProgress<int> kwargs

- ChebyshevSpline (3 ctors): nWorkers+progress threaded through per-piece Approx ctors
- ChebyshevSlider (1 ctor): nWorkers+progress threaded through per-slide Approx ctors
- ChebyshevTT (1 ctor): nWorkers accepted but ignored (D10); progress fires per
  TT-Cross sweep via new IProgress<int>? sweepProgress kwarg in TtCross
- Thread-safety contract documented on every ctor
- 6 tests in BuildPerfTests.cs

Test count: 958 -> 964 (+6)."
```

---

## Task 5: nWorkers validation edge cases + 3 tests

**Goal:** Verify all four classes' ctors propagate the `NormalizeNWorkers` validation. Three tests covering 0 and -2 across the four-class surface.

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/BuildPerfTests.cs` (append new test class)

**Python source:** N/A — exercises existing `_normalize_n_workers` validation end-to-end through public ctors.

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 3 tests**

Append to `tests/ChebyshevSharp.Tests/BuildPerfTests.cs`:

```csharp
// ======================================================================
// TestNWorkersValidation (Phase 6 Task 5)
// ======================================================================

public class TestNWorkersValidation
{
    private static double F(double[] p, object? _) => p[0];
    private static double F2(double[] p) => p[0];

    [Fact]
    public void Test_approx_nworkers_zero_throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(F, 1,
                new[] { new[] { 0.0, 1.0 } }, new[] { 5 }, nWorkers: 0));
    }

    [Fact]
    public void Test_spline_nworkers_minus_two_throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(F, 1,
                new[] { new[] { 0.0, 1.0 } }, new[] { 5 },
                new[] { Array.Empty<double>() }, nWorkers: -2));
    }

    [Fact]
    public void Test_slider_and_tt_nworkers_zero_throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSlider(F, 1,
                new[] { new[] { 0.0, 1.0 } }, new[] { 5 },
                new[] { new[] { 0 } }, new[] { 0.5 }, nWorkers: 0));

        Assert.Throws<ArgumentException>(() =>
            new ChebyshevTT(F2, 2,
                new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5, 5 }, nWorkers: 0));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~TestNWorkersValidation"
```

Expected: 3 tests passed.

- [ ] **Step 3: Run full suite**

```bash
dotnet test
```

Expected: 967 tests passing. 0 failures, 0 warnings.

- [ ] **Step 4: Commit**

```bash
git add tests/ChebyshevSharp.Tests/BuildPerfTests.cs
git commit -m "phase6: nWorkers validation edge cases on all four ctors

3 tests verifying ArgumentException on nWorkers=0 (Approx, Slider, TT)
and nWorkers=-2 (Spline).

Test count: 964 -> 967 (+3)."
```

---

## Task 6: Sensitivity helpers + ChebyshevApproximation.SobolIndices + 8 tests

**Goal:** Implement `Internal/Sensitivity.cs` (`ChebyshevCoefficientsND`, `ComputeSobolFromCoeffs`) and the public `SobolIndices()` instance method on `ChebyshevApproximation` returning a `SobolResult` record. 8 tests in a new `SobolIndicesTests.cs`.

**Files:**
- Modify: `src/ChebyshevSharp/Internal/Sensitivity.cs` (replace stub body with real implementation)
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (append public method `SobolIndices()` near other Phase 4 ergonomics methods)
- Create: `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs`

**Python source:** `ref/PyChebyshev/src/pychebyshev/_sensitivity.py:1-145` (full file: `_compute_chebyshev_coefficients`, `_chebyshev_norm_squared`, `_multi_index_norm_squared`, `_compute_sobol_from_coeffs`). `barycentric.py:1277-1340` (Approx `sobol_indices` calling into `_compute_sobol_from_coeffs`).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 8 failing tests**

Create `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestApproxSobolIndices (Phase 6 Task 6)
// ======================================================================

public class TestApproxSobolIndices
{
    [Fact]
    public void Test_additive_function_first_order_sums_to_one()
    {
        // f(x, y) = sin(x) + cos(y) — additive, no interaction term.
        // FirstOrder[0] + FirstOrder[1] ≈ 1; both TotalOrder ≈ FirstOrder (no mixing).
        static double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 16, 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0] + s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(s.FirstOrder[0], s.TotalOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(s.FirstOrder[1], s.TotalOrder[1], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_pure_first_dim_function()
    {
        // f(x, y) = sin(x) — constant in y.
        // FirstOrder[0] ≈ 1; FirstOrder[1] ≈ 0; TotalOrder[1] ≈ 0.
        static double F(double[] p, object? _) => Math.Sin(p[0]);
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 16, 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(0.0, s.FirstOrder[1], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(0.0, s.TotalOrder[1], rtol: 0, atol: 1e-10);
    }

    [Fact]
    public void Test_multiplicative_function_total_order_is_one()
    {
        // f(x, y) = x * y on [-1,1]^2 — pure interaction term, no additive part.
        // FirstOrder[*] ≈ 0; TotalOrder[0] ≈ TotalOrder[1] ≈ 1.
        static double F(double[] p, object? _) => p[0] * p[1];
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(0.0, s.FirstOrder[0], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(0.0, s.FirstOrder[1], rtol: 0, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[1], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_total_order_at_least_first_order()
    {
        // Invariant: FirstOrder[d] <= TotalOrder[d] for every d.
        static double F(double[] p, object? _) => Math.Sin(p[0] * p[1]) + Math.Cos(p[2]);
        var ap = new ChebyshevApproximation(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        for (int d = 0; d < 3; d++)
            Assert.True(s.FirstOrder[d] <= s.TotalOrder[d] + 1e-12,
                $"FirstOrder[{d}]={s.FirstOrder[d]} > TotalOrder[{d}]={s.TotalOrder[d]}");
    }

    [Fact]
    public void Test_dim_importance_ranking()
    {
        // f(x,y,z) = 100*sin(x) + 1*y + 0.01*z*z — clearly x > y > z by sensitivity.
        static double F(double[] p, object? _) => 100 * Math.Sin(p[0]) + p[1] + 0.01 * p[2] * p[2];
        var ap = new ChebyshevApproximation(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 12, 12, 12 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.True(s.TotalOrder[0] > s.TotalOrder[1]);
        Assert.True(s.TotalOrder[1] > s.TotalOrder[2]);
    }

    [Fact]
    public void Test_1d_function_first_order_equals_total_order_one()
    {
        // 1D function: FirstOrder[0] = TotalOrder[0] = 1 (no interaction possible).
        static double F(double[] p, object? _) => Math.Sin(p[0]);
        var ap = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-6, atol: 1e-6);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_constant_function_zero_variance()
    {
        // f(x, y) = 5 — constant; Variance = 0, indices = 0.
        static double F(double[] p, object? _) => 5.0;
        var ap = new ChebyshevApproximation(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        ap.Build(verbose: false);

        var s = ap.SobolIndices();
        Assert.Equal(0.0, s.Variance);
        Assert.Equal(0.0, s.FirstOrder[0]);
        Assert.Equal(0.0, s.TotalOrder[1]);
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        static double F(double[] p, object? _) => p[0];
        var ap = new ChebyshevApproximation(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 }, deferBuild: true);
        Assert.Throws<InvalidOperationException>(() => ap.SobolIndices());
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestApproxSobolIndices"
```

Expected: build error — `'ChebyshevApproximation' does not contain a definition for 'SobolIndices'`.

- [ ] **Step 3: Implement `Internal/Sensitivity.cs`**

Replace the stub body of `src/ChebyshevSharp/Internal/Sensitivity.cs` with:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Sobol sensitivity indices computed from Chebyshev spectral coefficients.
/// Mirrors PyChebyshev <c>_sensitivity.py</c> (v0.20.0).
/// </summary>
internal static class Sensitivity
{
    /// <summary>Chebyshev T_n inner product norm² under weight 1/√(1-x²) on [-1,1].</summary>
    private static double ChebyshevNormSquared(int n) => n == 0 ? Math.PI : Math.PI / 2.0;

    /// <summary>Multi-D inner product norm² = ∏ per-dim norms.</summary>
    private static double MultiIndexNormSquared(int[] alpha)
    {
        double r = 1.0;
        for (int i = 0; i < alpha.Length; i++) r *= ChebyshevNormSquared(alpha[i]);
        return r;
    }

    /// <summary>Convert flat row-major index → multi-index (one int per dimension).</summary>
    private static int[] UnravelIndex(long flat, int[] shape)
    {
        int n = shape.Length;
        var idx = new int[n];
        long rem = flat;
        for (int d = n - 1; d >= 0; d--)
        {
            idx[d] = (int)(rem % shape[d]);
            rem /= shape[d];
        }
        return idx;
    }

    /// <summary>
    /// Apply <see cref="BarycentricKernel.ChebyshevCoefficients1D"/> along every axis,
    /// matching PyChebyshev's <c>_compute_chebyshev_coefficients</c> convention
    /// (DCT-II per axis with c_0 halving, applied dim-by-dim).
    /// </summary>
    /// <param name="tensorValues">Row-major tensor of values at Type-I Chebyshev nodes.</param>
    /// <param name="shape">Per-dim node counts.</param>
    /// <returns>Row-major tensor of Chebyshev coefficients, same shape.</returns>
    internal static double[] ChebyshevCoefficientsND(double[] tensorValues, int[] shape)
    {
        int nDim = shape.Length;
        var coeffs = (double[])tensorValues.Clone();

        // Apply 1D DCT-II axis-by-axis.
        // For each axis d, compute the leading "outer" size (product of dims to the
        // left of d) and the trailing "inner" size (product of dims to the right).
        // Iterate over (outer, inner) coordinates, extract a 1D slice along d,
        // apply BarycentricKernel.ChebyshevCoefficients1D, write back.
        for (int d = 0; d < nDim; d++)
        {
            int n = shape[d];
            int outer = 1;
            for (int k = 0; k < d; k++) outer *= shape[k];
            int inner = 1;
            for (int k = d + 1; k < nDim; k++) inner *= shape[k];

            var slice = new double[n];
            for (int o = 0; o < outer; o++)
            {
                for (int j = 0; j < inner; j++)
                {
                    // Extract slice: coeffs[o, :, j] in (outer, n, inner) layout.
                    for (int i = 0; i < n; i++)
                        slice[i] = coeffs[(o * n + i) * inner + j];

                    var c = BarycentricKernel.ChebyshevCoefficients1D(slice);
                    // c[0] is already halved by ChebyshevCoefficients1D (matches PyChebyshev convention).

                    for (int i = 0; i < n; i++)
                        coeffs[(o * n + i) * inner + j] = c[i];
                }
            }
        }

        return coeffs;
    }

    /// <summary>
    /// Compute first- and total-order Sobol sensitivity indices from a Chebyshev
    /// coefficient tensor. Throws on NaN/Inf in input. Returns zero-filled
    /// <see cref="SobolResult"/> for constant functions (Variance == 0).
    /// </summary>
    internal static SobolResult ComputeSobolFromCoeffs(double[] coeffs, int[] shape)
    {
        for (int i = 0; i < coeffs.Length; i++)
            if (!double.IsFinite(coeffs[i]))
                throw new ArgumentException(
                    "coefficients contain NaN or Inf; SobolIndices() requires finite spectral coefficients");

        int nDim = shape.Length;
        var firstOrder = new double[nDim];
        var totalOrder = new double[nDim];
        double variance = 0;

        for (long flat = 0; flat < coeffs.Length; flat++)
        {
            var alpha = UnravelIndex(flat, shape);
            int nonzeroCount = 0;
            int firstNonzeroDim = -1;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) { nonzeroCount++; if (firstNonzeroDim == -1) firstNonzeroDim = d; }
            if (nonzeroCount == 0) continue;  // skip α = 0 (mean term).

            double c = coeffs[flat];
            if (c == 0) continue;
            double energy = c * c * MultiIndexNormSquared(alpha);
            variance += energy;
            for (int d = 0; d < nDim; d++)
                if (alpha[d] > 0) totalOrder[d] += energy;
            if (nonzeroCount == 1) firstOrder[firstNonzeroDim] += energy;
        }

        if (variance == 0)
            return new SobolResult(new double[nDim], new double[nDim], 0);
        for (int d = 0; d < nDim; d++)
        {
            firstOrder[d] /= variance;
            totalOrder[d] /= variance;
        }
        return new SobolResult(firstOrder, totalOrder, variance);
    }
}
```

- [ ] **Step 4: Add `SobolIndices()` to `ChebyshevApproximation`**

Append to `src/ChebyshevSharp/ChebyshevApproximation.cs`, near the other Phase 4 ergonomics methods (search for `IsConstructionFinished` for landing site):

```csharp
/// <summary>
/// Compute first- and total-order Sobol sensitivity indices directly from this
/// approximation's spectral Chebyshev coefficients. No Monte Carlo, no extra
/// function evaluations.
/// </summary>
/// <returns>A <see cref="SobolResult"/> with per-dim FirstOrder, TotalOrder, and total Variance.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
public SobolResult SobolIndices()
{
    if (TensorValues == null)
        throw new InvalidOperationException(
            "SobolIndices requires a built ChebyshevApproximation. Call Build() first.");
    var coeffs = Internal.Sensitivity.ChebyshevCoefficientsND(TensorValues, NNodes);
    return Internal.Sensitivity.ComputeSobolFromCoeffs(coeffs, NNodes);
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestApproxSobolIndices"
```

Expected: 8 tests passed.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: 975 tests passing (967 + 8). 0 failures, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/Sensitivity.cs src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/SobolIndicesTests.cs
git commit -m "phase6: Sobol indices via spectral coefficients (Approx surface)

- Internal/Sensitivity.cs: ChebyshevCoefficientsND (axis-wise DCT-II reuse),
  ComputeSobolFromCoeffs (variance decomposition with multi-D inner-product norm)
- ChebyshevApproximation.SobolIndices(): public instance method returning SobolResult
- 8 tests covering additive, pure-1d, multiplicative, ranking, 1D, constant,
  invariants, unbuilt-throws

Test count: 967 -> 975 (+8). Mirrors PyChebyshev _sensitivity.py + barycentric.py:1277."
```

---

## Task 7: ChebyshevSpline.SobolIndices aggregation + 4 tests

**Goal:** Add per-piece Sobol aggregation on `ChebyshevSpline` matching `spline.py:735-810` exactly: `vol × piece_variance` weighting, then divide by global variance at the end. 4 tests.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs` (append public `SobolIndices()` method)
- Modify: `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs` (append new test class)

**Python source:** `ref/PyChebyshev/src/pychebyshev/spline.py:735-810`.

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 4 failing tests**

Append to `tests/ChebyshevSharp.Tests/SobolIndicesTests.cs`:

```csharp
// ======================================================================
// TestSplineSobolIndices (Phase 6 Task 7)
// ======================================================================

public class TestSplineSobolIndices
{
    [Fact]
    public void Test_single_piece_matches_approx()
    {
        // Single-piece spline (no interior knots) matches Approx exactly.
        static double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 16, 16 };

        var ap = new ChebyshevApproximation(F, 2, domain, nNodes);
        ap.Build(verbose: false);
        var apSob = ap.SobolIndices();

        var sp = new ChebyshevSpline(F, 2, domain, nNodes,
            new[] { Array.Empty<double>(), Array.Empty<double>() });
        sp.Build(verbose: false);
        var spSob = sp.SobolIndices();

        TestFixtures.AssertClose(apSob.FirstOrder[0], spSob.FirstOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(apSob.FirstOrder[1], spSob.FirstOrder[1], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(apSob.TotalOrder[0], spSob.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_piecewise_abs_x_1d()
    {
        // f(x) = |x| on [-1,1] with knot at 0; per-piece smooth, both pieces equal volume.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            new[] { new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();
        // 1D: FirstOrder[0] = TotalOrder[0] = 1 (or 0 if Variance is 0; |x| is non-constant so Variance > 0).
        Assert.True(s.Variance > 0);
        TestFixtures.AssertClose(1.0, s.FirstOrder[0], rtol: 1e-10, atol: 1e-10);
        TestFixtures.AssertClose(1.0, s.TotalOrder[0], rtol: 1e-10, atol: 1e-10);
    }

    [Fact]
    public void Test_piecewise_abs_x_plus_abs_y_2d()
    {
        // f(x,y) = |x| + |y| with knots at 0 in both dims. Additive → both first-orders sum to 1.
        static double F(double[] p, object? _) => Math.Abs(p[0]) + Math.Abs(p[1]);
        var sp = new ChebyshevSpline(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 },
            new[] { new[] { 0.0 }, new[] { 0.0 } });
        sp.Build(verbose: false);

        var s = sp.SobolIndices();
        TestFixtures.AssertClose(1.0, s.FirstOrder[0] + s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
        // Both dims contribute roughly equally (symmetric function on symmetric domain).
        TestFixtures.AssertClose(s.FirstOrder[0], s.FirstOrder[1], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        static double F(double[] p, object? _) => p[0];
        var sp = new ChebyshevSpline(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 5 },
            new[] { Array.Empty<double>() });
        // sp not built — should throw.
        Assert.Throws<InvalidOperationException>(() => sp.SobolIndices());
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestSplineSobolIndices"
```

Expected: build error — `'ChebyshevSpline' does not contain a definition for 'SobolIndices'`.

- [ ] **Step 3: Implement Spline.SobolIndices**

Append to `src/ChebyshevSharp/ChebyshevSpline.cs`, near the other Phase 4 ergonomics methods:

```csharp
/// <summary>
/// Compute Sobol sensitivity indices aggregated across spline pieces.
/// Per-piece coefficients are computed under the Chebyshev measure on each piece's
/// local domain; per-piece contributions are weighted by domain volume × variance,
/// then normalized by global variance. For a single-piece spline, this reduces to
/// the <see cref="ChebyshevApproximation.SobolIndices"/> case.
/// </summary>
/// <returns>A <see cref="SobolResult"/> with per-dim FirstOrder, TotalOrder, global Variance.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
public SobolResult SobolIndices()
{
    if (Pieces == null || Pieces.Length == 0 || Pieces.Any(p => p == null || p.TensorValues == null))
        throw new InvalidOperationException(
            "SobolIndices requires a built ChebyshevSpline. Call Build() first.");

    int nDim = NumDimensions;
    var globalFirstOrder = new double[nDim];
    var globalTotalOrder = new double[nDim];
    double globalVariance = 0.0;

    foreach (var piece in Pieces)
    {
        if (piece == null) continue;
        double vol = 1.0;
        for (int d = 0; d < nDim; d++)
        {
            double lo = piece.Domain[d][0], hi = piece.Domain[d][1];
            vol *= (hi - lo);
        }
        var coeffs = Internal.Sensitivity.ChebyshevCoefficientsND(piece.TensorValues!, piece.NNodes);
        var pieceResult = Internal.Sensitivity.ComputeSobolFromCoeffs(coeffs, piece.NNodes);
        globalVariance += vol * pieceResult.Variance;
        for (int d = 0; d < nDim; d++)
        {
            globalFirstOrder[d] += vol * pieceResult.FirstOrder[d] * pieceResult.Variance;
            globalTotalOrder[d] += vol * pieceResult.TotalOrder[d] * pieceResult.Variance;
        }
    }

    if (globalVariance == 0)
        return new SobolResult(new double[nDim], new double[nDim], 0);
    for (int d = 0; d < nDim; d++)
    {
        globalFirstOrder[d] /= globalVariance;
        globalTotalOrder[d] /= globalVariance;
    }
    return new SobolResult(globalFirstOrder, globalTotalOrder, globalVariance);
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestSplineSobolIndices"
```

Expected: 4 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 979 tests passing (975 + 4). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/SobolIndicesTests.cs
git commit -m "phase6: Spline.SobolIndices via per-piece variance aggregation

Aggregation rule: global_variance += vol * piece_variance;
global_first[d] += vol * piece_first[d] * piece_variance;
divide by global_variance at the end. Matches Python spline.py:735-810
exactly. Single-piece spline reduces to Approx case.

4 tests including single-piece-equivalence with Approx.

Test count: 975 -> 979 (+4)."
```

---

## Task 8: ChebyshevSpline.AutoKnots + 10 tests

**Goal:** Add static factory `ChebyshevSpline.AutoKnots(...)` that scans each dim for curvature spikes, clusters and caps them, then constructs a Spline using the resulting knots via the existing `specialPoints` pipeline.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs` (append static AutoKnots + private static ScanForKnotsAlongDim helper)
- Create: `tests/ChebyshevSharp.Tests/AutoKnotsTests.cs`

**Python source:** `ref/PyChebyshev/src/pychebyshev/spline.py:2111-2210` (auto_knots classmethod with the scan/threshold/cluster algorithm).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 10 failing tests**

Create `tests/ChebyshevSharp.Tests/AutoKnotsTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestAutoKnots (Phase 6 Task 8)
// ======================================================================

public class TestAutoKnots
{
    [Fact]
    public void Test_abs_x_finds_knot_near_zero()
    {
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.True(sp.NumPieces >= 2,
            $"Expected at least 2 pieces (knot near 0); got {sp.NumPieces}");
        var knots0 = sp.Knots[0];
        Assert.Contains(knots0, k => Math.Abs(k) < 0.05);
    }

    [Fact]
    public void Test_relu_finds_knot_near_half()
    {
        static double F(double[] p, object? _) => Math.Max(0.0, p[0] - 0.5);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 8 });
        Assert.True(sp.NumPieces >= 2);
        var knots0 = sp.Knots[0];
        Assert.Contains(knots0, k => Math.Abs(k - 0.5) < 0.05);
    }

    [Fact]
    public void Test_2d_additive_abs_finds_knots_per_dim()
    {
        static double F(double[] p, object? _) => Math.Abs(p[0]) + Math.Abs(p[1]);
        var sp = ChebyshevSpline.AutoKnots(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        Assert.True(sp.Knots[0].Length >= 1);
        Assert.True(sp.Knots[1].Length >= 1);
    }

    [Fact]
    public void Test_smooth_function_finds_no_knots()
    {
        static double F(double[] p, object? _) => p[0] * p[0];
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        // Smooth function: |d²f| is constant, no spikes above threshold.
        Assert.Equal(0, sp.Knots[0].Length);
        Assert.Equal(1, sp.NumPieces);
    }

    [Fact]
    public void Test_high_threshold_finds_no_knots_for_abs()
    {
        // Threshold so high that even |x|'s spike is filtered out.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            thresholdFactor: 1000.0);
        Assert.Equal(0, sp.Knots[0].Length);
    }

    [Fact]
    public void Test_max_knots_per_dim_caps_count()
    {
        // f(x) with many bumps; cap at 1.
        static double F(double[] p, object? _) =>
            Math.Abs(p[0] - 0.2) + Math.Abs(p[0] - 0.5) + Math.Abs(p[0] - 0.8);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 8 },
            maxKnotsPerDim: 1);
        Assert.True(sp.Knots[0].Length <= 1);
    }

    [Fact]
    public void Test_n_scan_points_too_small_throws()
    {
        static double F(double[] p, object? _) => p[0];
        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 },
            nScanPoints: 2));
    }

    [Fact]
    public void Test_function_returning_nan_throws()
    {
        static double F(double[] p, object? _) => double.NaN;
        Assert.Throws<ArgumentException>(() => ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 }));
    }

    [Fact]
    public void Test_max_knots_zero_returns_no_knot_spline()
    {
        // maxKnotsPerDim=0 means "no auto-knots, just build a single-piece spline".
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 8 },
            maxKnotsPerDim: 0);
        Assert.Equal(0, sp.Knots[0].Length);
        Assert.Equal(1, sp.NumPieces);
    }

    [Fact]
    public void Test_result_is_fully_functional()
    {
        // The returned ChebyshevSpline must Eval correctly.
        static double F(double[] p, object? _) => Math.Abs(p[0]);
        var sp = ChebyshevSpline.AutoKnots(F, 1,
            new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        TestFixtures.AssertClose(0.5, sp.Eval(new[] { 0.5 }), rtol: 1e-3, atol: 1e-3);
        TestFixtures.AssertClose(0.5, sp.Eval(new[] { -0.5 }), rtol: 1e-3, atol: 1e-3);
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestAutoKnots"
```

Expected: build error — `'ChebyshevSpline' does not contain a definition for 'AutoKnots'`.

- [ ] **Step 3: Implement AutoKnots**

Append to `src/ChebyshevSharp/ChebyshevSpline.cs` near the other static factories:

```csharp
/// <summary>
/// Auto-place knots at function kinks via a curvature-spike scan, then build the
/// resulting <see cref="ChebyshevSpline"/>. Mirrors PyChebyshev <c>spline.py:2111</c>.
/// </summary>
/// <param name="function">f(point, additionalData) → double; must return finite at every scan point.</param>
/// <param name="numDimensions">Number of input dimensions.</param>
/// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
/// <param name="numNodes">Per-piece node counts; same shape as the regular ctor.</param>
/// <param name="maxOrderDerivative">Max derivative order. Default 2.</param>
/// <param name="additionalData">Optional user data threaded through f calls.</param>
/// <param name="descriptor">Optional free-form descriptor.</param>
/// <param name="thresholdFactor">Spike threshold = thresholdFactor × mean(|d²f|). Default 5.0.</param>
/// <param name="maxKnotsPerDim">Cap on knots per dimension. Default 5. Zero means no auto-knots.</param>
/// <param name="nScanPoints">Number of scan points per dim. Default 200; must be at least 3.</param>
/// <param name="nWorkers">See <see cref="ChebyshevSpline"/> ctor.</param>
/// <param name="progress">See <see cref="ChebyshevSpline"/> ctor.</param>
/// <param name="verbose">If true, print scan progress.</param>
/// <returns>A built ChebyshevSpline with the discovered knots.</returns>
/// <remarks>
/// When <paramref name="nWorkers"/> is non-null, <paramref name="function"/> may be
/// invoked concurrently from multiple threads. Functions that capture mutable state
/// must use locks or external synchronization, or pass <c>nWorkers: null</c>.
/// </remarks>
public static ChebyshevSpline AutoKnots(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] numNodes,
    int maxOrderDerivative = 2,
    object? additionalData = null,
    string? descriptor = null,
    double thresholdFactor = 5.0,
    int maxKnotsPerDim = 5,
    int nScanPoints = 200,
    int? nWorkers = null,
    IProgress<int>? progress = null,
    bool verbose = false)
{
    if (thresholdFactor <= 0)
        throw new ArgumentException("thresholdFactor must be > 0", nameof(thresholdFactor));
    if (maxKnotsPerDim < 0)
        throw new ArgumentException("maxKnotsPerDim must be >= 0", nameof(maxKnotsPerDim));
    if (nScanPoints < 3)
        throw new ArgumentException("nScanPoints must be at least 3 to compute a 2nd-derivative finite difference", nameof(nScanPoints));

    int? effectiveWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);

    // Scan each dim for curvature spikes; build per-dim knot arrays.
    var allKnots = new double[numDimensions][];
    for (int d = 0; d < numDimensions; d++)
    {
        if (maxKnotsPerDim == 0)
        {
            allKnots[d] = Array.Empty<double>();
            continue;
        }
        allKnots[d] = ScanForKnotsAlongDim(
            function, d, numDimensions, domain, additionalData,
            thresholdFactor, maxKnotsPerDim, nScanPoints,
            effectiveWorkers, progress);
    }

    // Construct the resulting spline.
    var sp = new ChebyshevSpline(function, numDimensions, domain, numNodes, allKnots,
        maxDerivativeOrder: maxOrderDerivative,
        additionalData: additionalData,
        nWorkers: nWorkers,
        progress: progress);
    sp.SetDescriptor(descriptor ?? string.Empty);
    sp.Build(verbose: verbose);
    return sp;
}

/// <summary>
/// Scan one dimension for second-derivative spikes; cluster spikes; cap to
/// maxKnotsPerDim. Returns the knot positions in that dimension's domain.
/// </summary>
private static double[] ScanForKnotsAlongDim(
    Func<double[], object?, double> function,
    int dim,
    int numDimensions,
    double[][] domain,
    object? additionalData,
    double thresholdFactor,
    int maxKnotsPerDim,
    int nScanPoints,
    int? effectiveWorkers,
    IProgress<int>? progress)
{
    double lo = domain[dim][0], hi = domain[dim][1];
    // Build sample points: this dim varies, others fixed at midpoint.
    var samplePoints = new double[nScanPoints][];
    double dx = (hi - lo) / (nScanPoints - 1);
    for (int i = 0; i < nScanPoints; i++)
    {
        var pt = new double[numDimensions];
        for (int k = 0; k < numDimensions; k++)
            pt[k] = (k == dim) ? (lo + i * dx) : 0.5 * (domain[k][0] + domain[k][1]);
        samplePoints[i] = pt;
    }

    // Evaluate (parallelized if requested).
    double[] ys = Internal.ParallelBuild.EvaluateInParallel(
        function, samplePoints, additionalData, effectiveWorkers, progress);

    // Reject non-finite values.
    for (int i = 0; i < nScanPoints; i++)
        if (!double.IsFinite(ys[i]))
            throw new ArgumentException(
                $"AutoKnots requires a finite-valued function over the entire domain " +
                $"(non-finite at scan point {i} of dim {dim})");

    // 2nd-derivative finite difference; pad boundaries with 0.
    var d2 = new double[nScanPoints];
    double h2 = dx * dx;
    for (int i = 1; i < nScanPoints - 1; i++)
        d2[i] = (ys[i + 1] - 2.0 * ys[i] + ys[i - 1]) / h2;

    // Compute mean(|d2|) over interior.
    double sumAbs = 0;
    int interiorCount = 0;
    for (int i = 1; i < nScanPoints - 1; i++)
    {
        sumAbs += Math.Abs(d2[i]);
        interiorCount++;
    }
    double meanD2 = interiorCount > 0 ? sumAbs / interiorCount : 0.0;
    if (meanD2 == 0) return Array.Empty<double>();
    double threshold = thresholdFactor * meanD2;

    // Identify spike indices.
    var spikes = new List<int>();
    for (int i = 1; i < nScanPoints - 1; i++)
        if (Math.Abs(d2[i]) > threshold) spikes.Add(i);
    if (spikes.Count == 0) return Array.Empty<double>();

    // Cluster spikes within radius = max(1, nScanPoints / (maxKnotsPerDim * 4)).
    int clusterRadius = Math.Max(1, nScanPoints / Math.Max(1, maxKnotsPerDim * 4));
    var clusterPeaks = new List<int>();
    int j = 0;
    while (j < spikes.Count)
    {
        int peak = spikes[j];
        double peakAbs = Math.Abs(d2[peak]);
        int k = j + 1;
        while (k < spikes.Count && spikes[k] - peak <= clusterRadius)
        {
            if (Math.Abs(d2[spikes[k]]) > peakAbs)
            {
                peak = spikes[k];
                peakAbs = Math.Abs(d2[peak]);
            }
            k++;
        }
        clusterPeaks.Add(peak);
        j = k;
    }

    // Sort by |d²| desc; cap at maxKnotsPerDim.
    clusterPeaks.Sort((a, b) => Math.Abs(d2[b]).CompareTo(Math.Abs(d2[a])));
    if (clusterPeaks.Count > maxKnotsPerDim)
        clusterPeaks.RemoveRange(maxKnotsPerDim, clusterPeaks.Count - maxKnotsPerDim);

    // Sort by position ascending and convert to domain coordinates.
    clusterPeaks.Sort();
    var knots = clusterPeaks.Select(idx => lo + idx * dx).ToArray();
    return knots;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestAutoKnots"
```

Expected: 10 tests passed.

- [ ] **Step 5: Run full suite**

```bash
dotnet test
```

Expected: 989 tests passing (979 + 10). 0 failures, 0 warnings.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/AutoKnotsTests.cs
git commit -m "phase6: ChebyshevSpline.AutoKnots — curvature-spike knot detection

Static factory mirroring spline.py:2111. Per-dim scan, 2nd-derivative FD,
threshold via thresholdFactor*mean(|d²|), cluster within radius, cap at
maxKnotsPerDim, build resulting spline through the existing specialPoints
pipeline. Defaults match Python: thresholdFactor=5.0, maxKnotsPerDim=5,
nScanPoints=200.

10 tests covering kink detection, smoothness no-detect, threshold sensitivity,
caps, validation, and result functionality.

Test count: 979 -> 989 (+10)."
```

---

## Task 9: TtSwapAdjacent + Reorder + DimOrder property + 8 tests

**Goal:** Add `TtSwapAdjacent` static helper to `Internal/TensorTrainAlgebra.cs`. Add `_dimOrder` field + `DimOrder` property + `Reorder` method to `ChebyshevTT`. 8 tests: 3 for `TtSwapAdjacent`, 5 for `Reorder`. (`WithAutoOrder` and `_dimOrder` threading land in Tasks 10–11.)

**Files:**
- Modify: `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` (append `TtSwapAdjacent`)
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (add `_dimOrder` field, `DimOrder` property, `Reorder` method; initialize `_dimOrder` to identity in two ctors)
- Create: `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs`

**Python source:** `_algebra.py:177-243` (`_tt_swap_adjacent`), `tensor_train.py:1136-1140` (`_dim_order` field init), `tensor_train.py:2349-2470` (`reorder` instance method).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 8 failing tests**

Create `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs`:

```csharp
using ChebyshevSharp.Internal;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestTtSwapAdjacent (Phase 6 Task 9)
// ======================================================================

public class TestTtSwapAdjacent
{
    private static TensorTrainKernel.TtCore[] Build3DCores()
    {
        // Build a small 3D TT-Cross result for swap testing.
        static double F(double[] p) => Math.Sin(p[0]) + p[1] * p[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, maxRank: 5, maxSweeps: 3);
        tt.Build(verbose: false, seed: 42);
        // Access cores via reflection-like internal helper or copy via algebra primitive.
        return GetCoresViaSerialization(tt);
    }

    /// <summary>Round-trip a TT through Save/Load and extract the cores. Test-only helper.</summary>
    private static TensorTrainKernel.TtCore[] GetCoresViaSerialization(ChebyshevTT tt)
    {
        var tmp = Path.GetTempFileName();
        try
        {
            tt.Save(tmp);
            var loaded = ChebyshevTT.Load(tmp);
            // Loaded TT exposes cores only through eval; for the swap test, we
            // construct cores directly via TensorTrainAlgebra.NegateCores reflection.
            // Simpler: just expose a test-only internal accessor on ChebyshevTT.
            // For this plan, assume an `internal TensorTrainKernel.TtCore[] GetCoeffCoresForTest()`
            // is added on ChebyshevTT in Task 9 Step 3.
            return loaded.GetCoeffCoresForTest();
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void Test_swap_is_self_inverse()
    {
        var cores = Build3DCores();
        // Swap (0,1) twice — result should equal original.
        var once = TensorTrainAlgebra.TtSwapAdjacent(cores, 0, maxRank: 10);
        var twice = TensorTrainAlgebra.TtSwapAdjacent(once, 0, maxRank: 10);

        // Compare via inner product: <cores, cores> ~ <twice, twice>; <cores, twice> ~ <cores, cores>.
        double a = TensorTrainAlgebra.InnerProductCores(cores, cores);
        double b = TensorTrainAlgebra.InnerProductCores(twice, twice);
        double c = TensorTrainAlgebra.InnerProductCores(cores, twice);
        TestFixtures.AssertClose(a, b, rtol: 1e-8, atol: 1e-8);
        TestFixtures.AssertClose(a, c, rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_swap_out_of_range_throws()
    {
        var cores = Build3DCores();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorTrainAlgebra.TtSwapAdjacent(cores, -1, maxRank: 10));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorTrainAlgebra.TtSwapAdjacent(cores, cores.Length - 1, maxRank: 10));
    }

    [Fact]
    public void Test_swap_changes_node_axis_lengths_in_pair()
    {
        // After swapping axes (i, i+1), the node lengths at positions i and i+1
        // should equal original n_b and n_a respectively (cores are 6×6×6 in this fixture
        // so they're equal — the test verifies shapes are valid post-swap).
        var cores = Build3DCores();
        int origN0 = cores[0].NNodes, origN1 = cores[1].NNodes;
        var swapped = TensorTrainAlgebra.TtSwapAdjacent(cores, 0, maxRank: 10);
        Assert.Equal(origN1, swapped[0].NNodes);
        Assert.Equal(origN0, swapped[1].NNodes);
    }
}

// ======================================================================
// TestReorder (Phase 6 Task 9)
// ======================================================================

public class TestReorder
{
    private static ChebyshevTT BuildTestTt()
    {
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2] * p[2];
        var tt = new ChebyshevTT(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 }, maxRank: 8, maxSweeps: 5);
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Test_reorder_identity_returns_equivalent_tt()
    {
        var tt = BuildTestTt();
        var reord = tt.Reorder(new[] { 0, 1, 2 });
        Assert.Equal(new[] { 0, 1, 2 }, reord.DimOrder);
        var pt = new[] { 0.3, -0.5, 0.7 };
        TestFixtures.AssertClose(tt.Eval(pt), reord.Eval(pt), rtol: 1e-8, atol: 1e-8);
    }

    [Fact]
    public void Test_reorder_round_trip_recovers_original()
    {
        var tt = BuildTestTt();
        var perm = new[] { 2, 0, 1 };
        var inv = new[] { 1, 2, 0 };  // inverse permutation
        var step1 = tt.Reorder(perm, maxRank: 16, tolerance: 1e-12);
        var step2 = step1.Reorder(inv, maxRank: 16, tolerance: 1e-12);
        Assert.Equal(new[] { 0, 1, 2 }, step2.DimOrder);
        var pt = new[] { 0.3, -0.5, 0.7 };
        TestFixtures.AssertClose(tt.Eval(pt), step2.Eval(pt), rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_reorder_changes_dim_order()
    {
        var tt = BuildTestTt();
        var reord = tt.Reorder(new[] { 1, 2, 0 });
        Assert.Equal(new[] { 1, 2, 0 }, reord.DimOrder);
    }

    [Fact]
    public void Test_reorder_invalid_permutation_throws()
    {
        var tt = BuildTestTt();
        Assert.Throws<ArgumentException>(() => tt.Reorder(new[] { 0, 1, 1 }));   // duplicate
        Assert.Throws<ArgumentException>(() => tt.Reorder(new[] { 0, 1 }));      // wrong length
        Assert.Throws<ArgumentException>(() => tt.Reorder(new[] { 0, 1, 5 }));   // out of range
    }

    [Fact]
    public void Test_dim_order_returns_clone()
    {
        var tt = BuildTestTt();
        int[] order = tt.DimOrder;
        order[0] = 99;
        Assert.Equal(0, tt.DimOrder[0]);  // mutation does not affect TT.
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestTtSwapAdjacent|FullyQualifiedName~TestReorder"
```

Expected: build error — `TtSwapAdjacent` not defined; `Reorder` not defined; `DimOrder` not defined; `GetCoeffCoresForTest` not defined.

- [ ] **Step 3: Add `TtSwapAdjacent` to `Internal/TensorTrainAlgebra.cs`**

Append to `src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs` (before the closing `}` of the class):

```csharp
/// <summary>
/// Swap adjacent storage axes <paramref name="i"/> and <paramref name="i+1"/>
/// of a TT in coefficient space via SVD truncation. Mirrors PyChebyshev
/// <c>_algebra.py:177</c>.
/// </summary>
/// <param name="cores">Coefficient cores. Not mutated; returns a fresh list.</param>
/// <param name="i">Position of the leftmost core in the swap pair (0 ≤ i &lt; cores.Length - 1).</param>
/// <param name="maxRank">Maximum rank for the SVD truncation between the swapped cores.</param>
/// <param name="tolerance">Relative singular-value cutoff (s_max × tolerance). Default 1e-12.</param>
/// <returns>New cores list with axes <c>i</c> and <c>i+1</c> swapped.</returns>
internal static TensorTrainKernel.TtCore[] TtSwapAdjacent(
    TensorTrainKernel.TtCore[] cores, int i, int maxRank, double tolerance = 1e-12)
{
    if (i < 0 || i >= cores.Length - 1)
        throw new ArgumentOutOfRangeException(nameof(i),
            $"i={i} out of range [0, {cores.Length - 1})");

    var newCores = new TensorTrainKernel.TtCore[cores.Length];
    for (int k = 0; k < cores.Length; k++) newCores[k] = cores[k].Copy();

    var A = newCores[i];        // (rL, nA, rM)
    var B = newCores[i + 1];    // (rM, nB, rR)
    int rL = A.RLeft, nA = A.NNodes, rM = A.RRight;
    int rM2 = B.RLeft, nB = B.NNodes, rR = B.RRight;
    if (rM != rM2)
        throw new ArgumentException($"core shape mismatch at {i}: A.RRight={rM}, B.RLeft={rM2}");

    // Form joint M[rL, nA, nB, rR] = Σ_rM A · B
    // M is stored row-major: M[l, a, b, r] at index ((l * nA + a) * nB + b) * rR + r
    var M = new double[rL * nA * nB * rR];
    for (int l = 0; l < rL; l++)
        for (int a = 0; a < nA; a++)
            for (int b = 0; b < nB; b++)
                for (int r = 0; r < rR; r++)
                {
                    double acc = 0;
                    for (int m = 0; m < rM; m++)
                        acc += A[l, a, m] * B[m, b, r];
                    M[((l * nA + a) * nB + b) * rR + r] = acc;
                }

    // Transpose middle axes: Mt[l, b, a, r] = M[l, a, b, r]
    var Mt = new double[rL * nB * nA * rR];
    for (int l = 0; l < rL; l++)
        for (int a = 0; a < nA; a++)
            for (int b = 0; b < nB; b++)
                for (int r = 0; r < rR; r++)
                    Mt[((l * nB + b) * nA + a) * rR + r] =
                        M[((l * nA + a) * nB + b) * rR + r];

    // Reshape to matrix: rows = (rL × nB), cols = (nA × rR)
    int rows = rL * nB;
    int cols = nA * rR;
    var matData = new double[rows, cols];
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            matData[row, col] = Mt[row * cols + col];

    var matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(matData);
    var svd = matrix.Svd(true);
    var U = svd.U;
    var S = svd.S;
    var Vh = svd.VT;

    double sMax = S.Count > 0 ? S[0] : 0.0;
    int keep = Math.Min(maxRank, S.Count);
    if (sMax > 0 && tolerance > 0)
    {
        double cutoff = sMax * tolerance;
        int keepByTol = 0;
        for (int k = 0; k < S.Count; k++) if (S[k] > cutoff) keepByTol++;
        keep = Math.Max(1, Math.Min(keep, keepByTol));
    }
    else
    {
        keep = Math.Max(1, keep);
    }

    // Repack: A' = U * S, shape (rL, nB, keep); B' = Vh, shape (keep, nA, rR).
    var aNewData = new double[rL * nB * keep];
    for (int row = 0; row < rows; row++)
        for (int k = 0; k < keep; k++)
            aNewData[row * keep + k] = U[row, k] * S[k];

    var bNewData = new double[keep * nA * rR];
    for (int k = 0; k < keep; k++)
        for (int col = 0; col < cols; col++)
            bNewData[k * cols + col] = Vh[k, col];

    newCores[i] = new TensorTrainKernel.TtCore(rL, nB, keep, aNewData);
    newCores[i + 1] = new TensorTrainKernel.TtCore(keep, nA, rR, bNewData);
    return newCores;
}
```

- [ ] **Step 4: Add `_dimOrder`, `DimOrder`, `Reorder`, and the test-only core accessor to `ChebyshevTT.cs`**

In `src/ChebyshevSharp/ChebyshevTT.cs`:

Near other private fields (after `_additionalData`), add:

```csharp
private int[] _dimOrder = Array.Empty<int>();
```

In the public ctor at line 113, after the existing field assignments, append:

```csharp
_dimOrder = Enumerable.Range(0, numDimensions).ToArray();
_nWorkers = Internal.ParallelBuild.NormalizeNWorkers(nWorkers);
_progress = progress;
```

In the private ctor at line 143 (used by Load), append:

```csharp
_dimOrder = Enumerable.Range(0, numDimensions).ToArray();  // overwritten by Load's v2 deserialization
```

Add the public read-only property near other Phase 4 ergonomics getters:

```csharp
/// <summary>
/// Storage permutation: <c>DimOrder[k]</c> is the original-dimension index stored
/// at TT position k. Identity by default; non-identity for TTs produced by
/// <see cref="WithAutoOrder"/> or <see cref="Reorder"/>. Returns a defensive clone;
/// mutating the returned array does not affect this TT.
/// </summary>
public int[] DimOrder => (int[])_dimOrder.Clone();
```

Add `Reorder`:

```csharp
/// <summary>
/// Realign storage to a target permutation via TT-swap (adjacent-axis SVDs in
/// coefficient space). Functional API; returns a new TT. Inherits all build
/// parameters (maxRank, tolerance, maxSweeps, descriptor, additionalData,
/// maxDerivativeOrder, Method) from this TT.
/// </summary>
/// <param name="newOrder">Target permutation; must be a permutation of [0, NumDimensions-1].</param>
/// <param name="maxRank">Optional override for swap-time SVD truncation. Default: this TT's maxRank.</param>
/// <param name="tolerance">Optional relative-tolerance cutoff. Default: this TT's tolerance.</param>
/// <returns>A new TT with <c>DimOrder == newOrder</c>.</returns>
/// <exception cref="ArgumentException">If <paramref name="newOrder"/> is not a valid permutation.</exception>
public ChebyshevTT Reorder(int[] newOrder, int? maxRank = null, double? tolerance = null)
{
    CheckBuilt();
    ValidatePermutation(newOrder, _numDimensions);
    int rank = maxRank ?? _maxRank;
    double tol = tolerance ?? _tolerance;

    // Bubble-sort current storage to newOrder, applying TtSwapAdjacent at each transposition.
    var currentOrder = (int[])_dimOrder.Clone();
    var cores = new Internal.TensorTrainKernel.TtCore[_coeffCores!.Length];
    for (int k = 0; k < cores.Length; k++) cores[k] = _coeffCores[k].Copy();

    for (int i = 0; i < _numDimensions - 1; i++)
        for (int j = 0; j < _numDimensions - 1 - i; j++)
        {
            // Position j in currentOrder should hold newOrder[j] at the end.
            // If currentOrder[j+1] is closer to its target than currentOrder[j], swap.
            int wantAtJ = newOrder[j];
            int idxOfWant = Array.IndexOf(currentOrder, wantAtJ);
            int idxOfCurrentJ = j;
            // Greedy: if currentOrder[j] != wantAtJ AND currentOrder[j+1] is wantAtJ
            // OR currentOrder[j] is further from its target, swap.
            if (currentOrder[j] != wantAtJ && idxOfWant > j)
            {
                // bubble wantAtJ leftward by one step.
                int sourcePos = idxOfWant;
                if (sourcePos > j)
                {
                    // Swap positions sourcePos-1 and sourcePos.
                    int swapAt = sourcePos - 1;
                    cores = Internal.TensorTrainAlgebra.TtSwapAdjacent(cores, swapAt, rank, tol);
                    (currentOrder[swapAt], currentOrder[swapAt + 1]) =
                        (currentOrder[swapAt + 1], currentOrder[swapAt]);
                    // Reset inner loop to redo this position with updated state.
                    j--;
                    continue;
                }
            }
        }

    // Build result via existing BuildResultFromCores helper (Phase 2).
    var newDomain = newOrder.Select(d => (double[])_domain[d].Clone()).ToArray();
    var newNNodes = newOrder.Select(d => _nNodes[d]).ToArray();
    var result = BuildResultFromCores(cores, newDomain, newNNodes);
    result._dimOrder = (int[])newOrder.Clone();
    result._descriptor = _descriptor;
    result._additionalData = _additionalData;
    return result;
}

private static void ValidatePermutation(int[] perm, int n)
{
    if (perm == null) throw new ArgumentNullException(nameof(perm));
    if (perm.Length != n)
        throw new ArgumentException(
            $"Permutation length {perm.Length} != numDimensions {n}", nameof(perm));
    var seen = new bool[n];
    foreach (int v in perm)
    {
        if (v < 0 || v >= n)
            throw new ArgumentException(
                $"Permutation entry {v} out of range [0, {n - 1}]", nameof(perm));
        if (seen[v])
            throw new ArgumentException($"Duplicate entry {v} in permutation", nameof(perm));
        seen[v] = true;
    }
}
```

Add the test-only core accessor (visible to the test project via `InternalsVisibleTo`):

```csharp
internal Internal.TensorTrainKernel.TtCore[] GetCoeffCoresForTest() => _coeffCores!;
```

(The `InternalsVisibleTo` attribute already grants the test project access to internal members; verify in `ChebyshevSharp.csproj`.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestTtSwapAdjacent|FullyQualifiedName~TestReorder"
```

Expected: 8 tests passed.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: 997 tests passing (989 + 8). 0 failures, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/TensorTrainAlgebra.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs
git commit -m "phase6: TtSwapAdjacent + Reorder + DimOrder property + _dimOrder field

- Internal/TensorTrainAlgebra.TtSwapAdjacent: coefficient-space adjacent-axis swap
  via joint contraction → middle-axis transpose → SVD-truncate. Mirrors Python
  _algebra.py:177-243.
- ChebyshevTT._dimOrder field: identity init in both ctors.
- ChebyshevTT.DimOrder: defensive-clone read-only property.
- ChebyshevTT.Reorder(newOrder, maxRank?, tolerance?): bubble-sort via TtSwapAdjacent;
  result inherits descriptor/additionalData/maxRank/tolerance/etc.
- ValidatePermutation guard on Reorder.

8 tests across TestTtSwapAdjacent (3) + TestReorder (5).

Test count: 989 -> 997 (+8)."
```

---

## Task 10: WithAutoOrder + JSON migration + 10 tests

**Goal:** Implement `ChebyshevTT.WithAutoOrder(...)` (greedy_swap + random methods); bump JSON schema to v2 with `dimOrder` field; backfill identity for v1 files. 10 tests covering the factory + JSON migration. Commit a v0.9.0 fixture file.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (append `WithAutoOrder` static factory + Save/Load updates)
- Modify: `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs` (append new test class)
- Create: `tests/ChebyshevSharp.Tests/fixtures/v0.9.0_sin3d_tt.json` (pre-Phase-6 fixture; generate one-off in Step 4)

**Python source:** `tensor_train.py:2687-2890` (`with_auto_order`), `tensor_train.py` Save/Load `_dim_order` handling (around `__setstate__` backfill).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 10 failing tests**

Append to `tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs`:

```csharp
// ======================================================================
// TestWithAutoOrder + TestJsonMigration (Phase 6 Task 10)
// ======================================================================

public class TestWithAutoOrder
{
    [Fact]
    public void Test_with_auto_order_lower_rank_function()
    {
        // f(x,y,z) = sin(x) + cos(y) + z*z is rank-low under canonical order;
        // we just verify WithAutoOrder produces a valid build with non-degenerate DimOrder.
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2] * p[2];
        var tt = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 }, maxRank: 5, maxSweeps: 3,
            nTrials: 3, method: "greedy_swap");
        Assert.Equal(3, tt.DimOrder.Length);
        Assert.Equal(new HashSet<int> { 0, 1, 2 }, new HashSet<int>(tt.DimOrder));
    }

    [Fact]
    public void Test_greedy_swap_deterministic()
    {
        static double F(double[] p) => Math.Sin(p[0]) + p[1] * p[2];
        var tt1 = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, nTrials: 3, method: "greedy_swap", seed: 42);
        var tt2 = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, nTrials: 3, method: "greedy_swap", seed: 99);
        // greedy_swap ignores seed (deterministic); both runs produce same DimOrder.
        Assert.Equal(tt1.DimOrder, tt2.DimOrder);
    }

    [Fact]
    public void Test_random_with_seed_reproducible()
    {
        static double F(double[] p) => p[0] * p[1] + p[2];
        var tt1 = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, nTrials: 3, method: "random", seed: 42);
        var tt2 = ChebyshevTT.WithAutoOrder(F, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 }, nTrials: 3, method: "random", seed: 42);
        Assert.Equal(tt1.DimOrder, tt2.DimOrder);  // bit-exact for same seed.
    }

    [Fact]
    public void Test_unknown_method_throws()
    {
        static double F(double[] p) => p[0];
        Assert.Throws<ArgumentException>(() => ChebyshevTT.WithAutoOrder(F, 1,
            new[] { new[] { 0.0, 1.0 } }, new[] { 5 }, method: "wat"));
    }

    [Fact]
    public void Test_n_trials_zero_returns_canonical()
    {
        static double F(double[] p) => Math.Sin(p[0]) + p[1];
        var tt = ChebyshevTT.WithAutoOrder(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, nTrials: 0);
        Assert.Equal(new[] { 0, 1 }, tt.DimOrder);
    }

    [Fact]
    public void Test_result_is_fully_functional()
    {
        static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var tt = ChebyshevTT.WithAutoOrder(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 }, nTrials: 2, method: "greedy_swap");
        var pt = new[] { 0.3, -0.4 };
        TestFixtures.AssertClose(F(pt), tt.Eval(pt), rtol: 1e-3, atol: 1e-3);
    }
}

public class TestJsonMigrationDimOrder
{
    [Fact]
    public void Test_save_writes_jsonversion_2_and_dimorder()
    {
        static double F(double[] p) => Math.Sin(p[0]) + p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, maxSweeps: 3);
        tt.Build(verbose: false, seed: 42);
        var tmp = Path.GetTempFileName();
        try
        {
            tt.Save(tmp);
            string json = File.ReadAllText(tmp);
            Assert.Contains("\"JsonVersion\":2", json);
            Assert.Contains("\"DimOrder\":[0,1]", json);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [Fact]
    public void Test_load_v2_round_trip_preserves_dimorder()
    {
        static double F(double[] p) => Math.Sin(p[0]) + p[1];
        var tt = new ChebyshevTT(F, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, maxSweeps: 3);
        tt.Build(verbose: false, seed: 42);
        var perm = new[] { 1, 0 };
        var reord = tt.Reorder(perm, maxRank: 16, tolerance: 1e-10);
        var tmp = Path.GetTempFileName();
        try
        {
            reord.Save(tmp);
            var loaded = ChebyshevTT.Load(tmp);
            Assert.Equal(perm, loaded.DimOrder);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [Fact]
    public void Test_load_v0_9_0_fixture_backfills_identity()
    {
        // The fixture file is committed at tests/fixtures/v0.9.0_sin3d_tt.json.
        // It was saved with v0.9.0 (no JsonVersion or DimOrder fields).
        // Load must succeed and DimOrder must default to identity [0, 1, 2].
        string fixturePath = Path.Combine(
            AppContext.BaseDirectory,
            "..", "..", "..", "..", "..", "tests", "ChebyshevSharp.Tests", "fixtures",
            "v0.9.0_sin3d_tt.json");
        var loaded = ChebyshevTT.Load(fixturePath);
        Assert.Equal(new[] { 0, 1, 2 }, loaded.DimOrder);
        Assert.Equal(3, loaded.NumDimensions);
    }

    [Fact]
    public void Test_dim_order_defensive_clone()
    {
        static double F(double[] p) => p[0];
        var tt = new ChebyshevTT(F, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 5 });
        tt.Build(verbose: false, seed: 42);
        int[] order = tt.DimOrder;
        order[0] = 42;
        Assert.Equal(0, tt.DimOrder[0]);  // mutation does not propagate.
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestWithAutoOrder|FullyQualifiedName~TestJsonMigrationDimOrder"
```

Expected: build error — `WithAutoOrder` not defined; the JSON `JsonVersion` and `DimOrder` fields don't exist; the fixture file doesn't exist.

- [ ] **Step 3: Implement `WithAutoOrder`**

Append to `src/ChebyshevSharp/ChebyshevTT.cs`:

```csharp
/// <summary>
/// Build a TT trying multiple dim orderings, returning the lowest-rank result.
/// TT-Cross compression depends on dim order; different orderings yield different
/// ranks for the same function. Mirrors PyChebyshev <c>tensor_train.py:2687</c>.
/// </summary>
/// <param name="function">f(point) → double in the original (user) dim order.</param>
/// <param name="numDimensions">Number of input dimensions.</param>
/// <param name="domain">Bounds for each dimension in original order.</param>
/// <param name="numNodes">Node counts per dimension in original order.</param>
/// <param name="maxRank">Maximum TT rank passed to each trial. Default 10.</param>
/// <param name="tolerance">Convergence tolerance for each trial. Default 1e-6.</param>
/// <param name="maxSweeps">Max TT-Cross sweeps per trial. Default 10.</param>
/// <param name="additionalData">Stored on the result for introspection; not threaded into f.</param>
/// <param name="nTrials">Number of swap iterations / random samples. Default 5.</param>
/// <param name="method">"greedy_swap" (default, deterministic) or "random".</param>
/// <param name="seed">Optional seed for "random". Ignored by "greedy_swap".</param>
/// <param name="progress">Optional per-sweep progress reporter (forwarded to each trial's Build).</param>
/// <param name="verbose">If true, print per-trial diagnostics.</param>
/// <returns>The lowest-total-rank TT among the tried permutations, with <c>DimOrder</c> set.</returns>
public static ChebyshevTT WithAutoOrder(
    Func<double[], double> function,
    int numDimensions,
    double[][] domain,
    int[] numNodes,
    int maxRank = 10,
    double tolerance = 1e-6,
    int maxSweeps = 10,
    object? additionalData = null,
    int nTrials = 5,
    string method = "greedy_swap",
    int? seed = null,
    IProgress<int>? progress = null,
    bool verbose = false)
{
    if (method != "greedy_swap" && method != "random")
        throw new ArgumentException(
            $"unknown method: '{method}' (use 'greedy_swap' or 'random')", nameof(method));

    ChebyshevTT BuildWith(int[] order)
    {
        var permDomain = order.Select(d => domain[d]).ToArray();
        var permNNodes = order.Select(d => numNodes[d]).ToArray();
        // Permuted f: caller passes a point in PERMUTED order; map back to original.
        Func<double[], double> permF = (point) =>
        {
            var orig = new double[numDimensions];
            for (int k = 0; k < numDimensions; k++) orig[order[k]] = point[k];
            return function(orig);
        };
        var tt = new ChebyshevTT(permF, numDimensions, permDomain, permNNodes,
            maxRank: maxRank, tolerance: tolerance, maxSweeps: maxSweeps,
            additionalData: additionalData, progress: progress);
        tt.Build(verbose: verbose, seed: seed);
        tt._dimOrder = (int[])order.Clone();
        return tt;
    }

    int RankSum(ChebyshevTT t)
    {
        int sum = 0;
        foreach (int r in t.TtRanks) sum += r;
        return sum;
    }

    int[] canonical = Enumerable.Range(0, numDimensions).ToArray();
    var bestTt = BuildWith(canonical);
    int bestScore = RankSum(bestTt);

    if (nTrials <= 0) return bestTt;

    if (method == "greedy_swap")
    {
        bool improved = true;
        int iter = 0;
        while (improved && iter < nTrials)
        {
            improved = false;
            for (int i = 0; i < numDimensions - 1; i++)
            {
                var trial = (int[])bestTt.DimOrder.Clone();
                (trial[i], trial[i + 1]) = (trial[i + 1], trial[i]);
                var candidateTt = BuildWith(trial);
                int candidateScore = RankSum(candidateTt);
                if (candidateScore < bestScore)
                {
                    bestTt = candidateTt;
                    bestScore = candidateScore;
                    improved = true;
                }
            }
            iter++;
        }
    }
    else  // method == "random"
    {
        var rng = new Random(seed ?? Environment.TickCount);
        for (int t = 0; t < nTrials; t++)
        {
            // Fisher-Yates shuffle of canonical order.
            var trial = (int[])canonical.Clone();
            for (int i = numDimensions - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (trial[i], trial[j]) = (trial[j], trial[i]);
            }
            var candidateTt = BuildWith(trial);
            int candidateScore = RankSum(candidateTt);
            if (candidateScore < bestScore)
            {
                bestTt = candidateTt;
                bestScore = candidateScore;
            }
        }
    }

    return bestTt;
}
```

- [ ] **Step 4: Update Save/Load for JSON v2 + commit fixture**

In `src/ChebyshevSharp/ChebyshevTT.cs:1677-1698` (the `TTSerializationState` DTO), add:

```csharp
public int? JsonVersion { get; set; }     // 2 for v0.10.0+; null/absent => v1.
public int[]? DimOrder { get; set; }      // present in v2; null in v1 (backfill identity).
```

In `Save()` (line 1337), add to the state:

```csharp
JsonVersion = 2,
DimOrder = (int[])_dimOrder.Clone(),
```

In `Load()` (line 1386), after `state` is deserialized, before the existing field assignments:

```csharp
int jsonVersion = state.JsonVersion ?? 1;
int[] dimOrder = state.DimOrder ?? Enumerable.Range(0, state.NumDimensions).ToArray();
```

Then assign to the result TT:

```csharp
tt._dimOrder = (int[])dimOrder.Clone();
```

To create the v0.9.0 fixture file, run a one-off generation step:

```bash
mkdir -p tests/ChebyshevSharp.Tests/fixtures
# Use the v0.9.0 release tag's TT.Save format. The simplest approach is to:
# 1. Check out v0.9.0 in a temp worktree, build a 3D sin TT with seed=42, save it.
# 2. Copy the resulting JSON file into tests/ChebyshevSharp.Tests/fixtures/v0.9.0_sin3d_tt.json.
# Alternatively, since v0.9.0 didn't write JsonVersion or DimOrder, we can hand-craft
# the v0.9.0 JSON by saving on a HEAD that omits the new fields, then committing it.

# Simplest in-place: create a v0.9.0-like file by serializing on this branch with the
# Save() method temporarily reverted to omit the new fields. To avoid that gymnastic,
# generate a JSON that mimics v0.9.0's output by reading any post-Phase-2 saved TT
# fixture and stripping any DimOrder/JsonVersion keys before committing.
```

**Recommended approach**: temporarily build a TT and Save it, then post-process the JSON to strip the new keys before committing as the v0.9.0 fixture:

```bash
dotnet run --project /tmp/gen_v090_fixture.csproj  # one-off helper
# OR — preferred — use a small C# scratch in the test project itself: write a method
# that calls tt.Save() on a HEAD that doesn't include JsonVersion/DimOrder yet.
# We do this by checking out the v0.9.0 release tag, running the save, then
# checking back out:
git stash --include-untracked
git checkout v0.9.0 -- src/ChebyshevSharp/ChebyshevTT.cs
dotnet build  # build with v0.9.0's ChebyshevTT.Save signature
# Run a test or main method that builds a 3D sin TT (seed=42) and saves to tmp.
# (See step 4a below for the exact code to add as a one-off generator.)
# Then:
git checkout HEAD -- src/ChebyshevSharp/ChebyshevTT.cs
git stash pop
mv /tmp/v0.9.0_sin3d_tt.json tests/ChebyshevSharp.Tests/fixtures/
git add tests/ChebyshevSharp.Tests/fixtures/v0.9.0_sin3d_tt.json
```

Step 4a (the actual generator — add as a one-off temp test, run once, then delete):

```csharp
[Fact(Skip = "Generator: run once to create the v0.9.0 fixture, then delete.")]
public void Generate_v090_fixture()
{
    static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + p[2] * p[2];
    var tt = new ChebyshevTT(F, 3,
        new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
        new[] { 6, 6, 6 }, maxRank: 5, maxSweeps: 3);
    tt.Build(verbose: false, seed: 42);
    tt.Save("/tmp/v0.9.0_sin3d_tt.json");
    // Move the file manually to tests/ChebyshevSharp.Tests/fixtures/.
}
```

After generating, ensure the resulting JSON does NOT contain `"JsonVersion"` or `"DimOrder"` keys (it shouldn't, because the generator runs against the v0.9.0 ChebyshevTT.cs Save signature — the temporary `git checkout v0.9.0 -- ...` step ensures that).

Add `<None Include="fixtures\**\*" CopyToOutputDirectory="PreserveNewest" />` to `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` so the fixture file is copied to the test output directory.

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestWithAutoOrder|FullyQualifiedName~TestJsonMigrationDimOrder"
```

Expected: 10 tests passed.

- [ ] **Step 6: Run full suite**

```bash
dotnet test
```

Expected: 1007 tests passing (997 + 10). 0 failures, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtAutoOrderTests.cs tests/ChebyshevSharp.Tests/fixtures/v0.9.0_sin3d_tt.json tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj
git commit -m "phase6: ChebyshevTT.WithAutoOrder + JSON v1->v2 migration

- ChebyshevTT.WithAutoOrder(...): static factory; greedy_swap (deterministic) and
  random (seeded) methods; selects lowest-total-rank build.
- JSON Save writes JsonVersion=2 + DimOrder; Load backfills identity for v1 files.
- v0.9.0 fixture committed at tests/fixtures/v0.9.0_sin3d_tt.json verifying
  the backfill path against an actual pre-Phase-6 save.

10 tests across TestWithAutoOrder (6) + TestJsonMigrationDimOrder (4).

Test count: 997 -> 1007 (+10). Mirrors tensor_train.py:2687 + __setstate__ backfill."
```

---

## Task 11: _dimOrder threading across TT public methods + 10 tests

**Goal:** Thread `_dimOrder` through every public TT method that takes a coordinate or returns a sub-TT. Add 10 tests verifying permuted-vs-canonical equivalence and the binary-algebra mismatch error.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (Eval/EvalBatch/EvalMulti/Slice/Extrude/ToDense/Integrate(partial)/algebra)
- Create: `tests/ChebyshevSharp.Tests/TtDimOrderTests.cs`

**Python source:** `tensor_train.py:1494-2105` (the threading sites; I cited specific line ranges in the spec §6.5 table).

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Write 10 failing tests**

Create `tests/ChebyshevSharp.Tests/TtDimOrderTests.cs`:

```csharp
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// ======================================================================
// TestTtDimOrderThreading (Phase 6 Task 11)
// ======================================================================

public class TestTtDimOrderThreading
{
    private static double F(double[] p) => Math.Sin(p[0]) + Math.Cos(p[1]) + 0.5 * p[2];

    private static (ChebyshevTT canonical, ChebyshevTT reord) Pair()
    {
        var domain = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var nNodes = new[] { 8, 8, 8 };
        var canonical = new ChebyshevTT(F, 3, domain, nNodes, maxRank: 8, maxSweeps: 5);
        canonical.Build(verbose: false, seed: 42);
        var reord = canonical.Reorder(new[] { 2, 0, 1 }, maxRank: 16, tolerance: 1e-12);
        return (canonical, reord);
    }

    [Fact]
    public void Test_eval_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pt = new[] { 0.3, -0.4, 0.5 };
        TestFixtures.AssertClose(canonical.Eval(pt), reord.Eval(pt), rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_eval_batch_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pts = new double[,]
        {
            { 0.3, -0.4, 0.5 },
            { -0.7, 0.1, 0.2 },
            { 0.0, 0.0, 0.0 },
        };
        var canonicalRes = canonical.EvalBatch(pts);
        var reordRes = reord.EvalBatch(pts);
        for (int i = 0; i < canonicalRes.Length; i++)
            TestFixtures.AssertClose(canonicalRes[i], reordRes[i], rtol: 1e-6, atol: 1e-6);
    }

    [Fact]
    public void Test_eval_multi_respects_dimorder()
    {
        var (canonical, reord) = Pair();
        var pt = new[] { 0.3, -0.4, 0.5 };
        var orders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };  // value + ∂/∂x[0]
        var canonicalRes = canonical.EvalMulti(pt, orders);
        var reordRes = reord.EvalMulti(pt, orders);
        for (int i = 0; i < canonicalRes.Length; i++)
            TestFixtures.AssertClose(canonicalRes[i], reordRes[i], rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_slice_updates_dimorder()
    {
        var (_, reord) = Pair();
        // reord has DimOrder = [2, 0, 1]; slice user-dim 0 → drop original dim 0.
        var sliced = reord.Slice(0, 0.5);
        Assert.Equal(2, sliced.NumDimensions);
        // sliced.DimOrder should be the surviving original dims renumbered to a permutation of [0, 1].
        Assert.Equal(new HashSet<int> { 0, 1 }, new HashSet<int>(sliced.DimOrder));
    }

    [Fact]
    public void Test_extrude_updates_dimorder()
    {
        var (_, reord) = Pair();
        var extruded = reord.Extrude(0, (-1.0, 1.0), 4);
        Assert.Equal(4, extruded.NumDimensions);
        // Result DimOrder: a permutation of [0..3].
        Assert.Equal(new HashSet<int> { 0, 1, 2, 3 }, new HashSet<int>(extruded.DimOrder));
    }

    [Fact]
    public void Test_to_dense_produces_tensor_in_original_dim_order()
    {
        var (canonical, reord) = Pair();
        var canonicalDense = canonical.ToDense();
        var reordDense = reord.ToDense();
        // Both should produce arrays of the same shape (since both have the same original-dim
        // axes and node counts). For a 3D 8x8x8 grid:
        Assert.Equal(canonicalDense.Length, reordDense.Length);
        // Numerical match (best effort given SVD truncation in reorder):
        for (int i = 0; i < canonicalDense.Length; i++)
            TestFixtures.AssertClose(canonicalDense[i], reordDense[i], rtol: 1e-3, atol: 1e-3);
    }

    [Fact]
    public void Test_partial_integrate_updates_dimorder()
    {
        var (_, reord) = Pair();
        // Partial integrate over user-dim 1 → result is 2D.
        var integrated = (ChebyshevTT)reord.Integrate(dims: new[] { 1 });
        Assert.Equal(2, integrated.NumDimensions);
        Assert.Equal(new HashSet<int> { 0, 1 }, new HashSet<int>(integrated.DimOrder));
    }

    [Fact]
    public void Test_unary_negation_preserves_dimorder()
    {
        var (_, reord) = Pair();
        var neg = -reord;
        Assert.Equal(reord.DimOrder, neg.DimOrder);
    }

    [Fact]
    public void Test_binary_add_matching_dimorder()
    {
        var (canonical, reord) = Pair();
        // Add reord + reord — same DimOrder, succeeds.
        var sum = reord + reord;
        Assert.Equal(reord.DimOrder, sum.DimOrder);
    }

    [Fact]
    public void Test_binary_add_mismatched_dimorder_throws()
    {
        var (canonical, reord) = Pair();
        var ex = Assert.Throws<ArgumentException>(() => canonical + reord);
        Assert.Contains("dim_order", ex.Message);
        Assert.Contains("Reorder", ex.Message);
    }
}
```

- [ ] **Step 2: Run failing tests**

```bash
dotnet test --filter "FullyQualifiedName~TestTtDimOrderThreading"
```

Expected: many test failures — `Eval` doesn't yet permute by `_dimOrder`; `Slice/Extrude/Integrate/ToDense` don't yet update `_dimOrder`; binary `+` doesn't yet check `_dimOrder` matching.

- [ ] **Step 3: Thread `_dimOrder` through `Eval`, `EvalBatch`, `EvalMulti`**

In `src/ChebyshevSharp/ChebyshevTT.cs:Eval` (line 274), at the top, before the existing logic:

```csharp
// If non-identity dim_order, remap user point to storage frame.
if (!IsIdentityDimOrder())
{
    var permPoint = new double[_numDimensions];
    for (int k = 0; k < _numDimensions; k++) permPoint[k] = point[_dimOrder[k]];
    point = permPoint;
}
```

Same prefix for `EvalBatch` (line 337):

```csharp
if (!IsIdentityDimOrder())
{
    var n = points.GetLength(0);
    var permPoints = new double[n, _numDimensions];
    for (int i = 0; i < n; i++)
        for (int k = 0; k < _numDimensions; k++)
            permPoints[i, k] = points[i, _dimOrder[k]];
    points = permPoints;
}
```

For `EvalMulti` (line 414): both `point` AND `derivativeOrders[k][...]` need permutation. Wrap the body so that after permutation, the existing logic runs unchanged. Use the helper `IsIdentityDimOrder()` (added below) to short-circuit.

Add the private helper near other helpers:

```csharp
private bool IsIdentityDimOrder()
{
    for (int i = 0; i < _dimOrder.Length; i++)
        if (_dimOrder[i] != i) return false;
    return true;
}
```

- [ ] **Step 4: Update `Slice` and `Extrude` to set `_dimOrder` on the result**

In `Slice` (line 1101), after calling `BuildResultFromCores` (line 1128), insert:

```csharp
// Update result _dimOrder: drop the storage-position-of-user-dim, then renumber.
int storagePos = !IsIdentityDimOrder() ? Array.IndexOf(_dimOrder, dim) : dim;
var newDimOrder = new int[_numDimensions - 1];
int writePos = 0;
for (int k = 0; k < _numDimensions; k++)
    if (k != storagePos) newDimOrder[writePos++] = _dimOrder[k];
// Renumber surviving dims so the result's DimOrder is a permutation of [0, n-2].
var newDimIndex = new int[_numDimensions];
int counter = 0;
for (int origDim = 0; origDim < _numDimensions; origDim++)
    if (origDim != _dimOrder[storagePos]) newDimIndex[origDim] = counter++;
for (int k = 0; k < newDimOrder.Length; k++)
    newDimOrder[k] = newDimIndex[newDimOrder[k]];
var sliced = BuildResultFromCores(newCores, newDomain, newNNodes);
sliced._dimOrder = newDimOrder;
return sliced;
```

(Replace the existing `return BuildResultFromCores(newCores, newDomain, newNNodes);` line at 1128.)

In `Extrude` (line 1071), update similarly: append `numDimensions` (the new user-dim index becomes the highest original-dim index after extrusion). Wrap the existing return with:

```csharp
var extruded = BuildResultFromCores(newCores, newDomainArr, newNNodes);
var newDimOrder = new int[_numDimensions + 1];
for (int k = 0; k < dim; k++) newDimOrder[k] = _dimOrder[k];
newDimOrder[dim] = _numDimensions;  // new dim's user index.
for (int k = dim; k < _numDimensions; k++) newDimOrder[k + 1] = _dimOrder[k];
extruded._dimOrder = newDimOrder;
return extruded;
```

- [ ] **Step 5: Update `ToDense` to transpose into original-dim order**

In `ToDense` (line 1047), after `TensorTrainExtrude.ToDenseEinsumChain(...)`, transpose the result if `_dimOrder` is non-identity:

```csharp
var dense = TensorTrainExtrude.ToDenseEinsumChain(_coeffCores!, _nNodes);
if (IsIdentityDimOrder()) return dense;

// Build a transposed copy: dense currently has axes in storage order (_dimOrder).
// We need axes in original-dim order: targetAxes[origDim] = storage_pos.
int n = _numDimensions;
var origNNodes = new int[n];
for (int k = 0; k < n; k++) origNNodes[_dimOrder[k]] = _nNodes[k];
long total = 1;
for (int k = 0; k < n; k++) total = checked(total * origNNodes[k]);
var result = new double[total];
var origIdx = new int[n];
var storageIdx = new int[n];
for (long flat = 0; flat < total; flat++)
{
    long rem = flat;
    for (int k = n - 1; k >= 0; k--)
    {
        origIdx[k] = (int)(rem % origNNodes[k]);
        rem /= origNNodes[k];
    }
    for (int k = 0; k < n; k++) storageIdx[k] = origIdx[_dimOrder[k]];
    long storageFlat = 0;
    for (int k = 0; k < n; k++) storageFlat = storageFlat * _nNodes[k] + storageIdx[k];
    result[flat] = dense[storageFlat];
}
return result;
```

- [ ] **Step 6: Update partial `Integrate` to set `_dimOrder` on the result**

In `Integrate` (line 607), at the partial-integration branch (after `BuildIntegrateResult` is called at line 711), update `_dimOrder` analogously to `Slice`. Locate where `keptDims` is computed (line 705); use that to derive the new `_dimOrder`:

```csharp
// keptDims contains the storage-frame positions that survive. Convert to user dims via _dimOrder.
var newDimOrder = new int[keptDims.Length];
for (int i = 0; i < keptDims.Length; i++)
    newDimOrder[i] = _dimOrder[keptDims[i]];
// Renumber: surviving original-dim indices become 0..k-1.
var sortedSurvivors = newDimOrder.Distinct().OrderBy(d => d).ToArray();
var dimIndex = new Dictionary<int, int>();
for (int i = 0; i < sortedSurvivors.Length; i++) dimIndex[sortedSurvivors[i]] = i;
for (int i = 0; i < newDimOrder.Length; i++) newDimOrder[i] = dimIndex[newDimOrder[i]];

var partialResult = BuildIntegrateResult(newCores.ToArray(), newDomain, newNNodes);
partialResult._dimOrder = newDimOrder;
return partialResult;
```

(Replace the line `return BuildIntegrateResult(newCores.ToArray(), newDomain, newNNodes);` at line 711.)

- [ ] **Step 7: Update unary algebra to inherit `_dimOrder`**

In `src/ChebyshevSharp/ChebyshevTT.cs:1162-1218` (operators `*`, `-` unary, `/`, in-place variants), after each `BuildResultFromCores` call, set:

```csharp
result._dimOrder = (int[])tt._dimOrder.Clone();
```

For in-place variants (e.g., `ScalarMulInPlace`), no change is needed — `_dimOrder` is unchanged on `this`.

For binary algebra (`+` operator on TTs, lines 1265+): in `CheckCompatible` (line 1228), append:

```csharp
// _dimOrder mismatch: refuse with a hint at Reorder.
for (int k = 0; k < a._numDimensions; k++)
    if (a._dimOrder[k] != b._dimOrder[k])
        throw new ArgumentException(
            $"dim_order mismatch at storage position {k}: {a._dimOrder[k]} vs {b._dimOrder[k]}. " +
            "Call Reorder() on one operand to align before adding/subtracting.");
```

Inherit `_dimOrder` on the binary-result TT analogously to unary.

- [ ] **Step 8: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~TestTtDimOrderThreading"
```

Expected: 10 tests passed.

- [ ] **Step 9: Run full suite**

```bash
dotnet test
```

Expected: 1017 tests passing (1007 + 10). 0 failures, 0 warnings.

- [ ] **Step 10: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtDimOrderTests.cs
git commit -m "phase6: thread _dimOrder through every public TT method

- Eval/EvalBatch/EvalMulti remap user point/derivOrders by _dimOrder before contraction
- Slice/Extrude/partial-Integrate update result _dimOrder via drop-and-renumber rule
- ToDense transposes result tensor into original-dim axis order when _dimOrder ≠ identity
- Unary algebra (*, -, /, in-place variants) preserve _dimOrder on result
- Binary algebra (+, -) require matching _dimOrder; mismatch throws with Reorder hint
- New private helper IsIdentityDimOrder() for fast-path detection

10 tests in TtDimOrderTests.cs covering full surface + binary mismatch.

Test count: 1007 -> 1017 (+10). Mirrors tensor_train.py:1494-2105 (v0.20.1)."
```

---

## Task 12: Docs + changelog + parity tags + CLAUDE.md (no new tests)

**Goal:** Bump csproj version + parity metadata; commit changelog v0.10.0 entry; create user-guide pages for parallel build and adaptive refinement; update toc.yml; update skip_csharp.txt; update CLAUDE.md status block. No new tests; `dotnet test` continues to pass at 1017/1017.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj` — `<Version>`, `<PyChebyshevParity>`, `<InformationalVersion>`
- Modify: `docs/docs/changelog.md` — v0.10.0 entry
- Create: `docs/docs/parallel-build.md` — new user-guide page
- Create: `docs/docs/adaptive-refinement.md` — new user-guide page
- Modify: `docs/docs/toc.yml` — add the two new pages
- Modify: `skip_csharp.txt` — Phase 6 entry
- Modify: `CLAUDE.md` — bump Status block

**Python source:** N/A — release prep.

### WORKTREE ENFORCEMENT (MANDATORY)

```bash
git rev-parse --show-toplevel
```

Expected: ends in `.worktrees/phase6-perf-and-adaptive`.

- [ ] **Step 1: Bump csproj**

In `src/ChebyshevSharp/ChebyshevSharp.csproj`, locate `<Version>0.9.0</Version>` and update to `<Version>0.10.0</Version>`. Update `<PyChebyshevParity>0.17.0</PyChebyshevParity>` to `<PyChebyshevParity>0.20.1</PyChebyshevParity>`. Update `<InformationalVersion>0.9.0+pychebyshev.0.17.0</InformationalVersion>` to `<InformationalVersion>0.10.0+pychebyshev.0.20.1</InformationalVersion>`.

- [ ] **Step 2: Add v0.10.0 changelog entry**

Prepend to `docs/docs/changelog.md` (immediately below the existing top heading, before the v0.9.0 entry):

```markdown
## [0.10.0] - 2026-04-?? — PyChebyshev parity v0.20.1

### Build performance (from PyChebyshev v0.19.0)

- `nWorkers` ctor kwarg on all four classes (`int?`): `null` (default) = sequential; `-1` = `Environment.ProcessorCount`; positive int = pool size.
- `IProgress<int>` ctor kwarg on all four classes: per-evaluation in Approx/Spline/Slider; per-sweep in TT.
- Thread-safety contract: when `nWorkers` is non-null, the user-supplied function may be invoked concurrently from multiple threads. Functions that capture mutable state must use locks or external synchronization, or pass `nWorkers: null`.
- TT accepts `nWorkers` for API symmetry but ignores it: TT-Cross is adaptive sampling, not pre-grid evaluation.

### Adaptive refinement (from PyChebyshev v0.20.0 + v0.20.1)

- `ChebyshevSpline.AutoKnots(...)` static factory: auto-place knots at function kinks via curvature-spike scan. Defaults: `thresholdFactor=5.0`, `maxKnotsPerDim=5`, `nScanPoints=200`.
- `SobolIndices()` instance method on `ChebyshevApproximation` and `ChebyshevSpline`: variance decomposition from spectral Chebyshev coefficients. Returns new `SobolResult` record (`FirstOrder`, `TotalOrder`, `Variance`). No Monte Carlo, no extra evaluations.
- `ChebyshevTT.WithAutoOrder(...)` static factory: heuristic dim ordering to minimize TT rank. Methods: `"greedy_swap"` (deterministic) and `"random"` (seeded).
- `ChebyshevTT.Reorder(newOrder, maxRank?, tolerance?)` instance method: TT-swap-based realignment via adjacent-axis SVDs in coefficient space.
- `ChebyshevTT.DimOrder` read-only property: storage permutation, identity by default.
- All TT public methods now thread `_dimOrder` correctly: `Eval`, `EvalBatch`, `EvalMulti`, `Slice`, `Extrude`, `ToDense`, partial `Integrate`, unary algebra. Binary algebra (`+`, `-`) requires matching `_dimOrder` and throws `ArgumentException` with a hint at `Reorder` on mismatch.

### JSON migration

- `ChebyshevTT` save format bumped to `"jsonVersion": 2` with new `"dimOrder"` field.
- v0.9.0 and earlier files load with identity `dimOrder` backfilled.

### Skipped (Python-only ergonomic features)

- `plot_convergence`, `plot_1d`, `plot_2d_surface`, `plot_2d_contour` (matplotlib helpers). Documented under "Python-only ergonomic features."

### Internal

- New `Internal/ParallelBuild.cs` — `NormalizeNWorkers`, `EvaluateInParallel`.
- New `Internal/Sensitivity.cs` — `ChebyshevCoefficientsND`, `ComputeSobolFromCoeffs`.
- `Internal/TensorTrainAlgebra.cs` extended with `TtSwapAdjacent` (coefficient-space adjacent-axis swap via SVD).

### Submodule and parity

- Submodule `ref/PyChebyshev` bumped from v0.18.0 to v0.20.1 (15 commits).
- `<PyChebyshevParity>` advances 0.17.0 → 0.20.1 (Phase 5's deliberate non-monotonic drop is now corrected forward; Phase 6 ships everything between v0.18 and v0.20.1).
- This is the **final phase** of the v0.20.1 port: ChebyshevSharp is feature-complete against PyChebyshev v0.20.1 modulo the deliberately skipped matplotlib plotting helpers.
```

- [ ] **Step 3: Create `docs/docs/parallel-build.md`**

```markdown
# Parallel Build & Progress Reporting

ChebyshevSharp v0.10.0 adds two ctor-time kwargs to all four interpolant classes:

- `nWorkers` (`int?`): null (sequential, default), `-1` (`Environment.ProcessorCount`), or positive int (`Parallel.For` pool size).
- `progress` (`IProgress<int>?`): per-evaluation cumulative count for Approx/Spline/Slider; per-sweep for TT.

## Thread-safety contract

When `nWorkers` is non-null, the user-supplied function may be invoked concurrently from multiple threads via `Parallel.For`. Functions that capture mutable state must use locks or external synchronization. Functions that are pure (no captured mutable state) need no special handling.

```csharp
// Pure function: thread-safe by construction.
double F(double[] p, object? _) => Math.Sin(p[0]) + Math.Cos(p[1]);
var ap = new ChebyshevApproximation(F, 2,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 32, 32 },
    nWorkers: -1);  // use all cores
ap.Build();
```

```csharp
// Captures shared state — wrap in lock if used with nWorkers != null.
int callCount = 0;
object lockObj = new();
double F2(double[] p, object? _)
{
    lock (lockObj) callCount++;
    return p[0] * p[0];
}
```

## Progress reporting

Progress reports are cumulative integer counts. The total is derivable upfront via the `GetNumEvaluationPoints()` getter (Phase 4) on Approx/Spline/Slider, or `maxSweeps` on TT.

```csharp
var counter = new Progress<int>(n => Console.Write($"\r{n} evaluations done"));
var ap = new ChebyshevApproximation(F, 2, ..., progress: counter);
ap.Build();
```

`progress: null` is a no-op; the parallel path skips the increment entirely.

## TT and `nWorkers`

`ChebyshevTT` accepts `nWorkers` for API symmetry but ignores it: TT-Cross is an adaptive sampling algorithm, not a pre-grid evaluation, so per-grid parallelism does not apply. `progress` on TT fires once per TT-Cross sweep.
```

- [ ] **Step 4: Create `docs/docs/adaptive-refinement.md`**

```markdown
# Adaptive Refinement

ChebyshevSharp v0.10.0 adds three adaptive-refinement APIs derived from PyChebyshev v0.20.0 + v0.20.1.

## ChebyshevSpline.AutoKnots

Auto-place knots at function kinks via a curvature-spike scan. Useful for piecewise-smooth functions like `|x|`, `max(0, x)`, or piecewise polynomials.

```csharp
double F(double[] p, object? _) => Math.Abs(p[0]);
var sp = ChebyshevSpline.AutoKnots(F, 1,
    new[] { new[] { -1.0, 1.0 } },
    new[] { 16 });
// Discovers a knot near x=0; the resulting Spline has 2 pieces.
```

Tuning kwargs: `thresholdFactor` (default 5.0), `maxKnotsPerDim` (default 5), `nScanPoints` (default 200).

## SobolIndices

Variance decomposition from spectral Chebyshev coefficients. No Monte Carlo, no extra evaluations beyond what's already in `TensorValues`.

```csharp
double F(double[] p, object? _) => Math.Sin(p[0]) + p[1] * p[2];
var ap = new ChebyshevApproximation(F, 3, ..., new[] { 16, 16, 16 });
ap.Build();
SobolResult s = ap.SobolIndices();
Console.WriteLine($"FirstOrder: [{string.Join(", ", s.FirstOrder)}]");
Console.WriteLine($"TotalOrder: [{string.Join(", ", s.TotalOrder)}]");
Console.WriteLine($"Variance: {s.Variance}");
```

`s.Variance == 0` indicates a constant function — the indices are zero and meaningless. For `ChebyshevSpline`, indices are aggregated across pieces (volume-weighted variance).

## ChebyshevTT.WithAutoOrder + Reorder

TT compression rank depends on dim order; some functions admit much lower-rank TTs under a non-canonical permutation.

```csharp
double F(double[] p) => Math.Sin(p[0] * p[2]) + Math.Cos(p[1]);
var tt = ChebyshevTT.WithAutoOrder(F, 3,
    new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    new[] { 16, 16, 16 },
    nTrials: 5, method: "greedy_swap");
Console.WriteLine($"DimOrder: [{string.Join(", ", tt.DimOrder)}]");
// Eval/Slice/Extrude/etc. transparently remap user coordinates by tt.DimOrder.
double v = tt.Eval(new[] { 0.3, -0.4, 0.5 });

// Manual realignment to a different permutation:
var realigned = tt.Reorder(new[] { 1, 2, 0 }, maxRank: 16, tolerance: 1e-10);
```

Binary algebra (`+`, `-`) between TTs requires matching `DimOrder`; call `Reorder` on one operand first if they differ.
```

- [ ] **Step 5: Update `docs/docs/toc.yml`**

In `docs/docs/toc.yml`, add entries for the two new pages near related pages (e.g., `parallel-build.md` near `performance.md`, `adaptive-refinement.md` near `calculus.md`):

```yaml
- name: Parallel Build & Progress
  href: parallel-build.md
- name: Adaptive Refinement
  href: adaptive-refinement.md
```

- [ ] **Step 6: Update `skip_csharp.txt`**

Append:

```
# Phase 6 (v0.10.0) — Build perf + adaptive refinement (PyChebyshev v0.19+v0.20.0+v0.20.1 parity)
# All ~67 PyChebyshev tests in the v0.19/v0.20.0/v0.20.1 windows ported as:
#   - tests/BuildPerfTests.cs (17 tests)
#   - tests/SobolIndicesTests.cs (12 tests)
#   - tests/AutoKnotsTests.cs (10 tests)
#   - tests/TtAutoOrderTests.cs (18 tests; combines TtSwapAdjacent + Reorder + WithAutoOrder + JSON migration)
#   - tests/TtDimOrderTests.cs (10 tests for full _dimOrder threading surface)
# Skipped (Option C from master spec): plot_convergence, plot_1d, plot_2d_surface, plot_2d_contour (matplotlib helpers).
```

- [ ] **Step 7: Update `CLAUDE.md`**

In the `## Status` block of `CLAUDE.md`, replace the existing entry to reflect Phase 6 completion. Change `946/946 passing` (or whatever the current number is) to `1017/1017 passing`. Add a sentence to the v0.9.0/v0.10.0 narrative:

```
v0.10.0 (Phase 6) ships build perf (nWorkers + IProgress<int>) and adaptive
refinement (AutoKnots, SobolIndices, ChebyshevTT.WithAutoOrder/Reorder/DimOrder
with full _dimOrder threading; new SobolResult record). PyChebyshev parity
tag advances 0.17.0 → 0.20.1, matching the bundled v0.19+v0.20.0+v0.20.1
upstream window. With Phase 6 complete, ChebyshevSharp is feature-complete
against PyChebyshev v0.20.1 (modulo deliberately skipped matplotlib helpers,
Option C). 6 of 6 phases complete — port complete.
`dotnet test` runs **1017/1017** passing.
```

- [ ] **Step 8: Verify build is clean and tests pass**

```bash
dotnet build && dotnet test
```

Expected: 1017 tests passed, build with 0 warnings.

- [ ] **Step 9: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSharp.csproj docs/docs/changelog.md docs/docs/parallel-build.md docs/docs/adaptive-refinement.md docs/docs/toc.yml skip_csharp.txt CLAUDE.md
git commit -m "phase6: docs, csproj, parity metadata, release prep for v0.10.0

- csproj: <Version> 0.9.0 -> 0.10.0; <PyChebyshevParity> 0.17.0 -> 0.20.1;
  <InformationalVersion> 0.10.0+pychebyshev.0.20.1
- changelog: v0.10.0 entry, two-tier convention; explicit notes on Option C
  matplotlib skip and on the parity-tag advance from 0.17.0 to 0.20.1
- docs/docs/parallel-build.md: nWorkers + IProgress + thread-safety contract examples
- docs/docs/adaptive-refinement.md: AutoKnots + SobolIndices + WithAutoOrder examples
- docs/docs/toc.yml: both new pages added
- skip_csharp.txt: Phase 6 entry; matplotlib-skip noted (Option C)
- CLAUDE.md: status block bumped (946 -> 1017; phases 1-6 of 6 complete; port complete)
- No new tests"
```

- [ ] **Step 10: Final verification**

```bash
git log --oneline | head -15
dotnet test
```

Expected: 12 commits prefixed `phase6:` since baseline. 1017 tests passing.

Return control to the user. Do not auto-create the PR or auto-merge — that's a user-confirmation gate (per Phases 3, 4, 5 pattern).

---

## Self-Review Checklist (for the writing-plans author, NOT the implementer)

After writing this plan, the writing-plans skill self-review confirmed:

**Spec coverage:**

- D1 (Parallel.For + thread-safety contract) → Tasks 2, 3, 4 implement and document.
- D2 (cumulative IProgress<int> via Interlocked.Increment) → Task 2 implements; Task 3 + 4 verify cross-class.
- D3 (SobolResult is a public record with Variance) → Task 1 stub; Task 6 populates; Task 7 reuses.
- D4 (AutoKnots defaults match Python: 5.0, 5, 200) → Task 8 signature.
- D5 (WithAutoOrder takes optional seed; ignored for greedy_swap) → Task 10 signature + tests.
- D6 (`_dimOrder` field initialized identity in entry-point ctors/factories; factory bypasses derive from source) → Task 9 ctor + Reorder; Task 11 threading rules.
- D7 (binary algebra requires matching `_dimOrder`; throws with Reorder hint on mismatch) → Task 11 Step 7.
- D8 (JSON v1→v2 migration; backfill identity on Load) → Task 10 implementation + v0.9.0 fixture test.
- D9 (single PR, ~67 tests; single submodule hop v0.18→v0.20.1) → Task 1 hop + Task 12 release prep.
- D10 (TT skips nWorkers parallelism; progress fires per-sweep) → Task 4 Step 5 + tests.

**Placeholder scan:** No "TBD", "implement later", "similar to Task N", or unspecified code blocks. Every test class has full code; every implementation step has full code. Two minor caveats:

1. Task 4 Step 3 says "Locate the `Build()` method (and any internal piece-construction site — look for `new ChebyshevApproximation(...)` inside Spline)" — this is a guided pattern-match the implementer must perform; the actual line numbers are stable in the v0.9.0 source but vary slightly with how Build() is laid out. This is acceptable per Phase 4/5 precedent (each phase's plan has 1-2 such "locate the call site" steps).
2. Task 10 Step 4 documents two paths for generating the v0.9.0 fixture file (a temporary `git checkout v0.9.0 -- src/ChebyshevSharp/ChebyshevTT.cs` followed by save, or a stripped JSON post-processed from a Phase 6 save). The implementer chooses the cleaner path; the test verifies the result regardless.

**Type consistency:**

- `IProgress<int>` parameter type used uniformly across all four ctors and `ParallelBuild.EvaluateInParallel`.
- `int?` for `nWorkers` consistent across signatures.
- `SobolResult(double[] FirstOrder, double[] TotalOrder, double Variance)` consistent across Sensitivity helper, Approx.SobolIndices, Spline.SobolIndices.
- `_dimOrder` field type `int[]` consistent across ctor init, `DimOrder` property, `Reorder`, `WithAutoOrder`, threading sites.
- `TtCore[]` and `TtCore.Copy()` usage consistent in `TtSwapAdjacent`, `Reorder`, `BuildResultFromCores`.
- `SobolResult` and `SobolIndices()` method names consistent across Approx and Spline.
- `WithAutoOrder` method ordering: `function, numDimensions, domain, numNodes, maxRank?, tolerance?, maxSweeps?, ...` matches `ChebyshevTT` ctor's argument order.

**Worktree enforcement:** Every task starts with the same WORKTREE ENFORCEMENT block.

**Test count progression:** Task headers and final summary table align: 950 + 0 + 3 + 5 + 6 + 3 + 8 + 4 + 10 + 8 + 10 + 10 + 0 = 1017. ✓

**Cross-class API symmetry:** Tasks 3 and 4 collectively add the same two ctor kwargs (`nWorkers`, `progress`) at the end of every public ctor signature for all four classes. Test names follow consistent patterns (`Test_<class>_parallel_matches_sequential`, `Test_<class>_progress_count_*`). Validation on `nWorkers` is centralized in `ParallelBuild.NormalizeNWorkers` and tested through every public ctor in Task 5. No drift.

**TT _dimOrder threading completeness:** Task 11 covers `Eval`, `EvalBatch`, `EvalMulti`, `Slice`, `Extrude`, `ToDense`, partial `Integrate`, unary algebra (`-`, `*`, `/`, in-place variants), binary algebra (`+`, `-`). The `Save`/`Load` are covered in Task 10. The `WithAutoOrder` and `Reorder` factory methods produce non-identity `_dimOrder` results in Tasks 9 + 10. Helper `IsIdentityDimOrder` short-circuits the identity case for performance.

**JSON v1→v2 migration:** Task 10 includes both an automated round-trip test (Save in v2 + Load preserves `dimOrder`) AND a fixture-file test (load a hand-committed v0.9.0 JSON file lacking the new fields, verify identity backfill). The fixture file generation is documented in Step 4 of Task 10 with a one-off generator `[Fact(Skip = "...")]` test that's deleted after the fixture is created.
