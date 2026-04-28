# Phase 1: Adaptive Constructor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `ChebyshevApproximation` and `ChebyshevSpline` to PyChebyshev v0.12.0 parity by adding error-driven auto-N construction (`errorThreshold`, `maxN`, nullable `nNodes`), the `GetOptimalN1` static helper, per-sub-interval nested `nNodes` for splines, and a `ChebyshevSpline.WithSpecialPoints` factory.

**Architecture:** The existing `Build()` method is split into a public dispatcher and two private workers: `BuildFixedGrid()` (the existing logic, unchanged) and `BuildWithThreshold()` (a new doubling loop in `Internal/AdaptiveBuild.cs`). The doubling loop picks the worst-contributing dim each iteration via a new `ErrorEstimatePerDim()` internal helper, and is capped at `maxN`. The user's *original* `nNodes` (with `null` sentinels intact) is preserved in `_originalNNodes` so a second `Build()` after tightening `errorThreshold` correctly re-runs the loop. JSON `Load()` backfills `_originalNNodes` from `nNodes` for pre-v0.5.0 files.

For Spline, the same threshold/maxN are threaded per-piece, and `nNodes` accepts a nested `int[][]` (per-dim, per-piece) form. Python's `ChebyshevApproximation(special_points=...)` returning a `ChebyshevSpline` is **not** mirrored — C# constructors cannot return a different type. Instead, a new `ChebyshevSpline.WithSpecialPoints(...)` static factory is the C#-idiomatic entry point. The `specialPoints` kwarg is intentionally absent from the `ChebyshevApproximation` ctor.

**Tech Stack:** C# 13 / .NET 8 + .NET 10 multi-target; xUnit; existing `BarycentricKernel.ChebyshevCoefficients1D` (DCT-II) for per-dim error estimation.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ChebyshevSharp/ChebyshevApproximation.cs` | Modify | Add ctor overload accepting `int?[]?`, `errorThreshold`, `maxN`. Add `GetErrorThreshold()`, `GetOptimalN1()`. Refactor `Build()` into a dispatcher. Add `_originalNNodes` field with JSON migration. Add `BuildWarning` property. |
| `src/ChebyshevSharp/Internal/AdaptiveBuild.cs` | Create | Doubling loop driver. Pure-static helper consumed by `ChebyshevApproximation.Build()`. |
| `src/ChebyshevSharp/ChebyshevSpline.cs` | Modify | Add ctor overload accepting `int?[]?`/nested `int[][]?` `nNodes`, optional `knots`, `errorThreshold`, `maxN`. Add `GetErrorThreshold()`. Add `static WithSpecialPoints(...)` factory. Thread threshold per-piece in `Build()`. |
| `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs` | Create | Port of Python `test_error_threshold.py` — ~30 tests across `ConstructorValidation`, `DoublingLoop`, `MaxNCap`, `GetErrorThreshold`, `GetOptimalN1`, `SplineErrorThreshold`. |
| `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs` | Create | Port of Python `test_special_points.py` — ~15 tests across `Validation`, `Accuracy1D`, `Accuracy2D`, `CrossFeature` (factory pattern, not dispatch). |
| `tests/ChebyshevSharp.Tests/Helpers/TestFixtures.cs` | Modify | Add small fixtures used across the new test files (e.g., `Sin2D`, `Sin3D`, `Abs1D`). Reuse existing `AssertClose`. |
| `src/ChebyshevSharp/ChebyshevSharp.csproj` | Modify | Bump `<Version>` to 0.5.0; add `<PyChebyshevParity>0.12.0</PyChebyshevParity>`; surface in `<Description>`. |
| `docs/docs/changelog.md` | Modify | Add v0.5.0 entry leading with parity claim. |
| `docs/docs/error-driven-construction.md` | Create | New user-guide page (motivation, algorithm, examples). |
| `docs/docs/special-points.md` | Create | New user-guide page (kink declaration, factory pattern). |
| `docs/docs/toc.yml` | Modify | Add new doc pages. |
| `skip_csharp.txt` | Modify | Update phase tracker with new test count. |
| `README.md` | Modify | Add/update PyChebyshev parity badge. |
| `CLAUDE.md` | Modify | Add new public API surface notes (`GetOptimalN1`, `WithSpecialPoints`, error-driven build). |

---

## Task 1: Submodule advance + project scaffolding

**Files:**
- Modify: `ref/PyChebyshev` (submodule pin)
- Create: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs` (empty stub)
- Create: `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs` (empty stub)

- [ ] **Step 1: Advance the PyChebyshev submodule to v0.12.0**

```bash
git -C ref/PyChebyshev fetch --tags origin
git -C ref/PyChebyshev checkout v0.12.0
git add ref/PyChebyshev
```

- [ ] **Step 2: Verify submodule state**

```bash
git -C ref/PyChebyshev describe --tags
```

Expected output: `v0.12.0`

- [ ] **Step 3: Create empty test file stubs to anchor `using` lines**

Create `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_error_threshold.py (PyChebyshev v0.11)
// Tests added incrementally across Phase 1 tasks.
public class ErrorThresholdTests
{
}
```

Create `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`:

```csharp
using System;
using Xunit;
using ChebyshevSharp;
using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

// Port of ref/PyChebyshev/tests/test_special_points.py (PyChebyshev v0.12)
// Tests added incrementally across Phase 1 tasks.
public class SpecialPointsTests
{
}
```

- [ ] **Step 4: Build and verify the empty test files compile**

Run: `dotnet build`
Expected: succeeds with zero new warnings; existing 613 tests still found.

- [ ] **Step 5: Run full existing test suite to confirm baseline**

Run: `dotnet test`
Expected: `Passed: 613`. No regressions.

- [ ] **Step 6: Commit**

```bash
git add ref/PyChebyshev tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs tests/ChebyshevSharp.Tests/SpecialPointsTests.cs
git commit -m "phase1: advance submodule to v0.12.0 and add test stubs"
```

---

## Task 2: Add `errorThreshold` / `maxN` / nullable `nNodes` to ChebyshevApproximation ctor (validation only — no doubling loop yet)

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs:60-79` (constructor)
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs:24` (`NNodes` becomes `int[]?` storage but exposed as `int[]`; add `int?[]` view)
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

**Design notes:**
- C# does not have Python's `None` in arrays. We use `int?[]` for the user-facing nullable form internally and project to `int[]` once resolved. Public `NNodes` keeps `int[]` post-resolve.
- New properties:
  - `public double? ErrorThreshold { get; internal set; }` — the threshold (null in fixed-N mode).
  - `public int MaxN { get; internal set; } = 64` — cap for the doubling loop.
  - `internal int?[] _OriginalNNodes` — preserves the user's intent (null sentinels intact).
- New backing storage `_resolvedNNodes` (int[]) replaces direct `NNodes` writes — but this task only adds the surface, no doubling. We resolve trivially: if all entries non-null, copy; otherwise leave `NNodes` as a placeholder ` []` until `Build()` resolves.

- [ ] **Step 1: Write failing tests for ctor validation**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class ConstructorValidation
{
    private static readonly Func<double[], object?, double> Sin2D = (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]);
    private static readonly double[][] UnitSq = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };

    [Fact]
    public void Test_explicit_n_unchanged()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, new[] { 11, 11 });
        Assert.Equal(new[] { 11, 11 }, cheb.NNodes);
        Assert.Null(cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_error_threshold_only()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(1e-6, cheb.ErrorThreshold);
    }

    [Fact]
    public void Test_neither_n_nor_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: null));
    }

    [Fact]
    public void Test_none_without_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: new int?[] { null, 11 }, errorThreshold: null));
    }

    [Fact]
    public void Test_max_n_default()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        Assert.Equal(64, cheb.MaxN);
    }

    [Fact]
    public void Test_max_n_custom()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 128);
        Assert.Equal(128, cheb.MaxN);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(-1)]
    public void Test_max_n_below_minimum_raises(int badMaxN)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: badMaxN));
        Assert.Contains("maxN must be at least 3", ex.Message);
    }

    [Fact]
    public void Test_max_n_equal_to_minimum_accepted()
    {
        var cheb = new ChebyshevApproximation(Sin2D, 2, UnitSq, nNodes: null, errorThreshold: 1e-6, maxN: 3);
        Assert.Equal(3, cheb.MaxN);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run: `dotnet test --filter "FullyQualifiedName~ErrorThresholdTests"`
Expected: build fails — `ChebyshevApproximation` has no overload taking `int?[]`/`errorThreshold`/`maxN`.

- [ ] **Step 3: Add new overload + properties to ChebyshevApproximation**

Modify `src/ChebyshevSharp/ChebyshevApproximation.cs`. After line 48 (the `NEvaluations` property) add:

```csharp
    /// <summary>Target supremum-norm error for auto-N construction. Null in fixed-N mode.</summary>
    public double? ErrorThreshold { get; internal set; }

    /// <summary>Maximum nodes per dimension for the auto-N doubling loop. Default 64.</summary>
    public int MaxN { get; internal set; } = 64;

    /// <summary>Warning emitted by Build() if maxN was reached before errorThreshold was satisfied. Null otherwise.</summary>
    public string? BuildWarning { get; internal set; }

    /// <summary>The user's original nNodes argument with null sentinels intact, used to dispatch a re-run of the doubling loop on a second Build() call.</summary>
    internal int?[] OriginalNNodes { get; set; } = Array.Empty<int?>();
```

After the existing constructor (line 79), add a new overload:

```csharp
    /// <summary>
    /// Create a new ChebyshevApproximation with optional error-driven auto-N construction.
    /// </summary>
    /// <param name="function">Function to approximate: f(point, data) -&gt; double.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds for each dimension as double[ndim][2].</param>
    /// <param name="nNodes">Number of Chebyshev nodes per dimension; null entries signal auto-N for that dim. Pass null to make every dim auto-N (requires errorThreshold).</param>
    /// <param name="errorThreshold">Target supremum-norm error. Required if any nNodes entry is null.</param>
    /// <param name="maxN">Cap on nodes per dimension during the doubling loop (default 64, must be at least 3).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public ChebyshevApproximation(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int?[]? nNodes = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2)
    {
        if (maxN < 3)
            throw new ArgumentException(
                $"maxN must be at least 3 (the initial N of the doubling loop), got maxN={maxN}. " +
                "For a grid smaller than 3 per dimension, pass nNodes explicitly.");

        // Normalize nNodes: null array means "all dims auto-N"
        int?[] resolved;
        if (nNodes == null)
        {
            if (errorThreshold == null)
                throw new ArgumentException(
                    "Must provide either nNodes (explicit) or errorThreshold (auto-N). Got neither.");
            resolved = new int?[numDimensions];
        }
        else
        {
            resolved = (int?[])nNodes.Clone();
            if (resolved.Any(n => n == null) && errorThreshold == null)
                throw new ArgumentException(
                    "Null entries in nNodes require errorThreshold to be set (auto-N mode).");
        }

        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        ErrorThreshold = errorThreshold;
        MaxN = maxN;
        MaxDerivativeOrder = maxDerivativeOrder;
        OriginalNNodes = (int?[])resolved.Clone();

        // If all entries are non-null, populate NNodes + nodes immediately (matches existing fixed-N behavior).
        if (resolved.All(n => n != null))
        {
            NNodes = resolved.Select(n => n!.Value).ToArray();
            NodeArrays = new double[numDimensions][];
            for (int d = 0; d < numDimensions; d++)
                NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], NNodes[d]);
        }
        else
        {
            // Auto-N path: NNodes left empty until Build() resolves.
            NNodes = Array.Empty<int>();
            NodeArrays = Array.Empty<double[]>();
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~ErrorThresholdTests"`
Expected: 9 tests pass (the 9 from Step 1).

- [ ] **Step 5: Run full suite to verify no regression**

Run: `dotnet test`
Expected: `Passed: 622` (613 existing + 9 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: add errorThreshold/maxN/nullable nNodes ctor overload (validation only)"
```

---

## Task 3: Refactor `Build()` into `BuildFixedGrid()` (no behavior change)

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs:88-...` (the existing Build method)

**Design note:** This task is a pure refactor. The body of the existing `Build()` becomes a private `BuildFixedGrid()` method. `Build()` becomes a dispatcher stub that always calls `BuildFixedGrid()` for now. The doubling loop branch is wired in Task 5. All existing tests must still pass.

- [ ] **Step 1: No new tests needed — existing tests verify Build() unchanged**

The 613 existing tests all exercise `Build()` and must continue to pass.

- [ ] **Step 2: Run baseline**

Run: `dotnet test`
Expected: `Passed: 622`.

- [ ] **Step 3: Refactor Build() into a dispatcher + BuildFixedGrid()**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, replace the existing `public void Build(bool verbose = true)` body with:

```csharp
    /// <summary>
    /// Build the Chebyshev approximation. Dispatches to the doubling loop if any
    /// dimension was constructed with a null entry in nNodes (auto-N), otherwise
    /// builds on the resolved fixed grid.
    /// </summary>
    /// <param name="verbose">If true, print build progress.</param>
    public void Build(bool verbose = true)
    {
        if (Function == null)
            throw new InvalidOperationException(
                "Cannot build: no function assigned. " +
                "This object was created via FromValues() or Load().");

        if (OriginalNNodes.Length > 0 && OriginalNNodes.Any(n => n == null))
        {
            // Doubling loop wired in Task 5.
            throw new NotImplementedException("Auto-N build not yet implemented (Task 5).");
        }

        BuildFixedGrid(verbose);
    }

    /// <summary>
    /// Build on the already-resolved (all-int) grid. The original Build() body,
    /// extracted so the doubling loop can call it once per iteration.
    /// </summary>
    internal void BuildFixedGrid(bool verbose = true)
    {
        // [PASTE THE EXISTING Build() BODY HERE — lines 95 through the end of the
        // existing method, starting with `int total = 1;` through the final
        // closing brace of the original method. Do not change any logic.]
    }
```

The intent: take the original method body (everything inside the curly braces of the existing `public void Build()`) and move it verbatim into `BuildFixedGrid()`. The new `Build()` only contains the dispatcher logic above.

- [ ] **Step 4: Run full suite to verify no regression**

Run: `dotnet test`
Expected: `Passed: 622`. The refactor is behavior-preserving for all fixed-N callers.

- [ ] **Step 5: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs
git commit -m "phase1: refactor Build() into dispatcher + BuildFixedGrid() (no behavior change)"
```

---

## Task 4: Add `ErrorEstimatePerDim()` internal helper (refactor existing ErrorEstimate to use it)

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs:422-455` (ErrorEstimate)
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

**Design note:** The doubling loop needs per-dimension error estimates to pick the worst-contributing dim. We expose this internally as `ErrorEstimatePerDim()` returning `double[]` (one entry per dim) and refactor `ErrorEstimate()` to be `ErrorEstimatePerDim().Sum()`.

- [ ] **Step 1: Write failing tests for ErrorEstimatePerDim**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class ErrorEstimatePerDimTests
{
    [Fact]
    public void Test_per_dim_returns_one_entry_per_dimension()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 });
        cheb.Build(verbose: false);
        double[] perDim = cheb.ErrorEstimatePerDim();
        Assert.Equal(2, perDim.Length);
        Assert.All(perDim, e => Assert.True(e >= 0.0));
    }

    [Fact]
    public void Test_per_dim_sum_equals_error_estimate()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 9, 11, 13 });
        cheb.Build(verbose: false);
        double[] perDim = cheb.ErrorEstimatePerDim();
        double total = perDim.Sum();
        Assert.Equal(cheb.ErrorEstimate(), total, precision: 14);
    }

    [Fact]
    public void Test_per_dim_throws_before_build()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0],
            1, new[] { new[] { -1.0, 1.0 } },
            new[] { 5 });
        Assert.Throws<InvalidOperationException>(() => cheb.ErrorEstimatePerDim());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~ErrorEstimatePerDimTests"`
Expected: build fails — `ErrorEstimatePerDim` is not defined.

- [ ] **Step 3: Add `ErrorEstimatePerDim` and refactor `ErrorEstimate`**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, add this method just above the existing `ErrorEstimate()`:

```csharp
    /// <summary>
    /// Compute per-dimension max last-coefficient magnitudes.
    /// Returns one entry per dimension; ErrorEstimate() returns the sum.
    /// Used by the auto-N doubling loop to pick the worst-contributing dim.
    /// </summary>
    /// <returns>Per-dimension last-coefficient magnitudes, one entry per dim.</returns>
    public double[] ErrorEstimatePerDim()
    {
        if (TensorValues == null)
            throw new InvalidOperationException("Call Build() first");

        var perDim = new double[NumDimensions];
        for (int d = 0; d < NumDimensions; d++)
        {
            double maxErrThisDim = 0.0;
            int[] otherShape = NNodes.Where((_, i) => i != d).ToArray();
            int otherTotal = 1;
            for (int i = 0; i < otherShape.Length; i++)
                otherTotal *= otherShape[i];

            for (int otherFlat = 0; otherFlat < otherTotal; otherFlat++)
            {
                double[] values1d = Extract1DSlice(TensorValues, NNodes, d, otherFlat, otherShape);
                double[] coeffs = BarycentricKernel.ChebyshevCoefficients1D(values1d);
                double lastCoeff = Math.Abs(coeffs[^1]);
                if (lastCoeff > maxErrThisDim)
                    maxErrThisDim = lastCoeff;
            }
            perDim[d] = maxErrThisDim;
        }
        return perDim;
    }
```

Replace the body of the existing `ErrorEstimate()` (lines 422-455) with:

```csharp
    /// <summary>
    /// Estimate the supremum-norm interpolation error.
    /// Sums per-dimension max last-coefficient magnitudes.
    /// </summary>
    /// <returns>Estimated maximum interpolation error.</returns>
    public double ErrorEstimate()
    {
        if (_cachedErrorEstimate.HasValue)
            return _cachedErrorEstimate.Value;
        double total = ErrorEstimatePerDim().Sum();
        _cachedErrorEstimate = total;
        return total;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~ErrorEstimatePerDimTests"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full suite to verify ErrorEstimate refactor didn't regress**

Run: `dotnet test`
Expected: `Passed: 625` (622 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: extract ErrorEstimatePerDim() helper (used by doubling loop)"
```

---

## Task 5: Implement the doubling loop (`Internal/AdaptiveBuild.cs`)

**Files:**
- Create: `src/ChebyshevSharp/Internal/AdaptiveBuild.cs`
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (Build() dispatcher to call it)
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

**Design notes:**
- `AdaptiveBuild.RunDoublingLoop(approx, verbose)` is a static method that:
  - Starts with `current[d] = OriginalNNodes[d] ?? 3` for each dim
  - Loops:
    - Set `approx.NNodes = current`, regenerate `NodeArrays`, call `BuildFixedGrid(verbose: false)`
    - Compute `perDim = ErrorEstimatePerDim()`, total = `perDim.Sum()`
    - If verbose: `Console.WriteLine($"[auto-N] nNodes={...}, error={total:e3}");`
    - Accumulate `totalEvals += approx.NEvaluations` and `totalBuildTime += approx.BuildTime`
    - If `total <= ErrorThreshold`, break
    - Pick worst auto-dim with `current[d] < MaxN`; if none, set `BuildWarning` and break
    - `current[worstDim] = Math.Min(2 * current[worstDim], MaxN)`
  - At end: write `approx.NEvaluations = totalEvals`, `approx.BuildTime = totalBuildTime`
- Auto dims = indices `i` where `OriginalNNodes[i] == null`.
- Tie-breaking on per-dim error: largest first; ties → lowest index (matches Python).
- `BuildWarning` is the C# replacement for Python's `warnings.warn(RuntimeWarning, ...)`.

- [ ] **Step 1: Write failing doubling-loop tests**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class DoublingLoopTests
{
    private static readonly double[][] UnitSq = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };

    [Fact]
    public void Test_1d_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-8);
        cheb.Build(verbose: false);
        Assert.True(cheb.NNodes[0] <= 64);
        Assert.True(cheb.ErrorEstimate() <= 1e-8);
    }

    [Fact]
    public void Test_2d_auto_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_3d_auto_converges()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_semi_variable_respects_fixed_dims()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]),
            3, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null, 15, 15 }, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(15, cheb.NNodes[1]);
        Assert.Equal(15, cheb.NNodes[2]);
    }

    [Fact]
    public void Test_already_accurate_stops_immediately()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => x[0] + x[1],  // linear; exact at N=3
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(3, cheb.NNodes[0]);
        Assert.Equal(3, cheb.NNodes[1]);
    }

    [Fact]
    public void Test_tight_threshold_eventual()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Exp(-x[0] * x[0]),
            1, new[] { new[] { -2.0, 2.0 } },
            nNodes: null, errorThreshold: 1e-12);
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-12);
    }

    [Fact]
    public void Test_max_n_cap_emits_warning_and_remains_usable()
    {
        // sin(20x) + cos(17x): non-aliased oscillation, can't satisfy 1e-12 with N=16.
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(20 * x[0]) + Math.Cos(17 * x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-12, maxN: 16);
        cheb.Build(verbose: false);

        Assert.NotNull(cheb.BuildWarning);
        Assert.Contains("maxN", cheb.BuildWarning);
        Assert.True(cheb.NNodes[0] <= 16);
        // Object still usable — eval returns finite value
        double v = cheb.VectorizedEval(new[] { 0.1 }, new[] { 0 });
        Assert.True(double.IsFinite(v));
    }

    [Fact]
    public void Test_no_warning_when_threshold_met()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, UnitSq, nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Null(cheb.BuildWarning);
    }

    [Fact]
    public void Test_rebuild_with_tighter_threshold_rebuilds_auto_dims()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-4);
        cheb.Build(verbose: false);
        int nFirst = cheb.NNodes[0];
        Assert.True(cheb.ErrorEstimate() <= 1e-4);

        cheb.ErrorThreshold = 1e-10;
        cheb.Build(verbose: false);
        Assert.True(cheb.ErrorEstimate() <= 1e-10);
        Assert.True(cheb.NNodes[0] >= nFirst);
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~DoublingLoopTests"`
Expected: 9 tests fail with `NotImplementedException` from the Build() dispatcher stub from Task 3.

- [ ] **Step 3: Create AdaptiveBuild.cs**

Create `src/ChebyshevSharp/Internal/AdaptiveBuild.cs`:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Doubling-loop driver for ChebyshevApproximation auto-N construction.
/// Iteratively grows the worst-contributing auto-dim until ErrorEstimate is
/// below ErrorThreshold or every auto-dim has hit MaxN.
/// </summary>
internal static class AdaptiveBuild
{
    /// <summary>
    /// Run the doubling loop on an approximation that has at least one null
    /// entry in OriginalNNodes. On return, the approximation is fully built
    /// (TensorValues populated, NNodes resolved to ints, NEvaluations and
    /// BuildTime accumulated across all iterations). If MaxN is reached
    /// before the threshold is satisfied, BuildWarning is set.
    /// </summary>
    public static void RunDoublingLoop(ChebyshevApproximation approx, bool verbose)
    {
        if (approx.ErrorThreshold == null)
            throw new InvalidOperationException("RunDoublingLoop requires ErrorThreshold to be set.");
        if (!approx.OriginalNNodes.Any(n => n == null))
            throw new InvalidOperationException("RunDoublingLoop requires at least one null entry in OriginalNNodes.");

        double threshold = approx.ErrorThreshold.Value;
        int maxN = approx.MaxN;
        int numDim = approx.NumDimensions;

        // Resolve: ints stay; nulls start at 3
        var current = new int[numDim];
        for (int d = 0; d < numDim; d++)
            current[d] = approx.OriginalNNodes[d] ?? 3;

        // Auto-dim indices (where OriginalNNodes[d] == null)
        var autoDims = Enumerable.Range(0, numDim).Where(i => approx.OriginalNNodes[i] == null).ToArray();

        int totalEvals = 0;
        double totalBuildTime = 0.0;
        approx.BuildWarning = null;

        while (true)
        {
            // Apply current grid
            approx.NNodes = (int[])current.Clone();
            approx.NodeArrays = new double[numDim][];
            for (int d = 0; d < numDim; d++)
                approx.NodeArrays[d] = BarycentricKernel.MakeNodesForDim(
                    approx.Domain[d][0], approx.Domain[d][1], current[d]);

            approx.BuildFixedGrid(verbose: false);
            totalEvals += approx.NEvaluations;
            totalBuildTime += approx.BuildTime;

            double[] perDim = approx.ErrorEstimatePerDim();
            double err = perDim.Sum();
            // Seed cache so a public ErrorEstimate() call after build hits cache.
            approx.SetCachedErrorEstimate(err);

            if (verbose)
                Console.WriteLine($"[auto-N] nNodes=[{string.Join(", ", current)}], error={err:e3}");

            if (err <= threshold)
                break;

            // Pick the worst auto-dim not at maxN. Tie: lowest index first.
            int worstDim = -1;
            double worstErr = -1.0;
            foreach (int d in autoDims)
            {
                if (current[d] >= maxN) continue;
                if (perDim[d] > worstErr)
                {
                    worstErr = perDim[d];
                    worstDim = d;
                }
            }

            if (worstDim < 0)
            {
                approx.BuildWarning =
                    $"maxN={maxN} reached on all auto dims before errorThreshold={threshold:e2} satisfied " +
                    $"(last error={err:e3}). Increase maxN or relax errorThreshold.";
                break;
            }

            current[worstDim] = Math.Min(2 * current[worstDim], maxN);
        }

        approx.NEvaluations = totalEvals;
        approx.BuildTime = totalBuildTime;
    }
}
```

- [ ] **Step 4: Wire AdaptiveBuild into Build() dispatcher; expose `SetCachedErrorEstimate`**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, replace the `throw new NotImplementedException(...)` line in `Build()` (added in Task 3) with:

```csharp
            ChebyshevSharp.Internal.AdaptiveBuild.RunDoublingLoop(this, verbose);
            return;
```

Below the existing `_cachedErrorEstimate` field, add an internal setter the doubling loop can call:

```csharp
    /// <summary>Internal hook for AdaptiveBuild to seed the error-estimate cache after each iteration.</summary>
    internal void SetCachedErrorEstimate(double value) => _cachedErrorEstimate = value;
```

- [ ] **Step 5: Run tests**

Run: `dotnet test --filter "FullyQualifiedName~DoublingLoopTests"`
Expected: 9 tests pass.

- [ ] **Step 6: Run full suite**

Run: `dotnet test`
Expected: `Passed: 634` (625 + 9 new).

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/AdaptiveBuild.cs src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: implement auto-N doubling loop (Internal/AdaptiveBuild.cs)"
```

---

## Task 6: Add `GetErrorThreshold()` accessor + `GetOptimalN1()` static helper

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs`
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class GetErrorThresholdTests
{
    [Fact]
    public void Test_returns_threshold_when_set()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, errorThreshold: 1e-6);
        cheb.Build(verbose: false);
        Assert.Equal(1e-6, cheb.GetErrorThreshold());
    }

    [Fact]
    public void Test_returns_null_when_not_set()
    {
        var cheb = new ChebyshevApproximation(
            (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]),
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 11, 11 });
        cheb.Build(verbose: false);
        Assert.Null(cheb.GetErrorThreshold());
    }
}

public class GetOptimalN1Tests
{
    [Fact]
    public void Test_returns_int_above_minimum()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(x[0]),
            (-1.0, 1.0),
            errorThreshold: 1e-8);
        Assert.True(n >= 3 && n <= 64);
    }

    [Fact]
    public void Test_smooth_low_freq_small_n()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => x[0],
            (-1.0, 1.0),
            errorThreshold: 1e-10);
        Assert.Equal(3, n);
    }

    [Fact]
    public void Test_high_freq_larger_n()
    {
        int nLow = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(x[0]) + Math.Cos(x[0]),
            (-1.0, 1.0), errorThreshold: 1e-8);
        int nHigh = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(10 * x[0]) + Math.Cos(10 * x[0]),
            (-1.0, 1.0), errorThreshold: 1e-8);
        Assert.True(nHigh > nLow);
    }

    [Fact]
    public void Test_respects_max_n()
    {
        int n = ChebyshevApproximation.GetOptimalN1(
            (x, _) => Math.Sin(50 * x[0]) + Math.Cos(43 * x[0]),
            (-1.0, 1.0),
            errorThreshold: 1e-14,
            maxN: 8);
        Assert.Equal(8, n);
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~GetErrorThresholdTests|FullyQualifiedName~GetOptimalN1Tests"`
Expected: build fails — `GetErrorThreshold` and `GetOptimalN1` are not defined.

- [ ] **Step 3: Implement both methods**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, add (after the existing `ErrorEstimate()` method):

```csharp
    /// <summary>Return the error threshold passed to the constructor, or null in fixed-N mode.</summary>
    public double? GetErrorThreshold() => ErrorThreshold;

    /// <summary>
    /// 1-D capacity estimator: the smallest N at which a 1-D Chebyshev build
    /// over <paramref name="domain"/> hits <paramref name="errorThreshold"/>.
    /// Useful as a sizing pass before committing to a multi-dimensional build.
    /// </summary>
    /// <param name="function">Function to approximate; signature f(point[1], data) -&gt; double.</param>
    /// <param name="domain">(lo, hi) bounds for the single dimension.</param>
    /// <param name="errorThreshold">Target supremum-norm error.</param>
    /// <param name="maxN">Cap on the returned N. Default 64. If the doubling loop cannot achieve <paramref name="errorThreshold"/> within this cap, returns <paramref name="maxN"/> with BuildWarning set on the temporary internal interpolant.</param>
    /// <returns>Resolved N on the single dimension.</returns>
    public static int GetOptimalN1(
        Func<double[], object?, double> function,
        (double lo, double hi) domain,
        double errorThreshold,
        int maxN = 64)
    {
        var cheb = new ChebyshevApproximation(
            function, 1, new[] { new[] { domain.lo, domain.hi } },
            nNodes: null, errorThreshold: errorThreshold, maxN: maxN);
        cheb.Build(verbose: false);
        return cheb.NNodes[0];
    }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test --filter "FullyQualifiedName~GetErrorThresholdTests|FullyQualifiedName~GetOptimalN1Tests"`
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite**

Run: `dotnet test`
Expected: `Passed: 640` (634 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: add GetErrorThreshold() accessor + GetOptimalN1() static helper"
```

---

## Task 7: JSON migration — `Load()` backfills `OriginalNNodes` for pre-v0.5.0 files

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (Save / Load / SerializationState)
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

**Design notes:**
- Add `OriginalNNodes` to `SerializationState` (nullable `int?[]?`).
- Save: write the field.
- Load: if the field is missing/null in the JSON, backfill from `NNodes` (treating fully-resolved as the user's original intent — pre-v0.5.0 files were always fixed-N).
- Bump `Version` field to `"0.5.0"` on Save.

- [ ] **Step 1: Write failing tests for migration**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class JsonMigrationTests
{
    [Fact]
    public void Test_save_load_roundtrip_auto_n()
    {
        string path = Path.GetTempFileName();
        try
        {
            var cheb = new ChebyshevApproximation(
                (x, _) => Math.Sin(x[0]),
                1, new[] { new[] { -1.0, 1.0 } },
                nNodes: null, errorThreshold: 1e-6);
            cheb.Build(verbose: false);
            cheb.Save(path);

            var loaded = ChebyshevApproximation.Load(path);
            Assert.Equal(cheb.NNodes, loaded.NNodes);
            Assert.Equal(cheb.OriginalNNodes.Length, loaded.OriginalNNodes.Length);
            Assert.Equal(cheb.OriginalNNodes[0], loaded.OriginalNNodes[0]);
            Assert.Equal(cheb.ErrorThreshold, loaded.ErrorThreshold);
            Assert.Equal(cheb.MaxN, loaded.MaxN);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_load_pre_v05_file_backfills_original_n_nodes()
    {
        // Simulate a pre-v0.5.0 JSON file by hand-crafting one without OriginalNNodes / ErrorThreshold / MaxN.
        string path = Path.GetTempFileName();
        try
        {
            // Save a fixed-N file the new way, then strip the new fields by re-serializing without them.
            var cheb = new ChebyshevApproximation(
                (x, _) => x[0] + x[1],
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                new[] { 5, 5 });
            cheb.Build(verbose: false);
            cheb.Save(path);

            string json = File.ReadAllText(path);
            // Remove the v0.5.0 fields to mimic an older file. (Keys must match property names.)
            string oldJson = System.Text.RegularExpressions.Regex.Replace(
                json, @",\s*""(OriginalNNodes|ErrorThreshold|MaxN)""\s*:\s*[^,}]+", "");
            File.WriteAllText(path, oldJson);

            var loaded = ChebyshevApproximation.Load(path);
            // OriginalNNodes backfilled from NNodes (fully-resolved fixed-N intent)
            Assert.Equal(2, loaded.OriginalNNodes.Length);
            Assert.Equal(5, loaded.OriginalNNodes[0]);
            Assert.Equal(5, loaded.OriginalNNodes[1]);
            Assert.Null(loaded.ErrorThreshold);
            Assert.Equal(64, loaded.MaxN);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~JsonMigrationTests"`
Expected: tests fail — Save doesn't write the new fields, Load doesn't backfill.

- [ ] **Step 3: Update `SerializationState` and Save/Load**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, find the `internal class SerializationState` (around line 1149) and add three properties:

```csharp
        public int?[]? OriginalNNodes { get; set; }
        public double? ErrorThreshold { get; set; }
        public int? MaxN { get; set; }
```

In `Save()`, populate the new fields and bump version. Replace:

```csharp
            BuildTime = BuildTime,
            NEvaluations = NEvaluations,
            Version = "0.1.0"
```

with:

```csharp
            BuildTime = BuildTime,
            NEvaluations = NEvaluations,
            OriginalNNodes = OriginalNNodes,
            ErrorThreshold = ErrorThreshold,
            MaxN = MaxN,
            Version = "0.5.0"
```

In `Load()`, after the `state` is deserialized and the `obj` is populated with existing fields, add backfill:

```csharp
        // v0.5.0 migration: OriginalNNodes / ErrorThreshold / MaxN may be absent in older files.
        if (state.OriginalNNodes != null)
            obj.OriginalNNodes = state.OriginalNNodes;
        else
            obj.OriginalNNodes = obj.NNodes.Select(n => (int?)n).ToArray();
        obj.ErrorThreshold = state.ErrorThreshold;
        obj.MaxN = state.MaxN ?? 64;
```

(Place this right before the existing `return obj;` in `Load()`.)

- [ ] **Step 4: Run tests**

Run: `dotnet test --filter "FullyQualifiedName~JsonMigrationTests"`
Expected: 2 tests pass.

- [ ] **Step 5: Run full suite**

Run: `dotnet test`
Expected: `Passed: 642` (640 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: JSON migration — Save/Load OriginalNNodes/ErrorThreshold/MaxN; backfill pre-v0.5.0 files"
```

---

## Task 8: ChebyshevSpline — accept `errorThreshold` / `maxN` per piece + optional `knots`

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Test: `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`

**Design notes:**
- Add a new ctor overload mirroring the Approximation one: `int?[]? nNodes`, `double? errorThreshold`, `int maxN`, optional `double[][]? knots`.
- If `knots == null`, default to `new double[numDimensions][] { Array.Empty<double>(), ... }` (single piece per dim).
- During `Build()`, when constructing each `ChebyshevApproximation` piece, pass through the threshold/maxN. The piece's `nNodes` for the threshold/maxN path is `null` (auto-N within the piece).
- Add a `double? GetErrorThreshold()` accessor and an `ErrorThreshold` property to mirror Approximation.
- Validation: at least one of `nNodes` (with all-int entries) or `errorThreshold` must be supplied. `maxN < 3` raises.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs`:

```csharp
public class SplineErrorThresholdTests
{
    private static readonly Func<double[], object?, double> Sin2D = (x, _) => Math.Sin(x[0]) + Math.Sin(x[1]);

    [Fact]
    public void Test_1d_with_knot()
    {
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null },
            knots: new[] { new[] { 0.0 } },
            errorThreshold: 1e-6);
        spl.Build(verbose: false);
        foreach (var piece in spl.Pieces)
        {
            Assert.NotNull(piece);
            Assert.True(piece!.ErrorEstimate() <= 1e-6);
        }
    }

    [Fact]
    public void Test_2d_no_knots_matches_flat()
    {
        var spl = new ChebyshevSpline(
            Sin2D,
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new int?[] { null, null },
            knots: new[] { Array.Empty<double>(), Array.Empty<double>() },
            errorThreshold: 1e-6);
        spl.Build(verbose: false);
        Assert.Single(spl.Pieces);
        Assert.True(spl.Pieces[0]!.ErrorEstimate() <= 1e-6);
    }

    [Fact]
    public void Test_explicit_n_still_works()
    {
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 15 },
            knots: new[] { new[] { 0.0 } });
        spl.Build(verbose: false);
        foreach (var piece in spl.Pieces)
            Assert.Equal(new[] { 15 }, piece!.NNodes);
    }

    [Fact]
    public void Test_spline_neither_n_nor_threshold_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: (int?[]?)null, knots: null));
    }

    [Fact]
    public void Test_spline_max_n_below_minimum_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodes: null, knots: null, errorThreshold: 1e-6, maxN: 2));
    }

    [Fact]
    public void Test_spline_knots_default_to_empty_per_dim()
    {
        var spl = new ChebyshevSpline(
            Sin2D, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: null, knots: null, errorThreshold: 1e-6);
        Assert.Equal(2, spl.Knots.Length);
        Assert.Empty(spl.Knots[0]);
        Assert.Empty(spl.Knots[1]);
        spl.Build(verbose: false);
        Assert.Single(spl.Pieces);
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~SplineErrorThresholdTests"`
Expected: build fails — Spline has no overload taking nullable nNodes / errorThreshold.

- [ ] **Step 3: Add Spline ctor overload + threshold properties**

In `src/ChebyshevSharp/ChebyshevSpline.cs`, after the existing properties section, add:

```csharp
    /// <summary>Target supremum-norm error for per-piece auto-N construction. Null in fixed-N mode.</summary>
    public double? ErrorThreshold { get; internal set; }

    /// <summary>Maximum nodes per dimension per piece for the auto-N doubling loop. Default 64.</summary>
    public int MaxN { get; internal set; } = 64;

    /// <summary>The user's original nNodes argument with null sentinels intact.</summary>
    internal int?[] OriginalNNodes { get; set; } = Array.Empty<int?>();
```

After the existing constructor (around line 98), add a new overload:

```csharp
    /// <summary>
    /// Create a piecewise Chebyshev spline with optional error-driven auto-N construction.
    /// </summary>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="nNodes">Number of nodes per dimension; null entries signal auto-N. Pass null to make every dim auto-N (requires errorThreshold).</param>
    /// <param name="knots">Interior knots per dimension. Null defaults to empty arrays (single piece per dim).</param>
    /// <param name="errorThreshold">Target supremum-norm error per piece. Required if any nNodes entry is null.</param>
    /// <param name="maxN">Cap on nodes per dimension during the doubling loop (default 64, must be at least 3).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int?[]? nNodes = null,
        double[][]? knots = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2)
    {
        if (maxN < 3)
            throw new ArgumentException(
                $"maxN must be at least 3 (the initial N of the doubling loop), got maxN={maxN}.");

        knots ??= Enumerable.Range(0, numDimensions).Select(_ => Array.Empty<double>()).ToArray();

        // Normalize nNodes
        int?[] resolvedOriginal;
        if (nNodes == null)
        {
            if (errorThreshold == null)
                throw new ArgumentException(
                    "Must provide either nNodes (explicit) or errorThreshold (auto-N). Got neither.");
            resolvedOriginal = new int?[numDimensions];
        }
        else
        {
            resolvedOriginal = (int?[])nNodes.Clone();
            if (resolvedOriginal.Any(n => n == null) && errorThreshold == null)
                throw new ArgumentException(
                    "Null entries in nNodes require errorThreshold to be set (auto-N mode).");
        }

        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        ErrorThreshold = errorThreshold;
        MaxN = maxN;
        MaxDerivativeOrder = maxDerivativeOrder;
        OriginalNNodes = (int?[])resolvedOriginal.Clone();

        // Keep public NNodes consistent with the resolved-or-zero state. For auto-N
        // dims we leave 0 as a placeholder (Build will populate per-piece).
        NNodes = resolvedOriginal.Select(n => n ?? 0).ToArray();

        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        Intervals = ComputeIntervals(numDimensions, domain, knots);
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        int totalPieces = 1;
        foreach (int s in Shape) totalPieces *= s;
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }

    /// <summary>Convenience overload accepting flat int[] nNodes (matches the original ctor signature).</summary>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[] nNodes,
        double[][]? knots = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2)
        : this(function, numDimensions, domain,
            nNodes.Select(n => (int?)n).ToArray(),
            knots, errorThreshold, maxN, maxDerivativeOrder)
    {
    }

    /// <summary>Return the error threshold passed to the constructor, or null in fixed-N mode.</summary>
    public double? GetErrorThreshold() => ErrorThreshold;
```

In the existing `Build()` method, when constructing each piece, pass through the threshold/maxN. Find the piece-construction site (around line 200-220 in the existing Build) and replace `new ChebyshevApproximation(Function, ...)` with one that threads errorThreshold:

```csharp
            // For auto-N dims, the piece's nNodes is null; for fixed-N dims, the user's int.
            int?[] pieceNNodes = OriginalNNodes.ToArray();
            var piece = new ChebyshevApproximation(
                Function, NumDimensions, pieceDomain,
                nNodes: pieceNNodes,
                errorThreshold: ErrorThreshold,
                maxN: MaxN,
                maxDerivativeOrder: MaxDerivativeOrder);
```

Note: this replacement only applies when `OriginalNNodes` has any null OR when `ErrorThreshold != null`. In the fixed-N case, the existing piece-construction code remains correct.

- [ ] **Step 4: Run Spline tests**

Run: `dotnet test --filter "FullyQualifiedName~SplineErrorThresholdTests"`
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite (verify existing 128 Spline tests still pass)**

Run: `dotnet test`
Expected: `Passed: 648` (642 + 6 new). Pay attention to any regression in `SplineTests` — the new overload must not change existing behavior.

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/ErrorThresholdTests.cs
git commit -m "phase1: ChebyshevSpline accepts errorThreshold/maxN per piece; default knots empty"
```

---

## Task 9: ChebyshevSpline — nested `int[][]` `nNodes` for per-sub-interval node counts

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Test: `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`

**Design notes:**
- Add a third ctor overload accepting `int[][] nNodes` where `nNodes[d][i]` is the node count for piece `i` along dim `d`.
- Validation:
  - `nNodes.Length == numDimensions`
  - For each dim `d`, `nNodes[d].Length == knots[d].Length + 1` (one entry per piece)
- Internally, this expands to per-piece overrides during `Build()`. When a piece is constructed, its `nNodes` becomes `[ nNodes[0][i0], nNodes[1][i1], ... ]` where `(i0, i1, ...)` is the piece's multi-index in `Shape`.
- This overload is incompatible with the auto-N path. Combination raises.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`:

```csharp
public class NestedNNodesTests
{
    [Fact]
    public void Test_nested_n_nodes_per_piece()
    {
        // 1D abs(x) with knot at 0; left piece uses 11 nodes, right piece 13.
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]),
            1, new[] { new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 11, 13 } },
            knots: new[] { new[] { 0.0 } });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 13 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_2d_per_sub_interval()
    {
        // 2D: dim 0 has knot at 0.2 (2 pieces with 7,9 nodes), dim 1 has no knot (1 piece, 11 nodes)
        var spl = new ChebyshevSpline(
            (x, _) => Math.Abs(x[0]) + x[1] * x[1] * x[1] * x[1],
            2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } },
            knots: new[] { new[] { 0.2 }, Array.Empty<double>() });
        spl.Build(verbose: false);
        Assert.Equal(2, spl.Pieces.Length);
        Assert.Equal(new[] { 7, 11 }, spl.Pieces[0]!.NNodes);
        Assert.Equal(new[] { 9, 11 }, spl.Pieces[1]!.NNodes);
    }

    [Fact]
    public void Test_nested_outer_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]),
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11, 11 } },  // only 1 entry, should be 2
                knots: new[] { new[] { 0.0 }, Array.Empty<double>() }));
        Assert.Contains("must have 2 entries", ex.Message);
    }

    [Fact]
    public void Test_nested_inner_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new ChebyshevSpline(
                (x, _) => Math.Abs(x[0]),
                1, new[] { new[] { -1.0, 1.0 } },
                nNodesNested: new[] { new[] { 11 } },  // 1 entry, should be 2 (since knots has 1 knot)
                knots: new[] { new[] { 0.0 } }));
        Assert.Contains("must have 2 entries", ex.Message);
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~NestedNNodesTests"`
Expected: build fails — no `nNodesNested` overload.

- [ ] **Step 3: Add nested-form ctor**

In `src/ChebyshevSharp/ChebyshevSpline.cs`, add a new field and a new ctor:

```csharp
    /// <summary>Per-piece, per-dim node counts (when constructed with nested nNodesNested form). Null otherwise.</summary>
    internal int[][]? NestedNNodes { get; set; }

    /// <summary>
    /// Create a piecewise Chebyshev spline with per-sub-interval node counts.
    /// </summary>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="nNodesNested">Nested array: nNodesNested[d][i] is the node count for piece i along dim d. Length per dim must equal knots[d].Length + 1.</param>
    /// <param name="knots">Interior knots per dimension. Required (no default) when using nested form.</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public ChebyshevSpline(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        int[][] nNodesNested,
        double[][] knots,
        int maxDerivativeOrder = 2)
    {
        if (nNodesNested.Length != numDimensions)
            throw new ArgumentException(
                $"nNodesNested must have {numDimensions} entries (one list per dim), got {nNodesNested.Length}");
        for (int d = 0; d < numDimensions; d++)
        {
            int expected = knots[d].Length + 1;
            if (nNodesNested[d].Length != expected)
                throw new ArgumentException(
                    $"nNodesNested[{d}] must have {expected} entries (one per piece), got {nNodesNested[d].Length}");
        }

        Function = function;
        NumDimensions = numDimensions;
        Domain = domain.Select(d => (double[])d.Clone()).ToArray();
        MaxDerivativeOrder = maxDerivativeOrder;
        MaxN = 64;
        ErrorThreshold = null;
        OriginalNNodes = Array.Empty<int?>();
        NestedNNodes = nNodesNested.Select(row => (int[])row.Clone()).ToArray();

        ValidateKnots(numDimensions, domain, knots);
        Knots = knots.Select(k => (double[])k.Clone()).ToArray();

        Intervals = ComputeIntervals(numDimensions, domain, knots);
        Shape = Intervals.Select(iv => iv.Length).ToArray();

        // Public NNodes is meaningful only in the flat-form case. Surface piece 0's
        // counts as a representative summary; full per-piece data lives in NestedNNodes.
        NNodes = nNodesNested.Select(row => row[0]).ToArray();

        int totalPieces = 1;
        foreach (int s in Shape) totalPieces *= s;
        Pieces = new ChebyshevApproximation?[totalPieces];

        Built = false;
        BuildTime = 0.0;
        _cachedErrorEstimate = null;
    }
```

In the existing `Build()` method, when constructing each piece, branch on `NestedNNodes`. Find the piece-construction site and replace it with:

```csharp
            int[] pieceN;
            if (NestedNNodes != null)
            {
                // Decode this piece's multi-index from the flat piece index.
                int[] multiIdx = UnravelMultiIndex(pieceFlat, Shape);
                pieceN = new int[NumDimensions];
                for (int d = 0; d < NumDimensions; d++)
                    pieceN[d] = NestedNNodes[d][multiIdx[d]];
            }
            else
            {
                pieceN = NNodes;  // existing flat path
            }

            var piece = new ChebyshevApproximation(
                Function, NumDimensions, pieceDomain, pieceN,
                MaxDerivativeOrder);
```

If `UnravelMultiIndex` does not exist already, add it as an internal static:

```csharp
    internal static int[] UnravelMultiIndex(int flat, int[] shape)
    {
        int[] idx = new int[shape.Length];
        int rem = flat;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        return idx;
    }
```

(If a similar helper already exists with a different name, reuse it instead.)

- [ ] **Step 4: Run nested tests**

Run: `dotnet test --filter "FullyQualifiedName~NestedNNodesTests"`
Expected: 4 tests pass.

- [ ] **Step 5: Run full suite (verify Spline tests still pass)**

Run: `dotnet test`
Expected: `Passed: 652` (648 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/SpecialPointsTests.cs
git commit -m "phase1: ChebyshevSpline accepts nested int[][] nNodes for per-sub-interval node counts"
```

---

## Task 10: Add `ChebyshevSpline.WithSpecialPoints` static factory

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Test: `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`

**Design notes:**
- `WithSpecialPoints` is a thin wrapper that: takes `specialPoints` (semantically identical to `knots` but a more user-friendly name in v0.12), validates, and dispatches to the appropriate ctor (nested-int[][] form if user supplied nested nNodes; flat-int[] form otherwise).
- Validation matches Python:
  - Outer length must equal `numDimensions`
  - Inner lists must be sorted
  - Each special point must be strictly inside its domain
  - Coinciding points raise
- Validation reuses `ValidateKnots` for the domain/sorted/duplicate checks; the "strictly inside" check is also there.

- [ ] **Step 1: Write failing tests**

Append to `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`:

```csharp
public class WithSpecialPointsTests
{
    private static readonly Func<double[], object?, double> Abs1D = (x, _) => Math.Abs(x[0]);

    [Fact]
    public void Test_factory_returns_spline_with_kink_as_knot()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        Assert.Equal(new[] { new[] { 0.0 } }, spl.Knots);
        Assert.Equal(2, spl.Pieces.Length);
    }

    [Fact]
    public void Test_abs_kink_reaches_machine_precision()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        for (double x = -0.95; x <= 0.95; x += 0.05)
        {
            if (Math.Abs(x) < 1e-8) continue;
            double v = spl.Eval(new[] { x }, new[] { 0 });
            TestFixtures.AssertClose(Math.Abs(x), v, atol: 1e-13);
        }
    }

    [Fact]
    public void Test_unsorted_points_raise()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.5, -0.5 } },
                nNodesNested: new[] { new[] { 11, 11, 11 } }));
        Assert.Contains("must be sorted", ex.Message);
    }

    [Fact]
    public void Test_point_on_boundary_raises()
    {
        Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 1.0 } },
                nNodesNested: new[] { new[] { 11, 11 } }));
    }

    [Fact]
    public void Test_outer_length_mismatch_raises()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ChebyshevSpline.WithSpecialPoints(
                (x, _) => Math.Abs(x[0]) + Math.Abs(x[1]),
                2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.0 } },  // missing dim 1
                nNodesNested: new[] { new[] { 11, 11 }, new[] { 13 } }));
        Assert.Contains("must have 2 entries", ex.Message);
    }

    [Fact]
    public void Test_factory_with_error_threshold()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            errorThreshold: 1e-10);
        spl.Build(verbose: false);
        TestFixtures.AssertClose(0.5, spl.Eval(new[] { 0.5 }, new[] { 0 }), atol: 1e-10);
    }
}
```

- [ ] **Step 2: Run to verify failures**

Run: `dotnet test --filter "FullyQualifiedName~WithSpecialPointsTests"`
Expected: build fails — `WithSpecialPoints` is not defined.

- [ ] **Step 3: Implement `WithSpecialPoints`**

In `src/ChebyshevSharp/ChebyshevSpline.cs`, add as a static method on the class:

```csharp
    /// <summary>
    /// Create a ChebyshevSpline with kinks declared via specialPoints (a more user-friendly
    /// name than knots when the function has known non-smooth points). Functionally equivalent
    /// to passing the same values as knots to a regular ChebyshevSpline constructor.
    /// </summary>
    /// <remarks>
    /// Python's ChebyshevApproximation(special_points=...) returns a ChebyshevSpline at
    /// construction time. C# constructors cannot return a different type; this static
    /// factory is the C#-idiomatic equivalent.
    /// </remarks>
    /// <param name="function">Function to approximate.</param>
    /// <param name="numDimensions">Number of input dimensions.</param>
    /// <param name="domain">Bounds per dimension.</param>
    /// <param name="specialPoints">Per-dim list of kink locations. Equivalent to knots; outer length must equal numDimensions.</param>
    /// <param name="nNodesNested">Per-sub-interval node counts (per dim, per piece). Mutually exclusive with errorThreshold.</param>
    /// <param name="nNodes">Flat per-dim node counts (shared across pieces). Mutually exclusive with nNodesNested and with errorThreshold.</param>
    /// <param name="errorThreshold">Target error per piece. Mutually exclusive with nNodes/nNodesNested.</param>
    /// <param name="maxN">Cap on doubling-loop nodes per dimension (default 64).</param>
    /// <param name="maxDerivativeOrder">Maximum derivative order to support (default 2).</param>
    public static ChebyshevSpline WithSpecialPoints(
        Func<double[], object?, double> function,
        int numDimensions,
        double[][] domain,
        double[][] specialPoints,
        int[][]? nNodesNested = null,
        int[]? nNodes = null,
        double? errorThreshold = null,
        int maxN = 64,
        int maxDerivativeOrder = 2)
    {
        if (specialPoints.Length != numDimensions)
            throw new ArgumentException(
                $"specialPoints must have {numDimensions} entries, got {specialPoints.Length}");

        // Validate using existing knots validator (sorted, strictly inside, no dupes).
        ValidateKnots(numDimensions, domain, specialPoints);

        int suppliedFormCount = (nNodesNested != null ? 1 : 0) + (nNodes != null ? 1 : 0) + (errorThreshold != null ? 1 : 0);
        if (suppliedFormCount == 0)
            throw new ArgumentException(
                "WithSpecialPoints requires exactly one of: nNodesNested, nNodes, or errorThreshold.");
        if (suppliedFormCount > 1)
            throw new ArgumentException(
                "WithSpecialPoints accepts only one of nNodesNested, nNodes, or errorThreshold (not multiple).");

        if (nNodesNested != null)
            return new ChebyshevSpline(function, numDimensions, domain, nNodesNested, specialPoints, maxDerivativeOrder);

        if (nNodes != null)
            return new ChebyshevSpline(function, numDimensions, domain, nNodes, specialPoints, errorThreshold: null, maxN: maxN, maxDerivativeOrder: maxDerivativeOrder);

        // errorThreshold path: every dim auto-N
        return new ChebyshevSpline(function, numDimensions, domain,
            nNodes: (int?[]?)null, knots: specialPoints,
            errorThreshold: errorThreshold, maxN: maxN, maxDerivativeOrder: maxDerivativeOrder);
    }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test --filter "FullyQualifiedName~WithSpecialPointsTests"`
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite**

Run: `dotnet test`
Expected: `Passed: 658` (652 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSpline.cs tests/ChebyshevSharp.Tests/SpecialPointsTests.cs
git commit -m "phase1: add ChebyshevSpline.WithSpecialPoints static factory"
```

---

## Task 11: Cross-feature tests for special points (algebra, save/load, integrate, extrude/slice)

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`

**Design note:** these are integration tests that confirm a Spline produced by `WithSpecialPoints` works correctly with the other features ChebyshevSpline already supports (no new code needed — these tests should pass with current implementation).

- [ ] **Step 1: Write tests**

Append to `tests/ChebyshevSharp.Tests/SpecialPointsTests.cs`:

```csharp
public class CrossFeatureTests
{
    private static readonly Func<double[], object?, double> Abs1D = (x, _) => Math.Abs(x[0]);

    [Fact]
    public void Test_save_load_roundtrip()
    {
        string path = Path.GetTempFileName();
        try
        {
            var spl = ChebyshevSpline.WithSpecialPoints(
                Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
                specialPoints: new[] { new[] { 0.0 } },
                nNodesNested: new[] { new[] { 11, 11 } });
            spl.Build(verbose: false);
            spl.Save(path);

            var loaded = ChebyshevSpline.Load(path);
            foreach (double x in new[] { -0.5, 0.2, 0.8 })
                TestFixtures.AssertClose(spl.Eval(new[] { x }, new[] { 0 }),
                                         loaded.Eval(new[] { x }, new[] { 0 }), atol: 1e-14);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Test_algebra_with_sibling()
    {
        var a = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        a.Build(verbose: false);
        var b = ChebyshevSpline.WithSpecialPoints(
            (x, _) => x[0] * x[0], 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        b.Build(verbose: false);

        var c = a + b;
        foreach (double x in new[] { -0.5, 0.3, 0.7 })
        {
            double expected = Math.Abs(x) + x * x;
            TestFixtures.AssertClose(expected, c.Eval(new[] { x }, new[] { 0 }), atol: 1e-12);
        }
    }

    [Fact]
    public void Test_integrate()
    {
        var spl = ChebyshevSpline.WithSpecialPoints(
            Abs1D, 1, new[] { new[] { -1.0, 1.0 } },
            specialPoints: new[] { new[] { 0.0 } },
            nNodesNested: new[] { new[] { 11, 11 } });
        spl.Build(verbose: false);
        // Integral of |x| over [-1, 1] is 1.0 exactly.
        TestFixtures.AssertClose(1.0, spl.Integrate(), atol: 1e-12);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `dotnet test --filter "FullyQualifiedName~CrossFeatureTests"`
Expected: 3 tests pass (existing Spline algebra/save/integrate already work).

- [ ] **Step 3: Run full suite**

Run: `dotnet test`
Expected: `Passed: 661` (658 + 3 new).

- [ ] **Step 4: Commit**

```bash
git add tests/ChebyshevSharp.Tests/SpecialPointsTests.cs
git commit -m "phase1: cross-feature tests for WithSpecialPoints (save/algebra/integrate)"
```

---

## Task 12: Documentation, parity metadata, version bump, release prep

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj`
- Create: `docs/docs/error-driven-construction.md`
- Create: `docs/docs/special-points.md`
- Modify: `docs/docs/toc.yml`
- Modify: `docs/docs/changelog.md`
- Modify: `skip_csharp.txt`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Bump version + add parity metadata in csproj**

Modify `src/ChebyshevSharp/ChebyshevSharp.csproj`. Find the `<Version>0.4.0</Version>` line and replace with:

```xml
    <Version>0.5.0</Version>
    <PyChebyshevParity>0.12.0</PyChebyshevParity>
    <Description>ChebyshevSharp 0.5.0 — multi-dimensional Chebyshev tensor interpolation with analytical derivatives. Feature parity with PyChebyshev v0.12.0.</Description>
    <InformationalVersion>0.5.0+pychebyshev.0.12.0</InformationalVersion>
```

(If `<Description>` and `<InformationalVersion>` already exist, replace them; otherwise add inside the existing `<PropertyGroup>`.)

- [ ] **Step 2: Build to confirm csproj changes**

Run: `dotnet build`
Expected: succeeds.

- [ ] **Step 3: Write `docs/docs/error-driven-construction.md`**

Create `docs/docs/error-driven-construction.md`:

```markdown
# Error-Driven Construction

`ChebyshevApproximation` and `ChebyshevSpline` accept an `errorThreshold`
parameter that drives an automatic node-count selection loop. Pass `null` for
each dimension you want auto-sized; the build doubles the worst-contributing
dim each iteration until `ErrorEstimate() <= errorThreshold` (or `maxN` is
reached, in which case `BuildWarning` is set).

## Quick Start

```csharp
using ChebyshevSharp;

// Auto-N: every dim resolves automatically.
var cheb = new ChebyshevApproximation(
    function: (x, _) => Math.Sin(x[0]) + Math.Cos(x[1]),
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: null,
    errorThreshold: 1e-8);
cheb.Build(verbose: true);  // [auto-N] nNodes=[3, 3], error=...
                            // [auto-N] nNodes=[6, 3], error=...
                            // ...

// Mixed: fix dim 1 at 15, auto-size dim 0 against the threshold.
var mixed = new ChebyshevApproximation(
    function: ...,
    numDimensions: 2,
    domain: ...,
    nNodes: new int?[] { null, 15 },
    errorThreshold: 1e-6);
```

## Capacity Pre-Check (`GetOptimalN1`)

Before committing to a multi-dimensional build, you can size a single
dimension:

```csharp
int n = ChebyshevApproximation.GetOptimalN1(
    function: (x, _) => Math.Sin(x[0]),
    domain: (-1.0, 1.0),
    errorThreshold: 1e-8);
// n is the smallest N at which a 1-D Chebyshev build hits 1e-8.
```

## Spline Per-Piece Threshold

`ChebyshevSpline` applies the threshold per piece — kink-adjacent pieces
refine more than smooth ones automatically:

```csharp
var spl = new ChebyshevSpline(
    function: (x, _) => Math.Abs(x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: null,
    knots: new[] { new[] { 0.0 } },
    errorThreshold: 1e-6);
spl.Build();
// Each piece in spl.Pieces hits the threshold.
```

## When the Cap Bites

If `maxN` is reached before the threshold is satisfied on any auto dim,
`BuildWarning` is set and the build returns the best result it could
achieve. The interpolant is still usable; you should either raise `maxN`
or relax the threshold.

```csharp
var cheb = new ChebyshevApproximation(
    function: (x, _) => Math.Sin(50 * x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: null,
    errorThreshold: 1e-12,
    maxN: 16);
cheb.Build();
if (cheb.BuildWarning != null) Console.Error.WriteLine(cheb.BuildWarning);
```
```

- [ ] **Step 4: Write `docs/docs/special-points.md`**

Create `docs/docs/special-points.md`:

```markdown
# Special Points (Kink Declaration)

`ChebyshevSpline.WithSpecialPoints` is the C# entry point for declaring
known kinks at construction time. Equivalent to passing the same values
as `knots` to a regular `ChebyshevSpline` constructor, but the name
matches PyChebyshev's `special_points` kwarg for cross-language
discoverability.

## Why Declare Kinks

Without a kink declaration, spectral methods plateau at low precision on
non-smooth functions (Gibbs phenomenon). Declaring the kink as a
sub-interval boundary restores spectral convergence on each piece.

```csharp
// Without kink declaration: plateaus around 1e-3 even at N=31.
var bad = new ChebyshevApproximation(
    (x, _) => Math.Abs(x[0]),
    1, new[] { new[] { -1.0, 1.0 } }, new[] { 31 });

// With kink declaration: machine precision at N=11 per piece.
var good = ChebyshevSpline.WithSpecialPoints(
    function: (x, _) => Math.Abs(x[0]),
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    specialPoints: new[] { new[] { 0.0 } },
    nNodesNested: new[] { new[] { 11, 11 } });
```

## API Note (Python ↔ C# Difference)

In Python, `ChebyshevApproximation(special_points=[[...]])` returns a
`ChebyshevSpline` at construction time, leveraging Python's `__new__`
polymorphism. C# constructors cannot return a different type, so the
`specialPoints` kwarg is intentionally absent from
`ChebyshevApproximation`'s constructor. Use `ChebyshevSpline.WithSpecialPoints(...)`
instead.

## Per-Sub-Interval Node Counts

Pass nested arrays to `nNodesNested` for per-piece refinement:

```csharp
var spl = ChebyshevSpline.WithSpecialPoints(
    function: ...,
    numDimensions: 2,
    domain: ...,
    specialPoints: new[] { new[] { 0.0 }, Array.Empty<double>() },
    nNodesNested: new[] { new[] { 7, 9 }, new[] { 11 } });
// Dim 0: 2 pieces (split at 0.0) with 7 and 9 nodes.
// Dim 1: 1 piece (no kink) with 11 nodes.
```
```

- [ ] **Step 5: Update `docs/docs/toc.yml`**

Open `docs/docs/toc.yml`. After the existing "Spline" entry add (preserve formatting):

```yaml
- name: Error-Driven Construction
  href: error-driven-construction.md
- name: Special Points
  href: special-points.md
```

- [ ] **Step 6: Update `docs/docs/changelog.md`**

Prepend to `docs/docs/changelog.md`:

```markdown
## [0.5.0] - 2026-04-27

### PyChebyshev parity: v0.12.0

#### Added — Error-Driven Construction (Python v0.11.0)

- `ChebyshevApproximation` and `ChebyshevSpline` constructors accept
  `errorThreshold` and `maxN` parameters. `nNodes` may be `int?[]` with
  `null` per dim signalling auto-N for that dimension.
- New `static int ChebyshevApproximation.GetOptimalN1(...)` 1-D capacity
  estimator.
- New `double? GetErrorThreshold()` accessor on Approximation and Spline.
- New `string? BuildWarning` property — set when `maxN` is reached before
  `errorThreshold` is satisfied (replaces Python's `RuntimeWarning`).
- New internal `Internal/AdaptiveBuild.cs` runs the doubling loop.
- New `double[] ErrorEstimatePerDim()` public method (was internal in
  Python; exposed here for symmetry with the new auto-N infrastructure).

#### Added — Special Points (Python v0.12.0)

- `ChebyshevSpline.WithSpecialPoints(...)` static factory: kink
  declaration with `specialPoints` (mirrors Python's `special_points` kwarg
  via a C#-idiomatic factory, since constructors cannot return a different
  type).
- `ChebyshevSpline` accepts a nested `int[][]` form for `nNodes`,
  per-sub-interval. `nNodesNested[d][i]` is the node count for piece `i`
  along dimension `d`.

#### Changed

- `Build()` is now a public dispatcher; the original behavior moved to a
  private `BuildFixedGrid()`. No behavior change for fixed-N callers.
- `ErrorEstimate()` is now backed by `ErrorEstimatePerDim().Sum()`. Cache
  semantics unchanged.
- JSON `Save`/`Load` format version bumped to "0.5.0". `Load` backfills
  `OriginalNNodes`, `ErrorThreshold`, `MaxN` for older files (default
  values: `OriginalNNodes` ← `NNodes`, `ErrorThreshold` ← null, `MaxN`
  ← 64).

#### Skipped

- Python's `ChebyshevApproximation(special_points=...)` constructor
  dispatch to `ChebyshevSpline` is not mirrored. C# constructors cannot
  return a different type. Use `ChebyshevSpline.WithSpecialPoints(...)`.
```

- [ ] **Step 7: Update `skip_csharp.txt`**

Append to `skip_csharp.txt`:

```
=========================================================
Phase 5 (PyChebyshev v0.5.0 release - parity v0.12.0): Adaptive Constructor
=========================================================
ErrorThresholdTests.cs: ~30 tests (port of test_error_threshold.py)
SpecialPointsTests.cs: ~15 tests (port of test_special_points.py)
Total: 48 new tests (613 -> 661 total)
```

- [ ] **Step 8: Update README badge**

Find the parity badge line in `README.md` (or add one near the top of the file if absent). Set it to:

```markdown
![PyChebyshev parity](https://img.shields.io/badge/PyChebyshev_parity-v0.12.0-blue)
```

- [ ] **Step 9: Update CLAUDE.md**

In `CLAUDE.md`, find the "Status" section and update the parity claim from "v0.10.1" to "v0.12.0". Update the test count from "613/613" to "661/661". Add a sentence under the Phase 1 (ChebyshevApproximation) class description noting the new `errorThreshold`/`maxN`/`GetOptimalN1` surface and `ErrorEstimatePerDim()` helper. Add a similar sentence under Phase 2 (ChebyshevSpline) noting `WithSpecialPoints` and nested `nNodes`.

- [ ] **Step 10: Final full-suite run**

Run: `dotnet build` (zero warnings expected)
Run: `dotnet test`
Expected: `Passed: 661`. Both targets (net8.0 and net10.0) pass.

- [ ] **Step 11: Commit + tag + push**

```bash
git add src/ChebyshevSharp/ChebyshevSharp.csproj docs/docs/error-driven-construction.md docs/docs/special-points.md docs/docs/toc.yml docs/docs/changelog.md skip_csharp.txt README.md CLAUDE.md
git commit -m "phase1: docs + parity metadata + v0.5.0 release prep (PyChebyshev v0.12.0 parity)"
git tag v0.5.0
git push origin main
git push origin v0.5.0
```

- [ ] **Step 12: Cut GitHub release (triggers NuGet publish)**

```bash
gh release create v0.5.0 --title "v0.5.0 — PyChebyshev v0.12.0 parity" --notes-from-tag
```

Verify: `publish.yml` workflow runs and pushes the package to NuGet.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by task |
|---|---|
| `errorThreshold` / `maxN` ctor on Approximation | Task 2 |
| `int?[]` nullable `nNodes` form | Task 2 |
| `BuildFixedGrid` / `BuildWithThreshold` split | Tasks 3 + 5 |
| `Internal/AdaptiveBuild.cs` | Task 5 |
| `_originalNNodes` field | Task 2 (added) + Task 5 (consumed) + Task 7 (persisted) |
| `_ErrorEstimatePerDim()` helper | Task 4 |
| `GetErrorThreshold()` accessor | Task 6 |
| `GetOptimalN1()` static | Task 6 |
| JSON `Load()` backfill | Task 7 |
| Spline `errorThreshold` / `maxN` per piece | Task 8 |
| Spline `Knots` defaults to empty | Task 8 |
| Spline nested `nNodes` (per-piece) | Task 9 |
| `WithSpecialPoints` static factory | Task 10 |
| Cross-feature integration tests | Task 11 |
| `<PyChebyshevParity>` MSBuild metadata | Task 12 |
| Docs: error-driven-construction.md, special-points.md | Task 12 |
| README badge regen | Task 12 |
| CLAUDE.md update | Task 12 |
| `skip_csharp.txt` update | Task 12 |
| Submodule advance | Task 1 |
| NuGet release | Task 12 |

**Type consistency check:**

- `OriginalNNodes` is `int?[]` everywhere (Task 2 declares, Tasks 5/7/8 consume).
- `ErrorThreshold` is `double?` everywhere (Task 2 on Approx, Task 8 on Spline).
- `MaxN` is `int` everywhere (default 64).
- `ErrorEstimatePerDim()` returns `double[]` (Task 4 declares, Task 5 consumes).
- `BuildWarning` is `string?` (Task 2 declares, Task 5 sets).
- `WithSpecialPoints` accepts `double[][] specialPoints` (matches `Knots` shape).
- `nNodesNested` is `int[][]` (Tasks 9 + 10 consistent).

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in the plan.

**Open decision resolved:** The Python-vs-C# `ChebyshevApproximation(special_points=...) → ChebyshevSpline` dispatch is intentionally **not** mirrored. C# users call `ChebyshevSpline.WithSpecialPoints(...)` directly. Documented in `special-points.md` (Task 12).
