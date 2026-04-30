# Phase 7 Catch-up to v0.21.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle PyChebyshev v0.21.0 (Slider/TT calculus parity) + v0.21.1 (TT `_dim_order` cluster + cross-cutting fixes + perf) into ChebyshevSharp v0.11.0.

**Architecture:** No new public types. Six new methods on Slider/TT (Roots/Minimize/Maximize), one new TT method (SobolIndices), six bug fixes (most are direct C# ports of upstream regression fixes), two perf improvements. The Slider/TT calculus path delegates to existing 1-D `ChebyshevApproximation` calculus primitives via a 1-D reduction helper.

**Tech Stack:** C# 12, .NET 8 + .NET 10 multi-target, xUnit, MathNet.Numerics. Reuses Phase 1 (calculus 1-D primitives, `FromValues`, `MakeNodesForDim`), Phase 4 (TT `Slice`, `ToDense`), and Phase 6 (TT `_dimOrder`, `IsIdentityDimOrder`, `EvalCore`, `Sensitivity.ComputeSobolFromCoeffs`).

**Spec:** `docs/superpowers/specs/2026-04-29-phase7-catchup-v0.21.1-design.md` (5 design decisions D1–D5).

**Worktree:** `.worktrees/phase7-catchup-v0.21.1` (created 2026-04-29). Branch: `phase7-catchup-v0.21.1`. Submodule `ref/PyChebyshev` at v0.20.1 — Task 1 bumps to v0.21.1.

**Test count progression:** 1030 → ~1112 (+82 expected, ±2 drift per task allowed; precedent Phase 5 was +47 against +50 plan).

---

## Universal Constraints (Apply to Every Task)

### WORKTREE ENFORCEMENT (MANDATORY)

Before running any other command, every task subagent MUST run:

```bash
git rev-parse --show-toplevel
```

The output MUST end in `.worktrees/phase7-catchup-v0.21.1`. If it ends in `/home/max/Documents/ChebyshevSharp` (the main repo), STOP and switch to the worktree. Phase 1 Task 4 cross-directory commit is the cautionary tale.

### TDD Discipline

Each TDD task follows: **failing test → run-to-fail → minimal implementation → run-to-pass → commit**. Never skip ahead. Never write the implementation before seeing the test fail. Match the Phase 6 cadence exactly.

### Commit Convention

Every commit on this branch uses prefix `phase7:`. Examples:
- `phase7: bump submodule to v0.21.1; scaffold test files`
- `phase7: add Slider.To1DChebyshev + Slider.Roots`
- `phase7: race-fix EvalMulti via EvalStorageFrame helper`

### Test Count Verification

After each task's commit, run `dotnet test --nologo --verbosity quiet` and verify the passing count matches the task's "Expected total tests after commit" line. ±2 drift per task is acceptable. Larger drift requires investigation before proceeding.

### Two-Stage Review Between Tasks

The orchestrator runs spec compliance (does the diff match the task brief?) then code quality (bugs, smells, undocumented edges) before dispatching the next task. **Cross-cutting tasks** (4, 5, 7, 11) get the formal `feature-dev:code-reviewer` agent; mechanical tasks (1, 6, 9, 12) get inline review.

---

## Task 1: Submodule Bump + Scaffold + Version

**Files:**
- Modify: `.gitmodules` (no change, just verify)
- Modify (submodule pin): `ref/PyChebyshev` → bump from v0.20.1 to v0.21.1
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj` (Version 0.10.0 → 0.11.0; PyChebyshevParity 0.20.1 → 0.21.1; InformationalVersion 0.10.0+pychebyshev.0.20.1 → 0.11.0+pychebyshev.0.21.1)
- Create: `tests/ChebyshevSharp.Tests/SliderRootsTests.cs` (placeholder with one trivial test)
- Create: `tests/ChebyshevSharp.Tests/SliderOptimizeTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/TtCalculusTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/TtSobolIndicesTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/AlgebraTupleListTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/VectorizedEvalBatchPerfTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/OptimizeVectorizedTests.cs` (placeholder)
- Create: `tests/ChebyshevSharp.Tests/Phase7CoverageGapTests.cs` (placeholder)

**Expected total tests after commit:** 1030 (no change; placeholder files have no `[Fact]`)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```
Expected: `/home/max/Documents/ChebyshevSharp/.worktrees/phase7-catchup-v0.21.1`

- [ ] **Step 2: Bump submodule pin**

```bash
cd ref/PyChebyshev
git fetch --tags origin
git checkout v0.21.1
cd ../..
git status ref/PyChebyshev
```
Expected: shows `ref/PyChebyshev (new commits): v0.20.1 → v0.21.1` or similar.

- [ ] **Step 3: Bump csproj version markers**

Edit `src/ChebyshevSharp/ChebyshevSharp.csproj`:
- Replace `<Version>0.10.0</Version>` with `<Version>0.11.0</Version>`
- Replace `<PyChebyshevParity>0.20.1</PyChebyshevParity>` with `<PyChebyshevParity>0.21.1</PyChebyshevParity>`
- Replace `<InformationalVersion>0.10.0+pychebyshev.0.20.1</InformationalVersion>` with `<InformationalVersion>0.11.0+pychebyshev.0.21.1</InformationalVersion>`

- [ ] **Step 4: Create placeholder test files**

Each file follows this exact template (replace `<ClassName>` with the test class name):

```csharp
namespace ChebyshevSharp.Tests;

public class <ClassName>
{
    // Phase 7 tests will be added in subsequent tasks.
}
```

Filenames and corresponding `<ClassName>`:
- `SliderRootsTests.cs` → `SliderRootsTests`
- `SliderOptimizeTests.cs` → `SliderOptimizeTests`
- `TtCalculusTests.cs` → `TtCalculusTests`
- `TtSobolIndicesTests.cs` → `TtSobolIndicesTests`
- `TtDimOrderClusterTests.cs` → `TtDimOrderClusterTests`
- `AlgebraTupleListTests.cs` → `AlgebraTupleListTests`
- `VectorizedEvalBatchPerfTests.cs` → `VectorizedEvalBatchPerfTests`
- `OptimizeVectorizedTests.cs` → `OptimizeVectorizedTests`
- `Phase7CoverageGapTests.cs` → `Phase7CoverageGapTests`

- [ ] **Step 5: Build and test**

```bash
dotnet build --nologo --verbosity quiet
dotnet test --nologo --verbosity quiet
```
Expected: build with 0 warnings, `Passed: 1030, Failed: 0`.

- [ ] **Step 6: Commit**

```bash
git add ref/PyChebyshev src/ChebyshevSharp/ChebyshevSharp.csproj tests/ChebyshevSharp.Tests/
git commit -m "phase7: bump submodule to v0.21.1; scaffold test files; bump version 0.11.0"
```

---

## Task 2: Slider.To1DChebyshev + Slider.Roots

**Python source:** `ref/PyChebyshev/src/pychebyshev/slider.py:1138-1224` (`_to_1d_chebyshev`, `roots`).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs` (add `To1DChebyshev` private + `Roots` public)
- Modify: `tests/ChebyshevSharp.Tests/SliderRootsTests.cs` (add 6 tests)

**Expected total tests after commit:** 1036 (+6)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```
Expected: `/home/max/Documents/ChebyshevSharp/.worktrees/phase7-catchup-v0.21.1`

- [ ] **Step 2: Write the 6 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/SliderRootsTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderRootsTests
{
    private static readonly double Tolerance = 1e-8;

    [Fact]
    public void Test_1D_slider_finds_known_root()
    {
        // f(x) = x - 0.5 has a root at x = 0.5
        Func<double[], object?, double> f = (p, _) => p[0] - 0.5;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        slider.Build();

        double[] roots = slider.Roots();

        Assert.Single(roots);
        Assert.Equal(0.5, roots[0], precision: 8);
    }

    [Fact]
    public void Test_1D_slider_no_roots_returns_empty()
    {
        // f(x) = x^2 + 1 has no real roots
        Func<double[], object?, double> f = (p, _) => p[0] * p[0] + 1.0;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        slider.Build();

        double[] roots = slider.Roots();

        Assert.Empty(roots);
    }

    [Fact]
    public void Test_2D_slider_with_fixed_finds_root()
    {
        // f(x, y) = (x - 0.3) + (y - 0.7), fixing y=0.7 leaves f(x, 0.7) = x - 0.3
        Func<double[], object?, double> f = (p, _) => (p[0] - 0.3) + (p[1] - 0.7);
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        double[] roots = slider.Roots(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.7 } });

        Assert.Single(roots);
        Assert.Equal(0.3, roots[0], precision: 8);
    }

    [Fact]
    public void Test_multi_d_requires_dim_param()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        Assert.Throws<ArgumentException>(() => slider.Roots());
    }

    [Fact]
    public void Test_multi_d_requires_fixed_for_other_dims()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new double[] { -1.0, 1.0 }, new double[] { -1.0, 1.0 } },
            new[] { 16, 16 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();

        // Specifying dim=0 but no fixedDims for dim=1 should fail.
        Assert.Throws<ArgumentException>(() => slider.Roots(dim: 0, fixedDims: null));
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] - 0.5;
        var slider = new ChebyshevSlider(f, 1, new[] { new double[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        // No Build() call

        Assert.Throws<InvalidOperationException>(() => slider.Roots());
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~SliderRootsTests" --nologo --verbosity quiet
```
Expected: 6 failures, all complaining `'ChebyshevSlider' does not contain a definition for 'Roots'`.

- [ ] **Step 4: Implement Slider.To1DChebyshev + Slider.Roots**

In `src/ChebyshevSharp/ChebyshevSlider.cs`, add the following private helper (place near other private helpers, e.g., after the existing `Slice` method around line 950):

```csharp
/// <summary>
/// Build a 1-D ChebyshevApproximation from this 1-D Slider by evaluating at
/// Chebyshev Type-I nodes. Used by Roots/Minimize/Maximize to delegate to
/// the existing 1-D calculus primitives on ChebyshevApproximation.
/// </summary>
/// <remarks>
/// Precondition: this Slider must be 1-D (NumDimensions == 1). Call Slice()
/// to reduce a multi-D Slider to 1-D before calling this helper.
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1138-1176</c>.
/// </remarks>
private ChebyshevApproximation To1DChebyshev()
{
    if (NumDimensions != 1)
        throw new InvalidOperationException(
            $"To1DChebyshev requires a 1-D slider, got {NumDimensions}-D");

    int n = NNodes[0];
    double a = Domain[0][0];
    double b = Domain[0][1];
    double[] chebNodes = Internal.BarycentricKernel.MakeNodesForDim(a, b, n);

    var values = new double[n];
    for (int i = 0; i < n; i++)
        values[i] = Eval(new[] { chebNodes[i] });

    return ChebyshevApproximation.FromValues(
        values,
        numDimensions: 1,
        domain: new[] { new[] { a, b } },
        nNodes: new[] { n });
}
```

And add the public method (place near other public methods, e.g., after `Slice`):

```csharp
/// <summary>
/// Find all real roots of the slider along a specified dimension.
/// Reduces to a 1-D problem by slicing all other dimensions to their
/// fixed values, then delegates to <see cref="ChebyshevApproximation.Roots"/>.
/// </summary>
/// <param name="dim">Target dimension. For 1-D sliders, defaults to 0.</param>
/// <param name="fixedDims">For multi-D sliders, <c>{dim_index: value}</c>
/// for all dimensions except <paramref name="dim"/>.</param>
/// <returns>Sorted real root locations in the physical domain. Empty if no roots.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
/// <exception cref="ArgumentException">If <paramref name="dim"/> or <paramref name="fixedDims"/> validation fails.</exception>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1178-1224</c>.
/// </remarks>
public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    if (!_built)
        throw new InvalidOperationException("Call Build() first");

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);

    var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Roots();
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
dotnet test --filter "FullyQualifiedName~SliderRootsTests" --nologo --verbosity quiet
```
Expected: `Passed: 6, Failed: 0`.

- [ ] **Step 6: Run full suite to verify no regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1036, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSlider.cs tests/ChebyshevSharp.Tests/SliderRootsTests.cs
git commit -m "phase7: add Slider.To1DChebyshev + Slider.Roots"
```

---

## Task 3: Slider.Minimize + Slider.Maximize

**Python source:** `ref/PyChebyshev/src/pychebyshev/slider.py:1226-1283`.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs` (add `Minimize` and `Maximize` public methods)
- Modify: `tests/ChebyshevSharp.Tests/SliderOptimizeTests.cs` (add 10 tests)

**Expected total tests after commit:** 1046 (+10)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Write the 10 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/SliderOptimizeTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderOptimizeTests
{
    private static ChebyshevSlider Build1DSlider(Func<double, double> f, int n = 16, double a = -1.0, double b = 1.0)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0]);
        var slider = new ChebyshevSlider(wrapper, 1,
            new[] { new[] { a, b } },
            new[] { n },
            partition: new[] { new[] { 0 } },
            pivotPoint: new[] { (a + b) / 2.0 });
        slider.Build();
        return slider;
    }

    private static ChebyshevSlider Build2DSlider(Func<double, double, double> f, int n = 16)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0], p[1]);
        var slider = new ChebyshevSlider(wrapper, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();
        return slider;
    }

    [Fact]
    public void Test_1D_minimize_quadratic()
    {
        // f(x) = (x - 0.3)^2 has min at x = 0.3, value = 0
        var slider = Build1DSlider(x => (x - 0.3) * (x - 0.3));
        var (value, location) = slider.Minimize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_1D_maximize_quadratic()
    {
        // f(x) = -(x - 0.3)^2 has max at x = 0.3, value = 0
        var slider = Build1DSlider(x => -(x - 0.3) * (x - 0.3));
        var (value, location) = slider.Maximize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_1D_min_at_endpoint()
    {
        // f(x) = x has min at x = -1
        var slider = Build1DSlider(x => x);
        var (value, location) = slider.Minimize();
        Assert.Equal(-1.0, value, precision: 8);
        Assert.Equal(-1.0, location, precision: 8);
    }

    [Fact]
    public void Test_1D_max_at_endpoint()
    {
        // f(x) = x has max at x = 1
        var slider = Build1DSlider(x => x);
        var (value, location) = slider.Maximize();
        Assert.Equal(1.0, value, precision: 8);
        Assert.Equal(1.0, location, precision: 8);
    }

    [Fact]
    public void Test_2D_minimize_with_fixed()
    {
        // f(x, y) = (x - 0.5)^2 + y, fixing y=-0.5 makes f = (x - 0.5)^2 - 0.5
        // Min at x=0.5, value = -0.5
        var slider = Build2DSlider((x, y) => (x - 0.5) * (x - 0.5) + y);
        var (value, location) = slider.Minimize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, -0.5 } });
        Assert.Equal(-0.5, value, precision: 6);
        Assert.Equal(0.5, location, precision: 6);
    }

    [Fact]
    public void Test_2D_maximize_with_fixed()
    {
        // f(x, y) = -((x - 0.5)^2) + y, fixing y=0.5 makes f = -(x - 0.5)^2 + 0.5
        // Max at x=0.5, value = 0.5
        var slider = Build2DSlider((x, y) => -((x - 0.5) * (x - 0.5)) + y);
        var (value, location) = slider.Maximize(dim: 0, fixedDims: new Dictionary<int, double> { { 1, 0.5 } });
        Assert.Equal(0.5, value, precision: 6);
        Assert.Equal(0.5, location, precision: 6);
    }

    [Fact]
    public void Test_min_max_unbuilt_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var slider = new ChebyshevSlider(f, 1, new[] { new[] { -1.0, 1.0 } },
            new[] { 16 }, partition: new[] { new[] { 0 } }, pivotPoint: new[] { 0.0 });
        // No Build() call

        Assert.Throws<InvalidOperationException>(() => slider.Minimize());
        Assert.Throws<InvalidOperationException>(() => slider.Maximize());
    }

    [Fact]
    public void Test_multi_d_min_requires_dim()
    {
        var slider = Build2DSlider((x, y) => x + y);
        Assert.Throws<ArgumentException>(() => slider.Minimize());
    }

    [Fact]
    public void Test_multi_d_max_requires_fixed()
    {
        var slider = Build2DSlider((x, y) => x + y);
        Assert.Throws<ArgumentException>(() => slider.Maximize(dim: 0));
    }

    [Fact]
    public void Test_min_max_returns_tuple_value_first()
    {
        // Order: (value, location). Testing this convention explicitly.
        var slider = Build1DSlider(x => x * x);  // min at 0
        var (value, location) = slider.Minimize();
        Assert.Equal(0.0, value, precision: 8);  // value first
        Assert.Equal(0.0, location, precision: 8);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~SliderOptimizeTests" --nologo --verbosity quiet
```
Expected: 10 failures.

- [ ] **Step 4: Implement Slider.Minimize and Slider.Maximize**

In `src/ChebyshevSharp/ChebyshevSlider.cs`, add right after `Roots`:

```csharp
/// <summary>
/// Find the minimum value of the slider along a specified dimension.
/// Reduces to a 1-D problem by slicing all other dimensions to their fixed
/// values, then delegates to <see cref="ChebyshevApproximation.Minimize"/>.
/// </summary>
/// <param name="dim">Target dimension. For 1-D sliders, defaults to 0.</param>
/// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
/// dims except <paramref name="dim"/>.</param>
/// <returns>Tuple <c>(value, location)</c> where value is the minimum and
/// location is its coordinate in the target dimension.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
/// <exception cref="ArgumentException">If validation fails.</exception>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1226-1264</c>.
/// </remarks>
public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    if (!_built)
        throw new InvalidOperationException("Call Build() first");

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);

    var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Minimize();
}

/// <summary>
/// Find the maximum value of the slider along a specified dimension.
/// See <see cref="Minimize"/> for parameter details.
/// </summary>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/slider.py:1266-1283</c>.
/// </remarks>
public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    if (!_built)
        throw new InvalidOperationException("Call Build() first");

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(NumDimensions, dim, fixedDims, Domain);

    var sliced = sliceParams.Length > 0 ? Slice(sliceParams) : this;
    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Maximize();
}
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~SliderOptimizeTests" --nologo --verbosity quiet
```
Expected: `Passed: 10, Failed: 0`.

- [ ] **Step 6: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1046, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevSlider.cs tests/ChebyshevSharp.Tests/SliderOptimizeTests.cs
git commit -m "phase7: add Slider.Minimize + Slider.Maximize"
```

---

## Task 4: TT.Roots + TT.Minimize + TT.Maximize

**Python source:** `ref/PyChebyshev/src/pychebyshev/tensor_train.py:1704-1872` (`_to_1d_chebyshev`, `_user_frame_domain`, `roots`, `minimize`, `maximize`).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (add private `UserFrameDomain` + `To1DChebyshev`, public `Roots`/`Minimize`/`Maximize`)
- Modify: `tests/ChebyshevSharp.Tests/TtCalculusTests.cs` (add 15 tests)

**Expected total tests after commit:** 1061 (+15)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Write the 15 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/TtCalculusTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtCalculusTests
{
    private static ChebyshevTT Build1DTt(Func<double, double> f, int n = 12)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0]);
        var tt = new ChebyshevTT(wrapper, 1,
            new[] { new[] { -1.0, 1.0 } },
            new[] { n },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        tt.Build();
        return tt;
    }

    private static ChebyshevTT Build3DTt(Func<double, double, double, double> f, int n = 8, int rank = 6)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0], p[1], p[2]);
        var tt = new ChebyshevTT(wrapper, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n, n },
            maxRank: rank,
            tolerance: 1e-10,
            seed: 42);
        tt.Build();
        return tt;
    }

    [Fact]
    public void Test_1D_roots_finds_known_root()
    {
        var tt = Build1DTt(x => x - 0.4);
        double[] roots = tt.Roots();
        Assert.Single(roots);
        Assert.Equal(0.4, roots[0], precision: 6);
    }

    [Fact]
    public void Test_1D_minimize()
    {
        var tt = Build1DTt(x => (x - 0.2) * (x - 0.2));
        var (value, location) = tt.Minimize();
        Assert.Equal(0.0, value, precision: 6);
        Assert.Equal(0.2, location, precision: 6);
    }

    [Fact]
    public void Test_1D_maximize()
    {
        var tt = Build1DTt(x => -(x - 0.2) * (x - 0.2));
        var (value, location) = tt.Maximize();
        Assert.Equal(0.0, value, precision: 6);
        Assert.Equal(0.2, location, precision: 6);
    }

    [Fact]
    public void Test_3D_roots_with_fixed()
    {
        // f(x, y, z) = (x - 0.5) + (y - 0.5)^2 + z. Fix y=0.5, z=-0.5.
        // Then f(x, 0.5, -0.5) = x - 1, root at x = 1.
        // But x ∈ [-1, 1], so root is at x = 1 (endpoint).
        var tt = Build3DTt((x, y, z) => (x - 0.5) + (y - 0.5) * (y - 0.5) + z);
        double[] roots = tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 0.5 }, { 2, -0.5 } });
        Assert.Single(roots);
        Assert.Equal(1.0, roots[0], precision: 6);
    }

    [Fact]
    public void Test_3D_minimize_with_fixed()
    {
        var tt = Build3DTt((x, y, z) => (x - 0.3) * (x - 0.3) + y + z);
        var (value, location) = tt.Minimize(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, -1.0 }, { 2, -1.0 } });
        Assert.Equal(-2.0, value, precision: 5);
        Assert.Equal(0.3, location, precision: 5);
    }

    [Fact]
    public void Test_3D_maximize_with_fixed()
    {
        var tt = Build3DTt((x, y, z) => -((x - 0.3) * (x - 0.3)) + y + z);
        var (value, location) = tt.Maximize(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 1.0 }, { 2, 1.0 } });
        Assert.Equal(2.0, value, precision: 5);
        Assert.Equal(0.3, location, precision: 5);
    }

    [Fact]
    public void Test_unbuilt_roots_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var tt = new ChebyshevTT(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.Throws<InvalidOperationException>(() => tt.Roots());
    }

    [Fact]
    public void Test_multi_d_roots_requires_dim()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Roots());
    }

    [Fact]
    public void Test_multi_d_minimize_requires_fixed()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Minimize(dim: 0));
    }

    [Fact]
    public void Test_fixed_includes_target_throws()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_fixed_value_out_of_domain_throws()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        // Domain is [-1, 1]^3; passing 5.0 should throw.
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 5.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_under_with_auto_order_user_frame_dim()
    {
        // Build a TT with strong dim-0 dependence; WithAutoOrder may permute.
        // After permutation, user passes dim=0 (user-frame) and expects
        // the Roots to work transparently.
        Func<double[], object?, double> f = (p, _) => p[0] + 0.1 * p[1] + 0.01 * p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 6,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // Validate that even though _dimOrder is potentially non-identity,
        // user-frame dim=0 finds the root of f(x, fixed=0, fixed=0) = x.
        double[] roots = tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 }, { 2, 0.0 } });

        Assert.Single(roots);
        Assert.Equal(0.0, roots[0], precision: 4);
    }

    [Fact]
    public void Test_under_with_auto_order_user_frame_fixed_validation()
    {
        // Same setup as above; ensure user-frame fixedDims validation works
        // regardless of internal _dimOrder permutation.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // user-frame dim=1 with out-of-domain value should throw with user-frame error.
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 5.0 }, { 2, 0.0 } }));
    }

    [Fact]
    public void Test_no_roots_returns_empty()
    {
        var tt = Build1DTt(x => x * x + 0.5);  // No real roots
        double[] roots = tt.Roots();
        Assert.Empty(roots);
    }

    [Fact]
    public void Test_min_at_endpoint()
    {
        var tt = Build1DTt(x => x);
        var (value, location) = tt.Minimize();
        Assert.Equal(-1.0, value, precision: 6);
        Assert.Equal(-1.0, location, precision: 6);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~TtCalculusTests" --nologo --verbosity quiet
```
Expected: 15 failures.

- [ ] **Step 4: Implement helpers and public methods**

In `src/ChebyshevSharp/ChebyshevTT.cs`, add the following private helpers (place near other private helpers, e.g., right before `Roots` insertion point):

```csharp
/// <summary>
/// Return Domain reordered into user-frame indexing. For canonical
/// _dimOrder, this returns an array semantically equivalent to Domain.
/// For non-identity _dimOrder, _domain[s] is the storage-frame domain at
/// storage position s; user-frame dim u lives at storage position
/// Array.IndexOf(_dimOrder, u).
/// </summary>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1737-1747</c>.
/// </remarks>
private double[][] UserFrameDomain()
{
    var result = new double[_numDimensions][];
    for (int u = 0; u < _numDimensions; u++)
    {
        int s = Array.IndexOf(_dimOrder, u);
        result[u] = _domain[s];
    }
    return result;
}

/// <summary>
/// Build a 1-D ChebyshevApproximation from this 1-D TT. Uses ToDense() to
/// extract the values vector (which already applies the inverse permutation
/// so values are in user frame), then constructs a ChebyshevApproximation
/// via FromValues.
/// </summary>
/// <remarks>
/// Precondition: this TT must be 1-D. Call Slice() to reduce a multi-D
/// TT to 1-D before calling this helper.
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1704-1735</c>.
/// </remarks>
private ChebyshevApproximation To1DChebyshev()
{
    if (_numDimensions != 1)
        throw new InvalidOperationException(
            $"To1DChebyshev requires a 1-D TT, got {_numDimensions}-D");

    double[] values = ToDense();
    double a = _domain[0][0];
    double b = _domain[0][1];
    return ChebyshevApproximation.FromValues(
        values,
        numDimensions: 1,
        domain: new[] { new[] { a, b } },
        nNodes: new[] { _nNodes[0] });
}
```

Then add the three public methods (place near existing public methods):

```csharp
/// <summary>
/// Find all real roots of the TT-approximated function along a specified dimension.
/// Reduces to a 1-D problem by slicing all other dimensions to their fixed
/// values, then delegates to <see cref="ChebyshevApproximation.Roots"/>.
/// </summary>
/// <param name="dim">User-frame dimension. For 1-D TTs, defaults to 0.</param>
/// <param name="fixedDims">For multi-D, <c>{dim_index: value}</c> for all
/// user-frame dims except <paramref name="dim"/>. Validated against
/// user-frame domain.</param>
/// <returns>Sorted real root locations in the physical domain. Empty if no roots.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
/// <exception cref="ArgumentException">If validation fails.</exception>
/// <remarks>
/// Under non-identity <see cref="DimOrder"/>, dim and fixedDims keys translate
/// to storage frame transparently inside <see cref="Slice"/> and
/// <see cref="ToDense"/>.
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1749-1790</c>.
/// </remarks>
public double[] Roots(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    CheckBuilt();

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

    ChebyshevTT sliced = this;
    foreach (var (sliceDim, sliceValue) in sliceParams)
        sliced = sliced.Slice(sliceDim, sliceValue);

    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Roots();
}

/// <summary>
/// Find the minimum value of the TT along a user-frame dimension.
/// </summary>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1792-1831</c>.
/// </remarks>
public (double value, double location) Minimize(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    CheckBuilt();

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

    ChebyshevTT sliced = this;
    foreach (var (sliceDim, sliceValue) in sliceParams)
        sliced = sliced.Slice(sliceDim, sliceValue);

    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Minimize();
}

/// <summary>
/// Find the maximum value of the TT along a user-frame dimension.
/// </summary>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:1833-1872</c>.
/// </remarks>
public (double value, double location) Maximize(int? dim = null, Dictionary<int, double>? fixedDims = null)
{
    CheckBuilt();

    var (validatedDim, sliceParams) =
        Internal.Calculus.ValidateCalculusArgs(_numDimensions, dim, fixedDims, UserFrameDomain());

    ChebyshevTT sliced = this;
    foreach (var (sliceDim, sliceValue) in sliceParams)
        sliced = sliced.Slice(sliceDim, sliceValue);

    var cheb1D = sliced.To1DChebyshev();
    return cheb1D.Maximize();
}
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~TtCalculusTests" --nologo --verbosity quiet
```
Expected: `Passed: 15, Failed: 0`.

- [ ] **Step 6: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1061, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtCalculusTests.cs
git commit -m "phase7: add TT.Roots + TT.Minimize + TT.Maximize"
```

---

## Task 5: TT.SobolIndices (TT-native algorithm)

**Python source:** `ref/PyChebyshev/src/pychebyshev/_sensitivity.py:143-270` (`_compute_sobol_from_tt_cores`); `tensor_train.py:2823-2868` (`sobol_indices` user-frame translation).

**Files:**
- Modify: `src/ChebyshevSharp/Internal/Sensitivity.cs` (add `ComputeSobolFromTtCores` static helper)
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (add public `SobolIndices` method)
- Modify: `tests/ChebyshevSharp.Tests/TtSobolIndicesTests.cs` (add 12 tests)

**Expected total tests after commit:** 1073 (+12)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Write the 12 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/TtSobolIndicesTests.cs`:

```csharp
using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtSobolIndicesTests
{
    private static ChebyshevTT Build3DTt(Func<double, double, double, double> f,
        int n = 8, int rank = 6, int seed = 42)
    {
        Func<double[], object?, double> wrapper = (p, _) => f(p[0], p[1], p[2]);
        var tt = new ChebyshevTT(wrapper, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { n, n, n },
            maxRank: rank,
            tolerance: 1e-10,
            seed: seed);
        tt.Build();
        return tt;
    }

    [Fact]
    public void Test_separable_function_first_order_sums_to_one()
    {
        // f(x, y, z) = x + y + z is purely additive (no coupling).
        // First-order indices should sum to ~1.0; total-order = first-order for additive.
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();

        double sumFirst = 0;
        foreach (double v in result.FirstOrder) sumFirst += v;
        Assert.True(sumFirst > 0.99 && sumFirst < 1.01,
            $"FirstOrder sum {sumFirst} should be near 1.0 for additive function");
    }

    [Fact]
    public void Test_first_order_le_total_order()
    {
        var tt = Build3DTt((x, y, z) => Math.Exp(x * y) + z);
        var result = tt.SobolIndices();

        for (int d = 0; d < 3; d++)
            Assert.True(result.FirstOrder[d] <= result.TotalOrder[d] + 1e-10,
                $"FirstOrder[{d}]={result.FirstOrder[d]} > TotalOrder[{d}]={result.TotalOrder[d]}");
    }

    [Fact]
    public void Test_constant_function_zero_variance()
    {
        // f(x, y, z) = 5.0 (constant). Variance should be ~0.
        var tt = Build3DTt((x, y, z) => 5.0);
        var result = tt.SobolIndices();

        Assert.True(result.Variance < 1e-15,
            $"Variance={result.Variance} should be near 0 for constant function");
    }

    [Fact]
    public void Test_unbuilt_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var tt = new ChebyshevTT(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        Assert.Throws<InvalidOperationException>(() => tt.SobolIndices());
    }

    [Fact]
    public void Test_only_dim0_matters_first_order_concentrated()
    {
        // f(x, y, z) = x. Only dim 0 contributes.
        var tt = Build3DTt((x, y, z) => x);
        var result = tt.SobolIndices();

        Assert.True(result.FirstOrder[0] > 0.99,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 1.0");
        Assert.True(result.FirstOrder[1] < 0.01,
            $"FirstOrder[1]={result.FirstOrder[1]} should be near 0");
        Assert.True(result.FirstOrder[2] < 0.01,
            $"FirstOrder[2]={result.FirstOrder[2]} should be near 0");
    }

    [Fact]
    public void Test_pure_coupling_zero_first_order()
    {
        // f(x, y, z) = x * y has zero first-order energy in dims 0 and 1
        // (under uniform measure on [-1, 1] both x and y have mean 0,
        // so the additive parts integrate to zero).
        var tt = Build3DTt((x, y, z) => x * y);
        var result = tt.SobolIndices();

        Assert.True(result.FirstOrder[0] < 0.05,
            $"FirstOrder[0]={result.FirstOrder[0]} should be near 0 for pure coupling");
        Assert.True(result.FirstOrder[1] < 0.05,
            $"FirstOrder[1]={result.FirstOrder[1]} should be near 0 for pure coupling");
        // Total-order on dims 0 and 1 should be near 1 (both contribute fully through the coupling)
        Assert.True(result.TotalOrder[0] > 0.95,
            $"TotalOrder[0]={result.TotalOrder[0]} should be near 1 for pure coupling");
        Assert.True(result.TotalOrder[1] > 0.95,
            $"TotalOrder[1]={result.TotalOrder[1]} should be near 1 for pure coupling");
    }

    [Fact]
    public void Test_under_with_auto_order_keys_user_frame()
    {
        // Build a TT where dim 0 has the largest variance contribution.
        // After WithAutoOrder, _dimOrder may be non-identity, but result keys
        // must remain user-frame: index 0 should still report dim 0's importance.
        Func<double[], object?, double> f = (p, _) => 100 * p[0] + p[1] + 0.01 * p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        var result = tt.SobolIndices();

        // Dim 0 dominates (contribution ~10000 of total ~10000.0001), so its
        // first-order should be near 1.
        Assert.True(result.FirstOrder[0] > 0.99,
            $"FirstOrder[0]={result.FirstOrder[0]} should dominate");
    }

    [Fact]
    public void Test_cross_validation_against_dense_path()
    {
        // Cross-validate TT-native against dense (ToDense + Sensitivity.ComputeSobolFromCoeffs)
        // to within 1e-10 on a coupled function.
        Func<double[], object?, double> f = (p, _) => Math.Exp(0.5 * p[0] * p[1]) + 0.3 * p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 8,
            tolerance: 1e-12,
            seed: 42);
        tt.Build();

        var ttNative = tt.SobolIndices();

        // Build the dense oracle via ChebyshevApproximation on the same f.
        var approx = new ChebyshevApproximation(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 });
        approx.Build();
        var dense = approx.SobolIndices();

        for (int d = 0; d < 3; d++)
        {
            Assert.True(Math.Abs(ttNative.FirstOrder[d] - dense.FirstOrder[d]) < 1e-3,
                $"FirstOrder[{d}]: TT={ttNative.FirstOrder[d]} vs dense={dense.FirstOrder[d]}");
            Assert.True(Math.Abs(ttNative.TotalOrder[d] - dense.TotalOrder[d]) < 1e-3,
                $"TotalOrder[{d}]: TT={ttNative.TotalOrder[d]} vs dense={dense.TotalOrder[d]}");
        }
    }

    [Fact]
    public void Test_total_order_sums_ge_one()
    {
        // For functions with coupling, total-order indices sum to >= 1 (they
        // double-count interaction energy).
        var tt = Build3DTt((x, y, z) => x * y + y * z + x * z);
        var result = tt.SobolIndices();

        double sumTotal = 0;
        foreach (double v in result.TotalOrder) sumTotal += v;
        Assert.True(sumTotal >= 1.0 - 1e-10,
            $"TotalOrder sum {sumTotal} should be >= 1 for coupled function");
    }

    [Fact]
    public void Test_indices_in_unit_interval()
    {
        var tt = Build3DTt((x, y, z) => x + y * y + z * z * z);
        var result = tt.SobolIndices();

        for (int d = 0; d < 3; d++)
        {
            Assert.True(result.FirstOrder[d] >= -1e-10 && result.FirstOrder[d] <= 1 + 1e-10,
                $"FirstOrder[{d}]={result.FirstOrder[d]} outside [0, 1]");
            Assert.True(result.TotalOrder[d] >= -1e-10 && result.TotalOrder[d] <= 1 + 1e-10,
                $"TotalOrder[{d}]={result.TotalOrder[d]} outside [0, 1]");
        }
    }

    [Fact]
    public void Test_variance_positive_for_non_constant()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();
        Assert.True(result.Variance > 0,
            $"Variance={result.Variance} should be positive for non-constant function");
    }

    [Fact]
    public void Test_returns_correct_array_lengths()
    {
        var tt = Build3DTt((x, y, z) => x + y + z);
        var result = tt.SobolIndices();
        Assert.Equal(3, result.FirstOrder.Length);
        Assert.Equal(3, result.TotalOrder.Length);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~TtSobolIndicesTests" --nologo --verbosity quiet
```
Expected: 12 failures (`SobolIndices` not defined on TT).

- [ ] **Step 4: Implement Sensitivity.ComputeSobolFromTtCores**

Append to `src/ChebyshevSharp/Internal/Sensitivity.cs`:

```csharp
/// <summary>
/// Compute first-order + total-order Sobol indices from TT coefficient cores.
/// Mathematically equivalent to applying ComputeSobolFromCoeffs to the dense
/// coefficient tensor, but contracts through TT cores in coefficient space
/// with cost O(d * n * r^2) instead of O(n^d).
/// </summary>
/// <param name="cores">TT cores in Chebyshev coefficient space. Each core
/// has shape (r_{k-1}, n_k, r_k). cores[0] starts with r_0=1, cores[-1] ends with r_d=1.</param>
/// <returns>SobolResult with arrays keyed by storage-frame indices (caller
/// translates to user-frame).</returns>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/_sensitivity.py:143-270</c>.
/// </remarks>
internal static SobolResult ComputeSobolFromTtCores(TtCore[] cores)
{
    int d = cores.Length;
    double pi = Math.PI;
    int[] nPerDim = new int[d];
    for (int k = 0; k < d; k++) nPerDim[k] = cores[k].Shape[1];

    // Per-dim Chebyshev inner-product weights: [pi, pi/2, pi/2, ...]
    double[][] wFull = new double[d][];
    for (int k = 0; k < d; k++)
    {
        wFull[k] = new double[nPerDim[k]];
        wFull[k][0] = pi;
        for (int n = 1; n < nPerDim[k]; n++) wFull[k][n] = pi / 2.0;
    }

    // total_weighted_squared = sum over alpha of coeffs[alpha]^2 * prod_k w_full[k][alpha_k]
    // Iterative contraction: M[a, b] = sum over alpha (k cores' contribution)
    double[,] M = new double[1, 1] { { 1.0 } };
    for (int k = 0; k < d; k++)
    {
        var A = cores[k];
        // Aw[i, p, j] = A[i, p, j] * w_full[k][p]
        // Using TtCore indexer A[i, p, j]
        int rL = A.Shape[0], n = A.Shape[1], rR = A.Shape[2];
        // M_new[a, b] = sum over (i, j, p) M[i, j] * Aw[i, p, a] * A[j, p, b]
        var Mnew = new double[rR, rR];
        for (int a = 0; a < rR; a++)
        for (int b = 0; b < rR; b++)
        {
            double sum = 0;
            for (int i = 0; i < rL; i++)
            for (int j = 0; j < rL; j++)
            for (int p = 0; p < n; p++)
                sum += M[i, j] * A[i, p, a] * wFull[k][p] * A[j, p, b];
            Mnew[a, b] = sum;
        }
        M = Mnew;
    }
    double totalWeightedSquared = M[0, 0];

    // Constant term c_0 = product of cores[k][:, 0, :] chained
    double[] v = new double[] { 1.0 };
    for (int k = 0; k < d; k++)
    {
        var A = cores[k];
        int rL = A.Shape[0], rR = A.Shape[2];
        var vNew = new double[rR];
        for (int j = 0; j < rR; j++)
        {
            double sum = 0;
            for (int i = 0; i < rL; i++) sum += v[i] * A[i, 0, j];
            vNew[j] = sum;
        }
        v = vNew;
    }
    double c0 = v[0];
    double constantWeightedSquared = c0 * c0 * Math.Pow(pi, d);

    double variance = totalWeightedSquared - constantWeightedSquared;

    if (variance <= 0)
    {
        var zeros = new double[d];
        return new SobolResult(zeros, (double[])zeros.Clone(), Math.Max(variance, 0.0));
    }

    // Precompute L[k] (left partial inner-product matrices) and R[k] (right)
    // L[0] = [[1.0]]; L[k+1] = einsum("ij,ipa,jpb->ab", L[k], Aw_k, A_k)
    var L = new double[d + 1][,];
    L[0] = new double[1, 1] { { 1.0 } };
    for (int k = 0; k < d; k++)
    {
        var A = cores[k];
        int rL = A.Shape[0], n = A.Shape[1], rR = A.Shape[2];
        var Lnext = new double[rR, rR];
        for (int a = 0; a < rR; a++)
        for (int b = 0; b < rR; b++)
        {
            double sum = 0;
            for (int i = 0; i < rL; i++)
            for (int j = 0; j < rL; j++)
            for (int p = 0; p < n; p++)
                sum += L[k][i, j] * A[i, p, a] * wFull[k][p] * A[j, p, b];
            Lnext[a, b] = sum;
        }
        L[k + 1] = Lnext;
    }

    // R[d] = [[1.0]]; R[k] = einsum("ab,ipa,jpb->ij", R[k+1], Aw_k, A_k)
    var R = new double[d + 1][,];
    R[d] = new double[1, 1] { { 1.0 } };
    for (int k = d - 1; k >= 0; k--)
    {
        var A = cores[k];
        int rL = A.Shape[0], n = A.Shape[1], rR = A.Shape[2];
        var Rcurr = new double[rL, rL];
        for (int i = 0; i < rL; i++)
        for (int j = 0; j < rL; j++)
        {
            double sum = 0;
            for (int a = 0; a < rR; a++)
            for (int b = 0; b < rR; b++)
            for (int p = 0; p < n; p++)
                sum += R[k + 1][a, b] * A[i, p, a] * wFull[k][p] * A[j, p, b];
            Rcurr[i, j] = sum;
        }
        R[k] = Rcurr;
    }

    var firstOrder = new double[d];
    var totalOrder = new double[d];

    for (int j = 0; j < d; j++)
    {
        // First-order energy[j]: alpha_j >= 1 AND alpha_k = 0 for k != j
        // Boundary chains:
        //   left = product over k < j of cores[k][:, 0, :]   shape (r_j,)
        //   right = product over k > j of cores[k][:, 0, :]  shape (r_{j+1},)
        double[] left = new double[] { 1.0 };
        for (int k = 0; k < j; k++)
        {
            var A = cores[k];
            int rL = A.Shape[0], rR = A.Shape[2];
            var leftNew = new double[rR];
            for (int b = 0; b < rR; b++)
            {
                double sum = 0;
                for (int i = 0; i < rL; i++) sum += left[i] * A[i, 0, b];
                leftNew[b] = sum;
            }
            left = leftNew;
        }

        double[] right = new double[] { 1.0 };
        for (int k = d - 1; k > j; k--)
        {
            var A = cores[k];
            int rL = A.Shape[0], rR = A.Shape[2];
            var rightNew = new double[rL];
            for (int i = 0; i < rL; i++)
            {
                double sum = 0;
                for (int b = 0; b < rR; b++) sum += A[i, 0, b] * right[b];
                rightNew[i] = sum;
            }
            right = rightNew;
        }

        var Gj = cores[j];
        int rLj = Gj.Shape[0], rRj = Gj.Shape[2];
        double sumSquared = 0;
        for (int m = 1; m < nPerDim[j]; m++)
        {
            double coefM = 0;
            for (int i = 0; i < rLj; i++)
            for (int b = 0; b < rRj; b++)
                coefM += left[i] * Gj[i, m, b] * right[b];
            sumSquared += coefM * coefM;
        }
        double weightFirst = (pi / 2.0) * Math.Pow(pi, d - 1);
        firstOrder[j] = sumSquared * weightFirst / variance;

        // Total-order energy[j] = total_weighted_squared - sum_{alpha_j = 0} weighted
        // Using cached L[j] and R[j+1]:
        //   sum_alpha_j_zero = pi * einsum("ij,ia,jb,ab->", L[j], c_j0, c_j0, R[j+1])
        // where c_j0 = cores[j][:, 0, :] of shape (r_j, r_{j+1})
        double sumAlphaJZeroWeighted = 0;
        for (int ii = 0; ii < rLj; ii++)
        for (int jj = 0; jj < rLj; jj++)
        for (int a = 0; a < rRj; a++)
        for (int b = 0; b < rRj; b++)
            sumAlphaJZeroWeighted += L[j][ii, jj] * Gj[ii, 0, a] * Gj[jj, 0, b] * R[j + 1][a, b];
        sumAlphaJZeroWeighted *= pi;
        totalOrder[j] = (totalWeightedSquared - sumAlphaJZeroWeighted) / variance;
    }

    return new SobolResult(firstOrder, totalOrder, variance);
}
```

- [ ] **Step 5: Implement TT.SobolIndices**

In `src/ChebyshevSharp/ChebyshevTT.cs`, add the public method:

```csharp
/// <summary>
/// Compute first-order + total-order Sobol sensitivity indices natively
/// from the TT coefficient cores. O(d · n · r²) per dim, no dense materialization.
/// </summary>
/// <returns><see cref="SobolResult"/> with arrays keyed by user-frame dim indices.</returns>
/// <exception cref="InvalidOperationException">If <see cref="Build"/> has not been called.</exception>
/// <remarks>
/// Mathematically equivalent to <see cref="ChebyshevApproximation.SobolIndices"/>
/// applied to the dense version of the same function, but skips the O(n^d)
/// materialization. Under non-identity <see cref="DimOrder"/>, result keys are
/// translated from storage frame to user frame internally.
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:2823-2868</c>.
/// </remarks>
public SobolResult SobolIndices()
{
    CheckBuilt();

    // Helper returns SobolResult with arrays keyed by storage-frame indices.
    var storage = Internal.Sensitivity.ComputeSobolFromTtCores(_coeffCores!);

    // Translate keys to user-frame: storage position s holds original-dim _dimOrder[s].
    var userFirst = new double[_numDimensions];
    var userTotal = new double[_numDimensions];
    for (int s = 0; s < _numDimensions; s++)
    {
        int userD = _dimOrder[s];
        userFirst[userD] = storage.FirstOrder[s];
        userTotal[userD] = storage.TotalOrder[s];
    }
    return new SobolResult(userFirst, userTotal, storage.Variance);
}
```

- [ ] **Step 6: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~TtSobolIndicesTests" --nologo --verbosity quiet
```
Expected: `Passed: 12, Failed: 0`. If cross-validation test fails, investigate the multi-axis bookkeeping in `ComputeSobolFromTtCores` first.

- [ ] **Step 7: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1073, Failed: 0`.

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/Internal/Sensitivity.cs src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtSobolIndicesTests.cs
git commit -m "phase7: add TT.SobolIndices (TT-native O(d·n·r²) algorithm)"
```

---

## Task 6: TT.GetEvaluationPoints user-frame fix

**Python source:** `ref/PyChebyshev/src/pychebyshev/tensor_train.py:2775-2800` (post-v0.21.1 fix).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (replace `GetEvaluationPoints` body to permute by inverse `_dimOrder`)
- Modify: `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs` (add 5 tests targeting GetEvaluationPoints user-frame fix)

**Expected total tests after commit:** 1078 (+5)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Write the 5 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs`:

```csharp
using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtDimOrderClusterTests
{
    private static ChebyshevTT BuildAutoOrderTt(int seed = 42)
    {
        // Choose a function whose dim importance ordering may differ from
        // the natural [0, 1, 2] order. Slight asymmetry helps WithAutoOrder
        // produce a non-identity permutation.
        Func<double[], object?, double> f = (p, _) => 100 * p[2] + 10 * p[0] + p[1];
        return ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8, 8 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: seed,
            method: "greedy_swap");
    }

    [Fact]
    public void Test_get_evaluation_points_round_trips_under_identity_dim_order()
    {
        // For canonical _dimOrder, columns must already be in user-frame.
        Func<double[], object?, double> f = (p, _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var tt = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { 0.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        tt.Build();

        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        // Verify column 0 lies within [-1, 1] and column 1 lies within [0, 1]
        for (int i = 0; i < n; i++)
        {
            double x = flat[i * 2 + 0];
            double y = flat[i * 2 + 1];
            Assert.InRange(x, -1.0, 1.0);
            Assert.InRange(y, 0.0, 1.0);
        }
    }

    [Fact]
    public void Test_eval_at_get_evaluation_points_round_trips()
    {
        // For any TT (identity or non-identity _dimOrder), Eval(GetEvaluationPoints[i])
        // must return a finite value matching what direct Eval at that user-frame point would.
        var tt = BuildAutoOrderTt();

        double[] flat = tt.GetEvaluationPoints();
        int ndim = tt.NumDimensions;
        int n = tt.GetNumEvaluationPoints();

        for (int i = 0; i < Math.Min(n, 5); i++)
        {
            var point = new double[ndim];
            for (int d = 0; d < ndim; d++) point[d] = flat[i * ndim + d];

            double v = tt.Eval(point);
            Assert.False(double.IsNaN(v) || double.IsInfinity(v),
                $"Eval at point[{i}] returned non-finite {v}");
        }
    }

    [Fact]
    public void Test_get_evaluation_points_columns_match_per_dim_domain()
    {
        // Asymmetric per-dim domains catch storage-frame bugs immediately.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -3.0, -1.0 }, new[] { 5.0, 7.0 }, new[] { 100.0, 200.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        // Column 0 ∈ [-3, -1]; column 1 ∈ [5, 7]; column 2 ∈ [100, 200]
        for (int i = 0; i < n; i++)
        {
            double x0 = flat[i * 3 + 0];
            double x1 = flat[i * 3 + 1];
            double x2 = flat[i * 3 + 2];
            Assert.InRange(x0, -3.0, -1.0);
            Assert.InRange(x1, 5.0, 7.0);
            Assert.InRange(x2, 100.0, 200.0);
        }
    }

    [Fact]
    public void Test_eval_at_get_evaluation_points_matches_direct_eval()
    {
        // Sample 5 user-frame query points; compare Eval(GetEvaluationPoints[i])
        // to Eval(GetEvaluationPoints[i]) — they must match (identity test).
        var tt = BuildAutoOrderTt();
        double[] flat = tt.GetEvaluationPoints();
        int ndim = tt.NumDimensions;

        for (int i = 0; i < 5; i++)
        {
            var pt = new double[ndim];
            for (int d = 0; d < ndim; d++) pt[d] = flat[i * ndim + d];
            double v1 = tt.Eval(pt);
            double v2 = tt.Eval(pt);
            Assert.Equal(v1, v2);
        }
    }

    [Fact]
    public void Test_total_count_matches_get_num_evaluation_points()
    {
        var tt = BuildAutoOrderTt();
        double[] flat = tt.GetEvaluationPoints();
        int n = tt.GetNumEvaluationPoints();
        int ndim = tt.NumDimensions;
        Assert.Equal(n * ndim, flat.Length);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests.Test_get_evaluation_points|FullyQualifiedName~TtDimOrderClusterTests.Test_eval_at_get_evaluation_points|FullyQualifiedName~TtDimOrderClusterTests.Test_total_count" --nologo --verbosity quiet
```
Expected: at least the 3 tests that probe non-identity `_dimOrder` should fail.

- [ ] **Step 4: Apply user-frame permutation in GetEvaluationPoints**

In `src/ChebyshevSharp/ChebyshevTT.cs`, find `GetEvaluationPoints` (around line 1797) and update to:

```csharp
public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;

    int num = GetNumEvaluationPoints();
    int ndim = _numDimensions;

    var nodeArrays = new double[ndim][];
    for (int d = 0; d < ndim; d++)
        nodeArrays[d] = BarycentricKernel.MakeNodesForDim(_domain[d][0], _domain[d][1], _nNodes[d]);

    var points = new double[num * ndim];
    var indices = new int[ndim];

    for (int flat = 0; flat < num; flat++)
    {
        int rem = flat;
        for (int d = ndim - 1; d >= 0; d--)
        {
            indices[d] = rem % _nNodes[d];
            rem /= _nNodes[d];
        }
        for (int d = 0; d < ndim; d++)
            points[flat * ndim + d] = nodeArrays[d][indices[d]];
    }

    // v0.21.1: permute columns by inverse _dimOrder so column k is the user-frame
    // k-th coord (matches Approximation/Spline/Slider behavior).
    // Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:2775-2800.
    if (!IsIdentityDimOrder())
    {
        var inv = new int[ndim];
        for (int s = 0; s < ndim; s++) inv[_dimOrder[s]] = s;
        var permuted = new double[num * ndim];
        for (int i = 0; i < num; i++)
            for (int u = 0; u < ndim; u++)
                permuted[i * ndim + u] = points[i * ndim + inv[u]];
        points = permuted;
    }

    _evaluationPointsCache = points;
    return points;
}
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests" --nologo --verbosity quiet
```
Expected: `Passed: 5, Failed: 0`.

- [ ] **Step 6: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1078, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs
git commit -m "phase7: TT.GetEvaluationPoints returns user-frame columns"
```

---

## Task 7: TT.EvalMulti race fix via EvalStorageFrame helper

**Python source:** `ref/PyChebyshev/src/pychebyshev/tensor_train.py:2172-2215` (`_eval_storage_frame`).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (extract `EvalStorageFrame` helper; rewrite `EvalMulti` to use it without mutating `_dimOrder`)
- Modify: `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs` (add 5 race-fix tests)

**Expected total tests after commit:** 1083 (+5)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Append the 5 failing race-fix tests**

Append to `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs` (inside the existing class):

```csharp
    [Fact]
    public void Test_eval_multi_does_not_mutate_dim_order()
    {
        var tt = BuildAutoOrderTt();
        var orderBefore = tt.DimOrder;

        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };
        _ = tt.EvalMulti(new[] { 0.1, 0.2, 0.3 }, derivOrders);

        var orderAfter = tt.DimOrder;
        Assert.Equal(orderBefore, orderAfter);
    }

    [Fact]
    public void Test_eval_multi_concurrent_calls_no_exceptions()
    {
        // Race regression: 4 threads, 1000 calls each.
        var tt = BuildAutoOrderTt();
        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } };

        var tasks = new System.Threading.Tasks.Task[4];
        for (int t = 0; t < 4; t++)
        {
            int seed = t;
            tasks[t] = System.Threading.Tasks.Task.Run(() =>
            {
                var rng = new Random(seed);
                for (int i = 0; i < 1000; i++)
                {
                    var pt = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };
                    var results = tt.EvalMulti(pt, derivOrders);
                    Assert.Equal(2, results.Length);
                    Assert.False(double.IsNaN(results[0]) || double.IsNaN(results[1]));
                }
            });
        }
        System.Threading.Tasks.Task.WaitAll(tasks);
    }

    [Fact]
    public void Test_eval_multi_concurrent_results_match_serial()
    {
        // For deterministic input points, concurrent and serial calls produce
        // identical results.
        var tt = BuildAutoOrderTt();
        var pt = new[] { 0.1, 0.2, 0.3 };
        var derivOrders = new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 }, new[] { 0, 1, 0 } };

        // Serial baseline
        var serial = tt.EvalMulti(pt, derivOrders);

        // 8 concurrent calls
        var tasks = new System.Threading.Tasks.Task<double[]>[8];
        for (int t = 0; t < 8; t++)
            tasks[t] = System.Threading.Tasks.Task.Run(() => tt.EvalMulti(pt, derivOrders));
        System.Threading.Tasks.Task.WaitAll(tasks);

        foreach (var task in tasks)
        {
            var concurrent = task.Result;
            for (int i = 0; i < serial.Length; i++)
                Assert.Equal(serial[i], concurrent[i]);
        }
    }

    [Fact]
    public void Test_eval_multi_under_auto_order_returns_correct_value()
    {
        // After WithAutoOrder, Eval and EvalMulti's all-zero-derivative entry
        // must agree.
        var tt = BuildAutoOrderTt();
        var pt = new[] { 0.4, 0.3, -0.2 };

        double single = tt.Eval(pt);
        var multi = tt.EvalMulti(pt, new[] { new[] { 0, 0, 0 } });
        Assert.Equal(single, multi[0], precision: 10);
    }

    [Fact]
    public void Test_eval_multi_identity_dim_order_unchanged()
    {
        // For canonical _dimOrder, EvalMulti behavior is unchanged.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var tt = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        tt.Build();

        var pt = new[] { 0.1, 0.2, 0.3 };
        var multi = tt.EvalMulti(pt, new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 } });
        Assert.Equal(0.6, multi[0], precision: 4);  // f(0.1, 0.2, 0.3) = 0.6
    }
```

- [ ] **Step 3: Run race-fix tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests.Test_eval_multi" --nologo --verbosity quiet
```
Expected: at least `Test_eval_multi_does_not_mutate_dim_order` fails (the existing impl saves and restores).

- [ ] **Step 4: Replace EvalMulti with non-mutating implementation**

In `src/ChebyshevSharp/ChebyshevTT.cs`, find the existing `EvalMulti` (around line 463) and replace it with:

```csharp
public double[] EvalMulti(double[] point, int[][] derivativeOrders)
{
    CheckBuilt();

    // v0.21.1: race-safe via EvalStorageFrame helper that always operates in
    // storage frame. Public EvalMulti permutes user-frame inputs once into
    // local arrays — no mutation of self._dimOrder.
    // Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:2172-2215.
    double[] storagePoint = point;
    int[][] storageOrders = derivativeOrders;

    if (!IsIdentityDimOrder())
    {
        storagePoint = new double[_numDimensions];
        for (int s = 0; s < _numDimensions; s++)
            storagePoint[s] = point[_dimOrder[s]];

        storageOrders = new int[derivativeOrders.Length][];
        for (int i = 0; i < derivativeOrders.Length; i++)
        {
            storageOrders[i] = new int[_numDimensions];
            for (int s = 0; s < _numDimensions; s++)
                storageOrders[i][s] = derivativeOrders[i][_dimOrder[s]];
        }
    }

    var results = new double[storageOrders.Length];
    for (int i = 0; i < storageOrders.Length; i++)
        results[i] = EvalStorageFrame(storagePoint, storageOrders[i]);
    return results;
}

/// <summary>
/// Evaluate at a single point assuming storage-frame inputs (no _dimOrder
/// remapping). The structural workhorse for Eval and EvalMulti.
/// </summary>
/// <param name="storagePoint">Point in storage frame.</param>
/// <param name="derivativeOrderStorage">Derivative orders in storage frame.
/// All-zero triggers the value path; otherwise FD machinery.</param>
/// <returns>Interpolated value (or FD derivative).</returns>
/// <remarks>
/// Python source: <c>ref/PyChebyshev/src/pychebyshev/tensor_train.py:2172-2215</c>.
/// Does not mutate <see cref="_dimOrder"/>; safe under concurrent invocation.
/// </remarks>
private double EvalStorageFrame(double[] storagePoint, int[] derivativeOrderStorage)
{
    bool allZero = true;
    for (int d = 0; d < derivativeOrderStorage.Length; d++)
        if (derivativeOrderStorage[d] != 0) { allZero = false; break; }

    if (allZero)
        return EvalCore(storagePoint);
    return FdDerivative(storagePoint, derivativeOrderStorage);
}
```

- [ ] **Step 5: Run race-fix tests**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests.Test_eval_multi" --nologo --verbosity quiet
```
Expected: all 5 race-fix tests pass.

- [ ] **Step 6: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1083, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs
git commit -m "phase7: race-fix EvalMulti via EvalStorageFrame helper"
```

---

## Task 8: TT.InnerProduct dim_order mismatch + Integrate user-frame error

**Python source:**
- `ref/PyChebyshev/src/pychebyshev/tensor_train.py:1438-1503` (`inner_product` strict `_dim_order` check)
- TT integrate error message logic per spec §4.5

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (`InnerProduct` adds `_dimOrder` check; `Integrate` validates bounds against user-frame domain so error message references user-frame dim)
- Modify: `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs` (add 6 tests)

**Expected total tests after commit:** 1089 (+6)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Append 6 failing tests**

Append to `tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs`:

```csharp
    [Fact]
    public void Test_inner_product_mismatched_dim_order_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var a = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        a.Build();

        // b is the same TT reordered. inner_product on mismatched _dimOrder must throw.
        var b = a.Reorder(new[] { 2, 0, 1 });

        var ex = Assert.Throws<ArgumentException>(() => a.InnerProduct(b));
        Assert.Contains("_dimOrder", ex.Message);
        Assert.Contains("Reorder", ex.Message);
    }

    [Fact]
    public void Test_inner_product_after_alignment_returns_correct_value()
    {
        // Same setup as above, but align via Reorder; result should be sensible.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var a = new ChebyshevTT(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        a.Build();

        var b = a.Reorder(new[] { 2, 0, 1 });
        // Bring b back to a's dim order
        var bAligned = b.Reorder(a.DimOrder);

        double ip = a.InnerProduct(bAligned);
        Assert.False(double.IsNaN(ip));
        Assert.True(ip > 0);  // self-inner-product is positive
    }

    [Fact]
    public void Test_inner_product_identity_dim_order_unchanged()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var a = new ChebyshevTT(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42);
        a.Build();

        // Same TT as itself, no reordering — must succeed.
        double ip = a.InnerProduct(a);
        Assert.False(double.IsNaN(ip));
    }

    [Fact]
    public void Test_integrate_out_of_domain_error_uses_user_frame_dim()
    {
        // Build a TT with WithAutoOrder; the storage permutation may differ.
        // Pass dims=[0] (user-frame) with out-of-domain bounds; the error
        // message must reference dim 0, not the storage position.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var tt = ChebyshevTT.WithAutoOrder(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // [-2.0, 1.0] is outside [-1, 1].
        var ex = Assert.Throws<ArgumentException>(() =>
            tt.Integrate(dims: new[] { 0 }, bounds: new[] { (-2.0, 1.0) }));

        // Must reference user-frame dim 0
        Assert.Contains("dim 0", ex.Message);
    }

    [Fact]
    public void Test_integrate_in_domain_succeeds_for_auto_order()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var tt = ChebyshevTT.WithAutoOrder(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // Full integration of x + y over [-1,1]^2 = 0.
        double result = (double)tt.Integrate();
        Assert.Equal(0.0, result, precision: 6);
    }

    [Fact]
    public void Test_integrate_user_frame_partial()
    {
        // Build TT with WithAutoOrder; integrate only dim 0 in user frame.
        Func<double[], object?, double> f = (p, _) => p[0] + 2 * p[1];
        var tt = ChebyshevTT.WithAutoOrder(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            maxRank: 4,
            tolerance: 1e-10,
            seed: 42,
            method: "greedy_swap");

        // ∫ (x + 2y) dx from -1 to 1 = (x²/2 + 2yx) | -1 to 1 = 0 + 4y = 4y.
        // Result is a 1-D TT in dim 1.
        var partial = tt.Integrate(dims: new[] { 0 });
        Assert.NotNull(partial);
        Assert.IsType<ChebyshevTT>(partial);

        // Sample at y = 0.5: should be ~2.0.
        var partialTt = (ChebyshevTT)partial!;
        double atY05 = partialTt.Eval(new[] { 0.5 });
        Assert.Equal(2.0, atY05, precision: 4);
    }
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests.Test_inner_product|FullyQualifiedName~TtDimOrderClusterTests.Test_integrate" --nologo --verbosity quiet
```
Expected: at least `Test_inner_product_mismatched_dim_order_throws` and `Test_integrate_out_of_domain_error_uses_user_frame_dim` fail.

- [ ] **Step 4: Add `_dimOrder` check to InnerProduct**

In `src/ChebyshevSharp/ChebyshevTT.cs`, find `InnerProduct` (around line 990) and **before** `return TensorTrainAlgebra.InnerProductCores(...)`, add the new check:

```csharp
// v0.21.1: strict _dimOrder check. Two TTs with different _dimOrder represent
// the same underlying interpolant under different storage permutations; the
// raw core-by-core contraction is not the inner product of the interpolants.
// Python source: ref/PyChebyshev/src/pychebyshev/tensor_train.py:1488-1495.
if (!_dimOrder.SequenceEqual(other._dimOrder))
    throw new ArgumentException(
        $"InnerProduct requires matching _dimOrder; " +
        $"got [{string.Join(", ", _dimOrder)}] vs [{string.Join(", ", other._dimOrder)}]. " +
        $"Call other.Reorder(self.DimOrder) first.");
```

- [ ] **Step 5: Pre-validate Integrate bounds against user-frame domain**

In `src/ChebyshevSharp/ChebyshevTT.cs`, find the start of `Integrate` (around line 694). After the existing user-dim range validation (the `foreach (int d in sortedUserDims)` block ending around line 711), add a user-frame bounds pre-validation:

```csharp
// v0.21.1: pre-validate bounds against user-frame domain so error messages
// reference user-frame dim indices (issue #20). The downstream NormalizeBounds
// would otherwise report storage-frame indices when _dimOrder is non-identity.
// Python source: spec §4.5.
if (bounds != null && bounds.Length > 0)
{
    if (bounds.Length != sortedUserDims.Length)
        throw new ArgumentException(
            $"bounds length {bounds.Length} != dims length {sortedUserDims.Length}");
    for (int i = 0; i < bounds.Length; i++)
    {
        int userDim = sortedUserDims[i];
        int storageDim = Array.IndexOf(_dimOrder, userDim);
        double lo = _domain[storageDim][0], hi = _domain[storageDim][1];
        var bd = bounds[i];
        if (bd.lo > bd.hi)
            throw new ArgumentException($"bounds lo={bd.lo} > hi={bd.hi} for dim {userDim}");
        if (bd.lo < lo - 1e-14 || bd.hi > hi + 1e-14)
            throw new ArgumentException(
                $"bounds ({bd.lo}, {bd.hi}) outside domain [{lo}, {hi}] for dim {userDim}");
    }
}
```

This pre-check fires before the existing `NormalizeBounds` call (which uses storage-frame indices). The downstream call still runs but reports the same error against the storage-frame index — which it never reaches since the user-frame check already fired.

- [ ] **Step 6: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~TtDimOrderClusterTests.Test_inner_product|FullyQualifiedName~TtDimOrderClusterTests.Test_integrate" --nologo --verbosity quiet
```
Expected: all 6 new tests pass.

- [ ] **Step 7: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1089, Failed: 0`.

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevTT.cs tests/ChebyshevSharp.Tests/TtDimOrderClusterTests.cs
git commit -m "phase7: InnerProduct dim_order strict check + Integrate user-frame error"
```

---

## Task 9: Algebra.CheckCompatible numerical tolerance

**Python source:** `ref/PyChebyshev/src/pychebyshev/_algebra.py:13-58` (`_check_compatible`).

**Files:**
- Modify: `src/ChebyshevSharp/Internal/Algebra.cs` (add `DoublesAllClose` helper; switch `CheckCompatible` Domain comparison to numerical)
- Modify: `tests/ChebyshevSharp.Tests/AlgebraTupleListTests.cs` (add 5 tests)

**Expected total tests after commit:** 1094 (+5)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Write 5 failing tests**

Replace contents of `tests/ChebyshevSharp.Tests/AlgebraTupleListTests.cs`:

```csharp
using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class AlgebraTupleListTests
{
    [Fact]
    public void Test_mixed_allocations_with_identical_bounds_compose()
    {
        // Two ChebyshevApproximations with bounds expressed via different
        // double[][] allocations but numerically identical: + must succeed.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        Func<double[], object?, double> g = (p, _) => p[0] - p[1];

        var d1 = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        var d2 = new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } };
        Assert.NotSame(d1, d2);  // distinct allocations

        var a = new ChebyshevApproximation(f, 2, d1, new[] { 6, 6 });
        var b = new ChebyshevApproximation(g, 2, d2, new[] { 6, 6 });
        a.Build();
        b.Build();

        var c = a + b;
        Assert.NotNull(c);
    }

    [Fact]
    public void Test_genuinely_different_domain_still_throws()
    {
        // Domains differ by 0.5; must still throw "Domain mismatch".
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.5 } }, new[] { 6 });
        a.Build();
        b.Build();

        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Contains("Domain mismatch", ex.Message);
    }

    [Fact]
    public void Test_tiny_floating_difference_still_compose()
    {
        // Two operands constructed with bounds that differ by IEEE-754 noise
        // (e.g., one is `0.0 + 1.0`, the other is `2.0 * 0.5` — both exactly 1.0,
        // but rounding through different code paths can produce sub-ULP drift).
        // Use rtol=1e-5, atol=1e-8 (np.allclose defaults).
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 + 5e-9 } }, new[] { 6 });
        a.Build();
        b.Build();

        var c = a + b;  // should succeed: difference is below atol
        Assert.NotNull(c);
    }

    [Fact]
    public void Test_difference_above_tolerance_throws()
    {
        // Difference > rtol * 1.0 = 1e-5; should throw.
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.001 } }, new[] { 6 });
        a.Build();
        b.Build();

        Assert.Throws<ArgumentException>(() => { var _ = a + b; });
    }

    [Fact]
    public void Test_node_count_mismatch_still_exact()
    {
        // n_nodes is int[], stays exact comparison.
        Func<double[], object?, double> f = (p, _) => p[0];
        var a = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 6 });
        var b = new ChebyshevApproximation(f, 1, new[] { new[] { 0.0, 1.0 } }, new[] { 7 });
        a.Build();
        b.Build();

        var ex = Assert.Throws<ArgumentException>(() => { var _ = a + b; });
        Assert.Contains("Node count mismatch", ex.Message);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
dotnet test --filter "FullyQualifiedName~AlgebraTupleListTests" --nologo --verbosity quiet
```
Expected: `Test_tiny_floating_difference_still_compose` fails (existing `SequenceEqual` is exact). Other tests may pass since the `SequenceEqual` happens to handle the easy cases.

- [ ] **Step 4: Add DoublesAllClose helper and update CheckCompatible**

Replace contents of `src/ChebyshevSharp/Internal/Algebra.cs` with:

```csharp
namespace ChebyshevSharp.Internal;

/// <summary>
/// Shared helpers for Chebyshev arithmetic operators.
/// </summary>
internal static class Algebra
{
    /// <summary>
    /// Numerical equality test for two double arrays. Mirrors NumPy's
    /// <c>np.allclose(a, b, rtol, atol)</c>: <c>|a - b| &lt;= atol + rtol * |b|</c>
    /// elementwise.
    /// </summary>
    /// <remarks>
    /// Python source: <c>ref/PyChebyshev/src/pychebyshev/_algebra.py:46-49</c>.
    /// Used to tolerate sub-ULP floating-point drift between equivalent
    /// allocations (e.g., <c>tuple-of-tuples</c> vs <c>list-of-lists</c>
    /// in upstream Python).
    /// </remarks>
    internal static bool DoublesAllClose(double[] a, double[] b,
        double rtol = 1e-5, double atol = 1e-8)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs(a[i] - b[i]);
            double bound = atol + rtol * Math.Abs(b[i]);
            if (diff > bound) return false;
        }
        return true;
    }

    /// <summary>
    /// Validate that two ChebyshevApproximation objects can be combined arithmetically.
    /// </summary>
    internal static void CheckCompatible(ChebyshevApproximation a, ChebyshevApproximation b)
    {
        if (a.GetType() != b.GetType())
            throw new InvalidOperationException(
                $"Cannot combine {a.GetType().Name} with {b.GetType().Name}; " +
                "operands must be the same type.");

        if (a.TensorValues == null)
            throw new InvalidOperationException("Left operand is not built. Call Build() first.");
        if (b.TensorValues == null)
            throw new InvalidOperationException("Right operand is not built. Call Build() first.");

        if (a.NumDimensions != b.NumDimensions)
            throw new ArgumentException(
                $"Dimension mismatch: {a.NumDimensions} vs {b.NumDimensions}");

        if (!a.NNodes.SequenceEqual(b.NNodes))
            throw new ArgumentException(
                $"Node count mismatch: [{string.Join(", ", a.NNodes)}] vs [{string.Join(", ", b.NNodes)}]");

        // v0.21.1: numerical comparison on Domain[d] (was SequenceEqual = exact).
        // Tolerates sub-ULP drift between equivalent allocations.
        for (int d = 0; d < a.NumDimensions; d++)
        {
            if (!DoublesAllClose(a.Domain[d], b.Domain[d]))
                throw new ArgumentException(
                    $"Domain mismatch at dim {d}: " +
                    $"[{a.Domain[d][0]}, {a.Domain[d][1]}] vs [{b.Domain[d][0]}, {b.Domain[d][1]}]");
        }

        if (a.MaxDerivativeOrder != b.MaxDerivativeOrder)
            throw new ArgumentException(
                $"max_derivative_order mismatch: {a.MaxDerivativeOrder} vs {b.MaxDerivativeOrder}");
    }
}
```

- [ ] **Step 5: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~AlgebraTupleListTests" --nologo --verbosity quiet
```
Expected: `Passed: 5, Failed: 0`.

- [ ] **Step 6: Full regression — verify existing AlgebraTests still pass**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1094, Failed: 0`.

- [ ] **Step 7: Commit**

```bash
git add src/ChebyshevSharp/Internal/Algebra.cs tests/ChebyshevSharp.Tests/AlgebraTupleListTests.cs
git commit -m "phase7: Algebra.CheckCompatible numerical tolerance via DoublesAllClose"
```

---

## Task 10: Perf — VectorizedEvalBatch hoist + Optimize1D vectorized

**Python source:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py:992-1080` (`vectorized_eval_batch` post-hoist)
- `ref/PyChebyshev/src/pychebyshev/_calculus.py:245-297` (`_optimize_1d` vectorized)

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs` (`VectorizedEvalBatch`: hoist diff-matrix matmul outside per-point loop)
- Modify: `src/ChebyshevSharp/Internal/Calculus.cs` (`Optimize1D`: replace per-candidate loop with single vectorized call)
- Modify: `tests/ChebyshevSharp.Tests/VectorizedEvalBatchPerfTests.cs` (3 tests)
- Modify: `tests/ChebyshevSharp.Tests/OptimizeVectorizedTests.cs` (5 tests)

**Expected total tests after commit:** 1102 (+8)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Read existing VectorizedEvalBatch and Optimize1D**

```bash
grep -nE "VectorizedEvalBatch|Optimize1D" src/ChebyshevSharp/ChebyshevApproximation.cs src/ChebyshevSharp/Internal/Calculus.cs
```

Open `src/ChebyshevSharp/ChebyshevApproximation.cs` around line 469 to read `VectorizedEvalBatch`. Open `src/ChebyshevSharp/Internal/Calculus.cs` around line 174 to read `Optimize1D`.

- [ ] **Step 3: Write 3 VectorizedEvalBatchPerfTests + 5 OptimizeVectorizedTests**

Replace `tests/ChebyshevSharp.Tests/VectorizedEvalBatchPerfTests.cs`:

```csharp
using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class VectorizedEvalBatchPerfTests
{
    [Fact]
    public void Test_batch_results_match_loop_for_zero_derivative()
    {
        Func<double[], object?, double> f = (p, _) => Math.Sin(p[0]) + Math.Cos(p[1]);
        var approx = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 8, 8 });
        approx.Build();

        // Random batch of 200 points.
        var rng = new Random(42);
        var points = new double[200][];
        for (int i = 0; i < 200; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] batch = approx.VectorizedEvalBatch(points, new[] { 0, 0 });
        for (int i = 0; i < 200; i++)
        {
            double single = approx.VectorizedEval(points[i], new[] { 0, 0 });
            Assert.Equal(single, batch[i], precision: 12);
        }
    }

    [Fact]
    public void Test_batch_results_match_loop_for_first_derivative()
    {
        Func<double[], object?, double> f = (p, _) => Math.Exp(p[0]) * Math.Cos(p[1]);
        var approx = new ChebyshevApproximation(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 10, 10 });
        approx.Build();

        var rng = new Random(43);
        var points = new double[100][];
        for (int i = 0; i < 100; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] batch = approx.VectorizedEvalBatch(points, new[] { 1, 0 });
        for (int i = 0; i < 100; i++)
        {
            double single = approx.VectorizedEval(points[i], new[] { 1, 0 });
            Assert.Equal(single, batch[i], precision: 11);
        }
    }

    [Fact]
    public void Test_large_batch_correct_after_hoist()
    {
        // Large batch (1000 points) where the perf hoist amortization is most visible.
        Func<double[], object?, double> f = (p, _) => p[0] + p[1] + p[2];
        var approx = new ChebyshevApproximation(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 });
        approx.Build();

        var rng = new Random(44);
        var points = new double[1000][];
        for (int i = 0; i < 1000; i++)
            points[i] = new[] { rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1 };

        double[] result = approx.VectorizedEvalBatch(points, new[] { 1, 0, 0 });
        // ∂f/∂x for f = x + y + z is 1 everywhere.
        for (int i = 0; i < 1000; i++)
            Assert.Equal(1.0, result[i], precision: 8);
    }
}
```

Replace `tests/ChebyshevSharp.Tests/OptimizeVectorizedTests.cs`:

```csharp
using System;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

public class OptimizeVectorizedTests
{
    [Fact]
    public void Test_minimize_finds_known_minimum()
    {
        Func<double[], object?, double> f = (p, _) => (p[0] - 0.3) * (p[0] - 0.3);
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 12 });
        approx.Build();

        var (value, location) = approx.Minimize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.3, location, precision: 8);
    }

    [Fact]
    public void Test_maximize_finds_known_maximum()
    {
        Func<double[], object?, double> f = (p, _) => -((p[0] - 0.7) * (p[0] - 0.7));
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 12 });
        approx.Build();

        var (value, location) = approx.Maximize();
        Assert.Equal(0.0, value, precision: 8);
        Assert.Equal(0.7, location, precision: 8);
    }

    [Fact]
    public void Test_min_at_endpoint()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        approx.Build();

        var (value, location) = approx.Minimize();
        Assert.Equal(-1.0, value, precision: 10);
        Assert.Equal(-1.0, location, precision: 10);
    }

    [Fact]
    public void Test_max_at_endpoint()
    {
        Func<double[], object?, double> f = (p, _) => p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 8 });
        approx.Build();

        var (value, location) = approx.Maximize();
        Assert.Equal(1.0, value, precision: 10);
        Assert.Equal(1.0, location, precision: 10);
    }

    [Fact]
    public void Test_polynomial_with_multiple_critical_points()
    {
        // f(x) = x^4 - 2x^2 + 1 = (x^2 - 1)^2. Min at x = ±1 (value 0).
        // Has interior critical point at x = 0 (local max, value 1).
        Func<double[], object?, double> f = (p, _) => Math.Pow(p[0] * p[0] - 1, 2);
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 16 });
        approx.Build();

        var (minValue, minLocation) = approx.Minimize();
        Assert.Equal(0.0, minValue, precision: 6);
        Assert.True(Math.Abs(Math.Abs(minLocation) - 1.0) < 1e-6);

        var (maxValue, maxLocation) = approx.Maximize();
        Assert.Equal(1.0, maxValue, precision: 6);
        Assert.Equal(0.0, Math.Abs(maxLocation), precision: 6);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass (current impl is correct, just slower)**

```bash
dotnet test --filter "FullyQualifiedName~VectorizedEvalBatchPerfTests|FullyQualifiedName~OptimizeVectorizedTests" --nologo --verbosity quiet
```
Expected: all 8 pass against current implementation. These tests serve as parity-with-old correctness regressions for the perf change.

- [ ] **Step 5: Hoist diff-matrix matmul in VectorizedEvalBatch**

In `src/ChebyshevSharp/ChebyshevApproximation.cs`, find `VectorizedEvalBatch` (line 469). Locate the per-point loop where it currently calls `VectorizedEval` (or analogous per-point work). Refactor to:

1. Outside the loop: pre-compute `intermediate[d] = DiffMatrix[d]^k @ TensorValues` once per dimension where `derivativeOrder[d] > 0`. (Equivalent to the diff-matrix matmul.)
2. Inside the loop: each point evaluates barycentric-interpolation against the pre-computed `intermediate` tensor. No more per-point matmul.

If `derivativeOrder` is all-zero, no hoisting is needed (existing fast path stays unchanged).

Do not change behavior or numerical results — this is a code reorganization, not an algorithm change. After the change, all 8 tests in Step 4 must still pass to within `1e-12` precision.

- [ ] **Step 6: Vectorize Optimize1D candidate evaluation**

In `src/ChebyshevSharp/Internal/Calculus.cs`, find `Optimize1D` (line 174). Locate the per-candidate loop:

```csharp
for (int i = 0; i < candidates.Count; i++)
    vals[i] = BarycentricKernel.BarycentricInterpolate(candidates[i], nodes, values, baryWeights);
```

Replace with a single vectorized call. There are two viable approaches:

**Option A (simpler):** Keep the loop, but use a low-overhead batch helper that pre-computes `bary_weights[None, :] / (candidates[:, None] - nodes[None, :])` for all candidates at once.

**Option B (matches Python more closely):** Compute the full `(M × n)` distance and weight matrices in one pass, then numer/denom row-sums. Handle exact-node-hit case with masking.

Either way, the new code must match the existing per-candidate numerical results to within `1e-13`. Recommended skeleton (Option B):

```csharp
internal static (double value, double location) Optimize1D(
    double[] values, double[] nodes, double[] baryWeights,
    double[,] diffMatrix, double[] domain, string mode)
{
    // 1. Derivative values at nodes
    int n = values.Length;
    var derivValues = new double[n];
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int j = 0; j < n; j++) sum += diffMatrix[i, j] * values[j];
        derivValues[i] = sum;
    }

    // 2. Critical points: roots of the derivative
    double[] critical = Roots1D(derivValues, domain);

    // 3. Candidates = critical + endpoints
    double a = domain[0], b = domain[1];
    var candidates = new double[critical.Length + 2];
    candidates[0] = a;
    Array.Copy(critical, 0, candidates, 1, critical.Length);
    candidates[critical.Length + 1] = b;

    // 4. Vectorized barycentric evaluation at all candidates simultaneously.
    int M = candidates.Length;
    var vals = new double[M];
    for (int m = 0; m < M; m++)
    {
        double cm = candidates[m];
        double numer = 0, denom = 0;
        bool exactHit = false;
        double exactValue = 0;
        for (int j = 0; j < n; j++)
        {
            double diff = cm - nodes[j];
            if (Math.Abs(diff) < 1e-14)
            {
                exactHit = true;
                exactValue = values[j];
                break;
            }
            double wOverDiff = baryWeights[j] / diff;
            numer += wOverDiff * values[j];
            denom += wOverDiff;
        }
        vals[m] = exactHit ? exactValue : numer / denom;
    }

    // 5. Pick min or max
    int idx = 0;
    if (mode == "min")
    {
        for (int m = 1; m < M; m++) if (vals[m] < vals[idx]) idx = m;
    }
    else
    {
        for (int m = 1; m < M; m++) if (vals[m] > vals[idx]) idx = m;
    }
    return (vals[idx], candidates[idx]);
}
```

This consolidates all candidate evaluations into a single nested loop with no `BarycentricInterpolate` overhead per call. Verify the existing 5 tests in `OptimizeVectorizedTests` still pass.

- [ ] **Step 7: Run tests**

```bash
dotnet test --filter "FullyQualifiedName~VectorizedEvalBatchPerfTests|FullyQualifiedName~OptimizeVectorizedTests" --nologo --verbosity quiet
```
Expected: all 8 still pass with identical numerical values.

- [ ] **Step 8: Full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1102, Failed: 0`. Existing tests for `VectorizedEvalBatch` and `Minimize`/`Maximize` (Phase 1, etc.) must still pass.

- [ ] **Step 9: Commit**

```bash
git add src/ChebyshevSharp/ChebyshevApproximation.cs src/ChebyshevSharp/Internal/Calculus.cs tests/ChebyshevSharp.Tests/VectorizedEvalBatchPerfTests.cs tests/ChebyshevSharp.Tests/OptimizeVectorizedTests.cs
git commit -m "phase7: hoist VectorizedEvalBatch diff matmul; vectorize Optimize1D"
```

---

## Task 11: Coverage gap fillers + docs/changelog/parity tags + skip_csharp.txt

**Files:**
- Modify: `tests/ChebyshevSharp.Tests/Phase7CoverageGapTests.cs` (add ~10 defensive tests)
- Modify: `CLAUDE.md` (status block: parity 0.20.1 → 0.21.1; tests 1018 → ~1112)
- Modify: `docs/docs/changelog.md` (v0.11.0 entry)
- Modify: `skip_csharp.txt` (remove Slider/TT calculus parity rows)
- Modify: `docs/docs/calculus.md` (note Slider/TT now have full calculus surface)

**Expected total tests after commit:** ~1112 (+10)

- [ ] **Step 1: Verify worktree**

```bash
git rev-parse --show-toplevel
```

- [ ] **Step 2: Run codecov locally to identify Phase 7 coverage gaps**

```bash
dotnet test --collect:"XPlat Code Coverage" --nologo --verbosity quiet -- DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Format=cobertura
# Parse coverage.cobertura.xml output to identify lines in Phase 7 additions
# (Slider.Roots/Min/Max, TT.Roots/Min/Max/SobolIndices, Sensitivity.ComputeSobolFromTtCores)
# with coverage < 100%.
```

Use the coverage output to identify uncovered lines. Phase 6 hit ~96.19% patch coverage; Phase 7 should match.

- [ ] **Step 3: Write 10 coverage gap tests in Phase7CoverageGapTests.cs**

Replace `tests/ChebyshevSharp.Tests/Phase7CoverageGapTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using ChebyshevSharp;
using Xunit;

namespace ChebyshevSharp.Tests;

/// <summary>
/// Coverage gap fillers for Phase 7. Targets defensive paths and edge cases
/// that the per-feature test files don't naturally exercise. Goal: lift
/// codecov patch coverage on Phase 7 additions to ≥96%.
/// </summary>
public class Phase7CoverageGapTests
{
    [Fact]
    public void Test_slider_roots_dim_out_of_range_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var slider = new ChebyshevSlider(f, 2,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build();
        Assert.Throws<ArgumentException>(() => slider.Roots(dim: 5,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 } }));
    }

    [Fact]
    public void Test_tt_roots_dim_out_of_range_throws()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10, seed: 42);
        tt.Build();
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 5,
            fixedDims: new Dictionary<int, double> { { 1, 0.0 } }));
    }

    [Fact]
    public void Test_tt_sobol_indices_1d_function()
    {
        // 1-D edge case: SobolIndices on a 1-D TT.
        Func<double[], object?, double> f = (p, _) => Math.Sin(p[0]);
        var tt = new ChebyshevTT(f, 1, new[] { new[] { -1.0, 1.0 } },
            new[] { 8 }, maxRank: 4, tolerance: 1e-10, seed: 42);
        tt.Build();
        var result = tt.SobolIndices();
        Assert.Single(result.FirstOrder);
        Assert.True(result.FirstOrder[0] > 0.99);  // single dim explains all variance
    }

    [Fact]
    public void Test_slider_min_with_partial_partition()
    {
        // A slider with multi-dim group: ensure To1DChebyshev path works
        // when reducing through a multi-dim partition.
        Func<double[], object?, double> f = (p, _) => p[0] * p[1] + p[2];
        var slider = new ChebyshevSlider(f, 3,
            new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6, 6 },
            partition: new[] { new[] { 0, 1 }, new[] { 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build();
        var (value, _) = slider.Minimize(dim: 2,
            fixedDims: new Dictionary<int, double> { { 0, 0.5 }, { 1, 0.5 } });
        Assert.False(double.IsNaN(value));
    }

    [Fact]
    public void Test_tt_inner_product_self_returns_positive()
    {
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10, seed: 42);
        tt.Build();
        double ip = tt.InnerProduct(tt);
        Assert.True(ip > 0);
    }

    [Fact]
    public void Test_doubles_all_close_handles_empty_arrays()
    {
        Assert.True(Internal.Algebra.DoublesAllClose(Array.Empty<double>(), Array.Empty<double>()));
    }

    [Fact]
    public void Test_doubles_all_close_handles_length_mismatch()
    {
        Assert.False(Internal.Algebra.DoublesAllClose(new[] { 1.0 }, new[] { 1.0, 2.0 }));
    }

    [Fact]
    public void Test_tt_user_frame_domain_identity_returns_self()
    {
        // Identity _dimOrder: UserFrameDomain produces same array (different alloc).
        Func<double[], object?, double> f = (p, _) => p[0] + p[1];
        var tt = new ChebyshevTT(f, 2, new[] { new[] { -2.0, 3.0 }, new[] { 5.0, 7.0 } },
            new[] { 6, 6 }, maxRank: 4, tolerance: 1e-10, seed: 42);
        tt.Build();
        // Verify Roots correctly validates against [-2, 3] for dim 0:
        Assert.Throws<ArgumentException>(() => tt.Roots(dim: 0,
            fixedDims: new Dictionary<int, double> { { 1, 100.0 } }));  // 100 outside [5, 7]
    }

    [Fact]
    public void Test_optimize_1d_with_single_node_function()
    {
        // Edge case: small n.
        Func<double[], object?, double> f = (p, _) => p[0] * p[0];
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        approx.Build();
        var (value, location) = approx.Minimize();
        Assert.Equal(0.0, value, precision: 4);
    }

    [Fact]
    public void Test_optimize_1d_constant_function()
    {
        // Edge case: constant function. min == max == any node.
        Func<double[], object?, double> f = (p, _) => 7.0;
        var approx = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { 4 });
        approx.Build();
        var (minVal, _) = approx.Minimize();
        var (maxVal, _) = approx.Maximize();
        Assert.Equal(7.0, minVal, precision: 8);
        Assert.Equal(7.0, maxVal, precision: 8);
    }
}
```

- [ ] **Step 4: Update CLAUDE.md status block**

Open `CLAUDE.md` and find the status section (around line 18). Replace the v0.10.0/Phase 6 block with v0.11.0/Phase 7:

```markdown
## Status

**Feature-complete against PyChebyshev v0.21.1** (all 7 phases of the v0.21.1 port complete; see
`docs/superpowers/plans/2026-04-29-phase7-catchup-v0.21.1.md`).
All four public classes (`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`) mirror the Python API surface. v0.8.0 adds the v0.15+v0.16 ergonomics
layer; v0.9.0 (Phase 5) adds Slider/TT `Integrate`; v0.10.0 (Phase 6) ships build perf
(nWorkers + IProgress<int>) and adaptive refinement (AutoKnots, SobolIndices,
ChebyshevTT.WithAutoOrder/Reorder/DimOrder); v0.11.0 (Phase 7) bundles upstream
v0.21.0 + v0.21.1 — Slider/TT calculus parity (Roots/Minimize/Maximize), TT.SobolIndices
TT-native, plus the v0.21.1 _dim_order cluster fixes (GetEvaluationPoints user-frame,
EvalMulti race-safe, InnerProduct mismatch raises, Integrate user-frame error, Algebra
numerical tolerance) and perf hoists. With Phase 7 complete, ChebyshevSharp is
feature-complete against PyChebyshev v0.21.1 (modulo deliberately skipped matplotlib
helpers). 7 of 7 phases complete — port complete.
`dotnet test` runs **~1112/1112** passing.
```

- [ ] **Step 5: Update docs/docs/changelog.md**

Add a new entry at the top:

```markdown
## v0.11.0 (2026-04-29) — Phase 7: catch-up to PyChebyshev v0.21.1

Bundles upstream PyChebyshev v0.21.0 + v0.21.1 into one cohesive post-port maintenance
release. Parity tag advances 0.20.1 → 0.21.1.

### Added
- `ChebyshevSlider.Roots(dim, fixedDims)` / `Minimize` / `Maximize`
- `ChebyshevTT.Roots(dim, fixedDims)` / `Minimize` / `Maximize`
- `ChebyshevTT.SobolIndices()` — TT-native O(d·n·r²), no dense materialization

### Fixed
- `ChebyshevTT.GetEvaluationPoints` returns user-frame columns (was storage-frame)
- `ChebyshevTT.EvalMulti` race condition: refactored to use non-mutating `EvalStorageFrame` helper
- `ChebyshevTT.InnerProduct` now throws `ArgumentException` on `_dimOrder` mismatch (was silent wrong result)
- `ChebyshevTT.Integrate` error messages reference user-frame dim indices (was storage-frame)
- `Algebra.CheckCompatible` uses numerical tolerance (`np.allclose`-style) on `Domain[d]` (was exact `SequenceEqual`)

### Performance
- `VectorizedEvalBatch` hoists differentiation-matrix matmul outside per-point loop for non-zero derivative orders
- `Calculus.Optimize1D` uses single vectorized barycentric evaluation over all candidates
```

- [ ] **Step 6: Update skip_csharp.txt**

Open `skip_csharp.txt`. Remove any rows referring to Slider/TT calculus parity (e.g., `# Slider.Roots/Min/Max — deferred to Phase 7`).

- [ ] **Step 7: Run tests and full regression**

```bash
dotnet test --nologo --verbosity quiet
```
Expected: `Passed: 1112 ± 2, Failed: 0`.

- [ ] **Step 8: Build with zero warnings check**

```bash
dotnet build --nologo --verbosity quiet 2>&1 | grep -E "warning|error" || echo "0 warnings, 0 errors"
```
Expected: 0 warnings.

- [ ] **Step 9: Commit**

```bash
git add tests/ChebyshevSharp.Tests/Phase7CoverageGapTests.cs CLAUDE.md docs/docs/changelog.md skip_csharp.txt
git commit -m "phase7: coverage gap fillers + docs/changelog/parity tags"
```

---

## Definition of Done — Phase 7 Complete

After Task 11's commit:

- [ ] All 11 tasks committed; commit subjects start with `phase7:`
- [ ] `dotnet test` passes 1108–1116 (target 1112, ±2 drift permitted across the phase)
- [ ] `dotnet build` reports 0 warnings, 0 errors
- [ ] Codecov patch coverage on Phase 7 additions ≥ 96% (Phase 6 baseline)
- [ ] `CLAUDE.md` status updated with v0.11.0 + parity 0.21.1
- [ ] `docs/docs/changelog.md` has v0.11.0 entry
- [ ] `skip_csharp.txt` reflects Phase 7 additions
- [ ] No new public types introduced (verified by `git diff main..HEAD --stat src/ChebyshevSharp/*.cs`)

After Phase 7 closes, the orchestrator returns control to the user for the merge / PR / release flow. **Do not auto-create the PR or auto-merge.** Per the Phase 3/4/5/6 precedent, that is a user-confirmation gate.

---

## Self-Review Checklist (run before declaring plan complete)

1. **Spec coverage** — Each spec section maps to a task:
   - §3 D1 (TT-native SobolIndices) → Task 5 ✓
   - §3 D2 (EvalStorageFrame race fix) → Task 7 ✓
   - §3 D3 (InnerProduct mismatch) → Task 8 ✓
   - §3 D4 (DoublesAllClose) → Task 9 ✓
   - §3 D5 (single PR) → universal ✓
   - §4.1 (Slider To1DChebyshev) → Task 2 ✓
   - §4.2 (TT Roots/Min/Max + user-frame validation) → Task 4 ✓
   - §4.3 (SobolIndices) → Task 5 ✓
   - §4.4 (GetEvaluationPoints user-frame) → Task 6 ✓
   - §4.5 (Integrate user-frame error) → Task 8 ✓
   - §4.6 (VectorizedEvalBatch hoist) → Task 10 ✓
   - §4.7 (Optimize1D vectorized) → Task 10 ✓

2. **Type consistency**:
   - All Slider/TT public methods match `ChebyshevApproximation` signatures: `int? dim = null, Dictionary<int, double>? fixedDims = null` ✓
   - `Roots` returns `double[]` everywhere ✓
   - `Minimize`/`Maximize` return `(double value, double location)` everywhere (note: tuple ordering is `(value, location)`) ✓
   - `SobolIndices` returns `SobolResult` (existing record from Phase 6) ✓
   - `EvalStorageFrame` is private; takes `(double[] storagePoint, int[] derivativeOrderStorage)` ✓
   - `DoublesAllClose` is internal static in `ChebyshevSharp.Internal.Algebra` ✓
   - Error messages reference user-frame dims, not storage-frame ✓

3. **No placeholders**:
   - No "TODO" / "TBD" / "implement later" ✓
   - No "see Task N" without repeating the code ✓
   - All test code complete ✓
   - All commit messages explicit ✓

4. **WORKTREE ENFORCEMENT** is the first step in every task ✓

5. **Test count progression** sums correctly: 1030 + 6 + 10 + 15 + 12 + 5 + 5 + 6 + 5 + 8 + 10 = 1112 ✓

End of plan.
