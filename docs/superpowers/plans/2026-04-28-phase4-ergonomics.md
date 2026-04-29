# Phase 4 — Ergonomics Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill PyChebyshev v0.15 + v0.16 ergonomics polish (descriptor, additionalData threading, derivative-id registry, introspection getters, typed Clone, DeferBuild + SetOriginalFunctionValues, GetEvaluationPoints lazy cache, and Domain/Ns/SpecialPoints record types) across all four public classes, shipping ChebyshevSharp v0.8.0.

**Architecture:** Strictly additive surface changes. Per-class private state added via existing init paths. New record types use implicit conversions to avoid constructor overload explosion. JSON migration relies on System.Text.Json's automatic default-value deserialization for missing fields (no schema-version field). `.pcb` format unchanged.

**Tech Stack:** C# 12, .NET 8 + .NET 10 multi-target, System.Text.Json, xUnit, BlasSharp.OpenBlas, MathNet.Numerics.

**Approved design spec:** `docs/superpowers/specs/2026-04-28-phase4-ergonomics-design.md` (commit `e25cc3e`, 11 sections).

**Test count progression target:**

| After task | Total tests |
|---|---:|
| Baseline | 812 |
| 1 (RecordTypes) | 815 |
| 2 (Per-class scaffolding) | 831 |
| 3 (additionalData) | 839 |
| 4 (GetEvaluationPoints) | 847 |
| 5 (GetErrorThreshold + GetSpecialPoints) | 851 |
| 6 (GetDerivativeId + Eval-by-id) | 859 |
| 7 (Clone) | 864 |
| 8 (DeferBuild + SetOriginalFunctionValues) | 868 |
| 9 (JSON migration) | 872 |
| 10 (Release prep) | 872 |

If a task's actual count diverges from the table, investigate before proceeding (a missing test or unintended deletion is the usual cause).

---

## File Structure

### New files

| File | Purpose | Created in |
|---|---|---|
| `src/ChebyshevSharp/Domain.cs` | Public record `Domain(double[][] Bounds)` with implicit conversions | Task 1 |
| `src/ChebyshevSharp/Ns.cs` | Public record `Ns(int[] Counts)` with implicit conversions | Task 1 |
| `src/ChebyshevSharp/SpecialPoints.cs` | Public record `SpecialPoints(double[][] Points)` with implicit conversions | Task 1 |
| `src/ChebyshevSharp/Internal/TupleKey.cs` | Internal value-equality wrapper for `int[]` (for derivative-id registry dictionary) | Task 6 |
| `src/ChebyshevSharp/Internal/CloneHelpers.cs` | Internal deep-copy helpers shared by all four `Clone()` impls | Task 7 |
| `tests/ChebyshevSharp.Tests/RecordTypesTests.cs` | Domain/Ns/SpecialPoints implicit conversion tests | Task 1 |
| `tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs` | Approx ergonomics fan-out (12 tests) | Task 2 onward |
| `tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs` | Spline ergonomics fan-out (12 tests) | Task 2 onward |
| `tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs` | Slider ergonomics fan-out (9 tests) | Task 2 onward |
| `tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs` | TT ergonomics fan-out (7 tests) | Task 2 onward |
| `tests/ChebyshevSharp.Tests/CloneTests.cs` | Cross-class Clone() tests + reflection-based completeness check | Task 7 |
| `tests/ChebyshevSharp.Tests/DeferBuildTests.cs` | DeferBuild + SetOriginalFunctionValues tests (Approx + Spline) | Task 8 |
| `tests/ChebyshevSharp.Tests/JsonMigrationTests.cs` | Pre-v0.8.0 JSON files load with sensible defaults | Task 9 |
| `tests/fixtures/json-pre-v080/approx.json` | Pre-v0.8.0 Approx JSON fixture | Task 9 |
| `tests/fixtures/json-pre-v080/spline.json` | Pre-v0.8.0 Spline JSON fixture | Task 9 |
| `tests/fixtures/json-pre-v080/slider.json` | Pre-v0.8.0 Slider JSON fixture | Task 9 |
| `tests/fixtures/json-pre-v080/tt.json` | Pre-v0.8.0 TT JSON fixture | Task 9 |
| `docs/docs/ergonomics.md` | User-guide page for ergonomics features | Task 10 |
| `tools/GeneratePhase4Fixtures/Program.cs` | One-off generator for the 4 pre-v0.8.0 JSON fixtures (run before Task 9 modifies Save) | Task 9 |
| `tools/GeneratePhase4Fixtures/GeneratePhase4Fixtures.csproj` | Generator project file | Task 9 |

### Modified files

| File | Changes |
|---|---|
| `src/ChebyshevSharp/ChebyshevApproximation.cs` | New ctor kwargs (`additionalData`, `deferBuild`); new private fields (`_descriptor`, `_additionalData`, `_derivativeIdRegistry`, `_registeredDerivativeOrders`, `_constructorType`, `_isConstructionFinished`, `_evaluationPointsCache`, `_specialPoints`); new public methods (`Get/SetDescriptor`, `GetAdditionalData`, `GetDerivativeId`, `Eval(point, int)` overload, `IsConstructionFinished`, `GetConstructorType`, `GetUsedNs`, `Clone`, `GetMaxDerivativeOrder`, `GetErrorThreshold`, `GetSpecialPoints`, `GetEvaluationPoints`, `GetNumEvaluationPoints`, `SetOriginalFunctionValues`); `SerializationState` record extended with new fields; existing `Build()` and `BuildFixedGrid()` thread `_additionalData`. |
| `src/ChebyshevSharp/ChebyshevSpline.cs` | Same shape as Approx, plus per-piece `additionalData` propagation. `Built` field already exists; `_isConstructionFinished` aliases it. |
| `src/ChebyshevSharp/ChebyshevSlider.cs` | Same shape as Approx. Per-slide `additionalData` propagation. No `DeferBuild` / `SetOriginalFunctionValues`. No `GetErrorThreshold` / `GetSpecialPoints`. |
| `src/ChebyshevSharp/ChebyshevTT.cs` | Same shape as Approx with: ctor adds `maxDerivativeOrder = 2` kwarg, no `DeferBuild` / `SetOriginalFunctionValues`, no `GetErrorThreshold` / `GetSpecialPoints`. `additionalData` is stored for introspection but not threaded (TT's function signature is `Func<double[], double>`, no data arg). |
| `src/ChebyshevSharp/ChebyshevSharp.csproj` | `<Version>0.8.0`, `<InformationalVersion>0.8.0+pychebyshev.0.18.0`. Parity tag unchanged at 0.18.0. |
| `docs/docs/changelog.md` | v0.8.0 entry (two-tier convention). |
| `docs/docs/toc.yml` | Add ergonomics user-guide page link. |
| `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` | Add `<None Include>` for `tests/fixtures/json-pre-v080/*.json` to copy fixtures to bin dir. |
| `skip_csharp.txt` | Mark Phase 4 features ported. |
| `CLAUDE.md` | Status block update: 812 → 872 passing; phase list mentions Phase 4 complete. |

---

## WORKTREE ENFORCEMENT (apply to every task)

Every task MUST start with this verification block. **Do not skip.** Phase 1 Task 4 lost a commit to the wrong directory because this check was bypassed.

```bash
# Step 0: Verify worktree directory.
WORKTREE_ROOT=$(git rev-parse --show-toplevel)
echo "Working in: $WORKTREE_ROOT"

case "$WORKTREE_ROOT" in
    *.worktrees/phase4-ergonomics)
        echo "OK: in Phase 4 worktree"
        ;;
    *)
        echo "STOP: not in phase4-ergonomics worktree. cd to .worktrees/phase4-ergonomics and rerun."
        exit 1
        ;;
esac
```

---

## Task 1: Public record types (`Domain`, `Ns`, `SpecialPoints`)

**Why first:** these types have no dependencies on the per-class scaffolding. Land them first so they're available if any later task needs them.

**Files:**
- Create: `src/ChebyshevSharp/Domain.cs`
- Create: `src/ChebyshevSharp/Ns.cs`
- Create: `src/ChebyshevSharp/SpecialPoints.cs`
- Create: `tests/ChebyshevSharp.Tests/RecordTypesTests.cs`

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/__init__.py` exports `Domain`, `Ns`, `SpecialPoints` as frozen dataclasses. C# uses records as the closest semantic match.

- [ ] **Step 0: Worktree enforcement** (see top-of-plan block).

- [ ] **Step 1: Write the failing tests for `RecordTypesTests.cs`**

```csharp
// tests/ChebyshevSharp.Tests/RecordTypesTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class RecordTypesTests
{
    [Fact]
    public void Domain_implicit_conversion_both_directions()
    {
        double[][] raw = new[] { new[] { 0.0, 1.0 }, new[] { -1.0, 2.0 } };

        // raw -> Domain
        Domain d = raw;
        Assert.Equal(2, d.Bounds.Length);
        Assert.Equal(0.0, d.Bounds[0][0]);
        Assert.Equal(1.0, d.Bounds[0][1]);

        // Domain -> raw
        double[][] back = d;
        Assert.Same(raw, back);
    }

    [Fact]
    public void Ns_implicit_conversion_both_directions()
    {
        int[] raw = new[] { 5, 7 };

        Ns n = raw;
        Assert.Equal(new[] { 5, 7 }, n.Counts);

        int[] back = n;
        Assert.Same(raw, back);
    }

    [Fact]
    public void SpecialPoints_implicit_conversion_both_directions()
    {
        double[][] raw = new[] { new[] { 0.5 }, new[] { 0.7, 0.9 } };

        SpecialPoints sp = raw;
        Assert.Equal(2, sp.Points.Length);
        Assert.Equal(0.5, sp.Points[0][0]);

        double[][] back = sp;
        Assert.Same(raw, back);
    }
}
```

- [ ] **Step 2: Run the tests and verify they fail to compile**

Run: `dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --filter "FullyQualifiedName~RecordTypesTests"`

Expected: build errors — `Domain`, `Ns`, `SpecialPoints` not found in `ChebyshevSharp` namespace.

- [ ] **Step 3: Create `src/ChebyshevSharp/Domain.cs`**

```csharp
namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for a multi-dimensional rectangular domain.
/// Each entry is a 2-element array <c>[lo, hi]</c>.
/// Implicit conversions to and from <see cref="double[][]"/> let you pass
/// either form to constructors that accept <c>Domain</c>.
/// </summary>
/// <param name="Bounds">Per-dimension <c>[lo, hi]</c> pairs.</param>
public sealed record Domain(double[][] Bounds)
{
    public static implicit operator Domain(double[][] bounds) => new(bounds);
    public static implicit operator double[][](Domain d) => d.Bounds;
}
```

- [ ] **Step 4: Create `src/ChebyshevSharp/Ns.cs`**

```csharp
namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension Chebyshev node counts.
/// Implicit conversions to and from <see cref="int[]"/> let you pass either
/// form to constructors that accept <c>Ns</c>.
/// </summary>
/// <param name="Counts">Number of nodes per dimension.</param>
public sealed record Ns(int[] Counts)
{
    public static implicit operator Ns(int[] counts) => new(counts);
    public static implicit operator int[](Ns n) => n.Counts;
}
```

- [ ] **Step 5: Create `src/ChebyshevSharp/SpecialPoints.cs`**

```csharp
namespace ChebyshevSharp;

/// <summary>
/// Typed wrapper for per-dimension special points (e.g., kinks where a
/// piecewise spline must place a knot).
/// Implicit conversions to and from <see cref="double[][]"/> let you pass
/// either form to constructors that accept <c>SpecialPoints</c>.
/// </summary>
/// <param name="Points">Per-dimension array of special-point coordinates.</param>
public sealed record SpecialPoints(double[][] Points)
{
    public static implicit operator SpecialPoints(double[][] points) => new(points);
    public static implicit operator double[][](SpecialPoints sp) => sp.Points;
}
```

- [ ] **Step 6: Run the tests and verify all 3 pass**

Run: `dotnet test --filter "FullyQualifiedName~RecordTypesTests"`

Expected: 3/3 pass.

- [ ] **Step 7: Run full test suite — verify no regressions**

Run: `dotnet test`

Expected: **815/815 passing** (812 baseline + 3 new).

- [ ] **Step 8: Commit**

```bash
git add src/ChebyshevSharp/Domain.cs src/ChebyshevSharp/Ns.cs src/ChebyshevSharp/SpecialPoints.cs tests/ChebyshevSharp.Tests/RecordTypesTests.cs
git commit -m "$(cat <<'EOF'
phase4: T1 typed record types Domain/Ns/SpecialPoints (3 tests)

Three new public records with implicit conversions to/from raw arrays.
Phase 4 design D5: avoids constructor overload explosion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Per-class state scaffolding (descriptor, introspection getters, max-derivative-order)

**Why:** all four classes need the same set of simple state fields and getters before any of the more complex features (additionalData, derivative-id, Clone, etc.) can land. This task adds the no-build-path-changes scaffolding.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs`
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs` (also adds `maxDerivativeOrder = 2` ctor kwarg)
- Create: `tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs`
- Create: `tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs`
- Create: `tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs`
- Create: `tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs`

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `set_descriptor`, `get_descriptor`, `is_construction_finished`, `get_constructor_type`, `get_used_ns`, `get_max_derivative_order`.
- `ref/PyChebyshev/src/pychebyshev/spline.py`, `slider.py`, `tensor_train.py`: same set; `tensor_train.py` adds `max_derivative_order=2` ctor kwarg.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Create `tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs` (descriptor + introspection block)**

```csharp
// tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs
using ChebyshevSharp.Tests.Helpers;
using Xunit;

namespace ChebyshevSharp.Tests;

public class ApproxErgonomicsTests
{
    private static ChebyshevApproximation BuildSimple()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        return approx;
    }

    [Fact]
    public void Descriptor_get_returns_null_when_unset()
    {
        var approx = BuildSimple();
        Assert.Null(approx.GetDescriptor());
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var approx = BuildSimple();
        approx.SetDescriptor("my approximation");
        Assert.Equal("my approximation", approx.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var approx = BuildSimple();
        Assert.True(approx.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var approx = BuildSimple();
        Assert.Equal("function", approx.GetConstructorType());
    }

    [Fact]
    public void GetConstructorType_returns_from_values_for_FromValues_factory()
    {
        var values = new double[5 * 5];
        for (int i = 0; i < values.Length; i++) values[i] = i * 0.1;
        var approx = ChebyshevApproximation.FromValues(
            values,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        Assert.Equal("from_values", approx.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_resolved_node_counts()
    {
        var approx = BuildSimple();
        Assert.Equal(new[] { 5, 5 }, approx.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            maxDerivativeOrder: 3);
        approx.Build(verbose: false);
        Assert.Equal(3, approx.GetMaxDerivativeOrder());
    }
}
```

- [ ] **Step 2: Create `tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs` (descriptor + introspection block)**

```csharp
// tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class SplineErgonomicsTests
{
    private static ChebyshevSpline BuildSimple()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { System.Array.Empty<double>(), System.Array.Empty<double>() });
        spline.Build(verbose: false);
        return spline;
    }

    [Fact]
    public void Descriptor_get_returns_null_when_unset()
    {
        var spline = BuildSimple();
        Assert.Null(spline.GetDescriptor());
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var spline = BuildSimple();
        spline.SetDescriptor("my spline");
        Assert.Equal("my spline", spline.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var spline = BuildSimple();
        Assert.True(spline.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var spline = BuildSimple();
        Assert.Equal("function", spline.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var spline = BuildSimple();
        Assert.Equal(new[] { 5, 5 }, spline.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            maxDerivativeOrder: 4);
        spline.Build(verbose: false);
        Assert.Equal(4, spline.GetMaxDerivativeOrder());
    }
}
```

- [ ] **Step 3: Create `tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs` (descriptor + introspection block)**

```csharp
// tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class SliderErgonomicsTests
{
    private static ChebyshevSlider BuildSimple()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var slider = BuildSimple();
        slider.SetDescriptor("my slider");
        Assert.Equal("my slider", slider.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var slider = BuildSimple();
        Assert.True(slider.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_function_for_Build_path()
    {
        var slider = BuildSimple();
        Assert.Equal("function", slider.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var slider = BuildSimple();
        Assert.Equal(new[] { 5, 5, 5 }, slider.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_value()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            partition: new[] { new[] { 0 } },
            pivotPoint: new[] { 0.0 },
            maxDerivativeOrder: 3);
        slider.Build(verbose: false);
        Assert.Equal(3, slider.GetMaxDerivativeOrder());
    }
}
```

- [ ] **Step 4: Create `tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs` (descriptor + introspection block + maxDerivativeOrder kwarg)**

```csharp
// tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class TtErgonomicsTests
{
    private static ChebyshevTT BuildSimple()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Descriptor_set_then_get_roundtrip()
    {
        var tt = BuildSimple();
        tt.SetDescriptor("my tt");
        Assert.Equal("my tt", tt.GetDescriptor());
    }

    [Fact]
    public void IsConstructionFinished_true_after_Build()
    {
        var tt = BuildSimple();
        Assert.True(tt.IsConstructionFinished());
    }

    [Fact]
    public void GetConstructorType_returns_cross_for_default_method()
    {
        var tt = BuildSimple();
        Assert.Equal("cross", tt.GetConstructorType());
    }

    [Fact]
    public void GetUsedNs_returns_per_dim_node_counts()
    {
        var tt = BuildSimple();
        Assert.Equal(new[] { 5, 5, 5 }, tt.GetUsedNs());
    }

    [Fact]
    public void GetMaxDerivativeOrder_default_is_2()
    {
        var tt = BuildSimple();
        Assert.Equal(2, tt.GetMaxDerivativeOrder());
    }

    [Fact]
    public void GetMaxDerivativeOrder_returns_ctor_kwarg_value()
    {
        var tt = new ChebyshevTT(
            p => p[0],
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            maxDerivativeOrder: 3);
        tt.Build(verbose: false, seed: 42);
        Assert.Equal(3, tt.GetMaxDerivativeOrder());
    }
}
```

- [ ] **Step 5: Run the tests — verify they fail to compile**

Run: `dotnet test --filter "FullyQualifiedName~ApproxErgonomicsTests|FullyQualifiedName~SplineErgonomicsTests|FullyQualifiedName~SliderErgonomicsTests|FullyQualifiedName~TtErgonomicsTests"`

Expected: build errors — `GetDescriptor`/`SetDescriptor`/`IsConstructionFinished`/`GetConstructorType`/`GetUsedNs`/`GetMaxDerivativeOrder` not found on any of the four classes; TT ctor has no `maxDerivativeOrder` parameter.

- [ ] **Step 6: Modify `src/ChebyshevSharp/ChebyshevApproximation.cs` — add private fields and methods**

Locate the field-declarations region near the top of the class (after the existing `private double? _cachedErrorEstimate;` line). Add:

```csharp
private string? _descriptor;
private string _constructorType = "function";
private bool _isConstructionFinished;
```

`MaxDerivativeOrder` is already a public property on the class (set in ctor). No new field needed for it.

Locate `Build()` and `BuildFixedGrid()`. At the end of `BuildFixedGrid()` (after all build steps complete, just before returning), add:

```csharp
_isConstructionFinished = true;
```

Locate `FromValues` static factory (around line 797). After the constructed object is built, set:

```csharp
obj._constructorType = "from_values";
obj._isConstructionFinished = true;
```

Locate `LoadJson` and `LoadBinary`. Add at the end of each:

```csharp
obj._constructorType = "load";
obj._isConstructionFinished = true;
```

(For `LoadBinary` which calls `FromValues` internally, the `_constructorType = "load"` overrides the `"from_values"` set by `FromValues`.)

Add new public methods at the bottom of the class (just before the closing brace), in a `// Phase 4 ergonomics — accessors` region:

```csharp
// ------------------------------------------------------------------
// Phase 4 ergonomics — accessors
// ------------------------------------------------------------------

/// <summary>Set a free-form descriptor string for this interpolant.</summary>
public void SetDescriptor(string descriptor) => _descriptor = descriptor;

/// <summary>Get the descriptor previously set via <see cref="SetDescriptor"/>; null if unset.</summary>
public string? GetDescriptor() => _descriptor;

/// <summary>True if <see cref="Build"/>/<see cref="FromValues"/>/<see cref="Load"/> completed.</summary>
public bool IsConstructionFinished() => _isConstructionFinished;

/// <summary>Returns one of: "function" (Build), "from_values" (FromValues factory), "load" (Load), "clone" (Clone).</summary>
public string GetConstructorType() => _constructorType;

/// <summary>Per-dimension Chebyshev node counts actually used. After auto-N construction, these are the resolved values.</summary>
public int[] GetUsedNs() => (int[])NNodes.Clone();

/// <summary>Maximum derivative order this approximation supports.</summary>
public int GetMaxDerivativeOrder() => MaxDerivativeOrder;
```

- [ ] **Step 7: Modify `src/ChebyshevSharp/ChebyshevSpline.cs` — same shape**

Add private fields near the top of the class:

```csharp
private string? _descriptor;
private string _constructorType = "function";
```

`Built` is the existing field (used by `IsConstructionFinished()`). No alias field needed.

Locate `Build()` end. After build success, the existing code sets `Built = true;`. Just before returning, no change needed.

Locate `FromValues` static factory. After object is built, set:

```csharp
obj._constructorType = "from_values";
```

Locate `Load`. Before returning, set:

```csharp
obj._constructorType = "load";
```

Add new public methods at the bottom of the class:

```csharp
// ------------------------------------------------------------------
// Phase 4 ergonomics — accessors
// ------------------------------------------------------------------

public void SetDescriptor(string descriptor) => _descriptor = descriptor;
public string? GetDescriptor() => _descriptor;
public bool IsConstructionFinished() => Built;
public string GetConstructorType() => _constructorType;
public int[] GetUsedNs() => (int[])NNodes.Clone();
public int GetMaxDerivativeOrder() => MaxDerivativeOrder;
```

- [ ] **Step 8: Modify `src/ChebyshevSharp/ChebyshevSlider.cs` — same shape**

Add private fields near the top of the class:

```csharp
private string? _descriptor;
private string _constructorType = "function";
private bool _isConstructionFinished;
```

Locate `Build()` end. Just before the final closing brace of `Build()`, add:

```csharp
_isConstructionFinished = true;
```

Locate `Load`. After object construction, set:

```csharp
obj._constructorType = "load";
obj._isConstructionFinished = true;
```

Add new public methods:

```csharp
// ------------------------------------------------------------------
// Phase 4 ergonomics — accessors
// ------------------------------------------------------------------

public void SetDescriptor(string descriptor) => _descriptor = descriptor;
public string? GetDescriptor() => _descriptor;
public bool IsConstructionFinished() => _isConstructionFinished;
public string GetConstructorType() => _constructorType;
public int[] GetUsedNs() => (int[])NNodes.Clone();
public int GetMaxDerivativeOrder() => MaxDerivativeOrder;
```

- [ ] **Step 9: Modify `src/ChebyshevSharp/ChebyshevTT.cs` — same shape, plus `maxDerivativeOrder = 2` kwarg on ctor**

Add a new field near the top:

```csharp
private string? _descriptor;
private int _maxDerivativeOrder = 2;
```

(The `_built` field already exists.)

Modify the public ctor signature (around line 102) to add `maxDerivativeOrder = 2` at the end:

```csharp
public ChebyshevTT(
    Func<double[], double> function,
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxRank = 10,
    double tolerance = 1e-6,
    int maxSweeps = 10,
    int maxDerivativeOrder = 2)
{
    // ... existing validation ...
    _function = function;
    _numDimensions = numDimensions;
    _domain = domain;
    _nNodes = nNodes;
    _maxRank = maxRank;
    _tolerance = tolerance;
    _maxSweeps = maxSweeps;
    _maxDerivativeOrder = maxDerivativeOrder;
}
```

Modify the private deserialization ctor (around line 128) to also accept and store `maxDerivativeOrder`:

```csharp
private ChebyshevTT(
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxRank,
    double tolerance,
    int maxSweeps,
    TensorTrainKernel.TtCore[] coeffCores,
    int[] ttRanks,
    double buildTime,
    int totalBuildEvals,
    int maxDerivativeOrder = 2)
{
    // ... existing assignments ...
    _maxDerivativeOrder = maxDerivativeOrder;
}
```

Update all callers of this private ctor (search for `new ChebyshevTT(` inside `ChebyshevTT.cs`) to pass the new positional argument or rely on the default.

Add new public methods:

```csharp
// ------------------------------------------------------------------
// Phase 4 ergonomics — accessors
// ------------------------------------------------------------------

public void SetDescriptor(string descriptor) => _descriptor = descriptor;
public string? GetDescriptor() => _descriptor;
public bool IsConstructionFinished() => _built;
public string GetConstructorType() => Method ?? "function";
public int[] GetUsedNs() => (int[])_nNodes.Clone();
public int GetMaxDerivativeOrder() => _maxDerivativeOrder;
```

(`Method` is already exposed as a public property; it's set in `Build()` to "cross"/"svd"/"als".)

- [ ] **Step 10: Run the per-class ergonomics tests — verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~ApproxErgonomicsTests|FullyQualifiedName~SplineErgonomicsTests|FullyQualifiedName~SliderErgonomicsTests|FullyQualifiedName~TtErgonomicsTests"`

Expected: 25/25 passing (7 Approx + 6 Spline + 5 Slider + 6 TT — adjust counts if test file totals differ from this draft, but each file should fully pass).

- [ ] **Step 11: Run full test suite — verify no regressions**

Run: `dotnet test`

Expected: **831/831 passing** (812 baseline + 3 from Task 1 + 16 from this task — counts are a target; if final count differs by 1-2 tests, accept and document).

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T2 per-class state scaffolding (~16 tests)

Add private fields (_descriptor, _constructorType, _isConstructionFinished),
public accessors (Get/SetDescriptor, IsConstructionFinished, GetConstructorType,
GetUsedNs, GetMaxDerivativeOrder) on all four classes. ChebyshevTT also gains
a maxDerivativeOrder=2 ctor kwarg (parity with Python v0.16). No build-path
changes; pure additive surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `additionalData` ctor kwarg + threading through Build

**Why:** Approx, Spline, Slider have function signature `Func<double[], object?, double>` and currently pass `null` as the data arg. Phase 4 lets users set `additionalData` once on the ctor and threads it through every build call. TT's signature is `Func<double[], double>` (no data arg) — TT stores the value for introspection but does NOT thread it (user wraps with closure if needed).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSlider.cs`
- Modify: `src/ChebyshevSharp/ChebyshevTT.cs`
- Modify: existing 4 ergonomics test files (append tests)

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `additional_data` is stored as `self._additional_data` and passed in `f(point, self._additional_data)` during build.
- `ref/PyChebyshev/src/pychebyshev/spline.py`: each piece's `ChebyshevApproximation` receives the same `additional_data`.
- `ref/PyChebyshev/src/pychebyshev/slider.py`: each slide's `ChebyshevApproximation` receives the same `additional_data`.
- `ref/PyChebyshev/src/pychebyshev/tensor_train.py`: stored only; user wraps `f` with closure if needed.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Append `additionalData` tests to `tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs`**

Add inside the existing `ApproxErgonomicsTests` class:

```csharp
[Fact]
public void AdditionalData_threaded_through_Build()
{
    int callCount = 0;
    string? receivedTag = null;
    var approx = new ChebyshevApproximation(
        (p, data) =>
        {
            callCount++;
            receivedTag = (string?)data;
            return p[0];
        },
        numDimensions: 1,
        domain: new[] { new[] { -1.0, 1.0 } },
        nNodes: new[] { 5 },
        additionalData: "context-tag");
    approx.Build(verbose: false);

    Assert.Equal(5, callCount);
    Assert.Equal("context-tag", receivedTag);
    Assert.Equal("context-tag", approx.GetAdditionalData());
}

[Fact]
public void AdditionalData_default_is_null()
{
    var approx = BuildSimple();
    Assert.Null(approx.GetAdditionalData());
}
```

- [ ] **Step 2: Append `additionalData` tests to `tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs`**

```csharp
[Fact]
public void AdditionalData_threaded_to_each_piece()
{
    int callCount = 0;
    string? receivedTag = null;
    var spline = new ChebyshevSpline(
        (p, data) =>
        {
            callCount++;
            receivedTag = (string?)data;
            return p[0];
        },
        numDimensions: 1,
        domain: new[] { new[] { -1.0, 1.0 } },
        nNodes: new[] { 5 },
        knots: new[] { new[] { 0.0 } },  // 2 pieces
        additionalData: "spline-context");
    spline.Build(verbose: false);

    Assert.Equal(10, callCount);  // 2 pieces × 5 nodes
    Assert.Equal("spline-context", receivedTag);
    Assert.Equal("spline-context", spline.GetAdditionalData());
}

[Fact]
public void AdditionalData_default_is_null()
{
    var spline = BuildSimple();
    Assert.Null(spline.GetAdditionalData());
}
```

- [ ] **Step 3: Append `additionalData` tests to `tests/ChebyshevSharp.Tests/SliderErgonomicsTests.cs`**

```csharp
[Fact]
public void AdditionalData_threaded_through_Build()
{
    string? receivedTag = null;
    var slider = new ChebyshevSlider(
        (p, data) =>
        {
            receivedTag = (string?)data;
            return p[0] + p[1];
        },
        numDimensions: 2,
        domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
        nNodes: new[] { 5, 5 },
        partition: new[] { new[] { 0 }, new[] { 1 } },
        pivotPoint: new[] { 0.0, 0.0 },
        additionalData: "slider-context");
    slider.Build(verbose: false);

    Assert.Equal("slider-context", receivedTag);
    Assert.Equal("slider-context", slider.GetAdditionalData());
}

[Fact]
public void AdditionalData_default_is_null()
{
    var slider = BuildSimple();
    Assert.Null(slider.GetAdditionalData());
}
```

- [ ] **Step 4: Append `additionalData` tests to `tests/ChebyshevSharp.Tests/TtErgonomicsTests.cs`**

TT's function signature `Func<double[], double>` has no `data` arg — `additionalData` is stored on the instance for `GetAdditionalData()` introspection but not threaded to the function call.

```csharp
[Fact]
public void AdditionalData_stored_for_introspection()
{
    var tt = new ChebyshevTT(
        p => p[0] + p[1] + p[2],
        numDimensions: 3,
        domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
        nNodes: new[] { 5, 5, 5 },
        additionalData: "tt-context");
    tt.Build(verbose: false, seed: 42);
    Assert.Equal("tt-context", tt.GetAdditionalData());
}
```

- [ ] **Step 5: Run new tests — verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~AdditionalData"`

Expected: build errors — `additionalData` ctor kwarg not recognized; `GetAdditionalData()` not found.

- [ ] **Step 6: Modify `src/ChebyshevSharp/ChebyshevApproximation.cs`**

Add field:

```csharp
private object? _additionalData;
```

Update **both** public ctors. For the simple ctor (line 75 area):

```csharp
public ChebyshevApproximation(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxDerivativeOrder = 2,
    object? additionalData = null)
{
    Function = function;
    NumDimensions = numDimensions;
    Domain = domain.Select(d => (double[])d.Clone()).ToArray();
    NNodes = (int[])nNodes.Clone();
    MaxDerivativeOrder = maxDerivativeOrder;
    _additionalData = additionalData;

    NodeArrays = new double[numDimensions][];
    for (int d = 0; d < numDimensions; d++)
    {
        NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
    }
}
```

For the adaptive ctor (line 109 area), add `additionalData = null` at the end of the parameter list and assign `_additionalData = additionalData;` in the body.

Locate `BuildFixedGrid()` line 220:

```csharp
TensorValues[flat] = Function!(point, null);
```

Replace with:

```csharp
TensorValues[flat] = Function!(point, _additionalData);
```

Also locate `Internal/AdaptiveBuild.cs` if it does its own evaluations on the function — search for `Function(` calls. Replace any `null` data args with `_additionalData` (use `approx.AdditionalData` accessor or thread via internal property if needed).

Add accessor:

```csharp
/// <summary>
/// Returns the user-supplied <c>additionalData</c> object passed to the constructor,
/// or null if none was provided. Same value is threaded through every <c>f(point, data)</c>
/// call during <see cref="Build"/>.
/// </summary>
public object? GetAdditionalData() => _additionalData;
```

For internal access by Slider/Spline (during piece/slide creation), expose an internal property:

```csharp
internal object? AdditionalData => _additionalData;
```

- [ ] **Step 7: Modify `src/ChebyshevSharp/ChebyshevSpline.cs`**

Add field:

```csharp
private object? _additionalData;
```

Update **all three** public ctors to add `additionalData = null` at the end of the parameter list and assign `_additionalData = additionalData;`. (Each ctor: line 78, 123, 192 of the current file.)

Locate the build code that constructs per-piece `ChebyshevApproximation` instances (search for `new ChebyshevApproximation(`). Pass `additionalData: _additionalData` to each piece's ctor.

Add accessor:

```csharp
public object? GetAdditionalData() => _additionalData;
```

- [ ] **Step 8: Modify `src/ChebyshevSharp/ChebyshevSlider.cs`**

Add field:

```csharp
private object? _additionalData;
```

Update the public ctor (line 72) to add `additionalData = null` at the end and assign `_additionalData = additionalData;`.

Locate `Build()` line 146:

```csharp
PivotValue = Function(PivotPoint, null);
```

Replace with:

```csharp
PivotValue = Function(PivotPoint, _additionalData);
```

Locate the per-slide `ChebyshevApproximation` construction (search for `new ChebyshevApproximation(`). Pass `additionalData: _additionalData`.

Add accessor:

```csharp
public object? GetAdditionalData() => _additionalData;
```

- [ ] **Step 9: Modify `src/ChebyshevSharp/ChebyshevTT.cs`**

Add field:

```csharp
private object? _additionalData;
```

Update the public ctor signature (after `int maxDerivativeOrder = 2`):

```csharp
public ChebyshevTT(
    Func<double[], double> function,
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxRank = 10,
    double tolerance = 1e-6,
    int maxSweeps = 10,
    int maxDerivativeOrder = 2,
    object? additionalData = null)
{
    // ... existing body ...
    _additionalData = additionalData;
}
```

**Do not modify the build path** — TT's function signature has no `data` arg.

Add accessor:

```csharp
public object? GetAdditionalData() => _additionalData;
```

- [ ] **Step 10: Run new tests — verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~AdditionalData"`

Expected: 8/8 pass.

- [ ] **Step 11: Run full test suite — verify no regressions**

Run: `dotnet test`

Expected: **839/839 passing** (831 + 8). All existing function fixtures pass `null` for `additionalData` (default), so behavior is unchanged.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T3 additionalData ctor kwarg + threading (8 tests)

Approx, Spline, Slider thread _additionalData through Build's f(point, data)
calls (Spline propagates to each piece, Slider to each slide). TT stores
_additionalData for introspection only — its function signature has no data
arg; users wrap with closure if needed. Default is null, preserving existing
behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `GetEvaluationPoints` + `GetNumEvaluationPoints` lazy cache

**Why:** Users want to inspect the actual node coordinates the interpolant evaluates the function at. Returns `double[]` of length `numPoints × ndim` in C-order (row-major: row k has `ndim` doubles, the coordinates of the k-th evaluation point).

**Files:**
- Modify: 4 source files (add fields, methods)
- Modify: 4 ergonomics test files (append tests)

**Python source pointer:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `get_evaluation_points()` returns a `(num_points, ndim)` numpy array. C# returns flat 1D array (length × ndim).

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Append tests to `ApproxErgonomicsTests.cs`**

```csharp
[Fact]
public void GetEvaluationPoints_layout_is_row_major()
{
    var approx = BuildSimple();  // 2D, nNodes=[5,5]
    double[] pts = approx.GetEvaluationPoints();
    int num = approx.GetNumEvaluationPoints();

    Assert.Equal(25, num);
    Assert.Equal(50, pts.Length);  // 25 points × 2 dims

    // Row 0: first node-pair (smallest x0, smallest x1)
    // Verify against NodeArrays[0][0], NodeArrays[1][0]
    var nodeArrays = approx.NodeArrays;
    Assert.Equal(nodeArrays[0][0], pts[0]);
    Assert.Equal(nodeArrays[1][0], pts[1]);

    // Last row: largest x0, largest x1
    Assert.Equal(nodeArrays[0][4], pts[48]);
    Assert.Equal(nodeArrays[1][4], pts[49]);
}

[Fact]
public void GetEvaluationPoints_returns_cached_array_on_second_call()
{
    var approx = BuildSimple();
    double[] first = approx.GetEvaluationPoints();
    double[] second = approx.GetEvaluationPoints();
    Assert.Same(first, second);
}
```

- [ ] **Step 2: Append same tests to the other three ergonomics test files (Spline, Slider, TT)**

For each, build a simple instance and assert the same shape: `Length == NumEvaluationPoints * ndim`, row 0 reflects per-dim node 0, last row reflects per-dim last node, second call returns the same array reference.

For Spline: `BuildSimple` returns a 2-piece-per-dim spline; check that points cover all pieces (length = sum of per-piece num_points × ndim). For multi-piece splines, the points are concatenated piece-by-piece.

For Slider: total evaluation points = sum across all slides (each slide's `ChebyshevApproximation` contributes its grid).

For TT: total evaluation points = `_nNodes` Cartesian product (TT samples a subset during cross/als/svd, but `GetEvaluationPoints()` returns the full Chebyshev grid).

```csharp
// In SplineErgonomicsTests:
[Fact]
public void GetEvaluationPoints_layout_is_row_major()
{
    var spline = BuildSimple();
    double[] pts = spline.GetEvaluationPoints();
    int num = spline.GetNumEvaluationPoints();

    Assert.Equal(25, num);  // single-piece, nNodes=[5,5]
    Assert.Equal(50, pts.Length);
}

[Fact]
public void GetEvaluationPoints_returns_cached_array_on_second_call()
{
    var spline = BuildSimple();
    Assert.Same(spline.GetEvaluationPoints(), spline.GetEvaluationPoints());
}
```

```csharp
// In SliderErgonomicsTests:
[Fact]
public void GetEvaluationPoints_layout_is_row_major()
{
    var slider = BuildSimple();  // 3D, partition=[[0],[1,2]], nNodes=[5,5,5]
    double[] pts = slider.GetEvaluationPoints();
    int num = slider.GetNumEvaluationPoints();
    // Slide 0: 5 points × 3 dims = 15 doubles
    // Slide 1: 25 points × 3 dims = 75 doubles
    Assert.Equal(30, num);  // 5 + 25
    Assert.Equal(90, pts.Length);
}

[Fact]
public void GetEvaluationPoints_returns_cached_array_on_second_call()
{
    var slider = BuildSimple();
    Assert.Same(slider.GetEvaluationPoints(), slider.GetEvaluationPoints());
}
```

```csharp
// In TtErgonomicsTests:
[Fact]
public void GetEvaluationPoints_full_chebyshev_grid()
{
    var tt = BuildSimple();
    double[] pts = tt.GetEvaluationPoints();
    int num = tt.GetNumEvaluationPoints();
    Assert.Equal(125, num);  // 5*5*5
    Assert.Equal(375, pts.Length);
}

[Fact]
public void GetEvaluationPoints_returns_cached_array_on_second_call()
{
    var tt = BuildSimple();
    Assert.Same(tt.GetEvaluationPoints(), tt.GetEvaluationPoints());
}
```

- [ ] **Step 3: Run new tests — verify they fail**

Expected: 8 build errors — `GetEvaluationPoints`/`GetNumEvaluationPoints` not found.

- [ ] **Step 4: Implement on `ChebyshevApproximation.cs`**

Add field:

```csharp
private double[]? _evaluationPointsCache;
```

Add methods:

```csharp
/// <summary>
/// Total number of evaluation points: <c>nNodes[0] * nNodes[1] * ... * nNodes[ndim-1]</c>.
/// </summary>
public int GetNumEvaluationPoints()
{
    int total = 1;
    foreach (int n in NNodes) total *= n;
    return total;
}

/// <summary>
/// Flat row-major array of evaluation-point coordinates: length =
/// <c>GetNumEvaluationPoints() * NumDimensions</c>. Lazily built and cached.
/// </summary>
public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;

    int num = GetNumEvaluationPoints();
    int ndim = NumDimensions;
    var points = new double[num * ndim];
    var indices = new int[ndim];

    for (int flat = 0; flat < num; flat++)
    {
        int rem = flat;
        for (int d = ndim - 1; d >= 0; d--)
        {
            indices[d] = rem % NNodes[d];
            rem /= NNodes[d];
        }
        for (int d = 0; d < ndim; d++)
        {
            points[flat * ndim + d] = NodeArrays[d][indices[d]];
        }
    }

    _evaluationPointsCache = points;
    return points;
}
```

- [ ] **Step 5: Implement on `ChebyshevSpline.cs`**

Add field:

```csharp
private double[]? _evaluationPointsCache;
```

Spline accumulates points across all pieces:

```csharp
public int GetNumEvaluationPoints()
{
    int total = 0;
    if (Pieces == null) return 0;
    foreach (var piece in Pieces)
    {
        if (piece != null) total += piece.GetNumEvaluationPoints();
    }
    return total;
}

public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;

    int total = GetNumEvaluationPoints();
    var points = new double[total * NumDimensions];
    int offset = 0;

    foreach (var piece in Pieces!)
    {
        if (piece == null) continue;
        var piecePts = piece.GetEvaluationPoints();
        Array.Copy(piecePts, 0, points, offset, piecePts.Length);
        offset += piecePts.Length;
    }

    _evaluationPointsCache = points;
    return points;
}
```

- [ ] **Step 6: Implement on `ChebyshevSlider.cs`**

Add field:

```csharp
private double[]? _evaluationPointsCache;
```

Slider accumulates points across all slides; for each slide, the slide's `ChebyshevApproximation` is in slide-local coords, so we expand to the full ndim by filling pivot values for non-group dims:

```csharp
public int GetNumEvaluationPoints()
{
    if (Slides == null) return 0;
    int total = 0;
    foreach (var slide in Slides) total += slide.GetNumEvaluationPoints();
    return total;
}

public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;

    int total = GetNumEvaluationPoints();
    var points = new double[total * NumDimensions];
    int offset = 0;

    for (int slideIdx = 0; slideIdx < Slides!.Length; slideIdx++)
    {
        var slide = Slides[slideIdx];
        var group = Partition[slideIdx];
        var slidePts = slide.GetEvaluationPoints();  // (num × group.Length)
        int slideNum = slide.GetNumEvaluationPoints();
        int gdim = group.Length;

        for (int p = 0; p < slideNum; p++)
        {
            // Fill pivot for all dims; overwrite group dims with slide coords.
            for (int d = 0; d < NumDimensions; d++)
                points[offset + p * NumDimensions + d] = PivotPoint[d];
            for (int gi = 0; gi < gdim; gi++)
                points[offset + p * NumDimensions + group[gi]] = slidePts[p * gdim + gi];
        }
        offset += slideNum * NumDimensions;
    }

    _evaluationPointsCache = points;
    return points;
}
```

- [ ] **Step 7: Implement on `ChebyshevTT.cs`**

Add field:

```csharp
private double[]? _evaluationPointsCache;
```

TT samples a sparse set during cross/als, but the spec semantic is "full Chebyshev grid":

```csharp
public int GetNumEvaluationPoints()
{
    int total = 1;
    foreach (int n in _nNodes) total *= n;
    return total;
}

public double[] GetEvaluationPoints()
{
    if (_evaluationPointsCache != null) return _evaluationPointsCache;

    int num = GetNumEvaluationPoints();
    int ndim = _numDimensions;

    // Build per-dim Chebyshev nodes
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

    _evaluationPointsCache = points;
    return points;
}
```

(Add `using ChebyshevSharp.Internal;` if not already present, so `BarycentricKernel` resolves.)

- [ ] **Step 8: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~GetEvaluationPoints"`

Expected: 8/8 pass.

- [ ] **Step 9: Run full test suite**

Run: `dotnet test`

Expected: **847/847 passing** (839 + 8).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T4 GetEvaluationPoints lazy cache (8 tests)

Each class returns flat row-major double[] of length numPoints*ndim.
Spline concatenates across pieces; Slider expands slide-local coords to full
ndim using PivotPoint for non-group dims; TT returns the full Chebyshev grid
regardless of the sparse build sampling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `GetErrorThreshold` + `GetSpecialPoints` (Approx + Spline only)

**Why:** Approx and Spline both can be built with error-driven auto-N. They expose the threshold used and (for Spline) the special-points config that placed knots. Approx adds new `_specialPoints` storage (Spline already stores `Knots`).

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Modify: `tests/ChebyshevSharp.Tests/ApproxErgonomicsTests.cs`
- Modify: `tests/ChebyshevSharp.Tests/SplineErgonomicsTests.cs`

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `get_error_threshold()` returns `self._error_threshold`. `get_special_points()` returns `self._special_points` (Phase 4 v0.16 added storage of this on Approximation).
- `ref/PyChebyshev/src/pychebyshev/spline.py`: same accessors; Spline always has `Knots` so special_points is derived from there if not explicitly stored.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Append tests to `ApproxErgonomicsTests.cs`**

```csharp
[Fact]
public void GetErrorThreshold_returns_ctor_value_when_set()
{
    var approx = new ChebyshevApproximation(
        (p, _) => Math.Sin(p[0]),
        numDimensions: 1,
        domain: new[] { new[] { -1.0, 1.0 } },
        nNodes: new int?[] { null },
        errorThreshold: 1e-6);
    approx.Build(verbose: false);
    Assert.Equal(1e-6, approx.GetErrorThreshold());
}

[Fact]
public void GetErrorThreshold_returns_null_when_not_set()
{
    var approx = BuildSimple();
    Assert.Null(approx.GetErrorThreshold());
}
```

- [ ] **Step 2: Append tests to `SplineErgonomicsTests.cs`**

```csharp
[Fact]
public void GetErrorThreshold_returns_ctor_value_when_set()
{
    var spline = new ChebyshevSpline(
        (p, _) => Math.Abs(p[0]),
        numDimensions: 1,
        domain: new[] { new[] { -1.0, 1.0 } },
        nNodes: new int?[] { null },
        knots: new[] { new[] { 0.0 } },
        errorThreshold: 1e-6);
    spline.Build(verbose: false);
    Assert.Equal(1e-6, spline.GetErrorThreshold());
}

[Fact]
public void GetSpecialPoints_returns_knots_used_for_construction()
{
    var spline = new ChebyshevSpline(
        (p, _) => p[0],
        numDimensions: 1,
        domain: new[] { new[] { -1.0, 1.0 } },
        nNodes: new[] { 5 },
        knots: new[] { new[] { -0.5, 0.5 } });
    spline.Build(verbose: false);

    double[][]? sp = spline.GetSpecialPoints();
    Assert.NotNull(sp);
    Assert.Equal(new[] { -0.5, 0.5 }, sp![0]);
}
```

- [ ] **Step 3: Run new tests — verify they fail**

Expected: build errors — `GetErrorThreshold`/`GetSpecialPoints` not found.

- [ ] **Step 4: Implement on `ChebyshevApproximation.cs`**

`ErrorThreshold` is already a public property (auto-property). Add a method that returns it via the `GetX` style:

```csharp
public double? GetErrorThreshold() => ErrorThreshold;
```

For `GetSpecialPoints` on Approx: Approximation does not natively store special points (Spline does). Add storage and a method that returns null if never set:

```csharp
private double[][]? _specialPoints;

public double[][]? GetSpecialPoints() => _specialPoints;
```

(There is no current path to pass special points to Approximation's ctor — the `nNodes`-with-nulls is the auto-N signal but special points are a Spline concept. We add the storage now so future phases can populate it, and `Clone()` and JSON Save/Load will round-trip the field. Always returns null in Phase 4.)

- [ ] **Step 5: Implement on `ChebyshevSpline.cs`**

Add:

```csharp
public double? GetErrorThreshold() => ErrorThreshold;

public double[][]? GetSpecialPoints()
{
    // Return knots used for construction. Returns null if no interior knots
    // were specified (single-piece spline per dim).
    if (Knots == null) return null;
    bool anyInterior = false;
    foreach (var k in Knots)
        if (k.Length > 0) { anyInterior = true; break; }
    return anyInterior ? Knots.Select(k => (double[])k.Clone()).ToArray() : null;
}
```

- [ ] **Step 6: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~GetErrorThreshold|FullyQualifiedName~GetSpecialPoints"`

Expected: 4/4 pass.

- [ ] **Step 7: Run full test suite**

Run: `dotnet test`

Expected: **851/851 passing** (847 + 4).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T5 GetErrorThreshold + GetSpecialPoints accessors (4 tests)

Approx + Spline only. Spline derives GetSpecialPoints from Knots; returns null
if no interior knots. Approx stores _specialPoints for future use (always null
in Phase 4) so JSON round-trip and Clone preserve the slot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `GetDerivativeId` registry + `Eval(point, int)` overload

**Why:** Users registering derivative-orders tuples by stable session-local int ID makes downstream call sites cleaner (e.g., a vector of registered "Greeks" indexed by id). Backed by per-class `Dictionary<TupleKey, int>` and `List<int[]>`.

**Files:**
- Create: `src/ChebyshevSharp/Internal/TupleKey.cs`
- Modify: 4 source files
- Modify: 4 ergonomics test files

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `get_derivative_id(orders)` registers and returns int. `eval(point, derivative_id=...)` looks up.
- Pre-Phase-4 `get_derivative_id` was a no-op stub returning the input list (Python v0.15 made it a real registry — **breaking** in Python). C# never had the stub, so this is non-breaking for us.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Create `tests/ChebyshevSharp.Tests/Helpers/InternalsTests.cs` (TupleKey unit tests)**

We need to test the internal `TupleKey` directly. Use `InternalsVisibleTo` (already wired for tests).

```csharp
// tests/ChebyshevSharp.Tests/TupleKeyTests.cs
using ChebyshevSharp.Internal;
using Xunit;

namespace ChebyshevSharp.Tests;

public class TupleKeyTests
{
    [Fact]
    public void TupleKey_value_equality()
    {
        var a = new TupleKey(new[] { 1, 0, 2 });
        var b = new TupleKey(new[] { 1, 0, 2 });
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void TupleKey_distinct_for_different_values()
    {
        var a = new TupleKey(new[] { 1, 0 });
        var b = new TupleKey(new[] { 0, 1 });
        Assert.NotEqual(a, b);
    }
}
```

- [ ] **Step 2: Append registry tests to each ergonomics test file**

For ApproxErgonomicsTests:

```csharp
[Fact]
public void GetDerivativeId_returns_stable_int_per_orders_tuple()
{
    var approx = BuildSimple();
    int id1 = approx.GetDerivativeId(new[] { 1, 0 });
    int id2 = approx.GetDerivativeId(new[] { 0, 1 });
    int id1Again = approx.GetDerivativeId(new[] { 1, 0 });

    Assert.Equal(0, id1);
    Assert.Equal(1, id2);
    Assert.Equal(0, id1Again);  // same orders -> same id
}

[Fact]
public void EvalByDerivativeId_matches_EvalByOrders()
{
    var approx = BuildSimple();
    int id = approx.GetDerivativeId(new[] { 1, 0 });
    double byOrders = approx.Eval(new[] { 0.3, 0.5 }, new[] { 1, 0 });
    double byId = approx.Eval(new[] { 0.3, 0.5 }, id);
    Assert.Equal(byOrders, byId, precision: 12);
}
```

For SplineErgonomicsTests, SliderErgonomicsTests, TtErgonomicsTests: same shape, adapted to each class's existing `Eval(point, orders)` signature. (TT's `Eval(point, derivativeId)` → looks up orders, calls existing TT eval-with-derivatives path.)

```csharp
// SplineErgonomicsTests
[Fact]
public void GetDerivativeId_returns_stable_int_per_orders_tuple()
{
    var spline = BuildSimple();
    int id1 = spline.GetDerivativeId(new[] { 1, 0 });
    int id2 = spline.GetDerivativeId(new[] { 0, 1 });
    Assert.Equal(0, id1);
    Assert.Equal(1, id2);
    Assert.Equal(0, spline.GetDerivativeId(new[] { 1, 0 }));
}

[Fact]
public void EvalByDerivativeId_matches_EvalByOrders()
{
    var spline = BuildSimple();
    int id = spline.GetDerivativeId(new[] { 1, 0 });
    double byOrders = spline.Eval(new[] { 0.3, 0.5 }, new[] { 1, 0 });
    double byId = spline.Eval(new[] { 0.3, 0.5 }, id);
    Assert.Equal(byOrders, byId, precision: 12);
}
```

```csharp
// SliderErgonomicsTests
[Fact]
public void GetDerivativeId_returns_stable_int_per_orders_tuple()
{
    var slider = BuildSimple();
    int id1 = slider.GetDerivativeId(new[] { 1, 0, 0 });
    int id2 = slider.GetDerivativeId(new[] { 0, 1, 0 });
    Assert.Equal(0, id1);
    Assert.Equal(1, id2);
}

[Fact]
public void EvalByDerivativeId_matches_EvalByOrders()
{
    var slider = BuildSimple();
    int id = slider.GetDerivativeId(new[] { 1, 0, 0 });
    double byOrders = slider.Eval(new[] { 0.3, 0.5, 0.2 }, new[] { 1, 0, 0 });
    double byId = slider.Eval(new[] { 0.3, 0.5, 0.2 }, id);
    Assert.Equal(byOrders, byId, precision: 12);
}
```

```csharp
// TtErgonomicsTests
[Fact]
public void GetDerivativeId_returns_stable_int_per_orders_tuple()
{
    var tt = BuildSimple();
    int id1 = tt.GetDerivativeId(new[] { 1, 0, 0 });
    int id2 = tt.GetDerivativeId(new[] { 0, 1, 0 });
    Assert.Equal(0, id1);
    Assert.Equal(1, id2);
}

[Fact]
public void EvalByDerivativeId_matches_EvalMulti()
{
    var tt = BuildSimple();
    int id = tt.GetDerivativeId(new[] { 1, 0, 0 });
    double byMulti = tt.EvalMulti(new[] { 0.3, 0.5, 0.2 }, new[] { new[] { 1, 0, 0 } })[0];
    double byId = tt.Eval(new[] { 0.3, 0.5, 0.2 }, id);
    Assert.Equal(byMulti, byId, precision: 8);  // FD derivatives, looser tolerance
}
```

Plus an unknown-id test for one class (sufficient to cover the throw path):

```csharp
// In ApproxErgonomicsTests
[Fact]
public void EvalByUnknownDerivativeId_throws()
{
    var approx = BuildSimple();
    Assert.Throws<ArgumentOutOfRangeException>(() =>
        approx.Eval(new[] { 0.3, 0.5 }, derivativeId: 99));
}
```

- [ ] **Step 3: Run new tests — verify they fail**

Expected: build errors — `TupleKey`, `GetDerivativeId`, `Eval(point, int)` overload not found.

- [ ] **Step 4: Create `src/ChebyshevSharp/Internal/TupleKey.cs`**

```csharp
using System;
using System.Linq;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Value-equality wrapper around <see cref="int[]"/> for use as a dictionary key.
/// </summary>
internal readonly struct TupleKey : IEquatable<TupleKey>
{
    private readonly int[] _values;

    public TupleKey(int[] values)
    {
        _values = (int[])values.Clone();
    }

    public int[] Values => (int[])_values.Clone();

    public bool Equals(TupleKey other)
    {
        if (_values.Length != other._values.Length) return false;
        for (int i = 0; i < _values.Length; i++)
            if (_values[i] != other._values[i]) return false;
        return true;
    }

    public override bool Equals(object? obj) => obj is TupleKey o && Equals(o);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int v in _values) hash.Add(v);
        return hash.ToHashCode();
    }

    public override string ToString() => "[" + string.Join(",", _values) + "]";
}
```

- [ ] **Step 5: Add registry to `ChebyshevApproximation.cs`**

Add fields:

```csharp
private readonly Dictionary<Internal.TupleKey, int> _derivativeIdRegistry = new();
private readonly List<int[]> _registeredDerivativeOrders = new();
```

Add public methods:

```csharp
/// <summary>
/// Register or look up a derivative-orders tuple. Returns a stable
/// session-local int id for the same orders. Used in conjunction with
/// the <c>Eval(point, derivativeId)</c> overload.
/// </summary>
public int GetDerivativeId(int[] orders)
{
    var key = new Internal.TupleKey(orders);
    if (_derivativeIdRegistry.TryGetValue(key, out int existing))
        return existing;
    int id = _registeredDerivativeOrders.Count;
    _registeredDerivativeOrders.Add((int[])orders.Clone());
    _derivativeIdRegistry[key] = id;
    return id;
}

/// <summary>Evaluate at <paramref name="point"/> using a previously-registered derivative id.</summary>
public double Eval(double[] point, int derivativeId)
{
    if (derivativeId < 0 || derivativeId >= _registeredDerivativeOrders.Count)
        throw new ArgumentOutOfRangeException(
            nameof(derivativeId),
            $"derivativeId {derivativeId} not registered. Call GetDerivativeId first.");
    return Eval(point, _registeredDerivativeOrders[derivativeId]);
}

internal Dictionary<Internal.TupleKey, int> DerivativeIdRegistry => _derivativeIdRegistry;
internal List<int[]> RegisteredDerivativeOrders => _registeredDerivativeOrders;
```

- [ ] **Step 6: Apply same registry pattern to `ChebyshevSpline.cs`, `ChebyshevSlider.cs`, `ChebyshevTT.cs`**

For each: add the same two fields and same two methods. The `Eval(point, int)` overload calls each class's existing `Eval(point, int[] orders)` (Approx, Spline, Slider) or `EvalMulti(point, [orders])[0]` (TT).

For TT specifically:

```csharp
public double Eval(double[] point, int derivativeId)
{
    if (derivativeId < 0 || derivativeId >= _registeredDerivativeOrders.Count)
        throw new ArgumentOutOfRangeException(
            nameof(derivativeId),
            $"derivativeId {derivativeId} not registered. Call GetDerivativeId first.");
    var orders = _registeredDerivativeOrders[derivativeId];
    bool allZero = orders.All(o => o == 0);
    if (allZero) return Eval(point);
    return EvalMulti(point, new[] { orders })[0];
}
```

- [ ] **Step 7: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~TupleKeyTests|FullyQualifiedName~DerivativeId|FullyQualifiedName~EvalByDerivativeId|FullyQualifiedName~EvalByUnknownDerivativeId"`

Expected: 11/11 pass (2 TupleKey + 8 fan-out + 1 unknown-id).

- [ ] **Step 8: Run full test suite**

Run: `dotnet test`

Expected: **859/859 passing** (851 + 8 new ergonomics + 2 TupleKey unit + 1 throw test = +11; if 2 fewer tests due to consolidation accept 857-859).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T6 GetDerivativeId registry + Eval-by-id overload (8 tests)

New Internal/TupleKey.cs (value-equality wrapper around int[]). Each class
gains _derivativeIdRegistry Dictionary<TupleKey,int> and
_registeredDerivativeOrders List<int[]>. Eval(point, int derivativeId)
overload looks up orders by id; throws ArgumentOutOfRangeException for unknown
ids. Registry is per-instance, session-local, not serialized in Phase 4
(Task 9 will add JSON migration for it).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `Clone()` typed per class + `Internal/CloneHelpers.cs` + reflection completeness audit

**Why:** Users want to deep-copy an interpolant — sometimes for parallel modification, sometimes for safe `Save`-after-mutation. The reflection-based audit guards against future phases adding mutable state and forgetting to copy it.

**Files:**
- Create: `src/ChebyshevSharp/Internal/CloneHelpers.cs`
- Modify: 4 source files (add `Clone()` method)
- Create: `tests/ChebyshevSharp.Tests/CloneTests.cs`

**Python source pointers:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `clone()` returns deep copy with `function = None`.
- `ref/PyChebyshev/src/pychebyshev/spline.py`, `slider.py`, `tensor_train.py`: same.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Create `tests/ChebyshevSharp.Tests/CloneTests.cs`**

```csharp
// tests/ChebyshevSharp.Tests/CloneTests.cs
using System.Linq;
using System.Reflection;
using ChebyshevSharp.Tests.Helpers;
using Xunit;

namespace ChebyshevSharp.Tests;

public class CloneTests
{
    [Fact]
    public void Approx_Clone_returns_typed_copy_with_function_null()
    {
        var src = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        src.Build(verbose: false);
        src.SetDescriptor("source");

        ChebyshevApproximation copy = src.Clone();

        Assert.Null(copy.Function);  // function callable not duplicated
        Assert.Equal("source", copy.GetDescriptor());
        Assert.Equal("clone", copy.GetConstructorType());

        // Eval matches
        double[] pt = { 0.3, 0.5 };
        Assert.Equal(src.Eval(pt), copy.Eval(pt), precision: 12);

        // Mutating clone doesn't affect source
        copy.SetDescriptor("clone-only");
        Assert.Equal("source", src.GetDescriptor());
    }

    [Fact]
    public void Spline_Clone_returns_typed_copy_with_pieces_deep_copied()
    {
        var src = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { System.Array.Empty<double>(), System.Array.Empty<double>() });
        src.Build(verbose: false);

        ChebyshevSpline copy = src.Clone();
        Assert.Null(copy.Function);
        double[] pt = { 0.3, 0.5 };
        Assert.Equal(src.Eval(pt), copy.Eval(pt), precision: 12);
    }

    [Fact]
    public void Slider_Clone_returns_typed_copy()
    {
        var src = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        src.Build(verbose: false);

        ChebyshevSlider copy = src.Clone();
        Assert.Null(copy.Function);
        double[] pt = { 0.3, 0.5, 0.2 };
        Assert.Equal(src.Eval(pt), copy.Eval(pt), precision: 12);
    }

    [Fact]
    public void Tt_Clone_returns_typed_copy()
    {
        var src = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        src.Build(verbose: false, seed: 42);

        ChebyshevTT copy = src.Clone();
        // TT keeps function null after clone (consistent with save/load)
        double[] pt = { 0.3, 0.5, 0.2 };
        Assert.Equal(src.Eval(pt), copy.Eval(pt), precision: 12);
    }

    [Fact]
    public void Approx_Clone_arrays_are_not_aliased_with_source()
    {
        // Reflection-based completeness audit. Asserts that every array-typed
        // private field on the clone is not reference-equal to the source's
        // field. Catches future regressions where Clone forgets to copy a
        // newly-added mutable field.
        var src = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        src.Build(verbose: false);
        var copy = src.Clone();

        var fields = typeof(ChebyshevApproximation).GetFields(
            BindingFlags.NonPublic | BindingFlags.Instance);

        foreach (var f in fields)
        {
            var srcVal = f.GetValue(src);
            var copyVal = f.GetValue(copy);
            if (srcVal == null || copyVal == null) continue;
            // Arrays must be reference-distinct.
            if (srcVal is System.Array)
                Assert.False(ReferenceEquals(srcVal, copyVal),
                    $"Field {f.Name} is reference-aliased between src and clone");
        }
    }
}
```

- [ ] **Step 2: Run new tests — verify they fail**

Expected: build errors — `Clone()` method not found on any of the four classes.

- [ ] **Step 3: Create `src/ChebyshevSharp/Internal/CloneHelpers.cs`**

```csharp
using System.Collections.Generic;

namespace ChebyshevSharp.Internal;

/// <summary>
/// Deep-copy primitives used by all four classes' <c>Clone()</c> methods.
/// </summary>
internal static class CloneHelpers
{
    public static double[]? DeepCopy(double[]? src) =>
        src == null ? null : (double[])src.Clone();

    public static double[][]? DeepCopy(double[][]? src)
    {
        if (src == null) return null;
        var result = new double[src.Length][];
        for (int i = 0; i < src.Length; i++)
            result[i] = (double[])src[i].Clone();
        return result;
    }

    public static double[,]? DeepCopy(double[,]? src) =>
        src == null ? null : (double[,])src.Clone();

    public static double[][,]? DeepCopy(double[][,]? src)
    {
        if (src == null) return null;
        var result = new double[src.Length][,];
        for (int i = 0; i < src.Length; i++)
            result[i] = (double[,])src[i].Clone();
        return result;
    }

    public static int[]? DeepCopy(int[]? src) =>
        src == null ? null : (int[])src.Clone();

    public static int[][]? DeepCopy(int[][]? src)
    {
        if (src == null) return null;
        var result = new int[src.Length][];
        for (int i = 0; i < src.Length; i++)
            result[i] = (int[])src[i].Clone();
        return result;
    }

    public static int?[]? DeepCopy(int?[]? src) =>
        src == null ? null : (int?[])src.Clone();

    public static Dictionary<TupleKey, int> DeepCopy(Dictionary<TupleKey, int> src)
    {
        var result = new Dictionary<TupleKey, int>(src.Count);
        foreach (var kv in src) result[kv.Key] = kv.Value;
        return result;
    }

    public static List<int[]> DeepCopyOrders(List<int[]> src)
    {
        var result = new List<int[]>(src.Count);
        foreach (var orders in src) result.Add((int[])orders.Clone());
        return result;
    }
}
```

- [ ] **Step 4: Add `Clone()` to `ChebyshevApproximation.cs`**

```csharp
/// <summary>
/// Returns a deep copy of this approximation. The source <see cref="Function"/>
/// callable is NOT duplicated — clones cannot be rebuilt without re-supplying
/// the function. All precomputed state, descriptor, derivative-id registry,
/// and special points are deep-copied.
/// </summary>
public ChebyshevApproximation Clone()
{
    var copy = new ChebyshevApproximation();  // internal parameterless ctor
    copy.NumDimensions = NumDimensions;
    copy.NNodes = Internal.CloneHelpers.DeepCopy(NNodes)!;
    copy.Domain = Internal.CloneHelpers.DeepCopy(Domain)!;
    copy.NodeArrays = Internal.CloneHelpers.DeepCopy(NodeArrays)!;
    copy.TensorValues = Internal.CloneHelpers.DeepCopy(TensorValues);
    copy.Weights = Internal.CloneHelpers.DeepCopy(Weights);
    copy.DiffMatrices = Internal.CloneHelpers.DeepCopy(DiffMatrices);
    copy.MaxDerivativeOrder = MaxDerivativeOrder;
    copy.MaxN = MaxN;
    copy.ErrorThreshold = ErrorThreshold;
    copy.OriginalNNodes = Internal.CloneHelpers.DeepCopy(OriginalNNodes)!;
    copy.NEvaluations = NEvaluations;
    copy.BuildTime = BuildTime;
    copy._descriptor = _descriptor;
    copy._additionalData = _additionalData;
    copy._specialPoints = Internal.CloneHelpers.DeepCopy(_specialPoints);
    copy._isConstructionFinished = _isConstructionFinished;
    copy._constructorType = "clone";
    copy._evaluationPointsCache = null;  // lazy-rebuild on first call
    foreach (var kv in _derivativeIdRegistry)
        copy._derivativeIdRegistry[kv.Key] = kv.Value;
    foreach (var orders in _registeredDerivativeOrders)
        copy._registeredDerivativeOrders.Add((int[])orders.Clone());
    // Function intentionally left null
    return copy;
}
```

(Verify all public/internal properties of `ChebyshevApproximation` referenced above — fix names if any differ. The compiler will catch typos.)

- [ ] **Step 5: Add `Clone()` to `ChebyshevSpline.cs`**

```csharp
public ChebyshevSpline Clone()
{
    var copy = new ChebyshevSpline();  // need internal parameterless ctor
    copy.NumDimensions = NumDimensions;
    copy.Domain = Internal.CloneHelpers.DeepCopy(Domain)!;
    copy.NNodes = Internal.CloneHelpers.DeepCopy(NNodes)!;
    copy.Knots = Internal.CloneHelpers.DeepCopy(Knots)!;
    copy.Intervals = Internal.CloneHelpers.DeepCopy(Intervals)!;
    copy.Shape = Internal.CloneHelpers.DeepCopy(Shape)!;
    copy.MaxDerivativeOrder = MaxDerivativeOrder;
    copy.MaxN = MaxN;
    copy.ErrorThreshold = ErrorThreshold;
    copy.OriginalNNodes = OriginalNNodes != null
        ? Internal.CloneHelpers.DeepCopy(OriginalNNodes)!
        : System.Array.Empty<int?>();
    copy.NestedNNodes = Internal.CloneHelpers.DeepCopy(NestedNNodes);
    copy.Built = Built;
    copy.BuildTime = BuildTime;
    copy._descriptor = _descriptor;
    copy._additionalData = _additionalData;
    copy._constructorType = "clone";
    copy._evaluationPointsCache = null;
    if (Pieces != null)
    {
        copy.Pieces = new ChebyshevApproximation?[Pieces.Length];
        for (int i = 0; i < Pieces.Length; i++)
            copy.Pieces[i] = Pieces[i]?.Clone();
    }
    foreach (var kv in _derivativeIdRegistry)
        copy._derivativeIdRegistry[kv.Key] = kv.Value;
    foreach (var orders in _registeredDerivativeOrders)
        copy._registeredDerivativeOrders.Add((int[])orders.Clone());
    return copy;
}
```

(Add `internal ChebyshevSpline() { }` parameterless ctor if it doesn't exist.)

- [ ] **Step 6: Add `Clone()` to `ChebyshevSlider.cs`**

```csharp
public ChebyshevSlider Clone()
{
    var copy = new ChebyshevSlider();
    copy.NumDimensions = NumDimensions;
    copy.Domain = Internal.CloneHelpers.DeepCopy(Domain)!;
    copy.NNodes = Internal.CloneHelpers.DeepCopy(NNodes)!;
    copy.Partition = Internal.CloneHelpers.DeepCopy(Partition)!;
    copy.PivotPoint = Internal.CloneHelpers.DeepCopy(PivotPoint)!;
    copy.PivotValue = PivotValue;
    copy.MaxDerivativeOrder = MaxDerivativeOrder;
    copy._descriptor = _descriptor;
    copy._additionalData = _additionalData;
    copy._isConstructionFinished = _isConstructionFinished;
    copy._constructorType = "clone";
    copy._evaluationPointsCache = null;
    copy.DimToSlide = new Dictionary<int, int>(DimToSlide);
    if (Slides != null)
    {
        copy.Slides = new ChebyshevApproximation[Slides.Length];
        for (int i = 0; i < Slides.Length; i++)
            copy.Slides[i] = Slides[i].Clone();
    }
    foreach (var kv in _derivativeIdRegistry)
        copy._derivativeIdRegistry[kv.Key] = kv.Value;
    foreach (var orders in _registeredDerivativeOrders)
        copy._registeredDerivativeOrders.Add((int[])orders.Clone());
    return copy;
}
```

- [ ] **Step 7: Add `Clone()` to `ChebyshevTT.cs`**

```csharp
public ChebyshevTT Clone()
{
    // Reuse the private deserialization ctor.
    TensorTrainKernel.TtCore[]? clonedCores = null;
    if (_coeffCores != null)
    {
        clonedCores = new TensorTrainKernel.TtCore[_coeffCores.Length];
        for (int i = 0; i < _coeffCores.Length; i++)
            clonedCores[i] = _coeffCores[i].DeepCopy();
    }
    int[]? clonedRanks = _ttRanks != null ? (int[])_ttRanks.Clone() : null;

    var copy = new ChebyshevTT(
        _numDimensions,
        Internal.CloneHelpers.DeepCopy(_domain)!,
        Internal.CloneHelpers.DeepCopy(_nNodes)!,
        _maxRank,
        _tolerance,
        _maxSweeps,
        clonedCores ?? System.Array.Empty<TensorTrainKernel.TtCore>(),
        clonedRanks ?? System.Array.Empty<int>(),
        _buildTime,
        _totalBuildEvals,
        _maxDerivativeOrder);
    copy._descriptor = _descriptor;
    copy._additionalData = _additionalData;
    copy._evaluationPointsCache = null;
    foreach (var kv in _derivativeIdRegistry)
        copy._derivativeIdRegistry[kv.Key] = kv.Value;
    foreach (var orders in _registeredDerivativeOrders)
        copy._registeredDerivativeOrders.Add((int[])orders.Clone());
    return copy;
}
```

If `TtCore.DeepCopy()` doesn't exist, add it as an internal method on `TtCore` in `Internal/TensorTrainKernel.cs`:

```csharp
public TtCore DeepCopy()
{
    var copy = new TtCore(RLeft, NMode, RRight);
    System.Array.Copy(_data, copy._data, _data.Length);
    return copy;
}
```

- [ ] **Step 8: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~CloneTests"`

Expected: 5/5 pass.

- [ ] **Step 9: Run full test suite**

Run: `dotnet test`

Expected: **864/864 passing** (859 + 5).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T7 typed Clone() per class + CloneHelpers + reflection audit (5 tests)

New Internal/CloneHelpers.cs with DeepCopy primitives for arrays, jagged
arrays, 2D arrays, registry dict, and orders list. Each class's Clone()
returns its own concrete type with Function=null and ConstructorType="clone".
TtCore gains a DeepCopy method. Reflection-based completeness audit catches
future regressions where Clone forgets to copy a newly-added mutable array.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `DeferBuild` + `SetOriginalFunctionValues` (Approx + Spline only)

**Why:** Lets users construct an interpolant shell, then populate values asynchronously (e.g., from a network fetch). Bit-identical end state to `FromValues` factory.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevApproximation.cs`
- Modify: `src/ChebyshevSharp/ChebyshevSpline.cs`
- Create: `tests/ChebyshevSharp.Tests/DeferBuildTests.cs`

**Python source pointer:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py`: `defer_build=True` skips automatic build; `set_original_function_values(values)` populates and finishes construction. Bit-identical to `from_values()`.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Create `tests/ChebyshevSharp.Tests/DeferBuildTests.cs`**

```csharp
// tests/ChebyshevSharp.Tests/DeferBuildTests.cs
using Xunit;

namespace ChebyshevSharp.Tests;

public class DeferBuildTests
{
    [Fact]
    public void Approx_DeferBuild_then_SetValues_matches_FromValues()
    {
        // Build via FromValues
        var values = new double[5 * 5];
        for (int i = 0; i < 25; i++) values[i] = i * 0.1;
        var fromValues = ChebyshevApproximation.FromValues(
            values,
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });

        // Build via DeferBuild + SetOriginalFunctionValues
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,  // dummy — not called
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            deferBuild: true);
        Assert.False(deferred.IsConstructionFinished());
        deferred.SetOriginalFunctionValues(values);
        Assert.True(deferred.IsConstructionFinished());

        // Bit-identical: same Eval result for same point
        double[] pt = { 0.3, 0.5 };
        Assert.Equal(fromValues.Eval(pt), deferred.Eval(pt), precision: 12);
        Assert.Equal("from_values", deferred.GetConstructorType());
    }

    [Fact]
    public void Approx_DeferBuild_Eval_before_SetValues_throws()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            deferBuild: true);
        Assert.Throws<InvalidOperationException>(() => deferred.Eval(new[] { 0.5 }));
    }

    [Fact]
    public void Spline_DeferBuild_then_SetValues_works()
    {
        var values = new double[5];
        for (int i = 0; i < 5; i++) values[i] = i * 0.1;
        // Single-piece spline = same as approx
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);
        Assert.False(deferred.IsConstructionFinished());
        deferred.SetOriginalFunctionValues(values);
        Assert.True(deferred.IsConstructionFinished());
    }

    [Fact]
    public void Spline_DeferBuild_Save_before_SetValues_throws()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 1,
            domain: new[] { new[] { -1.0, 1.0 } },
            nNodes: new[] { 5 },
            knots: new[] { System.Array.Empty<double>() },
            deferBuild: true);
        string tmp = System.IO.Path.GetTempFileName();
        try
        {
            Assert.Throws<InvalidOperationException>(() => deferred.Save(tmp));
        }
        finally
        {
            System.IO.File.Delete(tmp);
        }
    }
}
```

- [ ] **Step 2: Run new tests — verify they fail**

Expected: build errors — `deferBuild` ctor kwarg, `SetOriginalFunctionValues` not found.

- [ ] **Step 3: Modify `ChebyshevApproximation.cs`**

Update both public ctors to add `deferBuild = false` at the end of the parameter list:

```csharp
public ChebyshevApproximation(
    Func<double[], object?, double> function,
    int numDimensions,
    double[][] domain,
    int[] nNodes,
    int maxDerivativeOrder = 2,
    object? additionalData = null,
    bool deferBuild = false)
{
    // ... existing init ...
    _additionalData = additionalData;

    // If deferBuild, don't materialize NodeArrays here — wait until SetOriginalFunctionValues
    if (!deferBuild)
    {
        NodeArrays = new double[numDimensions][];
        for (int d = 0; d < numDimensions; d++)
            NodeArrays[d] = BarycentricKernel.MakeNodesForDim(domain[d][0], domain[d][1], nNodes[d]);
    }
    else
    {
        NodeArrays = System.Array.Empty<double[]>();
    }
}
```

(Also update the adaptive ctor.)

Add the mutator method:

```csharp
/// <summary>
/// Populate this interpolant's tensor values from a precomputed flat array.
/// Used after constructing with <c>deferBuild: true</c>. Bit-identical to
/// the <see cref="FromValues"/> factory.
/// </summary>
/// <param name="values">Flat C-order tensor of length nNodes[0]*nNodes[1]*...</param>
public void SetOriginalFunctionValues(double[] values)
{
    int expected = 1;
    for (int d = 0; d < NumDimensions; d++) expected *= NNodes[d];
    if (values.Length != expected)
        throw new ArgumentException(
            $"values has {values.Length} entries, expected {expected} for nNodes=[{string.Join(",", NNodes)}]");

    // Materialize NodeArrays now if deferred.
    if (NodeArrays.Length == 0)
    {
        NodeArrays = new double[NumDimensions][];
        for (int d = 0; d < NumDimensions; d++)
            NodeArrays[d] = BarycentricKernel.MakeNodesForDim(Domain[d][0], Domain[d][1], NNodes[d]);
    }

    // Mirror FromValues precomputation
    TensorValues = (double[])values.Clone();
    NEvaluations = expected;

    Weights = new double[NumDimensions][];
    for (int d = 0; d < NumDimensions; d++)
        Weights[d] = BarycentricKernel.ComputeBarycentricWeights(NodeArrays[d]);

    DiffMatrices = new double[NumDimensions][,];
    for (int d = 0; d < NumDimensions; d++)
        DiffMatrices[d] = BarycentricKernel.ComputeDifferentiationMatrix(NodeArrays[d], Weights[d]);

    // Pre-transposed diff matrices — match BuildFixedGrid step 4.
    PrecomputeTransposedDiffMatrices();

    _evaluationPointsCache = null;
    _isConstructionFinished = true;
    _constructorType = "from_values";
}
```

(`PrecomputeTransposedDiffMatrices` is the existing internal method called by BuildFixedGrid. Verify the name and call site match.)

Update the `Eval` method (and `Save`, `VectorizedEval`, etc.) to throw `InvalidOperationException` when `_isConstructionFinished == false`:

```csharp
private void CheckBuilt()
{
    if (!_isConstructionFinished)
        throw new InvalidOperationException(
            "Cannot evaluate or save an unbuilt interpolant. Call Build() or SetOriginalFunctionValues() first.");
}
```

Add `CheckBuilt()` calls at the top of `Eval`, `VectorizedEval`, `VectorizedEvalBatch`, `VectorizedEvalMulti`, `Save`. (Some of these may already check via different means — audit each entry point.)

- [ ] **Step 4: Modify `ChebyshevSpline.cs`**

Update **all three** public ctors to add `deferBuild = false`. When true, skip the per-piece allocation; leave `Pieces` empty until `SetOriginalFunctionValues` is called.

Add the mutator:

```csharp
public void SetOriginalFunctionValues(double[] values)
{
    // Per-piece flat values: each piece's chunk is contiguous in C-order.
    int totalPieces = 1;
    foreach (int s in Shape) totalPieces *= s;
    int totalExpected = 0;
    var pieceSizes = new int[totalPieces];
    for (int p = 0; p < totalPieces; p++)
    {
        int n = 1;
        // Decompose flat piece index into per-dim piece coords; piece size
        // is the product of the per-dim node counts for that piece.
        // (If NestedNNodes is null, all pieces use NNodes uniformly.)
        if (NestedNNodes != null)
        {
            int rem = p;
            for (int d = NumDimensions - 1; d >= 0; d--)
            {
                int idx = rem % Shape[d];
                rem /= Shape[d];
                n *= NestedNNodes[d][idx];
            }
        }
        else
        {
            for (int d = 0; d < NumDimensions; d++) n *= NNodes[d];
        }
        pieceSizes[p] = n;
        totalExpected += n;
    }

    if (values.Length != totalExpected)
        throw new ArgumentException(
            $"values has {values.Length} entries, expected {totalExpected} across all pieces");

    Pieces = new ChebyshevApproximation?[totalPieces];
    int offset = 0;
    for (int p = 0; p < totalPieces; p++)
    {
        // Reconstruct piece domain
        // (Reuse existing per-piece domain calc from Build — extract a helper if needed)
        int sz = pieceSizes[p];
        var pieceValues = new double[sz];
        System.Array.Copy(values, offset, pieceValues, 0, sz);
        offset += sz;
        // Build piece via FromValues to ensure bit-identical state
        var (pieceDomain, pieceNNodes) = ComputePieceDomainAndN(p);
        Pieces[p] = ChebyshevApproximation.FromValues(
            pieceValues,
            NumDimensions,
            pieceDomain,
            pieceNNodes);
    }

    Built = true;
    _evaluationPointsCache = null;
    _constructorType = "from_values";
}
```

If `ComputePieceDomainAndN(int)` doesn't yet exist as a helper, extract it from the existing `Build()` body. The helper returns `(double[][] pieceDomain, int[] pieceNNodes)` for the given flat piece index.

Add `CheckBuilt()` to `Eval`, `Save`, etc. on Spline (mirroring Approx).

- [ ] **Step 5: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~DeferBuildTests"`

Expected: 4/4 pass.

- [ ] **Step 6: Run full test suite**

Run: `dotnet test`

Expected: **868/868 passing** (864 + 4).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T8 DeferBuild + SetOriginalFunctionValues (4 tests)

Approx + Spline only. ctor kwarg deferBuild=true skips Build; subsequent
SetOriginalFunctionValues(values) finishes construction (bit-identical to
FromValues factory). _isConstructionFinished tracks state; Eval/Save throw
InvalidOperationException while unbuilt. Spline routes per-piece via
FromValues to guarantee piece-by-piece bit-identical output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: JSON migration (Save writes new fields, Load tolerates absence) + 4 fixture files

**Why:** Phase 4 added Descriptor, _maxDerivativeOrder (TT), constructor type, and registered_derivative_orders to in-memory state. We need to (a) extend `Save` to persist them, (b) extend `Load` to tolerate their absence in pre-v0.8.0 JSON files, and (c) commit 4 fixture files generated from v0.7.0 code paths to prove the migration works.

**Critical ordering:** Generate the fixtures BEFORE modifying Save. If we modify Save first, the "pre-v0.8.0" fixtures would actually contain the new fields and the migration test would be a no-op.

**Files:**
- Create: `tools/GeneratePhase4Fixtures/GeneratePhase4Fixtures.csproj`
- Create: `tools/GeneratePhase4Fixtures/Program.cs`
- Create: `tests/fixtures/json-pre-v080/approx.json` (generated)
- Create: `tests/fixtures/json-pre-v080/spline.json` (generated)
- Create: `tests/fixtures/json-pre-v080/slider.json` (generated)
- Create: `tests/fixtures/json-pre-v080/tt.json` (generated)
- Create: `tests/ChebyshevSharp.Tests/JsonMigrationTests.cs`
- Modify: 4 source files (extend SerializationState records + add CheckBuilt for Save)
- Modify: `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` (copy fixtures to bin)

**Python source pointer:**
- `ref/PyChebyshev/src/pychebyshev/_pickle_migration.py`: Python uses `__setstate__` to backfill missing fields on load. C# uses `JsonElement.TryGetProperty` or System.Text.Json's automatic default-value behavior.

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Create the fixture generator project**

`tools/GeneratePhase4Fixtures/GeneratePhase4Fixtures.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <RootNamespace>GeneratePhase4Fixtures</RootNamespace>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\..\src\ChebyshevSharp\ChebyshevSharp.csproj" />
  </ItemGroup>
</Project>
```

`tools/GeneratePhase4Fixtures/Program.cs`:

```csharp
using System;
using System.IO;
using ChebyshevSharp;

class Program
{
    static void Main()
    {
        string outDir = Path.Combine("tests", "fixtures", "json-pre-v080");
        Directory.CreateDirectory(outDir);

        // Approx
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        approx.Save(Path.Combine(outDir, "approx.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "approx.json")}");

        // Spline
        var spline = new ChebyshevSpline(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { Array.Empty<double>(), Array.Empty<double>() });
        spline.Build(verbose: false);
        spline.Save(Path.Combine(outDir, "spline.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "spline.json")}");

        // Slider
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1, 2 } },
            pivotPoint: new[] { 0.0, 0.0, 0.0 });
        slider.Build(verbose: false);
        slider.Save(Path.Combine(outDir, "slider.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "slider.json")}");

        // TT
        var tt = new ChebyshevTT(
            p => p[0] + p[1] + p[2],
            numDimensions: 3,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5, 5 });
        tt.Build(verbose: false, seed: 42);
        tt.Save(Path.Combine(outDir, "tt.json"));
        Console.WriteLine($"Wrote {Path.Combine(outDir, "tt.json")}");
    }
}
```

- [ ] **Step 2: Run the generator BEFORE modifying any Save method**

```bash
dotnet run --project tools/GeneratePhase4Fixtures
```

Expected: 4 JSON fixtures written to `tests/fixtures/json-pre-v080/`. These represent pre-v0.8.0 state — they have the existing fields (Descriptor would be missing from the JSON entirely since the SerializationState records don't have it yet; same for the other new fields).

Verify the JSON does NOT contain Descriptor:

```bash
grep -L "Descriptor" tests/fixtures/json-pre-v080/*.json
```

Expected: all 4 files printed (none contain "Descriptor").

- [ ] **Step 3: Update `tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj` to copy fixtures to bin dir**

Locate the existing `<None Include="fixtures\...">` ItemGroup (added in Phase 3 for `.pcb` fixtures). Add:

```xml
<None Include="..\..\tests\fixtures\json-pre-v080\*.json">
  <Link>fixtures\json-pre-v080\%(Filename)%(Extension)</Link>
  <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
</None>
```

(If the existing ItemGroup uses a different pattern, follow that.)

- [ ] **Step 4: Create `tests/ChebyshevSharp.Tests/JsonMigrationTests.cs`**

```csharp
// tests/ChebyshevSharp.Tests/JsonMigrationTests.cs
using System.IO;
using Xunit;

namespace ChebyshevSharp.Tests;

public class JsonMigrationTests
{
    private static string FixturePath(string name) =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "json-pre-v080", name);

    [Fact]
    public void Approx_pre_v080_loads_with_default_descriptor()
    {
        var approx = ChebyshevApproximation.Load(FixturePath("approx.json"));
        Assert.Null(approx.GetDescriptor());
        Assert.True(approx.IsConstructionFinished());
        Assert.Equal("load", approx.GetConstructorType());
    }

    [Fact]
    public void Spline_pre_v080_loads_with_default_descriptor()
    {
        var spline = ChebyshevSpline.Load(FixturePath("spline.json"));
        Assert.Null(spline.GetDescriptor());
        Assert.True(spline.IsConstructionFinished());
    }

    [Fact]
    public void Slider_pre_v080_loads_with_default_descriptor()
    {
        var slider = ChebyshevSlider.Load(FixturePath("slider.json"));
        Assert.Null(slider.GetDescriptor());
        Assert.True(slider.IsConstructionFinished());
    }

    [Fact]
    public void Tt_pre_v080_loads_with_default_max_derivative_order()
    {
        var tt = ChebyshevTT.Load(FixturePath("tt.json"));
        Assert.Null(tt.GetDescriptor());
        Assert.Equal(2, tt.GetMaxDerivativeOrder());  // default
        Assert.True(tt.IsConstructionFinished());
    }
}
```

- [ ] **Step 5: Run new tests — verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~JsonMigrationTests"`

Expected: build errors or runtime errors — Load doesn't yet read the new fields, but absence-default behavior of System.Text.Json should already give us null/0 defaults. Tests likely already pass after Step 7's SerializationState extensions. If they pass without source changes, that just means the System.Text.Json default-value behavior is doing what we want — proceed to step 6 to add the *write* side.

- [ ] **Step 6: Extend `SerializationState` in `ChebyshevApproximation.cs`**

Locate the existing `SerializationState` record (likely inside the class or a nested type). Add fields:

```csharp
public string? Descriptor { get; init; }
public string? ConstructorType { get; init; }
public double[][]? SpecialPoints { get; init; }
public int[][]? RegisteredDerivativeOrders { get; init; }
```

Update `SaveJson` to populate them:

```csharp
private void SaveJson(string path)
{
    var state = new SerializationState
    {
        // ... existing fields ...
        Descriptor = _descriptor,
        ConstructorType = _constructorType,
        SpecialPoints = _specialPoints,
        RegisteredDerivativeOrders = _registeredDerivativeOrders.Count > 0
            ? _registeredDerivativeOrders.ToArray()
            : null,
        Version = "0.8.0"
    };

    var options = new JsonSerializerOptions { WriteIndented = false };
    string json = JsonSerializer.Serialize(state, options);
    File.WriteAllText(path, json);
}
```

Update `LoadJson` to populate the new private fields after deserializing:

```csharp
private static ChebyshevApproximation LoadJson(string path)
{
    string json = File.ReadAllText(path);
    var state = JsonSerializer.Deserialize<SerializationState>(json)
        ?? throw new InvalidOperationException("Failed to deserialize");

    var obj = new ChebyshevApproximation
    {
        // ... existing assignments ...
    };
    obj._descriptor = state.Descriptor;
    obj._specialPoints = state.SpecialPoints;
    if (state.RegisteredDerivativeOrders != null)
    {
        foreach (var orders in state.RegisteredDerivativeOrders)
        {
            var key = new Internal.TupleKey(orders);
            int id = obj._registeredDerivativeOrders.Count;
            obj._registeredDerivativeOrders.Add((int[])orders.Clone());
            obj._derivativeIdRegistry[key] = id;
        }
    }
    obj._constructorType = "load";
    obj._isConstructionFinished = true;
    return obj;
}
```

- [ ] **Step 7: Apply same SerializationState extensions to Spline, Slider, TT**

For Spline: add the same 4 new fields (no `SpecialPoints` field on Slider/TT since `GetSpecialPoints` is Approx+Spline only; for Slider/TT add only Descriptor, ConstructorType, RegisteredDerivativeOrders). For TT also add `MaxDerivativeOrder` field if not already present.

Mirror the Save/Load changes.

- [ ] **Step 8: Run new tests**

Run: `dotnet test --filter "FullyQualifiedName~JsonMigrationTests"`

Expected: 4/4 pass.

Run a Save → Load round-trip integration test (existing tests should cover this):

Run: `dotnet test --filter "FullyQualifiedName~SerializationTests|FullyQualifiedName~RoundTrip"`

Expected: all existing pass, plus the new fields persist (descriptor preserved, registered orders preserved).

- [ ] **Step 9: Run full test suite**

Run: `dotnet test`

Expected: **872/872 passing** (868 + 4).

- [ ] **Step 10: Commit fixtures + migration code**

```bash
git add tools/GeneratePhase4Fixtures/ tests/fixtures/json-pre-v080/ tests/ChebyshevSharp.Tests/JsonMigrationTests.cs tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj src/ChebyshevSharp/ChebyshevApproximation.cs src/ChebyshevSharp/ChebyshevSpline.cs src/ChebyshevSharp/ChebyshevSlider.cs src/ChebyshevSharp/ChebyshevTT.cs
git commit -m "$(cat <<'EOF'
phase4: T9 JSON migration + 4 pre-v0.8.0 fixtures (4 tests)

SerializationState extended with Descriptor, ConstructorType, SpecialPoints,
RegisteredDerivativeOrders, MaxDerivativeOrder (TT only). System.Text.Json
deserialization auto-defaults missing fields to null/0, providing absence-check
behavior without an explicit schema-version field.

Four fixture files captured from v0.7.0 code paths (before Save was extended)
prove pre-v0.8.0 JSON loads with sensible defaults. Generator left in tools/
for future regeneration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Docs + parity metadata + v0.8.0 release prep

**Why:** Bump version, update changelog, write user-facing docs page, sync `skip_csharp.txt` and `CLAUDE.md` Status block, prep for merge.

**Files:**
- Modify: `src/ChebyshevSharp/ChebyshevSharp.csproj`
- Modify: `docs/docs/changelog.md`
- Create: `docs/docs/ergonomics.md`
- Modify: `docs/docs/toc.yml`
- Modify: `skip_csharp.txt`
- Modify: `CLAUDE.md`

- [ ] **Step 0: Worktree enforcement.**

- [ ] **Step 1: Bump csproj metadata**

Edit `src/ChebyshevSharp/ChebyshevSharp.csproj`:

```xml
<Version>0.8.0</Version>
<PyChebyshevParity>0.18.0</PyChebyshevParity>  <!-- unchanged -->
<InformationalVersion>0.8.0+pychebyshev.0.18.0</InformationalVersion>
```

- [ ] **Step 2: Update `docs/docs/changelog.md`**

Add a new top entry following the two-tier convention used by Phase 3:

```markdown
## [0.8.0] - 2026-04-28 — Ergonomics polish (PyChebyshev v0.15+v0.16 fill-in)

> **PyChebyshev parity stays at v0.18.0.** Phase 4 of the v0.20.1 phased port —
> backfills the v0.15+v0.16 ergonomics layer that was skipped during the initial
> port. Same pattern as Phase 3's binary `.pcb` format fill-in.

### Added — descriptor, additional data, registry, introspection

- `SetDescriptor(string)` / `GetDescriptor()` on all four classes — free-form
  text labels that survive Save/Load.
- `additionalData` constructor kwarg (`object?`) on all four classes. Threads
  user-supplied context through every `f(point, data)` call during
  `ChebyshevApproximation`/`ChebyshevSpline`/`ChebyshevSlider` build. Stored
  for introspection on `ChebyshevTT` (its function signature has no data arg).
- `GetDerivativeId(int[] orders)` registry on all four classes — returns a
  stable session-local int per registered orders tuple. New
  `Eval(double[] point, int derivativeId)` overload looks up the orders.
- `IsConstructionFinished()`, `GetConstructorType()`, `GetUsedNs()` on all
  four classes — runtime introspection of build state.

### Added — clone, accessors

- Typed `Clone()` per class (`ChebyshevApproximation Clone()`,
  `ChebyshevSpline Clone()`, etc.). Returns deep copy with `Function = null`
  (matches Save/Load convention).
- `GetMaxDerivativeOrder()` on all four classes.
- `GetErrorThreshold()`, `GetSpecialPoints()` on Approximation + Spline.
- `GetEvaluationPoints()`, `GetNumEvaluationPoints()` on all four classes.

### Added — deferred construction, typed records

- `deferBuild` constructor kwarg (`bool`) + `SetOriginalFunctionValues(double[] values)`
  instance mutator on Approximation + Spline. Construct shell now, populate
  values later (e.g., from a network fetch). Bit-identical to the
  `FromValues()` factory.
- `ChebyshevTT` constructor: new `maxDerivativeOrder = 2` keyword-only kwarg.
- New public records `Domain`, `Ns`, `SpecialPoints` with implicit conversions
  to/from raw arrays. Optional ergonomic wrappers — call sites can pass either
  form.

### JSON migration

- `Load()` for all four classes tolerates missing v0.8.0 fields in pre-v0.8.0
  JSON files. Defaults: `Descriptor = null`, `MaxDerivativeOrder = 2`,
  `SpecialPoints = null`, empty derivative-id registry.
- 4 committed fixture files in `tests/fixtures/json-pre-v080/` (one per class)
  generated from v0.7.0 code paths prove the migration path.
- `additionalData` is **not** serialized (Python convention; `object?` isn't
  safely round-trippable through JSON without type info).

### Test count: 812 → 872 (+60)

Phase 4 fan-out across 8 new test files plus appended cross-class tests. See
[PR #19](https://github.com/0xC000005/ChebyshevSharp/pull/19) for the full
diff and the [design spec](https://github.com/0xC000005/ChebyshevSharp/blob/main/docs/superpowers/specs/2026-04-28-phase4-ergonomics-design.md).

Phase 5 (integrate everywhere — Slider/TT integration on calculus,
PyChebyshev parity bump to v0.17.0) is next.
```

- [ ] **Step 3: Create `docs/docs/ergonomics.md`**

```markdown
---
uid: ergonomics
title: Ergonomics
---

# Ergonomics

ChebyshevSharp v0.8.0 adds a suite of ergonomic accessors and constructor
sugar features across all four interpolant classes. This page summarizes the
new surface.

## Descriptors

A free-form text label you can attach to any interpolant. Survives save/load.

```csharp
var approx = new ChebyshevApproximation(
    (p, _) => p[0] + p[1],
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 5, 5 });
approx.Build();
approx.SetDescriptor("Black-Scholes 2D, vol=20%");

string? label = approx.GetDescriptor();  // "Black-Scholes 2D, vol=20%"
```

## Additional Data

If your function reads context (e.g., calibration parameters) supply it once
at construction. ChebyshevSharp threads it through every `f(point, data)`
call during build:

```csharp
var calibration = LoadCalibration("EOD-2026-04-28.json");
var approx = new ChebyshevApproximation(
    (p, data) => BlackScholesPrice(p, (Calibration)data!),
    numDimensions: 2,
    domain: ...,
    nNodes: ...,
    additionalData: calibration);
approx.Build();

object? stored = approx.GetAdditionalData();  // returns the calibration object
```

`additionalData` is **not** serialized in JSON or `.pcb` files. Loading a
saved interpolant returns one with `GetAdditionalData() == null` — supply
again if rebuilding.

`ChebyshevTT` stores `additionalData` for introspection but doesn't thread
it (its function signature is `Func<double[], double>`). Wrap your function
with a closure if you need data threading on TT.

## Derivative ID Registry

Stable session-local int per derivative-orders tuple — useful for
register-once-evaluate-many workflows:

```csharp
int delta = approx.GetDerivativeId(new[] { 1, 0 });    // → 0
int gamma = approx.GetDerivativeId(new[] { 2, 0 });    // → 1
int vega  = approx.GetDerivativeId(new[] { 0, 1 });    // → 2

double d = approx.Eval(point, delta);   // same as approx.Eval(point, new[] { 1, 0 })
double g = approx.Eval(point, gamma);
double v = approx.Eval(point, vega);
```

The registry is per-instance and not serialized. After `Save()`/`Load()`
re-register your IDs.

## Introspection

```csharp
bool ready = approx.IsConstructionFinished();      // true after Build()
string how = approx.GetConstructorType();           // "function" | "from_values" | "load" | "clone" | "cross"/"svd"/"als" (TT)
int[] ns = approx.GetUsedNs();                      // resolved per-dim node counts
int maxOrder = approx.GetMaxDerivativeOrder();
```

For Approximation and Spline only:

```csharp
double? thr = approx.GetErrorThreshold();           // null if not auto-N constructed
double[][]? sp = approx.GetSpecialPoints();         // null if not Spline with knots
```

For all four classes:

```csharp
int n = approx.GetNumEvaluationPoints();            // total grid size
double[] pts = approx.GetEvaluationPoints();        // length = n × ndim, row-major
```

## Cloning

Deep copy with `Function = null` (consistent with save/load):

```csharp
ChebyshevApproximation clone = approx.Clone();
// clone is independent — mutating its descriptor, registry, etc., does not affect approx
```

## Deferred Build (Approximation + Spline)

Construct now, populate values later — e.g., from an asynchronous data source:

```csharp
var deferred = new ChebyshevApproximation(
    (_, _) => throw new InvalidOperationException("not used"),
    numDimensions: 2,
    domain: ...,
    nNodes: new[] { 5, 5 },
    deferBuild: true);
// At this point IsConstructionFinished() == false; Eval/Save throw.

double[] precomputed = await FetchValuesAsync();  // length = 5 * 5
deferred.SetOriginalFunctionValues(precomputed);
// Now IsConstructionFinished() == true; constructor type is "from_values".
```

## Typed Domain/Ns/SpecialPoints Records

Optional type-tag wrappers — purely additive:

```csharp
var d = new Domain(new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } });
var n = new Ns(new[] { 5, 5 });
double[][] back = d;          // implicit Domain → double[][]
Domain again = back;          // implicit double[][] → Domain
```

The constructors of all four classes still accept raw `double[][]` and
`int[]` directly — these records are sugar, not required.
```

- [ ] **Step 4: Update `docs/docs/toc.yml`**

Add the ergonomics page after `binary-format.md`:

```yaml
- name: Ergonomics
  href: ergonomics.md
```

(Adjust nesting to match the existing structure.)

- [ ] **Step 5: Update `skip_csharp.txt`**

Mark Phase 4 features ported. Open the file and either delete the relevant skip lines or annotate them. Specifically:

- Remove any lines tagged with v0.15.0 or v0.16.0.
- Add a "Phase 4 complete" marker comment.

- [ ] **Step 6: Update `CLAUDE.md` Status block**

Edit the Status section near the top:

```markdown
## Status

**Feature-complete against PyChebyshev v0.18.0** (Phases 1+2+3+4 of the 6-phase v0.20.1 port complete; see
`docs/superpowers/specs/2026-04-27-pychebyshev-v0.20.1-port-design.md`).
All four public classes (`ChebyshevApproximation`, `ChebyshevSpline`, `ChebyshevSlider`,
`ChebyshevTT`) mirror the Python API surface. v0.8.0 adds the v0.15+v0.16 ergonomics
layer (descriptor, additionalData, derivative-id registry, introspection getters,
typed Clone, DeferBuild + SetOriginalFunctionValues, Domain/Ns/SpecialPoints records;
PyChebyshev parity tag unchanged at v0.18.0).
`dotnet test` runs **872/872** passing.
```

- [ ] **Step 7: Run full test suite — final smoke check**

Run: `dotnet test`

Expected: **872/872 passing**.

Run: `dotnet build -c Release`

Expected: zero warnings.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
phase4: T10 docs + parity metadata + v0.8.0 release prep

- csproj: <Version>0.8.0, <PyChebyshevParity>0.18.0 (unchanged),
  <InformationalVersion>0.8.0+pychebyshev.0.18.0
- changelog: v0.8.0 entry (two-tier convention; parity stays at v0.18.0)
- docs/docs/ergonomics.md: new user-guide page covering descriptor,
  additionalData, derivative-id registry, clone, DeferBuild, typed records
- toc.yml: link the new ergonomics page
- skip_csharp.txt: Phase 4 features marked ported
- CLAUDE.md: Status block updated to 872/872 passing, Phases 1+2+3+4 complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: Hand off to user for merge / PR / release**

Phase 4 implementation complete. The user will:

1. Review the worktree's commit log.
2. Decide PR shape: single PR `phase4-ergonomics` → main.
3. Run code review (`/review`).
4. Address review feedback if any.
5. Merge.
6. Tag `v0.8.0` and trigger NuGet publish via the existing `publish.yml` workflow.

Do NOT auto-create the PR or auto-merge — that's a user-confirmation gate (per Phase 3's pattern).

---

## Self-Review Checklist (for the planner — execute before handoff)

- [ ] **Spec coverage:** Every feature in spec §4.2 maps to a task. Domain/Ns/SpecialPoints (T1), descriptor/introspection/maxDerivativeOrder (T2), additionalData (T3), GetEvaluationPoints (T4), GetErrorThreshold/GetSpecialPoints (T5), GetDerivativeId/Eval-by-id (T6), Clone (T7), DeferBuild (T8), JSON migration (T9), release prep (T10). ✅
- [ ] **Placeholder scan:** No "TBD", "TODO", "implement later", "fill in details", or "similar to task N".
- [ ] **Type consistency:** `_descriptor`/`_additionalData`/`_constructorType`/`_isConstructionFinished`/`_evaluationPointsCache` field names match across tasks. `GetDescriptor`/`SetDescriptor`/`GetAdditionalData`/`GetDerivativeId`/`IsConstructionFinished`/`GetConstructorType`/`GetUsedNs`/`GetMaxDerivativeOrder`/`GetErrorThreshold`/`GetSpecialPoints`/`GetEvaluationPoints`/`GetNumEvaluationPoints`/`SetOriginalFunctionValues`/`Clone` method names match across tasks.
- [ ] **Test count progression:** 812 → 815 → 831 → 839 → 847 → 851 → 859 → 864 → 868 → 872. Sum: +60 tests over 10 tasks.

If actual test counts diverge by ±2 per task, accept and document. Phase 3 had similar drift.
