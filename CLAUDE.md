# CLAUDE.md

This file provides guidance to Claude Code when working on the ChebyshevSharp project.

## Project Objective

ChebyshevSharp is a **C# port of PyChebyshev** — a multi-dimensional Chebyshev tensor
interpolation library with analytical derivatives. The goal is feature-complete parity
with the Python reference implementation.

**Reference implementation**: `ref/PyChebyshev/` (git submodule pointing to the Python library).
Always consult the Python source when implementing or verifying behavior.

## Commands

```bash
# Build (produces net8.0 and net10.0 DLLs)
dotnet build

# Run tests
dotnet test

# Run tests with verbose output
dotnet test --verbosity normal

# Run a single test class
dotnet test --filter "FullyQualifiedName~BarycentricTests"

# Run a single test
dotnet test --filter "FullyQualifiedName~BarycentricTests.Placeholder_ProjectBuilds"

# Pack for NuGet
dotnet pack src/ChebyshevSharp --configuration Release

# Clean build artifacts
dotnet clean

# Build and preview docs locally (requires: dotnet tool install --global docfx)
docfx docs/docfx.json --serve        # http://localhost:8080
docfx docs/docfx.json                # Build only, output to docs/_site/
```

## Architecture

### Reference Mapping: Python → C#

Every C# class maps directly to a Python source file. **Always read the Python source
before implementing the C# equivalent.**

| C# Class | Python Source | Lines | Priority |
|----------|-------------|-------|----------|
| `ChebyshevApproximation` | `ref/PyChebyshev/src/pychebyshev/barycentric.py` | 1500 | **Phase 1** |
| `ChebyshevSpline` | `ref/PyChebyshev/src/pychebyshev/spline.py` | 1452 | **Phase 2** |
| `ChebyshevSlider` | `ref/PyChebyshev/src/pychebyshev/slider.py` | 856 | **Phase 3** |
| `ChebyshevTT` | `ref/PyChebyshev/src/pychebyshev/tensor_train.py` | 1343 | **Phase 4** |
| Internal helpers | `ref/PyChebyshev/src/pychebyshev/_algebra.py` | 55 | With Phase 1 |
| Internal helpers | `ref/PyChebyshev/src/pychebyshev/_extrude_slice.py` | 92 | With Phase 1 |
| Internal helpers | `ref/PyChebyshev/src/pychebyshev/_calculus.py` | 339 | With Phase 1 |

### Reference Tests: Python → C#

Port tests in the same order as the classes. Each Python test file maps to a C# test class.

| C# Test Class | Python Test Source | Tests | Priority |
|---------------|-------------------|-------|----------|
| `BarycentricTests` | `ref/PyChebyshev/tests/test_barycentric.py` | 48 | **Phase 1** |
| `FromValuesTests` | `ref/PyChebyshev/tests/test_from_values.py` | 65 | **Phase 1** |
| `AlgebraTests` | `ref/PyChebyshev/tests/test_algebra.py` | 77 | **Phase 1** |
| `ExtrudeSliceTests` | `ref/PyChebyshev/tests/test_extrude_slice.py` | 63 | **Phase 1** |
| `CalculusTests` | `ref/PyChebyshev/tests/test_calculus.py` | 74 | **Phase 1** |
| `SplineTests` | `ref/PyChebyshev/tests/test_spline.py` | 55 | **Phase 2** |
| `SliderTests` | `ref/PyChebyshev/tests/test_slider.py` | 40 | **Phase 3** |
| `TensorTrainTests` | `ref/PyChebyshev/tests/test_tensor_train.py` | 35 | **Phase 4** |

**Total target: 457 passing tests** (matching PyChebyshev exactly).

Shared test fixtures are in `ref/PyChebyshev/tests/conftest.py` — read this file
to understand the Black-Scholes functions and fixture setup used across all test files.

### Project Structure

```
ChebyshevSharp/
├── ChebyshevSharp.slnx                 # Solution file
├── ref/PyChebyshev/                    # Python reference (git submodule, READ-ONLY)
├── src/ChebyshevSharp/                 # Library source
│   ├── ChebyshevSharp.csproj           # Multi-target net8.0;net10.0
│   ├── ChebyshevApproximation.cs       # Core class
│   ├── ChebyshevSpline.cs              # Piecewise Chebyshev
│   ├── ChebyshevSlider.cs              # Sliding Technique
│   ├── ChebyshevTT.cs                  # Tensor Train
│   └── Internal/                       # Internal helpers (not public API)
│       ├── BarycentricKernel.cs         # Weights, diff matrices, interpolation
│       ├── Algebra.cs                   # Operator dispatch, compatibility checks
│       ├── ExtrudeSlice.cs              # Extrusion/slicing helpers
│       └── Calculus.cs                  # Fejér weights, rootfinding, optimization
├── tests/ChebyshevSharp.Tests/         # xUnit test project
│   ├── BarycentricTests.cs
│   ├── FromValuesTests.cs
│   ├── AlgebraTests.cs
│   ├── ExtrudeSliceTests.cs
│   ├── CalculusTests.cs
│   ├── SplineTests.cs
│   ├── SliderTests.cs
│   ├── TensorTrainTests.cs
│   └── Helpers/
│       └── BlackScholes.cs              # Port of conftest.py BS functions
├── .github/workflows/
│   ├── test.yml                         # CI: tests on .NET 8 + 10
│   └── publish.yml                      # CD: publish to NuGet on release
└── skip_csharp.txt                      # Feature parity tracker (see below)
```

## Implementation Phases

### Phase 1: ChebyshevApproximation (core)

This is the foundation. Everything else depends on it.

**Read these Python files first:**
- `ref/PyChebyshev/src/pychebyshev/barycentric.py` — the entire file
- `ref/PyChebyshev/src/pychebyshev/_algebra.py` — operator helpers
- `ref/PyChebyshev/src/pychebyshev/_extrude_slice.py` — extrude/slice helpers
- `ref/PyChebyshev/src/pychebyshev/_calculus.py` — integration/roots/optimize helpers
- `ref/PyChebyshev/tests/conftest.py` — shared fixtures and Black-Scholes functions

**Implement in this order within Phase 1:**

1. **Internal/BarycentricKernel.cs** — free functions:
   - `ComputeBarycentricWeights(double[] nodes)` ← Python `compute_barycentric_weights()`
   - `ComputeDifferentiationMatrix(double[] nodes, double[] weights)` ← Python `compute_differentiation_matrix()`
   - `BarycentricInterpolate(double x, double[] nodes, double[] values, double[] weights)` ← Python `barycentric_interpolate()`
   - `BarycentricDerivativeAnalytical(...)` ← Python `barycentric_derivative_analytical()`

2. **ChebyshevApproximation.cs** — constructor + build + eval:
   - Constructor: `ChebyshevApproximation(Func<double[], object, double> function, int ndim, double[][] domains, int[] orders)`
   - `Build(bool verbose = true)` — generate Chebyshev nodes, evaluate function on tensor grid, pre-compute weights and diff matrices
   - `Eval(double[] point, int[] derivativeOrder)` — loop-based evaluation (Python `eval()`)
   - `VectorizedEval(double[] point, int[] derivativeOrder)` — BLAS-style reshape trick (Python `vectorized_eval()`)
   - `VectorizedEvalBatch(double[,] points, int[] derivativeOrder)` — batch evaluation
   - `VectorizedEvalMulti(double[] point, int[][] derivativeOrders)` — multi-output (price + Greeks)
   - `ErrorEstimate()` — DCT-II based Chebyshev coefficient decay check

3. **Serialization** — save/load:
   - `Save(string path)` / `Load(string path)` — binary serialization (use `System.Text.Json` or `MessagePack`)
   - Custom serialization that matches Python's pickle contract (same data, different format)

4. **Static factory methods:**
   - `Nodes(int ndim, double[][] domains, int[] orders)` ← Python `nodes()`
   - `FromValues(int ndim, double[][] domains, int[] orders, double[,...] values)` ← Python `from_values()`
   - Must be **bit-identical** to `Build()` (all precomputed data depends only on node positions)

5. **Arithmetic operators** (Internal/Algebra.cs):
   - `operator +`, `-`, `*`, `/` (CT ± CT, CT * scalar, CT / scalar)
   - Unary `-`
   - `+=`, `-=`, `*=`, `/=`
   - `_FromGrid()` factory using internal constructor (Python `_from_grid()`)

6. **Extrusion and slicing** (Internal/ExtrudeSlice.cs):
   - `Extrude(...)` — add dimension where function is constant
   - `Slice(...)` — fix dimension at value via barycentric contraction
   - Fast path when slice value coincides with a node

7. **Calculus** (Internal/Calculus.cs):
   - `Integrate(int[]? dims, (double, double)[]? bounds)` — Fejér-1 quadrature via DCT-III
   - `Roots(int? dim, (int, double)[]? fixed)` — colleague matrix eigenvalue method
   - `Minimize(...)` / `Maximize(...)` — derivative roots + endpoint evaluation

**Port these test files (in order):**
- `test_barycentric.py` → `BarycentricTests.cs` (48 tests)
- `test_from_values.py` → `FromValuesTests.cs` (65 tests)
- `test_algebra.py` → `AlgebraTests.cs` (77 tests — partial, ChebyshevApproximation only)
- `test_extrude_slice.py` → `ExtrudeSliceTests.cs` (63 tests — partial)
- `test_calculus.py` → `CalculusTests.cs` (74 tests — partial)

### Phase 2: ChebyshevSpline

**Read:** `ref/PyChebyshev/src/pychebyshev/spline.py`

- Piecewise Chebyshev with user-specified knots at singularities
- Each piece is an independent `ChebyshevApproximation`
- Routing via `Array.BinarySearch` (equivalent to `np.searchsorted`)
- All operations: eval, eval_multi, eval_batch, error_estimate, save/load,
  nodes, from_values, extrude, slice, integrate, roots, minimize, maximize,
  arithmetic operators
- **Port:** `test_spline.py` (55 tests) + remaining algebra/extrude-slice/calculus tests

### Phase 3: ChebyshevSlider

**Read:** `ref/PyChebyshev/src/pychebyshev/slider.py`

- Sliding Technique: partition dimensions into groups, build per-group Chebyshev interpolants
- Additive decomposition around a pivot point
- All operations: eval, eval_multi, error_estimate, save/load, extrude, slice,
  arithmetic operators
- **Port:** `test_slider.py` (40 tests) + remaining algebra/extrude-slice tests

### Phase 4: ChebyshevTT

**Read:** `ref/PyChebyshev/src/pychebyshev/tensor_train.py`

- Tensor Train decomposition via TT-Cross (alternating sweeps with maxvol pivoting)
- TT-SVD mode for pre-computed tensors
- Batch eval via einsum-like contractions
- Finite-difference derivatives (NOT spectral)
- All operations: eval, eval_batch, eval_multi, error_estimate, save/load
- No algebra, extrude/slice, or calculus (not implemented in Python either)
- **Port:** `test_tensor_train.py` (35 tests)

## Validation Strategy

### Cross-language correctness

The C# implementation must produce **numerically identical results** to PyChebyshev
(within floating-point tolerance). For every test:

1. Read the Python test in `ref/PyChebyshev/tests/`
2. Understand what it asserts (accuracy tolerance, exact values, error messages)
3. Port the test to C# with the **same inputs, same expected outputs, same tolerances**
4. If the Python test uses `pytest.approx(x, rel=1e-6)`, use `Assert.Equal(x, actual, precision: 6)` or a custom tolerance helper

### Test helpers

Create `tests/ChebyshevSharp.Tests/Helpers/BlackScholes.cs` ported from
`ref/PyChebyshev/tests/conftest.py`. This contains:
- `BsPrice(double S, double K, double sigma, double T, double r)` — analytical BS call price
- `BsDelta(...)`, `BsGamma(...)`, `BsVega(...)`, `BsRho(...)` — analytical Greeks
- `BsPrice5D(double S, double K, double sigma, double T, double r)` — same but 5-arg for 5D tests
- Helper functions that create standard test interpolants (sin 3D, BS 3D, BS 5D, etc.)

### Skip list for feature parity tracking

Maintain `skip_csharp.txt` in the repo root. Each line is a test that is not yet
implemented. Remove lines as features are completed. Format:

```
# Phase 2: ChebyshevSpline (not yet implemented)
SplineTests.*
# Phase 3: ChebyshevSlider (not yet implemented)
SliderTests.*
# Phase 4: ChebyshevTT (not yet implemented)
TensorTrainTests.*
```

The skip list doubles as a TODO tracker and progress indicator.

## Key Technical Details

### NumPy → C# Equivalents

| NumPy | C# Equivalent |
|-------|---------------|
| `np.ndarray` | `double[]` (1D), `double[,]` (2D), or jagged `double[][]` |
| `np.zeros(shape)` | `new double[n]` (auto-zeroed) |
| `np.cos`, `np.sin` | `Math.Cos`, `Math.Sin` |
| `np.meshgrid(..., indexing='ij')` | Manual nested loops or LINQ |
| `np.tensordot(A, w, axes=([d],[0]))` | Manual contraction loop or reshape+BLAS |
| `np.polynomial.chebyshev.chebpts1(n)` | `cos((2i-1)*π/(2n))` for i=1..n |
| `np.linalg.eig` | Use `MathNet.Numerics` or implement companion matrix eigenvalues |
| `scipy.fft.dct(x, type=2)` | Implement DCT-II directly or use a library |
| `scipy.fft.dct(x, type=3)` | Implement DCT-III (inverse DCT-II scaled) |
| `np.searchsorted` | `Array.BinarySearch` |
| `pickle.dump/load` | `System.Text.Json` or binary serialization |

### BLAS Evaluation Strategy

The key performance insight from PyChebyshev: `vectorized_eval()` reshapes the tensor
contraction into a matrix-vector multiply per dimension, routed through BLAS GEMV.

In C#, use one of:
- Raw `double[]` with manual BLAS-style loops (simplest, no dependencies)
- `System.Numerics.Tensors.TensorPrimitives` for vectorized operations
- `MathNet.Numerics` BLAS bindings if needed

**Start with raw arrays.** Optimize later if benchmarks warrant it.

### Chebyshev Nodes

PyChebyshev uses **Type I Chebyshev nodes** (roots of T_n):
```
x_i = cos((2i - 1) * π / (2n)), for i = 1, ..., n
```
Mapped to domain [a, b]: `node = (a + b) / 2 + (b - a) / 2 * x_i`

These are from `numpy.polynomial.chebyshev.chebpts1(n)` which returns them in
ascending order. PyChebyshev stores nodes in ascending order (smallest first).

### Error Estimation

Uses DCT-II to extract Chebyshev coefficients, then checks the magnitude of the
last coefficient. The DCT-II implementation must match NumPy/SciPy conventions.
Read `ref/PyChebyshev/src/pychebyshev/barycentric.py` method `_chebyshev_coefficients_1d()`
and `error_estimate()` carefully.

### Serialization Format

Do NOT try to read Python pickle files. ChebyshevSharp uses its own serialization
format. The contract is: save all pre-computed data (nodes, weights, diff matrices,
tensor values, domains, orders) so that `Load()` produces a fully functional interpolant
without needing the original function.

## Coding Conventions

- **Namespace**: `ChebyshevSharp` (public API), `ChebyshevSharp.Internal` (helpers)
- **Naming**: PascalCase for public methods (`VectorizedEval`), camelCase for locals
- **Nullability**: enabled (`<Nullable>enable</Nullable>`) — use `?` annotations properly
- **XML docs**: required on all public classes and methods (NumPy-style docstrings → XML `<summary>`)
- **No unnecessary dependencies**: prefer raw arrays over heavy libraries. Only add
  a NuGet dependency if reimplementing would be error-prone (e.g., eigenvalue decomposition)
- **Tests**: use xUnit `[Fact]` for simple tests, `[Theory]` with `[InlineData]` for parameterized
- **Tolerances**: match the Python test tolerances exactly. Use a helper:
  ```csharp
  static void AssertClose(double expected, double actual, double rtol = 1e-10, double atol = 1e-14)
  {
      Assert.True(Math.Abs(expected - actual) <= atol + rtol * Math.Abs(expected),
          $"Expected {expected}, got {actual} (rtol={rtol}, atol={atol})");
  }
  ```

## DO NOT

- Modify anything in `ref/PyChebyshev/` — it is a read-only reference
- Port `_jit.py` or `fast_eval()` — these are deprecated in Python
- Use `float` — always use `double` for numerical work
- Add GUI, web, or console app projects — this is a library only
- Skip writing tests — every public method must have test coverage matching Python

## CI/CD

### Workflows (`.github/workflows/`)

- **`test.yml`** — Runs `dotnet test` on .NET 8 and .NET 10 on push/PR to main.
  Collects code coverage via `coverlet` and uploads to Codecov on the .NET 10 run.
  Summary job `All Tests Passed` gates branch protection.

- **`publish.yml`** — Triggers on GitHub release creation. Runs `dotnet pack` in
  Release configuration and pushes the `.nupkg` to nuget.org via `secrets.NUGET_API_KEY`.

- **`dependabot-automerge.yml`** — Auto-approves and merges Dependabot PRs for
  patch/minor version bumps after tests pass. Major bumps require manual review.

- **`docs.yml`** — Builds documentation with DocFX and deploys to GitHub Pages
  at https://0xc000005.github.io/ChebyshevSharp/. Triggers on push to main when
  `docs/` or `src/` files change. API reference is auto-generated from XML doc
  comments in the source code.

### Documentation (`docs/`)

DocFX-based documentation site deployed to GitHub Pages.

- `docs/docfx.json` — DocFX configuration (metadata extraction from `.csproj`, build settings)
- `docs/index.md` — Landing page
- `docs/docs/` — User guide articles (introduction, getting-started)
- `docs/api/` — Auto-generated from `<summary>` XML doc comments in source code
- `docs/_site/` — Build output (gitignored)

To add a new user guide page: create a `.md` file in `docs/docs/`, add it to `docs/docs/toc.yml`.

### Dependabot (`.github/dependabot.yml`)

Configured for weekly checks on:
- `nuget` — NuGet package updates (xUnit, coverlet, etc.)
- `github-actions` — Action version updates (checkout, setup-dotnet, codecov, etc.)

### Branch Protection

Ruleset "Protect main" is active with:
- No branch deletion
- No force pushes
- Required status check: `All Tests Passed` (GitHub Actions integration ID 15368)
- Admin bypass enabled (repo owner can push directly)

### Codecov

Code coverage collected via `coverlet.collector` (already a dependency in the test
project). The test workflow uploads `coverage.cobertura.xml` to Codecov. To enable:
add the `CODECOV_TOKEN` secret in the repo's Settings → Secrets → Actions.

### Secrets Required

| Secret | Where to get it | Used by |
|--------|----------------|---------|
| `NUGET_API_KEY` | [nuget.org → API Keys](https://www.nuget.org/account/apikeys) | `publish.yml` |
| `CODECOV_TOKEN` | [codecov.io](https://app.codecov.io/) after adding the repo | `test.yml` |

## Progress Tracking

After completing each phase, update `skip_csharp.txt` to remove passing test categories.
When all 457 tests pass, the port is feature-complete. The definition of done for each phase:

- [ ] All corresponding Python tests ported and passing
- [ ] `dotnet build` succeeds with zero warnings
- [ ] `dotnet test` passes all non-skipped tests
- [ ] XML documentation on all public API members
- [ ] `skip_csharp.txt` updated

## Release Process

1. Update `<Version>` in `src/ChebyshevSharp/ChebyshevSharp.csproj`
2. Update `docs/docs/changelog.md`
3. Commit, push to main
4. `gh release create vX.Y.Z` — triggers `publish.yml` → NuGet
