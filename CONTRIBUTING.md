# Contributing to ChebyshevSharp

ChebyshevSharp is a numerical library. Correctness, reproducibility, and clear
documentation matter more than broad API churn. Small, focused pull requests are
easier to review and safer to release.

## Project Philosophy

- Preserve PyChebyshev parity when porting upstream behavior.
- Document deliberate differences, especially mathematical conventions such as
  Type I roots versus Lobatto/extrema grids.
- Prefer explicit overflow, shape, and domain validation over silent truncation.
- Add examples, XML comments, and docs for user-facing behavior changes.
- Keep performance work measurable with benchmarks or before/after numbers.

## Before Opening an Issue

Search existing issues, pull requests, and discussions first. If the behavior is
a usage question, start with the docs pages linked from the README and open a
GitHub Discussion with the smallest code snippet that shows where you are stuck.

For bug reports, include:

- ChebyshevSharp version and .NET SDK version.
- Operating system and CPU architecture.
- A minimal runnable example with domain, node counts, and build method.
- Expected behavior, actual behavior, and any exception text.

For numerical accuracy reports, also include the reference value, how it was
computed, and whether the problem reproduces with PyChebyshev or another
independent implementation.

Security reports should not be filed publicly. Follow `SECURITY.md`.

## Local Validation

Run the focused tests while iterating, then run the standard suite before opening
a pull request:

```bash
dotnet restore
dotnet format --verify-no-changes --verbosity minimal
dotnet build --configuration Release --no-restore
dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage"
dotnet pack src/ChebyshevSharp --configuration Release --no-build --output artifacts/packages
docfx docs/docfx.json
```

Install DocFX first if needed:

```bash
dotnet tool install --global docfx
```

For targeted iteration:

```bash
dotnet test --filter "FullyQualifiedName~TtSobolIndicesTests"
```

## Coverage and Mutation Requirements

Codecov runs on pull requests. The current policy is:

- `codecov/patch` target: 90% patch coverage.
- `codecov/project` target: automatic baseline with a 1% allowed drop.
- Modified coverable lines should have focused tests. Do not lower coverage
  thresholds to make a PR pass.

Mutation testing uses Stryker.NET for high-risk numerical paths:

```bash
dotnet tool install --global dotnet-stryker
dotnet stryker --config-file stryker-config.json
```

The configured Stryker thresholds are high 80, low 60, break 0. Treat mutation
score as a release-readiness signal for now: changes in tensor shape arithmetic,
Tensor Train kernels, slicing/extrusion, and other numerical core paths should
not introduce unexplained surviving mutants or reduce the baseline without a PR
note.

## Pull Request Checklist

- Link the relevant issue with `Fixes #123` or `Refs #123`.
- Summarize the behavior change and any mathematical convention involved.
- List commands run locally.
- Add or update tests that fail before the fix and pass after it.
- Update docs, examples, and changelog for public API or numerical behavior.
- Include benchmark results when performance claims are part of the PR.

## Pull Request Scope and Review

Prefer focused pull requests. Split unrelated code, docs, benchmark, and release
changes unless they must ship together. Large numerical changes should explain
the affected algorithm, the reference behavior, and the regression tests that
cover the risk.

When receiving review feedback, respond with either a code change, a clarifying
question, or a short explanation of why the existing behavior is intentional.
Keep the thread tied to the technical point under discussion.

Docs-only changes normally do not require a version bump or NuGet release. Bump
the package version only when installable package contents, public behavior, or
NuGet-visible metadata must change immediately.

## Documentation Style

Keep docs practical and source-backed. Use short examples, define formulas before
using them, cite papers or upstream source files for algorithms, and call out
where ChebyshevSharp intentionally differs from PyChebyshev or MoCaX.
