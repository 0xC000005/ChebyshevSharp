---
title: Contributing
---

# Contributing

ChebyshevSharp is a numerical library. Contributions should keep mathematical
behavior explicit, tested, and documented.

## Philosophy

- Preserve PyChebyshev parity unless a C#-specific difference is intentional.
- Document convention choices, such as Type I roots versus Lobatto/extrema grids.
- Prefer clear validation errors over overflow, shape mismatch, or silent
  truncation.
- Keep PRs focused; split unrelated docs, performance, and API changes.

## Issues

Search existing issues, pull requests, and discussions before opening a new one.
Usage questions belong in GitHub Discussions. Bug reports should include a
minimal runnable example, ChebyshevSharp version, .NET SDK version, OS, expected
behavior, and actual behavior.

Numerical accuracy reports should include the function, domain, node counts,
construction method, tolerance, and an independent reference value. State whether
the behavior matches or differs from PyChebyshev when that comparison is
available.

Security issues should follow the repository `SECURITY.md` file, not public
issues.

## Required Checks

Run these before opening a PR:

```bash
dotnet restore
dotnet format --verify-no-changes --verbosity minimal
dotnet build --configuration Release --no-restore
dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage"
dotnet pack src/ChebyshevSharp --configuration Release --no-build --output artifacts/packages
docfx docs/docfx.json
```

Install DocFX with `dotnet tool install --global docfx` if it is not already
available.

## Coverage and Mutation Testing

Codecov enforces a 90% patch-coverage target and allows at most a 1% project
coverage drop. Add focused tests for modified coverable lines.

Stryker.NET mutation testing targets high-risk numerical code:

```bash
dotnet tool install --global dotnet-stryker
dotnet stryker --config-file stryker-config.json
```

Current thresholds are high 80, low 60, break 0. For numerical core changes,
address surviving mutants in touched code or explain why they are equivalent,
unreachable, or intentionally deferred.

## Pull Requests

Link the relevant issue, summarize the behavior change, list validation commands,
and update docs/changelog for public behavior. Include benchmark evidence for
performance claims.

Keep pull requests focused. For numerical changes, describe the algorithmic
risk, the reference implementation or paper used, and the regression tests that
cover the change. Docs-only changes normally do not require a version bump.

The repository root `CONTRIBUTING.md` contains the full contributor guide.
