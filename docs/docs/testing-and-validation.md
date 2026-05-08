# Testing and Validation

Run the standard validation suite before opening a PR:

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
dotnet tool install --global docfx --version 2.78.5
```

Run examples when changing public workflows:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

The [Examples](examples.md) page explains what each project is meant to validate
and how the output should be interpreted.

CI runs the examples in the validation job. The test matrix pins the selected
SDK inside each job: .NET 8 builds the library target, and .NET 10 runs the
full xUnit suite with coverage upload.

## CI Gates

Every pull request to `main` runs:

- `Format, Pack, and Docs`: restore, format verification, Release build,
  runnable examples, package validation, and DocFX.
- `.NET 8 library build`: builds `src/ChebyshevSharp/ChebyshevSharp.csproj`
  for `net8.0`.
- `.NET 10 tests`: builds and runs the xUnit test project for `net10.0` with
  coverage collection and Codecov upload.
- `All Tests Passed`: summary gate that fails if either validation or tests
  fail.

Wait for the separate `codecov/patch` status before merging. A completed GitHub
Actions run is not enough if Codecov is still pending.

Codecov runs on pull requests with a 90% patch target and a 1% allowed project
coverage drop. Modified coverable lines should have focused tests. The
configuration treats missing coverage reports as success so docs-only or
community-only changes do not need fake source edits.

Property tests live in `tests/ChebyshevSharp.Tests/PropertyTests.cs` and run
with the normal xUnit suite. They use deterministic FsCheck seeds and small
grids so failures are reproducible and fast to debug.

Mutation testing is available as a manual or scheduled workflow. To run it locally:

```bash
dotnet tool install --global dotnet-stryker --version 4.14.1
dotnet stryker --config-file stryker-config.json
```

The mutation workflow targets high-risk numerical code first. Current thresholds
are high 80, low 60, break 0. Treat mutation score as a release-readiness signal:
changes in tensor shape arithmetic, Tensor Train kernels, slicing/extrusion, or
other numerical core paths should not introduce unexplained surviving mutants.

## Release Gate

Publishing to NuGet is tied to GitHub releases. The publish workflow verifies
that the release tag, such as `v0.13.1`, matches the `<Version>` in
`src/ChebyshevSharp/ChebyshevSharp.csproj` before restore, build, test, pack, and
`dotnet nuget push`. Documentation-only changes normally do not need a package
version bump unless they change NuGet-visible metadata or package contents.

## Documentation Source Checks

When changing citations, mathematical references, or README links, verify the
sources before release:

- Rebuild the site with `docfx docs/docfx.json`.
- Check project-site links against the generated `docs/_site` output because
  the public GitHub Pages site can lag an open PR.
- Verify DOI-backed references against Crossref metadata, not only HTTP status.
- Verify book citations against publisher, author, library, or ISBN pages.
- Record the commands and evidence in the PR notes.
