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
dotnet tool install --global docfx
```

Run examples when changing public workflows:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

Property tests live in `tests/ChebyshevSharp.Tests/PropertyTests.cs` and run with the normal xUnit suite. They use deterministic FsCheck seeds and small grids so failures are reproducible and fast to debug.

Mutation testing is available as a manual or scheduled workflow. To run it locally:

```bash
dotnet tool install --global dotnet-stryker
dotnet stryker --config-file stryker-config.json
```

The mutation workflow targets high-risk numerical code first. Current thresholds
are high 80, low 60, break 0. Treat mutation score as a release-readiness signal:
changes in tensor shape arithmetic, Tensor Train kernels, slicing/extrusion, or
other numerical core paths should not introduce unexplained surviving mutants.

Codecov runs on pull requests with a 90% patch target and a 1% allowed project
coverage drop. Modified coverable lines should have focused tests.

## Documentation Source Checks

When changing citations, mathematical references, or README links, verify the
sources before release:

- Rebuild the site with `docfx docs/docfx.json`.
- Check project-site links against the generated `docs/_site` output because
  the public GitHub Pages site can lag an open PR.
- Verify DOI-backed references against Crossref metadata, not only HTTP status.
- Verify book citations against publisher, author, library, or ISBN pages.
- Record the commands and evidence in the release plan or PR notes.
