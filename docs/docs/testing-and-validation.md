# Testing and Validation

Run the standard validation suite before opening a PR:

```bash
dotnet format --verify-no-changes --verbosity minimal
dotnet build --verbosity minimal
dotnet test --verbosity minimal
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

Property tests live in `tests/ChebyshevSharp.Tests/PropertyTests.cs` and run with the normal xUnit suite. They use deterministic FsCheck seeds and small grids so failures are reproducible and fast to debug.

Mutation testing is available as a manual or scheduled workflow. To run it locally:

```bash
dotnet tool install --global dotnet-stryker
dotnet stryker --config-file stryker-config.json
```

The mutation workflow targets high-risk numerical code first. Treat the first score as a baseline; do not use it as a hard release gate until runtime and flake behavior are understood.
