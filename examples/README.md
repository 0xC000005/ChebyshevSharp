# Examples

Runnable examples are included as small console projects. Run them from the
repository root after installing the .NET 10 SDK:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/SliderPartitionValidation/SliderPartitionValidation.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

## QuickStart

`QuickStart` builds a dense 2D `ChebyshevApproximation`, evaluates a point,
computes first derivatives, and compares the interpolated values with the
original function. Use it as the smallest runnable example for the standard
build/evaluate/check workflow.

## SliderPartitionValidation

`SliderPartitionValidation` builds two 8D `ChebyshevSlider` models for the same
pairwise function. One partition groups the interacting variable pairs; the other
uses singleton groups that intentionally miss those interactions. The output
compares dense-grid size, build calls, held-out errors, error diagnostics, and a
first derivative.

## TensorTrainHighDim

`TensorTrainHighDim` builds a 7D `ChebyshevTT` with TT-Cross, reports the dense
grid size that the TT representation avoids, and checks one held-out point
against the original model. Use it when evaluating whether TT-Cross is a better
starting point than dense tensor construction for a high-dimensional smooth
model.
