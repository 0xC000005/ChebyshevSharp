# Examples

Runnable examples are included as small console projects:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

`QuickStart` builds a dense 2D interpolant with analytical derivatives.
`TensorTrainHighDim` builds a 7D Tensor Train with TT-Cross and reports the
dense-grid size that the TT representation avoids.
