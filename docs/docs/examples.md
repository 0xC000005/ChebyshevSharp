---
title: Examples
---

# Examples

The repository includes three runnable console projects. They are intended to be
small enough to read in one sitting and concrete enough to verify that the local
toolchain, package reference, and public API are working.

Run the examples from the repository root:

```bash
dotnet run --project examples/QuickStart/QuickStart.csproj
dotnet run --project examples/SliderPartitionValidation/SliderPartitionValidation.csproj
dotnet run --project examples/TensorTrainHighDim/TensorTrainHighDim.csproj
```

## QuickStart

`examples/QuickStart` demonstrates the standard dense workflow:

1. Define a smooth 2D function on a bounded domain.
2. Build a `ChebyshevApproximation` on a tensor-product Chebyshev grid.
3. Evaluate the value and first derivatives at an interior point.
4. Compare the interpolated values with the original function.
5. Print the `ErrorEstimate()` diagnostic.

Use this example before adapting the larger snippets in
[Getting Started](getting-started.md).

## SliderPartitionValidation

`examples/SliderPartitionValidation` demonstrates the high-dimensional Slider
workflow:

1. Define an 8D model with known pairwise interactions.
2. Build one `ChebyshevSlider` with the correct pair grouping.
3. Build a second slider with an intentionally weak singleton grouping.
4. Compare build costs against the dense grid size.
5. Validate both sliders on held-out points and compare a derivative.

Use this example when deciding whether a proposed partition is a modelling
assumption you can defend. A small `ErrorEstimate()` does not prove the partition
captures cross-group interactions; held-out points far from the pivot are the
more important check.

## TensorTrainHighDim

`examples/TensorTrainHighDim` demonstrates the high-dimensional TT-Cross
workflow:

1. Define a smooth 7D coupled model.
2. Build a `ChebyshevTT` with `method: "cross"`.
3. Compare the dense grid size with the number of TT-Cross evaluations.
4. Inspect TT ranks and compression ratio.
5. Check one held-out point against the original function.

Use this example when the dense grid size `prod(n_i)` is the limiting factor and
the model is expected to have moderate numerical TT ranks. If the TT ranks grow
to the rank cap or the held-out error is too high, increase the rank cap, adjust
the node counts, or use a dense/spline/slider representation instead.

## When to run them

Run the examples when changing public workflows, constructor defaults,
serialization behavior, domain validation, or documentation snippets that show
first-use code. CI runs both projects in the `Format, Pack, and Docs` job, so a
PR that breaks the examples should fail before merge.
