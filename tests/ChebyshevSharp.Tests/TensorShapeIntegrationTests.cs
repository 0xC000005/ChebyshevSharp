using ChebyshevSharp.Internal;

namespace ChebyshevSharp.Tests;

public class TensorShapeIntegrationTests
{
    private static readonly double[][] UnitSquare =
    [
        [-1.0, 1.0],
        [-1.0, 1.0]
    ];

    private static readonly int[] OverflowingGridShape = [46341, 46341];

    [Fact]
    public void Approx_FromValues_Throws_Overflow_For_Grid_Size_Above_Int_MaxValue()
    {
        var ex = Assert.Throws<OverflowException>(() =>
            ChebyshevApproximation.FromValues(
                [0.0],
                numDimensions: 2,
                domain: UnitSquare,
                nNodes: OverflowingGridShape));

        Assert.Contains("FromValues", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Fact]
    public void Approx_SetOriginalFunctionValues_Throws_Overflow_For_Deferred_Huge_Grid()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 2,
            domain: UnitSquare,
            nNodes: OverflowingGridShape,
            deferBuild: true);

        var ex = Assert.Throws<OverflowException>(() =>
            deferred.SetOriginalFunctionValues([0.0]));

        Assert.Contains("SetOriginalFunctionValues", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Fact]
    public void Approx_GetNumEvaluationPoints_Throws_Overflow_For_Deferred_Huge_Grid()
    {
        var deferred = new ChebyshevApproximation(
            (_, _) => 0.0,
            numDimensions: 2,
            domain: UnitSquare,
            nNodes: OverflowingGridShape,
            deferBuild: true);

        var ex = Assert.Throws<OverflowException>(() => deferred.GetNumEvaluationPoints());

        Assert.Contains("GetNumEvaluationPoints", ex.Message);
        Assert.Contains("int.MaxValue", ex.Message);
    }

    [Fact]
    public void Spline_FromValues_Throws_Overflow_For_Piece_Grid_Size_Above_Int_MaxValue()
    {
        var ex = Assert.Throws<OverflowException>(() =>
            ChebyshevSpline.FromValues(
                [[0.0]],
                numDimensions: 2,
                domain: UnitSquare,
                nNodes: OverflowingGridShape,
                knots: [[], []]));

        Assert.Contains("FromValues", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Fact]
    public void Spline_SetOriginalFunctionValues_Throws_Overflow_For_Deferred_Huge_Grid()
    {
        var deferred = new ChebyshevSpline(
            (_, _) => 0.0,
            numDimensions: 2,
            domain: UnitSquare,
            nNodes: OverflowingGridShape,
            knots: [[], []],
            deferBuild: true);

        var ex = Assert.Throws<OverflowException>(() =>
            deferred.SetOriginalFunctionValues([0.0]));

        Assert.Contains("SetOriginalFunctionValues", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Fact]
    public void ExtrudeTensor_Throws_Overflow_When_New_Shape_Is_Too_Large()
    {
        var ex = Assert.Throws<OverflowException>(() =>
            ExtrudeSlice.ExtrudeTensor([0.0], OverflowingGridShape, axis: 0, nNew: 2));

        Assert.Contains("ExtrudeTensor", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Fact]
    public void Slider_TotalBuildEvals_Throws_Overflow_For_Huge_Slide_Group()
    {
        var slider = new ChebyshevSlider(
            (_, _) => 0.0,
            numDimensions: 2,
            domain: UnitSquare,
            nNodes: OverflowingGridShape,
            partition: [[0, 1]],
            pivotPoint: [0.0, 0.0]);

        var ex = Assert.Throws<OverflowException>(() => slider.TotalBuildEvals);

        Assert.Contains("TotalBuildEvals", ex.Message);
        Assert.Contains("46341", ex.Message);
    }

    [Theory]
    [InlineData("take")]
    [InlineData("tensordot")]
    [InlineData("matmul")]
    public void Barycentric_Axis_Kernels_Throw_Overflow_For_Huge_Result_Shape(string operation)
    {
        int[] shape = [46341, 46341, 2];

        var ex = Assert.Throws<OverflowException>(() =>
        {
            switch (operation)
            {
                case "take":
                    BarycentricKernel.TakeAlongAxis([0.0], shape, axis: 2, index: 0);
                    break;
                case "tensordot":
                    BarycentricKernel.TensordotVector([0.0], shape, axis: 2, weights: [1.0, 1.0]);
                    break;
                case "matmul":
                    BarycentricKernel.MatmulAlongAxis([0.0], shape, axis: 2, new double[2, 2]);
                    break;
            }
        });

        Assert.Contains(operation == "take" ? "TakeAlongAxis" :
            operation == "tensordot" ? "TensordotVector" : "MatmulAlongAxis",
            ex.Message);
        Assert.Contains("46341", ex.Message);
    }
}
