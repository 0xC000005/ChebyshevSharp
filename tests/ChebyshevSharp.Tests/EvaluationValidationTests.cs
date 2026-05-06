using ChebyshevSharp.Tests.Helpers;

namespace ChebyshevSharp.Tests;

public class EvaluationDerivativeOrderValidationTests
{
    [Fact]
    public void Approximation_rejects_negative_derivative_orders()
    {
        var cheb = TestFixtures.ChebSin3D;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            cheb.VectorizedEval([0.1, 0.3, 1.7], [-1, 0, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            cheb.VectorizedEvalBatch([[0.1, 0.3, 1.7]], [-1, 0, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            cheb.VectorizedEvalMulti([0.1, 0.3, 1.7], [[-1, 0, 0]]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            cheb.GetDerivativeId([-1, 0, 0]));
    }

    [Fact]
    public void Spline_rejects_negative_derivative_orders()
    {
        var spline = TestFixtures.SplineAbs1D;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.Eval([0.25], [-1]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.EvalBatch([[0.25]], [-1]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.EvalMulti([0.25], [[-1]]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.GetDerivativeId([-1]));
    }

    [Fact]
    public void Slider_rejects_negative_derivative_orders()
    {
        var slider = TestFixtures.AlgebraSliderF;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slider.Eval([0.1, 0.2, 0.3], [-1, 0, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slider.EvalMulti([0.1, 0.2, 0.3], [[-1, 0, 0]]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slider.GetDerivativeId([-1, 0, 0]));
    }

    [Fact]
    public void TensorTrain_rejects_negative_derivative_orders()
    {
        var tt = TestFixtures.TtSin3D;

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.EvalMulti([0.1, 0.2, 0.3], [[-1, 0, 0]]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.GetDerivativeId([-1, 0, 0]));
    }
}
