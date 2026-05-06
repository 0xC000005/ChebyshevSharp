using ChebyshevSharp.Tests.Helpers;
using Xunit;

namespace ChebyshevSharp.Tests;

public class EvaluationValidationTests
{
    private static ChebyshevApproximation BuildApproximation()
    {
        var approx = new ChebyshevApproximation(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        approx.Build(verbose: false);
        return approx;
    }

    private static ChebyshevSpline BuildSpline()
    {
        var spline = new ChebyshevSpline(
            (p, _) => p[0] * p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            knots: new[] { new[] { 0.0 }, Array.Empty<double>() });
        spline.Build(verbose: false);
        return spline;
    }

    private static ChebyshevSlider BuildSlider()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });
        slider.Build(verbose: false);
        return slider;
    }

    private static ChebyshevTT BuildTensorTrain()
    {
        var tt = new ChebyshevTT(
            p => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 });
        tt.Build(verbose: false, seed: 42);
        return tt;
    }

    [Fact]
    public void Approximation_eval_rejects_non_finite_point()
    {
        var approx = BuildApproximation();

        Assert.Throws<ArgumentException>(() =>
            approx.Eval(new[] { double.NaN, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            approx.VectorizedEval(new[] { 0.0, double.PositiveInfinity }, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            approx.VectorizedEvalMulti(new[] { 0.0, double.NegativeInfinity }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Approximation_eval_rejects_out_of_domain_point()
    {
        var approx = BuildApproximation();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            approx.Eval(new[] { -1.1, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            approx.VectorizedEval(new[] { 0.0, 1.1 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            approx.VectorizedEvalMulti(new[] { 0.0, -1.1 }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Approximation_batch_eval_rejects_non_finite_point()
    {
        var approx = BuildApproximation();

        Assert.Throws<ArgumentException>(() =>
            approx.VectorizedEvalBatch(
                new[] { new[] { 0.0, 0.0 }, new[] { double.NaN, 0.0 } },
                new[] { 0, 0 }));
    }

    [Fact]
    public void Approximation_batch_eval_rejects_out_of_domain_point()
    {
        var approx = BuildApproximation();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            approx.VectorizedEvalBatch(
                new[] { new[] { 0.0, 0.0 }, new[] { 1.1, 0.0 } },
                new[] { 0, 0 }));
    }

    [Fact]
    public void Approximation_eval_rejects_null_or_wrong_shape_points()
    {
        var approx = BuildApproximation();

        Assert.Throws<ArgumentNullException>(() =>
            approx.Eval(null!, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            approx.VectorizedEval(new[] { 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentNullException>(() =>
            approx.VectorizedEvalBatch(null!, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            approx.VectorizedEvalBatch(new double[][] { null! }, new[] { 0, 0 }));
    }

    [Fact]
    public void Spline_eval_rejects_non_finite_point_before_piece_routing()
    {
        var spline = BuildSpline();

        Assert.Throws<ArgumentException>(() =>
            spline.Eval(new[] { double.NaN, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            spline.EvalMulti(new[] { 0.0, double.PositiveInfinity }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Spline_eval_rejects_out_of_domain_point_before_piece_routing()
    {
        var spline = BuildSpline();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.Eval(new[] { -1.1, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.EvalMulti(new[] { 0.0, 1.1 }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Spline_batch_eval_rejects_non_finite_point_before_piece_routing()
    {
        var spline = BuildSpline();

        Assert.Throws<ArgumentException>(() =>
            spline.EvalBatch(
                new[] { new[] { 0.0, 0.0 }, new[] { double.NegativeInfinity, 0.0 } },
                new[] { 0, 0 }));
    }

    [Fact]
    public void Spline_batch_eval_rejects_out_of_domain_point_before_piece_routing()
    {
        var spline = BuildSpline();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            spline.EvalBatch(
                new[] { new[] { 0.0, 0.0 }, new[] { 0.0, -1.1 } },
                new[] { 0, 0 }));
    }

    [Fact]
    public void Slider_eval_rejects_non_finite_point()
    {
        var slider = BuildSlider();

        Assert.Throws<ArgumentException>(() =>
            slider.Eval(new[] { double.NaN, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentException>(() =>
            slider.EvalMulti(new[] { 0.0, double.PositiveInfinity }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Slider_eval_rejects_out_of_domain_point()
    {
        var slider = BuildSlider();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slider.Eval(new[] { -1.1, 0.0 }, new[] { 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            slider.EvalMulti(new[] { 0.0, 1.1 }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void Slider_eval_multi_rejects_unbuilt_before_domain_validation()
    {
        var slider = new ChebyshevSlider(
            (p, _) => p[0] + p[1],
            numDimensions: 2,
            domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
            nNodes: new[] { 5, 5 },
            partition: new[] { new[] { 0 }, new[] { 1 } },
            pivotPoint: new[] { 0.0, 0.0 });

        Assert.Throws<InvalidOperationException>(() =>
            slider.EvalMulti(new[] { 0.0, 1.1 }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void TensorTrain_eval_rejects_non_finite_point()
    {
        var tt = BuildTensorTrain();

        Assert.Throws<ArgumentException>(() =>
            tt.Eval(new[] { double.NaN, 0.0 }));
        Assert.Throws<ArgumentException>(() =>
            tt.EvalMulti(new[] { 0.0, double.PositiveInfinity }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void TensorTrain_eval_rejects_out_of_domain_point()
    {
        var tt = BuildTensorTrain();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.Eval(new[] { -1.1, 0.0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            tt.EvalMulti(new[] { 0.0, 1.1 }, new[] { new[] { 0, 0 } }));
    }

    [Fact]
    public void TensorTrain_eval_rejects_out_of_domain_user_point_after_reorder()
    {
        static double F(double[] p) => p[0] + p[1];
        var tt = new ChebyshevTT(
            F,
            numDimensions: 2,
            domain: new[] { new[] { 0.0, 1.0 }, new[] { 10.0, 20.0 } },
            nNodes: new[] { 5, 5 },
            maxRank: 4);
        tt.Build(verbose: false, method: "svd");
        var reord = tt.Reorder(new[] { 1, 0 }, maxRank: 4, tolerance: 1e-12);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            reord.Eval(new[] { 15.0, 0.5 }));
    }

    [Fact]
    public void TensorTrain_batch_eval_rejects_non_finite_point()
    {
        var tt = BuildTensorTrain();
        var points = new double[,] { { 0.0, 0.0 }, { double.NegativeInfinity, 0.0 } };

        Assert.Throws<ArgumentException>(() => tt.EvalBatch(points));
    }

    [Fact]
    public void TensorTrain_batch_eval_rejects_out_of_domain_point()
    {
        var tt = BuildTensorTrain();
        var points = new double[,] { { 0.0, 0.0 }, { 1.1, 0.0 } };

        Assert.Throws<ArgumentOutOfRangeException>(() => tt.EvalBatch(points));
    }

    [Fact]
    public void TensorTrain_batch_eval_rejects_wrong_column_count()
    {
        var tt = BuildTensorTrain();
        var points = new double[,] { { 0.0 } };

        Assert.Throws<ArgumentException>(() => tt.EvalBatch(points));
    }

    [Fact]
    public void TensorTrain_batch_eval_rejects_null_matrix()
    {
        var tt = BuildTensorTrain();

        Assert.Throws<ArgumentNullException>(() => tt.EvalBatch(null!));
    }
}

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
