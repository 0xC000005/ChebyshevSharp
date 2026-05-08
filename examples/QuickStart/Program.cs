using ChebyshevSharp;

Console.WriteLine("QuickStart: dense 2D Chebyshev approximation");
Console.WriteLine();

static double Surface(double[] x, object? _)
    => Math.Sin(x[0]) * Math.Cos(x[1]) + 0.1 * x[0] * x[1];

static double SurfaceDx(double[] x)
    => Math.Cos(x[0]) * Math.Cos(x[1]) + 0.1 * x[1];

static double SurfaceDy(double[] x)
    => -Math.Sin(x[0]) * Math.Sin(x[1]) + 0.1 * x[0];

double[][] domain =
[
    [-1.0, 1.0],
    [-1.0, 1.0]
];
int[] nNodes = [13, 13];

var approximation = new ChebyshevApproximation(
    function: Surface,
    numDimensions: 2,
    domain: domain,
    nNodes: nNodes);

approximation.Build(verbose: false);

double[] point = [0.35, -0.25];
double value = approximation.VectorizedEval(point, [0, 0]);
double dfdx = approximation.VectorizedEval(point, [1, 0]);
double dfdy = approximation.VectorizedEval(point, [0, 1]);
double errorEstimate = approximation.ErrorEstimate();

double exactValue = Surface(point, null);
double exactDfdx = SurfaceDx(point);
double exactDfdy = SurfaceDy(point);

Console.WriteLine($"f({point[0]:F2}, {point[1]:F2}) = {value:F8}");
Console.WriteLine($"  exact value     = {exactValue:F8}");
Console.WriteLine($"  absolute error  = {Math.Abs(value - exactValue):E2}");
Console.WriteLine($"df/dx             = {dfdx:F8}");
Console.WriteLine($"  exact df/dx     = {exactDfdx:F8}");
Console.WriteLine($"  absolute error  = {Math.Abs(dfdx - exactDfdx):E2}");
Console.WriteLine($"df/dy             = {dfdy:F8}");
Console.WriteLine($"  exact df/dy     = {exactDfdy:F8}");
Console.WriteLine($"  absolute error  = {Math.Abs(dfdy - exactDfdy):E2}");
Console.WriteLine($"error diagnostic  = {errorEstimate:E2}");
