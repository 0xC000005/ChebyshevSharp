using ChebyshevSharp;

static double Surface(double[] x, object? _)
    => Math.Sin(x[0]) * Math.Cos(x[1]) + 0.1 * x[0] * x[1];

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

Console.WriteLine($"f({point[0]:F2}, {point[1]:F2}) = {value:F8}");
Console.WriteLine($"df/dx = {dfdx:F8}");
Console.WriteLine($"df/dy = {dfdy:F8}");
Console.WriteLine($"estimated error = {errorEstimate:E2}");
