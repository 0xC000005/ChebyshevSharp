using ChebyshevSharp;

const int dimensions = 7;
const int nodesPerDimension = 9;

static double CoupledModel(double[] x)
    => Math.Sin(x[0] + 0.25 * x[1])
       + 0.2 * x[2] * x[3]
       + Math.Cos(x[4] - x[5])
       + 0.1 * x[6] * x[0];

double[][] domain = Enumerable.Range(0, dimensions)
    .Select(_ => new[] { -1.0, 1.0 })
    .ToArray();

int[] nNodes = Enumerable.Repeat(nodesPerDimension, dimensions).ToArray();
long denseValues = nNodes.Aggregate(1L, (acc, n) => checked(acc * n));

var tt = new ChebyshevTT(
    function: CoupledModel,
    numDimensions: dimensions,
    domain: domain,
    nNodes: nNodes,
    maxRank: 8,
    tolerance: 1e-6,
    maxSweeps: 5);

tt.Build(verbose: false, seed: 123, method: "cross");

double[] point = [0.25, -0.4, 0.1, 0.7, -0.2, 0.3, 0.5];
double value = tt.Eval(point);

Console.WriteLine($"dense grid would contain {denseValues:N0} values");
Console.WriteLine($"TT evaluations used: {tt.TotalBuildEvals:N0}");
Console.WriteLine($"TT ranks: [{string.Join(", ", tt.TtRanks)}]");
Console.WriteLine($"compression ratio: {tt.CompressionRatio:F1}x");
Console.WriteLine($"f(point) = {value:F8}");
