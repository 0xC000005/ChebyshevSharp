using ChebyshevSharp;

Console.WriteLine("SliderPartitionValidation: 8D ChebyshevSlider partition check");
Console.WriteLine();

const int dimensions = 8;
const int nodesPerDimension = 13;

static double PairwiseModel(double[] x, object? _)
    => Math.Sin(x[0] + 0.5 * x[1])
       + 0.35 * x[2] * x[3]
       + Math.Exp(0.2 * x[4] - 0.15 * x[5])
       + 0.2 * Math.Cos(x[6] - x[7]);

static double PairwiseDx0(double[] x)
    => Math.Cos(x[0] + 0.5 * x[1]);

double[][] domain = Enumerable.Range(0, dimensions)
    .Select(_ => new[] { -1.0, 1.0 })
    .ToArray();
int[] nNodes = Enumerable.Repeat(nodesPerDimension, dimensions).ToArray();
double[] pivot = new double[dimensions];

int[][] groupedPartition =
[
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7]
];
int[][] singletonPartition = Enumerable.Range(0, dimensions)
    .Select(i => new[] { i })
    .ToArray();

long denseValues = nNodes.Aggregate(1L, (acc, n) => checked(acc * n));
double denseGiB = denseValues * sizeof(double) / (1024.0 * 1024.0 * 1024.0);

var grouped = new ChebyshevSlider(
    function: PairwiseModel,
    numDimensions: dimensions,
    domain: domain,
    nNodes: nNodes,
    partition: groupedPartition,
    pivotPoint: pivot);

var singletons = new ChebyshevSlider(
    function: PairwiseModel,
    numDimensions: dimensions,
    domain: domain,
    nNodes: nNodes,
    partition: singletonPartition,
    pivotPoint: pivot);

grouped.Build(verbose: false);
singletons.Build(verbose: false);

double maxGroupedAbs = 0.0;
double maxSingletonAbs = 0.0;
var rng = new Random(123);

for (int i = 0; i < 200; i++)
{
    double[] point = RandomPoint(rng, dimensions);
    double exact = PairwiseModel(point, null);
    double groupedValue = grouped.Eval(point, new int[dimensions]);
    double singletonValue = singletons.Eval(point, new int[dimensions]);

    maxGroupedAbs = Math.Max(maxGroupedAbs, Math.Abs(groupedValue - exact));
    maxSingletonAbs = Math.Max(maxSingletonAbs, Math.Abs(singletonValue - exact));
}

double[] checkPoint = [0.4, -0.6, 0.2, 0.7, -0.5, 0.25, -0.3, 0.8];
int[] dx0 = new int[dimensions];
dx0[0] = 1;

double groupedDx0 = grouped.Eval(checkPoint, dx0);
double singletonDx0 = singletons.Eval(checkPoint, dx0);
double exactDx0 = PairwiseDx0(checkPoint);

Console.WriteLine($"dense grid would contain {denseValues:N0} doubles (~{denseGiB:F1} GiB)");
Console.WriteLine($"grouped partition build calls: {grouped.TotalBuildEvals + 1:N0}");
Console.WriteLine($"singleton partition build calls: {singletons.TotalBuildEvals + 1:N0}");
Console.WriteLine();
Console.WriteLine($"grouped held-out max abs error:   {maxGroupedAbs:E2}");
Console.WriteLine($"singleton held-out max abs error: {maxSingletonAbs:E2}");
Console.WriteLine($"grouped error diagnostic:         {grouped.ErrorEstimate():E2}");
Console.WriteLine($"singleton error diagnostic:       {singletons.ErrorEstimate():E2}");
Console.WriteLine();
Console.WriteLine($"d/dx0 exact at check point:     {exactDx0:F8}");
Console.WriteLine($"d/dx0 grouped partition:        {groupedDx0:F8}");
Console.WriteLine($"d/dx0 singleton partition:      {singletonDx0:F8}");

static double[] RandomPoint(Random rng, int dimensions)
{
    double[] point = new double[dimensions];
    for (int d = 0; d < dimensions; d++)
        point[d] = -1.0 + 2.0 * rng.NextDouble();
    return point;
}
