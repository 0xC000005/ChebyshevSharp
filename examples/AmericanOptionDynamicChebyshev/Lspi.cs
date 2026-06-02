namespace AmericanOptionDynamicChebyshev;

public sealed record LspiSettings(
    int PathCount = 10_000,
    int ExerciseSteps = 50,
    int Seed = 12345,
    int MaxPolicyIterations = 12);

public sealed record LspiResult(
    double Price,
    double StandardError,
    int BoundaryDecisionCount,
    int PolicyIterations,
    int FeatureCount,
    int PathCount,
    int ExerciseSteps);

public sealed class LspiAmericanOptionPricer
{
    public LspiResult Price(
        AmericanOptionRequest request,
        LspiSettings settings)
    {
        Validate(request, settings);

        double maturityYears = (request.MaturityDate.Date - request.ValuationDate.Date).TotalDays / 365.0;
        double dt = maturityYears / settings.ExerciseSteps;
        double drift = (request.RiskFreeRate - request.DividendYield - 0.5 * request.Volatility * request.Volatility) * dt;
        double diffusion = request.Volatility * Math.Sqrt(dt);
        double discount = Math.Exp(-request.RiskFreeRate * dt);

        double[,] trainingPaths = SimulatePaths(request.Spot, drift, diffusion, settings.PathCount, settings.ExerciseSteps, settings.Seed);
        int featureCount = Basis(request, settings, request.Spot, step: 0).Length;
        double[] weights = new double[featureCount];
        int iterations = 0;

        for (; iterations < settings.MaxPolicyIterations; iterations++)
        {
            double[] nextWeights = PolicyEvaluation(request, settings, trainingPaths, weights, discount);
            double maxChange = nextWeights.Zip(weights, (a, b) => Math.Abs(a - b)).Max();
            weights = nextWeights;
            if (maxChange < 1e-8)
            {
                iterations++;
                break;
            }
        }

        double[,] evaluationPaths = SimulatePaths(
            request.Spot,
            drift,
            diffusion,
            settings.PathCount,
            settings.ExerciseSteps,
            settings.Seed + 7_919);
        double[] presentValues = EvaluatePolicy(request, settings, evaluationPaths, weights, discount);
        double price = presentValues.Average();
        double variance = presentValues.Sum(value => (value - price) * (value - price))
            / Math.Max(1, settings.PathCount - 1);

        return new LspiResult(
            Price: price,
            StandardError: Math.Sqrt(variance / settings.PathCount),
            BoundaryDecisionCount: CountBoundaryDecisions(request, settings, evaluationPaths, weights),
            PolicyIterations: iterations,
            FeatureCount: featureCount,
            PathCount: settings.PathCount,
            ExerciseSteps: settings.ExerciseSteps);
    }

    private static double[] PolicyEvaluation(
        AmericanOptionRequest request,
        LspiSettings settings,
        double[,] paths,
        double[] targetWeights,
        double discount)
    {
        int featureCount = targetWeights.Length;
        var matrix = new double[featureCount, featureCount];
        var rhs = new double[featureCount];

        for (int path = 0; path < settings.PathCount; path++)
        {
            for (int step = 0; step < settings.ExerciseSteps; step++)
            {
                double currentSpot = paths[path, step];
                double nextSpot = paths[path, step + 1];
                double[] phi = Basis(request, settings, currentSpot, step);
                double nextPayoff = Payoff(request, nextSpot);
                double nextContinuation = EvaluateBasis(targetWeights, request, settings, nextSpot, step + 1);
                bool nextPolicyContinues = step + 1 < settings.ExerciseSteps && nextContinuation >= nextPayoff;

                double[] continuationPhi = nextPolicyContinues
                    ? Basis(request, settings, nextSpot, step + 1)
                    : new double[phi.Length];
                double exerciseReward = nextPolicyContinues ? 0.0 : discount * nextPayoff;

                for (int row = 0; row < phi.Length; row++)
                {
                    rhs[row] += exerciseReward * phi[row];
                    for (int col = 0; col < phi.Length; col++)
                    {
                        matrix[row, col] += phi[row] * (phi[col] - discount * continuationPhi[col]);
                    }
                }
            }
        }

        AddRidge(matrix, 1e-10);
        return Solve(matrix, rhs);
    }

    private static double[] EvaluatePolicy(
        AmericanOptionRequest request,
        LspiSettings settings,
        double[,] paths,
        double[] weights,
        double discount)
    {
        int pathCount = paths.GetLength(0);
        int exerciseSteps = paths.GetLength(1) - 1;
        var presentValues = new double[pathCount];

        for (int path = 0; path < pathCount; path++)
        {
            bool exercised = false;
            for (int step = 0; step < exerciseSteps; step++)
            {
                double spot = paths[path, step];
                double payoff = Payoff(request, spot);
                double continuation = EvaluateBasis(weights, request, settings, spot, step);
                if (payoff > 0.0 && payoff > continuation)
                {
                    presentValues[path] = payoff * Math.Pow(discount, step);
                    exercised = true;
                    break;
                }
            }

            if (!exercised)
            {
                double payoff = Payoff(request, paths[path, exerciseSteps]);
                presentValues[path] = payoff * Math.Pow(discount, exerciseSteps);
            }
        }

        return presentValues;
    }

    private static int CountBoundaryDecisions(
        AmericanOptionRequest request,
        LspiSettings settings,
        double[,] paths,
        double[] weights)
    {
        int count = 0;
        int pathCount = paths.GetLength(0);
        int exerciseSteps = paths.GetLength(1) - 1;
        for (int path = 0; path < pathCount; path++)
        {
            for (int step = 0; step < exerciseSteps; step++)
            {
                double spot = paths[path, step];
                double payoff = Payoff(request, spot);
                double continuation = EvaluateBasis(weights, request, settings, spot, step);
                if (payoff > 0.0 && payoff > continuation)
                {
                    count++;
                    break;
                }
            }
        }

        return count;
    }

    private static double[,] SimulatePaths(
        double spot,
        double drift,
        double diffusion,
        int pathCount,
        int exerciseSteps,
        int seed)
    {
        var rng = new NormalRandom(seed);
        var paths = new double[pathCount, exerciseSteps + 1];

        for (int path = 0; path < pathCount; path++)
        {
            paths[path, 0] = spot;
            for (int step = 1; step <= exerciseSteps; step++)
            {
                paths[path, step] = paths[path, step - 1] * Math.Exp(drift + diffusion * rng.Next());
            }
        }

        return paths;
    }

    private static double Payoff(AmericanOptionRequest request, double spot)
    {
        return request.Right switch
        {
            VanillaOptionRight.Call => Math.Max(spot - request.Strike, 0.0),
            VanillaOptionRight.Put => Math.Max(request.Strike - spot, 0.0),
            _ => throw new ArgumentOutOfRangeException(nameof(request), "Unsupported option right."),
        };
    }

    private static double[] Basis(
        AmericanOptionRequest request,
        LspiSettings settings,
        double spot,
        int step)
    {
        double x = spot / request.Strike;
        double exp = Math.Exp(-0.5 * x);
        double remaining = Math.Max(settings.ExerciseSteps - step, 1);
        double time = step / (double)settings.ExerciseSteps;
        double normalizedRemaining = remaining / settings.ExerciseSteps;

        return
        [
            1.0,
            exp,
            exp * (1.0 - x),
            exp * (1.0 - 2.0 * x + 0.5 * x * x),
            Math.Sin(0.5 * Math.PI * normalizedRemaining),
            Math.Log(1.0 + remaining),
            time * time,
        ];
    }

    private static double EvaluateBasis(
        double[] coefficients,
        AmericanOptionRequest request,
        LspiSettings settings,
        double spot,
        int step)
    {
        double[] basis = Basis(request, settings, spot, step);
        double value = 0.0;
        for (int i = 0; i < coefficients.Length; i++)
        {
            value += coefficients[i] * basis[i];
        }

        return value;
    }

    private static void AddRidge(double[,] matrix, double ridge)
    {
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            matrix[i, i] += ridge;
        }
    }

    private static double[] Solve(double[,] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var a = new double[n, n + 1];
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                a[row, col] = matrix[row, col];
            }

            a[row, n] = rhs[row];
        }

        for (int pivot = 0; pivot < n; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < n; row++)
            {
                if (Math.Abs(a[row, pivot]) > Math.Abs(a[best, pivot]))
                {
                    best = row;
                }
            }

            if (Math.Abs(a[best, pivot]) < 1e-14)
            {
                return new double[n];
            }

            if (best != pivot)
            {
                for (int col = pivot; col <= n; col++)
                {
                    (a[pivot, col], a[best, col]) = (a[best, col], a[pivot, col]);
                }
            }

            double scale = a[pivot, pivot];
            for (int col = pivot; col <= n; col++)
            {
                a[pivot, col] /= scale;
            }

            for (int row = 0; row < n; row++)
            {
                if (row == pivot)
                {
                    continue;
                }

                double factor = a[row, pivot];
                for (int col = pivot; col <= n; col++)
                {
                    a[row, col] -= factor * a[pivot, col];
                }
            }
        }

        var solution = new double[n];
        for (int i = 0; i < n; i++)
        {
            solution[i] = a[i, n];
        }

        return solution;
    }

    private static void Validate(
        AmericanOptionRequest request,
        LspiSettings settings)
    {
        if (request.MaturityDate <= request.ValuationDate)
        {
            throw new ArgumentException("Maturity date must be after valuation date.", nameof(request));
        }

        if (settings.PathCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Path count must be positive.");
        }

        if (settings.ExerciseSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Exercise steps must be positive.");
        }

        if (settings.MaxPolicyIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Max policy iterations must be positive.");
        }
    }

    private sealed class NormalRandom(int seed)
    {
        private readonly Random _random = new(seed);
        private bool _hasSpare;
        private double _spare;

        public double Next()
        {
            if (_hasSpare)
            {
                _hasSpare = false;
                return _spare;
            }

            double u;
            double v;
            double s;
            do
            {
                u = 2.0 * _random.NextDouble() - 1.0;
                v = 2.0 * _random.NextDouble() - 1.0;
                s = u * u + v * v;
            }
            while (s <= 0.0 || s >= 1.0);

            double multiplier = Math.Sqrt(-2.0 * Math.Log(s) / s);
            _spare = v * multiplier;
            _hasSpare = true;
            return u * multiplier;
        }
    }
}
