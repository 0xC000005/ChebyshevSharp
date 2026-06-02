namespace AmericanOptionDynamicChebyshev;

public sealed record RegressionMonteCarloSettings(
    int PathCount = 10_000,
    int ExerciseSteps = 50,
    int Seed = 12345);

public sealed record RegressionMonteCarloResult(
    double Price,
    double StandardError,
    double ExercisedPathFraction,
    int PathCount,
    int ExerciseSteps);

public sealed class LongstaffSchwartzAmericanOptionPricer
{
    public RegressionMonteCarloResult Price(
        AmericanOptionRequest request,
        RegressionMonteCarloSettings settings)
    {
        Validate(request, settings);

        double maturityYears = (request.MaturityDate.Date - request.ValuationDate.Date).TotalDays / 365.0;
        double dt = maturityYears / settings.ExerciseSteps;
        double drift = (request.RiskFreeRate - request.DividendYield - 0.5 * request.Volatility * request.Volatility) * dt;
        double diffusion = request.Volatility * Math.Sqrt(dt);
        double discountRate = request.RiskFreeRate * dt;

        double[,] paths = SimulatePaths(request.Spot, drift, diffusion, settings);
        var cashflows = new double[settings.PathCount];
        var cashflowStep = new int[settings.PathCount];

        for (int path = 0; path < settings.PathCount; path++)
        {
            cashflows[path] = Payoff(request, paths[path, settings.ExerciseSteps]);
            cashflowStep[path] = settings.ExerciseSteps;
        }

        for (int step = settings.ExerciseSteps - 1; step >= 1; step--)
        {
            double[] coefficients = RegressContinuation(request, paths, cashflows, cashflowStep, step, discountRate);

            for (int path = 0; path < settings.PathCount; path++)
            {
                double immediate = Payoff(request, paths[path, step]);
                if (immediate <= 0.0)
                {
                    continue;
                }

                double continuation = EvaluateBasis(coefficients, paths[path, step] / request.Strike);
                if (immediate > continuation)
                {
                    cashflows[path] = immediate;
                    cashflowStep[path] = step;
                }
            }
        }

        var presentValues = new double[settings.PathCount];
        int earlyExerciseCount = 0;
        for (int path = 0; path < settings.PathCount; path++)
        {
            presentValues[path] = cashflows[path] * Math.Exp(-discountRate * cashflowStep[path]);
            if (cashflowStep[path] < settings.ExerciseSteps)
            {
                earlyExerciseCount++;
            }
        }

        double mean = presentValues.Average();
        double variance = presentValues.Sum(value => (value - mean) * (value - mean))
            / Math.Max(1, settings.PathCount - 1);

        return new RegressionMonteCarloResult(
            Price: mean,
            StandardError: Math.Sqrt(variance / settings.PathCount),
            ExercisedPathFraction: earlyExerciseCount / (double)settings.PathCount,
            PathCount: settings.PathCount,
            ExerciseSteps: settings.ExerciseSteps);
    }

    private static double[,] SimulatePaths(
        double spot,
        double drift,
        double diffusion,
        RegressionMonteCarloSettings settings)
    {
        var rng = new NormalRandom(settings.Seed);
        var paths = new double[settings.PathCount, settings.ExerciseSteps + 1];

        for (int path = 0; path < settings.PathCount; path++)
        {
            paths[path, 0] = spot;
            for (int step = 1; step <= settings.ExerciseSteps; step++)
            {
                paths[path, step] = paths[path, step - 1] * Math.Exp(drift + diffusion * rng.Next());
            }
        }

        return paths;
    }

    private static double[] RegressContinuation(
        AmericanOptionRequest request,
        double[,] paths,
        double[] cashflows,
        int[] cashflowStep,
        int step,
        double discountRate)
    {
        var xtx = new double[3, 3];
        var xty = new double[3];
        int rows = 0;

        for (int path = 0; path < cashflows.Length; path++)
        {
            double spot = paths[path, step];
            if (Payoff(request, spot) <= 0.0)
            {
                continue;
            }

            double x = spot / request.Strike;
            double[] basis = [1.0, x, x * x];
            double y = cashflows[path] * Math.Exp(-discountRate * (cashflowStep[path] - step));

            for (int row = 0; row < basis.Length; row++)
            {
                xty[row] += basis[row] * y;
                for (int col = 0; col < basis.Length; col++)
                {
                    xtx[row, col] += basis[row] * basis[col];
                }
            }

            rows++;
        }

        return rows >= 3 ? Solve3x3(xtx, xty) : [0.0, 0.0, 0.0];
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

    private static double EvaluateBasis(double[] coefficients, double x)
        => coefficients[0] + coefficients[1] * x + coefficients[2] * x * x;

    private static double[] Solve3x3(double[,] matrix, double[] rhs)
    {
        var a = new double[3, 4];
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                a[row, col] = matrix[row, col];
            }

            a[row, 3] = rhs[row];
        }

        for (int pivot = 0; pivot < 3; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < 3; row++)
            {
                if (Math.Abs(a[row, pivot]) > Math.Abs(a[best, pivot]))
                {
                    best = row;
                }
            }

            if (Math.Abs(a[best, pivot]) < 1e-14)
            {
                return [0.0, 0.0, 0.0];
            }

            if (best != pivot)
            {
                for (int col = pivot; col < 4; col++)
                {
                    (a[pivot, col], a[best, col]) = (a[best, col], a[pivot, col]);
                }
            }

            double scale = a[pivot, pivot];
            for (int col = pivot; col < 4; col++)
            {
                a[pivot, col] /= scale;
            }

            for (int row = 0; row < 3; row++)
            {
                if (row == pivot)
                {
                    continue;
                }

                double factor = a[row, pivot];
                for (int col = pivot; col < 4; col++)
                {
                    a[row, col] -= factor * a[pivot, col];
                }
            }
        }

        return [a[0, 3], a[1, 3], a[2, 3]];
    }

    private static void Validate(
        AmericanOptionRequest request,
        RegressionMonteCarloSettings settings)
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
