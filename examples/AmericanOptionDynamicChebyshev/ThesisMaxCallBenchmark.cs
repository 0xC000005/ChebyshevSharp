namespace AmericanOptionDynamicChebyshev;

public static class ThesisMaxCallBenchmark
{
    public static IReadOnlyList<ThesisMaxCallRow> Run()
    {
        var settings = new ThesisMaxCallSettings(
            AssetCount: 2,
            Strike: 100.0,
            RiskFreeRate: 0.05,
            DividendYield: 0.10,
            Volatility: 0.20,
            MaturityYears: 3.0,
            ExerciseSteps: 9,
            PathCount: 200_000,
            Seed: 24681357);

        return
        [
            RunCase(settings, initialSpot: 90.0, thesisLongstaffSchwartz: 8.063, thesisStandardError: 0.010),
            RunCase(settings, initialSpot: 100.0, thesisLongstaffSchwartz: 13.861, thesisStandardError: 0.012),
            RunCase(settings, initialSpot: 110.0, thesisLongstaffSchwartz: 21.333, thesisStandardError: 0.014),
        ];
    }

    private static ThesisMaxCallRow RunCase(
        ThesisMaxCallSettings settings,
        double initialSpot,
        double thesisLongstaffSchwartz,
        double thesisStandardError)
    {
        double price = Price(settings, initialSpot);
        return new ThesisMaxCallRow(
            AssetCount: settings.AssetCount,
            InitialSpot: initialSpot,
            ThesisLongstaffSchwartz: thesisLongstaffSchwartz,
            ThesisStandardError: thesisStandardError,
            ChebyshevSharpLongstaffSchwartz: price,
            AbsoluteDifference: Math.Abs(price - thesisLongstaffSchwartz),
            RelativeDifference: Math.Abs(price - thesisLongstaffSchwartz) / thesisLongstaffSchwartz);
    }

    private static double Price(ThesisMaxCallSettings settings, double initialSpot)
    {
        double dt = settings.MaturityYears / settings.ExerciseSteps;
        double drift = (settings.RiskFreeRate - settings.DividendYield - 0.5 * settings.Volatility * settings.Volatility) * dt;
        double diffusion = settings.Volatility * Math.Sqrt(dt);
        double discountRate = settings.RiskFreeRate * dt;
        double discount = Math.Exp(-discountRate);

        double[,,] paths = SimulatePaths(settings, initialSpot, drift, diffusion);
        var cashflows = new double[settings.PathCount];
        var cashflowStep = new int[settings.PathCount];

        for (int path = 0; path < settings.PathCount; path++)
        {
            cashflows[path] = Payoff(settings, paths, path, settings.ExerciseSteps);
            cashflowStep[path] = settings.ExerciseSteps;
        }

        for (int step = settings.ExerciseSteps - 1; step >= 1; step--)
        {
            double[] coefficients = RegressContinuation(settings, paths, cashflows, cashflowStep, step, discountRate);
            for (int path = 0; path < settings.PathCount; path++)
            {
                double immediate = Payoff(settings, paths, path, step);
                if (immediate <= 0.0)
                {
                    continue;
                }

                double continuation = Dot(coefficients, Basis(settings, paths, path, step));
                if (immediate > continuation)
                {
                    cashflows[path] = immediate;
                    cashflowStep[path] = step;
                }
            }
        }

        double sum = 0.0;
        for (int path = 0; path < settings.PathCount; path++)
        {
            sum += cashflows[path] * Math.Pow(discount, cashflowStep[path]);
        }

        return sum / settings.PathCount;
    }

    private static double[,,] SimulatePaths(
        ThesisMaxCallSettings settings,
        double initialSpot,
        double drift,
        double diffusion)
    {
        var rng = new NormalRandom(settings.Seed);
        var paths = new double[settings.PathCount, settings.ExerciseSteps + 1, settings.AssetCount];

        for (int path = 0; path < settings.PathCount; path++)
        {
            for (int asset = 0; asset < settings.AssetCount; asset++)
            {
                paths[path, 0, asset] = initialSpot;
            }

            for (int step = 1; step <= settings.ExerciseSteps; step++)
            {
                for (int asset = 0; asset < settings.AssetCount; asset++)
                {
                    paths[path, step, asset] = paths[path, step - 1, asset] * Math.Exp(drift + diffusion * rng.Next());
                }
            }
        }

        return paths;
    }

    private static double[] RegressContinuation(
        ThesisMaxCallSettings settings,
        double[,,] paths,
        double[] cashflows,
        int[] cashflowStep,
        int step,
        double discountRate)
    {
        int basisCount = Basis(settings, paths, path: 0, step).Length;
        var xtx = new double[basisCount, basisCount];
        var xty = new double[basisCount];

        for (int path = 0; path < settings.PathCount; path++)
        {
            double immediate = Payoff(settings, paths, path, step);
            if (immediate <= 0.0)
            {
                continue;
            }

            double[] basis = Basis(settings, paths, path, step);
            double y = cashflows[path] * Math.Exp(-discountRate * (cashflowStep[path] - step));

            for (int row = 0; row < basisCount; row++)
            {
                xty[row] += basis[row] * y;
                for (int col = 0; col < basisCount; col++)
                {
                    xtx[row, col] += basis[row] * basis[col];
                }
            }
        }

        return Solve(xtx, xty);
    }

    private static double Payoff(
        ThesisMaxCallSettings settings,
        double[,,] paths,
        int path,
        int step)
    {
        double maxAsset = double.NegativeInfinity;
        for (int asset = 0; asset < settings.AssetCount; asset++)
        {
            maxAsset = Math.Max(maxAsset, paths[path, step, asset]);
        }

        return Math.Max(maxAsset - settings.Strike, 0.0);
    }

    private static double[] Basis(
        ThesisMaxCallSettings settings,
        double[,,] paths,
        int path,
        int step)
    {
        double first = double.NegativeInfinity;
        double second = double.NegativeInfinity;
        for (int asset = 0; asset < settings.AssetCount; asset++)
        {
            double value = paths[path, step, asset] / settings.Strike;
            if (value >= first)
            {
                second = first;
                first = value;
            }
            else if (value > second)
            {
                second = value;
            }
        }

        double first2 = first * first;
        double second2 = second * second;
        return
        [
            1.0,
            first,
            second,
            first2,
            second2,
            first * second,
            first2 * first,
            second2 * second,
            first2 * second,
            first * second2,
        ];
    }

    private static double Dot(double[] left, double[] right)
    {
        double sum = 0.0;
        for (int i = 0; i < left.Length; i++)
        {
            sum += left[i] * right[i];
        }

        return sum;
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

public sealed record ThesisMaxCallRow(
    int AssetCount,
    double InitialSpot,
    double ThesisLongstaffSchwartz,
    double ThesisStandardError,
    double ChebyshevSharpLongstaffSchwartz,
    double AbsoluteDifference,
    double RelativeDifference);

public sealed record ThesisMaxCallSettings(
    int AssetCount,
    double Strike,
    double RiskFreeRate,
    double DividendYield,
    double Volatility,
    double MaturityYears,
    int ExerciseSteps,
    int PathCount,
    int Seed);
