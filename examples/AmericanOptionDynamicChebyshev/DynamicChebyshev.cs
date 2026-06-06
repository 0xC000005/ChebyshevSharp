using System.Diagnostics;
using ChebyshevSharp;

namespace AmericanOptionDynamicChebyshev;

public sealed record DynamicChebyshevSettings(
    int ExerciseSteps = 80,
    int SpotNodeCount = 81,
    double SpotLower = 5.0,
    double SpotUpper = 250.0,
    int QuadratureOrder = 8,
    bool ClosedFormTerminalStep = false,
    // Experimental, and a DOCUMENTED NEGATIVE RESULT: splitting the smooth continuation at the
    // exercise boundary does not improve Greeks. The kink lives in the price max(payoff, C), which
    // is applied exactly outside the interpolant; the continuation C is smooth at the boundary, so a
    // knot there resolves no singularity and only relocates the Gamma query onto the ill-conditioned
    // Chebyshev piece edge (it yields a non-physical negative Gamma one tick above the knot). Kept,
    // default-off, for reproducibility; superseded by the planned front-fixing variant. See the
    // "Why the boundary Gamma is weak" section of the American-option case study.
    bool BoundarySplit = false);

public sealed record DynamicChebyshevResult(
    double Price,
    double Delta,
    double Gamma,
    int BuildEvaluations,
    double BuildTimeSeconds,
    int ExerciseSteps,
    int SpotNodeCount,
    int QuadratureOrder);

public sealed record DynamicChebyshevEvaluation(
    double Price,
    double Delta,
    double Gamma);

public sealed class DynamicChebyshevAmericanOptionModel
{
    private readonly AmericanOptionRequest _request;
    private readonly DynamicChebyshevSettings _settings;
    private readonly Func<double, int, double> _continuation;

    internal DynamicChebyshevAmericanOptionModel(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings,
        Func<double, int, double> continuation,
        int buildEvaluations,
        double buildTimeSeconds)
    {
        _request = request;
        _settings = settings;
        _continuation = continuation;
        BuildEvaluations = buildEvaluations;
        BuildTimeSeconds = buildTimeSeconds;
    }

    public int BuildEvaluations { get; }

    public double BuildTimeSeconds { get; }

    public double Payoff(double spot)
    {
        ValidateSpot(spot);
        return DynamicChebyshevAmericanOptionPricer.Payoff(_request, spot);
    }

    public double Continuation(double spot)
    {
        ValidateSpot(spot);
        return _continuation(spot, 0);
    }

    public DynamicChebyshevEvaluation Evaluate(double spot)
    {
        ValidateSpot(spot);

        double payoff = Payoff(spot);
        double continuationAtSpot = Continuation(spot);
        bool continueAtSpot = continuationAtSpot >= payoff;

        double price = Math.Max(payoff, continuationAtSpot);
        double delta = continueAtSpot
            ? _continuation(spot, 1)
            : DynamicChebyshevAmericanOptionPricer.PayoffDelta(_request, spot);
        double gamma = continueAtSpot
            ? _continuation(spot, 2)
            : 0.0;

        return new DynamicChebyshevEvaluation(price, delta, gamma);
    }

    private void ValidateSpot(double spot)
    {
        if (spot < _settings.SpotLower || spot > _settings.SpotUpper)
        {
            throw new ArgumentOutOfRangeException(nameof(spot), "Spot must be inside the Dynamic Chebyshev spot domain.");
        }
    }
}

public sealed class DynamicChebyshevAmericanOptionPricer
{
    private static readonly double[] HermiteNodes8 =
    [
        -2.930637420257244,
        -1.981656756695843,
        -1.157193712446780,
        -0.3811869902073221,
        0.3811869902073221,
        1.157193712446780,
        1.981656756695843,
        2.930637420257244,
    ];

    private static readonly double[] HermiteWeights8 =
    [
        0.0001996040722113676,
        0.01707798300741348,
        0.2078023258148919,
        0.6611470125582413,
        0.6611470125582413,
        0.2078023258148919,
        0.01707798300741348,
        0.0001996040722113676,
    ];

    public DynamicChebyshevResult Price(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings)
    {
        DynamicChebyshevAmericanOptionModel model = Build(request, settings);
        DynamicChebyshevEvaluation evaluation = model.Evaluate(request.Spot);

        return new DynamicChebyshevResult(
            Price: evaluation.Price,
            Delta: evaluation.Delta,
            Gamma: evaluation.Gamma,
            BuildEvaluations: model.BuildEvaluations,
            BuildTimeSeconds: model.BuildTimeSeconds,
            ExerciseSteps: settings.ExerciseSteps,
            SpotNodeCount: settings.SpotNodeCount,
            QuadratureOrder: settings.QuadratureOrder);
    }

    public DynamicChebyshevAmericanOptionModel Build(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings)
    {
        Validate(request, settings);

        double maturityYears = (request.MaturityDate.Date - request.ValuationDate.Date).TotalDays / 365.0;
        double dt = maturityYears / settings.ExerciseSteps;
        double discount = Math.Exp(-request.RiskFreeRate * dt);
        double drift = (request.RiskFreeRate - request.DividendYield - 0.5 * request.Volatility * request.Volatility) * dt;
        double diffusion = request.Volatility * Math.Sqrt(dt);

        var stopwatch = Stopwatch.StartNew();
        int buildEvaluations = 0;
        Func<double, double> nextValue = spot => Payoff(request, spot);
        ChebyshevApproximation? firstContinuation = null;
        Func<double, double>? firstContinuationFunction = null;

        for (int step = settings.ExerciseSteps - 1; step >= 0; step--)
        {
            Func<double, double> valueAtNextStep = nextValue;
            bool closedFormTerminal = settings.ClosedFormTerminalStep
                && step == settings.ExerciseSteps - 1
                && request.Right == VanillaOptionRight.Put;
            Func<double, double> continuationFunction = closedFormTerminal
                ? spot => EuropeanPutPrice(
                    spot, request.Strike, dt, request.RiskFreeRate, request.Volatility, request.DividendYield)
                : spot => ContinuationValue(spot, drift, diffusion, discount, valueAtNextStep);
            ChebyshevApproximation continuation = BuildContinuationApproximation(
                request,
                settings,
                continuationFunction,
                maxDerivativeOrder: 2);

            buildEvaluations += continuation.NEvaluations;
            nextValue = spot => Math.Max(
                Payoff(request, spot),
                EvaluateApproximation(continuation, spot, settings, derivativeOrder: 0));

            if (step == 0)
            {
                firstContinuation = continuation;
                firstContinuationFunction = continuationFunction;
            }
        }

        stopwatch.Stop();

        Debug.Assert(firstContinuation is not null);
        Debug.Assert(firstContinuationFunction is not null);
        ChebyshevApproximation firstApprox = firstContinuation!;
        Func<double, double> firstFunc = firstContinuationFunction!;

        Func<double, int, double> continuationCurve;
        if (settings.BoundarySplit && request.Right == VanillaOptionRight.Put)
        {
            double GlobalContinuation(double spot) =>
                EvaluateApproximation(firstApprox, spot, settings, derivativeOrder: 0);
            double PayoffAt(double spot) => Payoff(request, spot);
            double boundary = FindExerciseBoundary(GlobalContinuation, PayoffAt, settings.SpotLower, request.Strike);

            ChebyshevSpline spline = BuildContinuationSpline(settings, firstFunc, boundary);
            buildEvaluations += spline.NumPieces * settings.SpotNodeCount;
            continuationCurve = (spot, order) => EvaluateSpline(spline, spot, settings, order, boundary);
        }
        else
        {
            continuationCurve = (spot, order) => EvaluateApproximation(firstApprox, spot, settings, order);
        }

        return new DynamicChebyshevAmericanOptionModel(
            request,
            settings,
            continuationCurve,
            buildEvaluations,
            stopwatch.Elapsed.TotalSeconds);
    }

    private static ChebyshevApproximation BuildContinuationApproximation(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings,
        Func<double, double> continuation,
        int maxDerivativeOrder)
    {
        double Function(double[] point, object? _)
            => continuation(point[0]);

        var approximation = new ChebyshevApproximation(
            Function,
            numDimensions: 1,
            domain: [[settings.SpotLower, settings.SpotUpper]],
            nNodes: [settings.SpotNodeCount],
            maxDerivativeOrder: maxDerivativeOrder);
        approximation.Build(verbose: false);
        return approximation;
    }

    private static double ContinuationValue(
        double spot,
        double drift,
        double diffusion,
        double discount,
        Func<double, double> nextValue)
    {
        double sum = 0.0;
        for (int i = 0; i < HermiteNodes8.Length; i++)
        {
            double nextSpot = spot * Math.Exp(drift + Math.Sqrt(2.0) * diffusion * HermiteNodes8[i]);
            sum += HermiteWeights8[i] * nextValue(nextSpot);
        }

        return discount * sum / Math.Sqrt(Math.PI);
    }

    internal static double EvaluateApproximation(
        ChebyshevApproximation approximation,
        double spot,
        DynamicChebyshevSettings settings,
        int derivativeOrder)
    {
        double clamped = Math.Clamp(spot, settings.SpotLower, settings.SpotUpper);
        return approximation.VectorizedEval([clamped], [derivativeOrder]);
    }

    internal static double Payoff(AmericanOptionRequest request, double spot)
    {
        return request.Right switch
        {
            VanillaOptionRight.Call => Math.Max(spot - request.Strike, 0.0),
            VanillaOptionRight.Put => Math.Max(request.Strike - spot, 0.0),
            _ => throw new ArgumentOutOfRangeException(nameof(request), "Unsupported option right."),
        };
    }

    internal static double PayoffDelta(AmericanOptionRequest request, double spot)
    {
        return request.Right switch
        {
            VanillaOptionRight.Call => spot > request.Strike ? 1.0 : 0.0,
            VanillaOptionRight.Put => spot < request.Strike ? -1.0 : 0.0,
            _ => throw new ArgumentOutOfRangeException(nameof(request), "Unsupported option right."),
        };
    }

    /// <summary>
    /// Closed-form Black-Scholes European put value. Used to seed the Dynamic Chebyshev
    /// recursion's terminal step with an exact, smooth continuation instead of quadrature
    /// against the kinked payoff (avoids the strike-corner error at the worst-fitted step).
    /// </summary>
    internal static double EuropeanPutPrice(double s, double k, double t, double r, double sigma, double q)
    {
        if (t <= 0.0 || sigma <= 0.0)
        {
            return Math.Max(k - s, 0.0);
        }

        double sqrtT = Math.Sqrt(t);
        double d1 = (Math.Log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        return k * Math.Exp(-r * t) * NormalCdf(-d2) - s * Math.Exp(-q * t) * NormalCdf(-d1);
    }

    private static double NormalCdf(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    private static double Erf(double x)
    {
        // Abramowitz & Stegun 7.1.26 (|error| <= 1.5e-7) -- ample for a European value.
        double sign = x < 0.0 ? -1.0 : 1.0;
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * z);
        double poly = t * (0.254829592
            + t * (-0.284496736
            + t * (1.421413741
            + t * (-1.453152027
            + t * 1.061405429))));
        return sign * (1.0 - poly * Math.Exp(-z * z));
    }

    /// <summary>
    /// Locates the early-exercise boundary B (where continuation = payoff) by Brent root-finding,
    /// using the already-present Math.NET solver. For a put, the gap continuation - payoff changes
    /// sign from negative (exercise region, S &lt; B) to positive (continuation region, S &gt; B).
    /// </summary>
    internal static double FindExerciseBoundary(
        DynamicChebyshevAmericanOptionModel model, double lo, double hi)
        => FindExerciseBoundary(model.Continuation, model.Payoff, lo, hi);

    internal static double FindExerciseBoundary(
        Func<double, double> continuation, Func<double, double> payoff, double lo, double hi)
    {
        double Gap(double spot) => continuation(spot) - payoff(spot);
        return MathNet.Numerics.RootFinding.Brent.FindRoot(Gap, lo, hi, accuracy: 1e-8, maxIterations: 100);
    }

    private static ChebyshevSpline BuildContinuationSpline(
        DynamicChebyshevSettings settings, Func<double, double> continuation, double knot)
    {
        double Function(double[] point, object? _) => continuation(point[0]);

        var spline = new ChebyshevSpline(
            Function,
            numDimensions: 1,
            domain: [[settings.SpotLower, settings.SpotUpper]],
            nNodes: [settings.SpotNodeCount],
            knots: [[knot]],
            maxDerivativeOrder: 2);
        spline.Build(verbose: false);
        return spline;
    }

    private static double EvaluateSpline(
        ChebyshevSpline spline,
        double spot,
        DynamicChebyshevSettings settings,
        int derivativeOrder,
        double knot)
    {
        double clamped = Math.Clamp(spot, settings.SpotLower, settings.SpotUpper);

        // Derivatives are undefined exactly at the knot (adjacent pieces disagree there); nudge onto
        // the continuation side so Greeks remain queryable right at the boundary.
        if (derivativeOrder > 0 && Math.Abs(clamped - knot) < 1e-9)
        {
            clamped = knot + 1e-7;
        }

        return spline.Eval([clamped], [derivativeOrder]);
    }

    private static void Validate(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings)
    {
        if (request.MaturityDate <= request.ValuationDate)
        {
            throw new ArgumentException("Maturity date must be after valuation date.", nameof(request));
        }

        if (settings.ExerciseSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Exercise steps must be positive.");
        }

        if (settings.SpotNodeCount < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Spot node count must be at least 3.");
        }

        if (settings.SpotLower <= 0.0 || settings.SpotUpper <= settings.SpotLower)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Spot domain must satisfy 0 < lower < upper.");
        }

        if (request.Spot < settings.SpotLower || request.Spot > settings.SpotUpper)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Spot must be inside the Dynamic Chebyshev spot domain.");
        }

        if (settings.QuadratureOrder != 8)
        {
            throw new NotSupportedException("The example currently supports 8-point Gauss-Hermite quadrature.");
        }
    }
}
