using System.Diagnostics;
using ChebyshevSharp;

namespace AmericanOptionDynamicChebyshev;

public sealed record DynamicChebyshevSettings(
    int ExerciseSteps = 80,
    int SpotNodeCount = 81,
    double SpotLower = 5.0,
    double SpotUpper = 250.0,
    int QuadratureOrder = 8);

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
    private readonly ChebyshevApproximation _firstContinuation;

    internal DynamicChebyshevAmericanOptionModel(
        AmericanOptionRequest request,
        DynamicChebyshevSettings settings,
        ChebyshevApproximation firstContinuation,
        int buildEvaluations,
        double buildTimeSeconds)
    {
        _request = request;
        _settings = settings;
        _firstContinuation = firstContinuation;
        BuildEvaluations = buildEvaluations;
        BuildTimeSeconds = buildTimeSeconds;
    }

    public int BuildEvaluations { get; }

    public double BuildTimeSeconds { get; }

    public DynamicChebyshevEvaluation Evaluate(double spot)
    {
        if (spot < _settings.SpotLower || spot > _settings.SpotUpper)
        {
            throw new ArgumentOutOfRangeException(nameof(spot), "Spot must be inside the Dynamic Chebyshev spot domain.");
        }

        double payoff = DynamicChebyshevAmericanOptionPricer.Payoff(_request, spot);
        double continuationAtSpot = DynamicChebyshevAmericanOptionPricer.EvaluateApproximation(
            _firstContinuation,
            spot,
            _settings,
            derivativeOrder: 0);
        bool continueAtSpot = continuationAtSpot >= payoff;

        double price = Math.Max(payoff, continuationAtSpot);
        double delta = continueAtSpot
            ? DynamicChebyshevAmericanOptionPricer.EvaluateApproximation(
                _firstContinuation,
                spot,
                _settings,
                derivativeOrder: 1)
            : DynamicChebyshevAmericanOptionPricer.PayoffDelta(_request, spot);
        double gamma = continueAtSpot
            ? DynamicChebyshevAmericanOptionPricer.EvaluateApproximation(
                _firstContinuation,
                spot,
                _settings,
                derivativeOrder: 2)
            : 0.0;

        return new DynamicChebyshevEvaluation(price, delta, gamma);
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

        for (int step = settings.ExerciseSteps - 1; step >= 0; step--)
        {
            Func<double, double> valueAtNextStep = nextValue;
            ChebyshevApproximation continuation = BuildContinuationApproximation(
                request,
                settings,
                spot => ContinuationValue(spot, drift, diffusion, discount, valueAtNextStep),
                maxDerivativeOrder: 2);

            buildEvaluations += continuation.NEvaluations;
            nextValue = spot => Math.Max(
                Payoff(request, spot),
                EvaluateApproximation(continuation, spot, settings, derivativeOrder: 0));

            if (step == 0)
            {
                firstContinuation = continuation;
            }
        }

        stopwatch.Stop();

        Debug.Assert(firstContinuation is not null);

        return new DynamicChebyshevAmericanOptionModel(
            request,
            settings,
            firstContinuation,
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
