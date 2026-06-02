using QLNet;
using QDate = QLNet.Date;

namespace AmericanOptionDynamicChebyshev;

public enum VanillaOptionRight
{
    Call,
    Put,
}

public enum AmericanOptionReferenceEngine
{
    FiniteDifference,
    CoxRossRubinstein,
}

public sealed record AmericanOptionRequest(
    DateTime ValuationDate,
    DateTime MaturityDate,
    VanillaOptionRight Right,
    double Spot,
    double Strike,
    double RiskFreeRate,
    double DividendYield,
    double Volatility,
    AmericanOptionReferenceEngine Engine = AmericanOptionReferenceEngine.FiniteDifference,
    int TimeSteps = 300,
    int GridPoints = 300,
    double SpotBump = 0.5);

public sealed record AmericanOptionResult(
    double Price,
    double EuropeanPrice,
    double EarlyExercisePremium,
    double Delta,
    double Gamma,
    AmericanOptionReferenceEngine Engine);

public interface IAmericanOptionReferencePricer
{
    AmericanOptionResult Price(AmericanOptionRequest request);
}

public sealed class QlNetAmericanOptionReferencePricer : IAmericanOptionReferencePricer
{
    private static readonly object QlNetLock = new();

    public AmericanOptionResult Price(AmericanOptionRequest request)
    {
        Validate(request);

        lock (QlNetLock)
        {
            double americanPrice = PriceAmericanOnly(request);
            double europeanPrice = PriceEuropeanOnly(request);
            (double delta, double gamma) = ComputeSpotGreeks(request, americanPrice);

            return new AmericanOptionResult(
                Price: americanPrice,
                EuropeanPrice: europeanPrice,
                EarlyExercisePremium: americanPrice - europeanPrice,
                Delta: delta,
                Gamma: gamma,
                Engine: request.Engine);
        }
    }

    private static (double Delta, double Gamma) ComputeSpotGreeks(
        AmericanOptionRequest request,
        double basePrice)
    {
        double bump = request.SpotBump;
        AmericanOptionRequest downRequest = request with { Spot = request.Spot - bump };
        AmericanOptionRequest upRequest = request with { Spot = request.Spot + bump };

        double down = PriceAmericanOnly(downRequest);
        double up = PriceAmericanOnly(upRequest);

        double delta = (up - down) / (2.0 * bump);
        double gamma = (up - 2.0 * basePrice + down) / (bump * bump);
        return (delta, gamma);
    }

    private static double PriceAmericanOnly(AmericanOptionRequest request)
    {
        QDate valuationDate = ToQlDate(request.ValuationDate);
        QDate maturityDate = ToQlDate(request.MaturityDate);
        Settings.setEvaluationDate(valuationDate);

        GeneralizedBlackScholesProcess process = BuildProcess(request, valuationDate);
        StrikedTypePayoff payoff = new PlainVanillaPayoff(ToQlOptionType(request.Right), request.Strike);
        var option = new VanillaOption(
            payoff,
            new AmericanExercise(valuationDate, maturityDate, payoffAtExpiry: true));
        option.setPricingEngine(CreateAmericanEngine(request, process));
        return option.NPV();
    }

    private static double PriceEuropeanOnly(AmericanOptionRequest request)
    {
        QDate valuationDate = ToQlDate(request.ValuationDate);
        QDate maturityDate = ToQlDate(request.MaturityDate);
        Settings.setEvaluationDate(valuationDate);

        GeneralizedBlackScholesProcess process = BuildProcess(request, valuationDate);
        StrikedTypePayoff payoff = new PlainVanillaPayoff(ToQlOptionType(request.Right), request.Strike);
        var option = new VanillaOption(payoff, new EuropeanExercise(maturityDate));
        option.setPricingEngine(new AnalyticEuropeanEngine(process));
        return option.NPV();
    }

    private static IPricingEngine CreateAmericanEngine(
        AmericanOptionRequest request,
        GeneralizedBlackScholesProcess process)
    {
        return request.Engine switch
        {
            AmericanOptionReferenceEngine.FiniteDifference =>
                new MakeFdBlackScholesVanillaEngine(process)
                    .withTGrid(request.TimeSteps)
                    .withXGrid(request.GridPoints)
                    .withDampingSteps(0)
                    .getAsPricingEngine(),

            AmericanOptionReferenceEngine.CoxRossRubinstein =>
                new BinomialVanillaEngine<CoxRossRubinstein>(process, request.TimeSteps),

            _ => throw new ArgumentOutOfRangeException(nameof(request), "Unsupported reference engine."),
        };
    }

    private static GeneralizedBlackScholesProcess BuildProcess(
        AmericanOptionRequest request,
        QDate valuationDate)
    {
        var calendar = new NullCalendar();
        var dayCounter = new Actual365Fixed();

        var spot = new Handle<Quote>(new SimpleQuote(request.Spot));
        var dividend = new Handle<YieldTermStructure>(
            new FlatForward(valuationDate, request.DividendYield, dayCounter));
        var riskFree = new Handle<YieldTermStructure>(
            new FlatForward(valuationDate, request.RiskFreeRate, dayCounter));
        var volatility = new Handle<BlackVolTermStructure>(
            new BlackConstantVol(valuationDate, calendar, request.Volatility, dayCounter));

        return new BlackScholesMertonProcess(spot, dividend, riskFree, volatility);
    }

    private static Option.Type ToQlOptionType(VanillaOptionRight right)
    {
        return right switch
        {
            VanillaOptionRight.Call => Option.Type.Call,
            VanillaOptionRight.Put => Option.Type.Put,
            _ => throw new ArgumentOutOfRangeException(nameof(right), right, "Unsupported option right."),
        };
    }

    private static void Validate(AmericanOptionRequest request)
    {
        if (request.MaturityDate <= request.ValuationDate)
        {
            throw new ArgumentException("Maturity date must be after valuation date.", nameof(request));
        }

        if (!double.IsFinite(request.Spot) || request.Spot <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Spot must be finite and positive.");
        }

        if (!double.IsFinite(request.Strike) || request.Strike <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Strike must be finite and positive.");
        }

        if (!double.IsFinite(request.RiskFreeRate))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Risk-free rate must be finite.");
        }

        if (!double.IsFinite(request.DividendYield))
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Dividend yield must be finite.");
        }

        if (!double.IsFinite(request.Volatility) || request.Volatility <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Volatility must be finite and positive.");
        }

        if (request.TimeSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Time steps must be positive.");
        }

        if (request.GridPoints <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Grid points must be positive.");
        }

        if (!double.IsFinite(request.SpotBump) || request.SpotBump <= 0.0 || request.SpotBump >= request.Spot)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Spot bump must be finite, positive, and smaller than spot.");
        }
    }

    private static QDate ToQlDate(DateTime date)
        => new(date.Day, date.Month, date.Year);
}

public static class AmericanOptionScenarios
{
    public static AmericanOptionRequest StandardPut(
        double spot = 100.0,
        double strike = 100.0,
        double riskFreeRate = 0.05,
        double dividendYield = 0.0,
        double volatility = 0.20)
    {
        DateTime valuationDate = new(2026, 5, 15);
        return new AmericanOptionRequest(
            ValuationDate: valuationDate,
            MaturityDate: valuationDate.AddYears(1),
            Right: VanillaOptionRight.Put,
            Spot: spot,
            Strike: strike,
            RiskFreeRate: riskFreeRate,
            DividendYield: dividendYield,
            Volatility: volatility);
    }

    public static AmericanOptionRequest StandardCall(
        double spot = 100.0,
        double strike = 100.0,
        double riskFreeRate = 0.05,
        double dividendYield = 0.02,
        double volatility = 0.20)
    {
        DateTime valuationDate = new(2026, 5, 15);
        return new AmericanOptionRequest(
            ValuationDate: valuationDate,
            MaturityDate: valuationDate.AddYears(1),
            Right: VanillaOptionRight.Call,
            Spot: spot,
            Strike: strike,
            RiskFreeRate: riskFreeRate,
            DividendYield: dividendYield,
            Volatility: volatility);
    }
}
