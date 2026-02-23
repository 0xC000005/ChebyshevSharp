namespace ChebyshevSharp.Tests.Helpers;

/// <summary>
/// Analytical Black-Scholes functions for test validation.
/// Ported from ref/PyChebyshev/tests/conftest.py.
/// </summary>
public static class BlackScholes
{
    private static readonly double SqrtTwoPi = Math.Sqrt(2.0 * Math.PI);

    public static double NormalCdf(double x)
    {
        // Abramowitz and Stegun approximation 26.2.17
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        double ax = Math.Abs(x) / Math.Sqrt(2.0);
        double t = 1.0 / (1.0 + p * ax);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-ax * ax);
        return 0.5 * (1.0 + sign * y);
    }

    public static double NormalPdf(double x)
    {
        return Math.Exp(-0.5 * x * x) / SqrtTwoPi;
    }

    public static double BsCallPrice(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        double d2 = d1 - sigma * Math.Sqrt(T);
        return S * Math.Exp(-q * T) * NormalCdf(d1) - K * Math.Exp(-r * T) * NormalCdf(d2);
    }

    public static double BsCallDelta(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        return Math.Exp(-q * T) * NormalCdf(d1);
    }

    public static double BsCallGamma(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        return Math.Exp(-q * T) * NormalPdf(d1) / (S * sigma * Math.Sqrt(T));
    }

    public static double BsCallVega(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        return S * Math.Exp(-q * T) * NormalPdf(d1) * Math.Sqrt(T);
    }

    public static double BsCallRho(double S, double K, double T, double r, double sigma, double q = 0.0)
    {
        double d1 = (Math.Log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
        double d2 = d1 - sigma * Math.Sqrt(T);
        return K * T * Math.Exp(-r * T) * NormalCdf(d2);
    }
}
