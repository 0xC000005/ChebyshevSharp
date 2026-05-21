using FixedRateBondSurrogate;

FixedRateBondExample.Run(Console.Out);

public static class FixedRateBondExample
{
    public static void Run(TextWriter output)
    {
        YieldCurveFixture fixture = FixedRateBondMarketData.LoadDefaultCurveFixture();
        FixedRateBondRequest request = FixedRateBondMarketData.RegularTenYearFromFixture(fixture);
        IFixedRateBondReferencePricer pricer = new QlNetFixedRateBondReferencePricer();
        FixedRateBondResult result = pricer.Price(request);

        output.WriteLine("Fixed-rate bond reference pricer");
        output.WriteLine();
        output.WriteLine($"Curve fixture : {fixture.FixtureId}");
        output.WriteLine($"Curve date    : {fixture.Source.CurveDate:yyyy-MM-dd}");
        output.WriteLine($"Curve source  : {fixture.Source.Institution}");
        output.WriteLine();
        output.WriteLine($"Valuation date : {request.ValuationDate:yyyy-MM-dd}");
        output.WriteLine($"Effective date : {request.EffectiveDate:yyyy-MM-dd}");
        output.WriteLine($"Maturity date  : {request.MaturityDate:yyyy-MM-dd}");
        output.WriteLine($"Coupon         : {request.Coupon:P2}");
        output.WriteLine($"Notional       : {request.Notional:N2}");
        output.WriteLine();
        output.WriteLine($"Dirty price    : {result.DirtyPrice:F8}");
        output.WriteLine($"Clean price    : {result.CleanPrice:F8}");
        output.WriteLine($"Accrued amount : {result.AccruedAmount:F8}");
        output.WriteLine($"NPV            : {result.NetPresentValue:F8}");
        output.WriteLine($"Cashflows      : {result.Cashflows.Count}");
    }
}
