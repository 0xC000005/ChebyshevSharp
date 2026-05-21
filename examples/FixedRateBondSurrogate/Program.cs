using FixedRateBondSurrogate;

FixedRateBondExample.Run(Console.Out);

public static class FixedRateBondExample
{
    public static void Run(TextWriter output)
    {
        FixedRateBondRequest request = FixedRateBondScenarios.RegularTenYear();
        IFixedRateBondReferencePricer pricer = new QlNetFixedRateBondReferencePricer();
        FixedRateBondResult result = pricer.Price(request);

        output.WriteLine("Fixed-rate bond reference pricer");
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
