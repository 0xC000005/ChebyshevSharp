using FixedRateBondSurrogate;

FixedRateBondRequest request = FixedRateBondScenarios.RegularTenYear();
IFixedRateBondReferencePricer pricer = new QlNetFixedRateBondReferencePricer();
FixedRateBondResult result = pricer.Price(request);

Console.WriteLine("Fixed-rate bond reference pricer");
Console.WriteLine();
Console.WriteLine($"Valuation date : {request.ValuationDate:yyyy-MM-dd}");
Console.WriteLine($"Effective date : {request.EffectiveDate:yyyy-MM-dd}");
Console.WriteLine($"Maturity date  : {request.MaturityDate:yyyy-MM-dd}");
Console.WriteLine($"Coupon         : {request.Coupon:P2}");
Console.WriteLine($"Notional       : {request.Notional:N2}");
Console.WriteLine();
Console.WriteLine($"Dirty price    : {result.DirtyPrice:F8}");
Console.WriteLine($"Clean price    : {result.CleanPrice:F8}");
Console.WriteLine($"Accrued amount : {result.AccruedAmount:F8}");
Console.WriteLine($"NPV            : {result.NetPresentValue:F8}");
Console.WriteLine($"Cashflows      : {result.Cashflows.Count}");
