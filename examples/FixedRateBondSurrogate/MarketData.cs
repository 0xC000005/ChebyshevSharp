using System.Text.Json;
using System.Text.Json.Serialization;

namespace FixedRateBondSurrogate;

public sealed record YieldCurveFixture(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("fixture_id")] string FixtureId,
    [property: JsonPropertyName("source")] YieldCurveSourceMetadata Source,
    [property: JsonPropertyName("instrument_family")] string InstrumentFamily,
    [property: JsonPropertyName("rate_kind")] string RateKind,
    [property: JsonPropertyName("units")] string Units,
    [property: JsonPropertyName("compounding")] string Compounding,
    [property: JsonPropertyName("day_count")] string DayCount,
    [property: JsonPropertyName("interpolation")] string Interpolation,
    [property: JsonPropertyName("original_fields")] IReadOnlyList<string> OriginalFields,
    [property: JsonPropertyName("points")] IReadOnlyList<YieldCurvePoint> Points);

public sealed record YieldCurveSourceMetadata(
    [property: JsonPropertyName("institution")] string Institution,
    [property: JsonPropertyName("source_url")] string SourceUrl,
    [property: JsonPropertyName("source_page")] string SourcePage,
    [property: JsonPropertyName("download_date")] DateTime DownloadDate,
    [property: JsonPropertyName("curve_date")] DateTime CurveDate,
    [property: JsonPropertyName("source_note")] string SourceNote);

public sealed record YieldCurvePoint(
    [property: JsonPropertyName("maturity_years")] int MaturityYears,
    [property: JsonPropertyName("field")] string Field,
    [property: JsonPropertyName("zero_yield_percent")] double ZeroYieldPercent);

public static class FixedRateBondMarketData
{
    public const string DefaultFixtureFileName = "fed-nominal-yield-curve-2026-05-15.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = false,
    };

    public static YieldCurveFixture LoadDefaultCurveFixture()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Data", DefaultFixtureFileName);
        return LoadCurveFixture(path);
    }

    public static YieldCurveFixture LoadCurveFixture(string path)
    {
        using FileStream stream = File.OpenRead(path);
        YieldCurveFixture? fixture = JsonSerializer.Deserialize<YieldCurveFixture>(stream, JsonOptions);
        if (fixture is null)
        {
            throw new InvalidDataException($"Unable to deserialize yield-curve fixture '{path}'.");
        }

        Validate(fixture);
        return fixture;
    }

    public static IReadOnlyList<ZeroRatePillar> ToZeroRatePillars(
        YieldCurveFixture fixture,
        DateTime valuationDate)
    {
        Validate(fixture);

        var pillars = new List<ZeroRatePillar>(fixture.Points.Count + 1)
        {
            new(valuationDate.Date, PercentToDecimal(fixture.Points[0].ZeroYieldPercent)),
        };

        foreach (YieldCurvePoint point in fixture.Points)
        {
            pillars.Add(new ZeroRatePillar(
                valuationDate.Date.AddYears(point.MaturityYears),
                PercentToDecimal(point.ZeroYieldPercent)));
        }

        return pillars;
    }

    public static FixedRateBondRequest RegularTenYearFromFixture(
        YieldCurveFixture fixture,
        double coupon = 0.045,
        double notional = 100.0)
    {
        DateTime curveDate = fixture.Source.CurveDate.Date;

        return new FixedRateBondRequest(
            ValuationDate: curveDate,
            EffectiveDate: curveDate,
            MaturityDate: curveDate.AddYears(10),
            Coupon: coupon,
            Notional: notional,
            ZeroCurve: ToZeroRatePillars(fixture, curveDate));
    }

    public static void Validate(YieldCurveFixture fixture)
    {
        if (fixture.RateKind != "zero_coupon_yield")
        {
            throw new InvalidDataException("The fixture must contain zero-coupon yields.");
        }

        if (fixture.Units != "percent")
        {
            throw new InvalidDataException("The fixture must express rates in percent.");
        }

        if (fixture.Compounding != "continuous")
        {
            throw new InvalidDataException("The fixture must use continuous compounding.");
        }

        if (fixture.Points.Count == 0)
        {
            throw new InvalidDataException("The fixture must contain at least one curve point.");
        }

        int previousMaturity = 0;
        foreach (YieldCurvePoint point in fixture.Points)
        {
            if (point.MaturityYears <= previousMaturity)
            {
                throw new InvalidDataException("Curve maturities must be strictly increasing.");
            }

            if (!double.IsFinite(point.ZeroYieldPercent))
            {
                throw new InvalidDataException("Curve yields must be finite.");
            }

            string expectedField = $"SVENY{point.MaturityYears:00}";
            if (point.Field != expectedField)
            {
                throw new InvalidDataException($"Expected field {expectedField} for {point.MaturityYears}Y point.");
            }

            previousMaturity = point.MaturityYears;
        }
    }

    private static double PercentToDecimal(double percent)
        => percent / 100.0;
}
