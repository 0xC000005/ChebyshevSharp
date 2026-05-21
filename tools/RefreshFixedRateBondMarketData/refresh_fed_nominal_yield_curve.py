#!/usr/bin/env python3
"""Refresh the fixed-rate bond example's Federal Reserve zero-yield fixture."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


FED_NOMINAL_YIELD_CURVE_CSV = (
    "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "FixedRateBondSurrogate"
    / "Data"
    / "fed-nominal-yield-curve-2026-05-15.json"
)
DENSE_SEMIANNUAL_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "FixedRateBondSurrogate"
    / "Data"
    / "fed-nominal-yield-curve-semiannual-2026-05-15.json"
)
SELECTED_MATURITIES = (1, 2, 3, 5, 7, 10, 20, 30)
SVENSSON_PARAMETER_FIELDS = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Federal Reserve nominal yield curve CSV and write a "
            "small normalized zero-coupon-yield fixture for the finance example."
        )
    )
    parser.add_argument(
        "--curve-date",
        default="2026-05-15",
        help="Curve date to pin in YYYY-MM-DD form. Use 'latest' for latest complete row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to the selected fixture path "
            f"({DEFAULT_OUTPUT} or {DENSE_SEMIANNUAL_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--density",
        choices=("selected-annual", "semiannual-svensson"),
        default="selected-annual",
        help=(
            "selected-annual writes the published annual SVENY fields used by the compact "
            "surrogate reproduction. semiannual-svensson samples the fitted Fed curve every "
            "six months from the published Svensson-style parameters."
        ),
    )
    parser.add_argument(
        "--download-date",
        default=dt.date.today().isoformat(),
        help="Download date to record in YYYY-MM-DD form. Defaults to today's date.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional local Federal Reserve CSV path. When omitted, the script downloads the CSV.",
    )
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        output = DENSE_SEMIANNUAL_OUTPUT if args.density == "semiannual-svensson" else DEFAULT_OUTPUT

    rows = (
        list(read_rows(args.input_csv.read_text(encoding="utf-8-sig")))
        if args.input_csv is not None
        else list(fetch_rows(FED_NOMINAL_YIELD_CURVE_CSV))
    )
    row = select_curve_row(rows, args.curve_date, args.density)
    download_date = dt.date.fromisoformat(args.download_date)
    fixture = build_fixture(row, download_date, args.density)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def fetch_rows(url: str) -> Iterable[dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ChebyshevSharp fixed-rate bond fixture refresh",
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8-sig")

    yield from read_rows(text)


def read_rows(text: str) -> Iterable[dict[str, str]]:
    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    yield from csv.DictReader(lines[header_index:])


def select_curve_row(
    rows: list[dict[str, str]],
    curve_date: str,
    density: str,
) -> dict[str, str]:
    complete_rows = [row for row in rows if is_complete_zero_curve_row(row, density)]

    if curve_date == "latest":
        if not complete_rows:
            raise ValueError("No complete zero-yield rows found.")
        return complete_rows[-1]

    for row in complete_rows:
        if row["Date"] == curve_date:
            return row

    raise ValueError(f"No complete zero-yield row found for {curve_date}.")


def is_complete_zero_curve_row(row: dict[str, str], density: str) -> bool:
    required_fields = [field_name(year) for year in SELECTED_MATURITIES]
    if density == "semiannual-svensson":
        required_fields += list(SVENSSON_PARAMETER_FIELDS)

    return all(parse_optional_float(row.get(field)) is not None for field in required_fields)


def build_fixture(row: dict[str, str], download_date: dt.date, density: str) -> dict[str, object]:
    if density == "semiannual-svensson":
        return build_dense_semiannual_fixture(row, download_date)

    curve_date = row["Date"]
    points = [
        {
            "maturity_years": year,
            "field": field_name(year),
            "zero_yield_percent": parse_required_float(row[field_name(year)]),
        }
        for year in SELECTED_MATURITIES
    ]

    return {
        "name": "Federal Reserve nominal zero yield curve",
        "fixture_id": f"fed-nominal-yield-curve-{curve_date}",
        "source": {
            "institution": "Board of Governors of the Federal Reserve System",
            "source_url": FED_NOMINAL_YIELD_CURVE_CSV,
            "source_page": "https://www.federalreserve.gov/data/nominal-yield-curve.htm",
            "download_date": download_date.isoformat(),
            "curve_date": curve_date,
            "source_note": (
                "Federal Reserve nominal yield curve staff research product; "
                "not an official statistical release."
            ),
        },
        "instrument_family": "hypothetical_treasury_zero_coupon_yield",
        "rate_kind": "zero_coupon_yield",
        "units": "percent",
        "compounding": "continuous",
        "day_count": "Actual/365",
        "interpolation": "linear_in_zero_rates",
        "original_fields": [field_name(year) for year in SELECTED_MATURITIES],
        "points": points,
    }


def build_dense_semiannual_fixture(row: dict[str, str], download_date: dt.date) -> dict[str, object]:
    curve_date = row["Date"]
    valuation_date = dt.date.fromisoformat(curve_date)
    parameters = {
        field: parse_required_float(row[field])
        for field in SVENSSON_PARAMETER_FIELDS
    }
    points = []

    for months in range(6, 361, 6):
        pillar_date = add_months(valuation_date, months)
        maturity_years = (pillar_date - valuation_date).days / 365.0
        points.append(
            {
                "maturity_years": maturity_years,
                "maturity_months": months,
                "field": f"SVENY_SVENSSON_{months:04d}M",
                "zero_yield_percent": round(svensson_zero_yield_percent(maturity_years, parameters), 10),
            }
        )

    return {
        "name": "Federal Reserve nominal zero yield curve, semiannual Svensson sample",
        "fixture_id": f"fed-nominal-yield-curve-semiannual-{curve_date}",
        "source": {
            "institution": "Board of Governors of the Federal Reserve System",
            "source_url": FED_NOMINAL_YIELD_CURVE_CSV,
            "source_page": "https://www.federalreserve.gov/data/nominal-yield-curve.htm",
            "download_date": download_date.isoformat(),
            "curve_date": curve_date,
            "source_note": (
                "Federal Reserve nominal yield curve staff research product; "
                "not an official statistical release. Semiannual points are derived from "
                "the published fitted nominal yield-curve parameters for the pinned curve date "
                "using Actual/365 year fractions to the pillar dates."
            ),
        },
        "instrument_family": "hypothetical_treasury_zero_coupon_yield",
        "rate_kind": "zero_coupon_yield",
        "units": "percent",
        "compounding": "continuous",
        "day_count": "Actual/365",
        "interpolation": (
            "semiannual Svensson parameter sample; QLNet example linearly interpolates "
            "zero rates between sampled pillars"
        ),
        "original_fields": list(SVENSSON_PARAMETER_FIELDS)
        + [field_name(year) for year in range(1, 31)],
        "source_model": {
            "family": "Svensson fitted nominal zero-yield curve",
            "formula": (
                "beta0 + beta1*L(t,tau1) + beta2*(L(t,tau1)-exp(-t/tau1)) + "
                "beta3*(L(t,tau2)-exp(-t/tau2)), L(t,tau)=(1-exp(-t/tau))/(t/tau)"
            ),
            "time_variable": "Actual/365 year fraction from curve_date to the semiannual pillar date",
            "parameters": parameters,
        },
        "points": points,
    }


def svensson_zero_yield_percent(maturity_years: float, parameters: dict[str, float]) -> float:
    tau1 = parameters["TAU1"]
    tau2 = parameters["TAU2"]
    loading1 = exponential_loading(maturity_years, tau1)
    loading2 = exponential_loading(maturity_years, tau2)

    return (
        parameters["BETA0"]
        + parameters["BETA1"] * loading1
        + parameters["BETA2"] * (loading1 - math.exp(-maturity_years / tau1))
        + parameters["BETA3"] * (loading2 - math.exp(-maturity_years / tau2))
    )


def exponential_loading(maturity_years: float, tau: float) -> float:
    x = maturity_years / tau
    return (1.0 - math.exp(-x)) / x


def add_months(date: dt.date, months: int) -> dt.date:
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    day = min(date.day, days_in_month(year, month))
    return dt.date(year, month, day)


def days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)

    return (next_month - dt.timedelta(days=1)).day


def field_name(year: int) -> str:
    return f"SVENY{year:02d}"


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "NA":
        return None

    result = float(value)
    if not math.isfinite(result):
        return None

    return result


def parse_required_float(value: str) -> float:
    result = parse_optional_float(value)
    if result is None:
        raise ValueError(f"Expected finite numeric value, got {value!r}.")

    return result


if __name__ == "__main__":
    sys.exit(main())
