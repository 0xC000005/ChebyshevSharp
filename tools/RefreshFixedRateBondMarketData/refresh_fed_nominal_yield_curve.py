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
SELECTED_MATURITIES = (1, 2, 3, 5, 7, 10, 20, 30)


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
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--download-date",
        default=dt.date.today().isoformat(),
        help="Download date to record in YYYY-MM-DD form. Defaults to today's date.",
    )
    args = parser.parse_args(argv)

    rows = list(fetch_rows(FED_NOMINAL_YIELD_CURVE_CSV))
    row = select_curve_row(rows, args.curve_date)
    download_date = dt.date.fromisoformat(args.download_date)
    fixture = build_fixture(row, download_date)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
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

    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    yield from csv.DictReader(lines[header_index:])


def select_curve_row(rows: list[dict[str, str]], curve_date: str) -> dict[str, str]:
    complete_rows = [row for row in rows if is_complete_zero_curve_row(row)]

    if curve_date == "latest":
        if not complete_rows:
            raise ValueError("No complete zero-yield rows found.")
        return complete_rows[-1]

    for row in complete_rows:
        if row["Date"] == curve_date:
            return row

    raise ValueError(f"No complete zero-yield row found for {curve_date}.")


def is_complete_zero_curve_row(row: dict[str, str]) -> bool:
    return all(parse_optional_float(row.get(field_name(year))) is not None for year in SELECTED_MATURITIES)


def build_fixture(row: dict[str, str], download_date: dt.date) -> dict[str, object]:
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
