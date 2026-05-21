#!/usr/bin/env python3
"""Generate Phase 6 fixed-rate bond maturity-sensitivity evidence plots.

The script keeps pricing in the C# QLNet-backed example and uses Python only to
persist CSV data and render a lightweight SVG. It has no third-party
dependencies.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanPoint:
    boundary_date: str
    offset_days: int
    maturity_date: str
    cashflow_count: int
    dirty_price: float
    central_slope_per_year: float | None
    second_difference: float | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 6 maturity schedule-sensitivity CSV and SVG evidence.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to two levels above this script.")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV output path. Defaults under docs/research/fixed-rate-bond-surrogate/data.")
    parser.add_argument(
        "--svg-out",
        type=Path,
        default=None,
        help="SVG output path. Defaults under docs/research/fixed-rate-bond-surrogate/images.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    csv_out = args.csv_out or repo_root / "docs/research/fixed-rate-bond-surrogate/data/phase-6-maturity-scan.csv"
    svg_out = args.svg_out or repo_root / "docs/research/fixed-rate-bond-surrogate/images/phase-6-maturity-sensitivity.svg"

    csv_text = run_scan(repo_root)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.write_text(csv_text, encoding="utf-8")

    points = parse_scan(csv_text)
    focus_boundary = find_focus_boundary(points)
    focus_points = [point for point in points if point.boundary_date == focus_boundary]

    svg_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.write_text(render_svg(focus_boundary, focus_points), encoding="utf-8")

    print(f"Wrote {csv_out}")
    print(f"Wrote {svg_out}")
    print(f"Focused boundary: {focus_boundary}")


def run_scan(repo_root: Path) -> str:
    command = [
        "dotnet",
        "run",
        "--project",
        "examples/FixedRateBondSurrogate/FixedRateBondSurrogate.csproj",
        "--",
        "--naive-maturity-scan-csv",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def parse_scan(csv_text: str) -> list[ScanPoint]:
    points: list[ScanPoint] = []
    reader = csv.DictReader(csv_text.splitlines())
    for row in reader:
        points.append(
            ScanPoint(
                boundary_date=row["boundary_date"],
                offset_days=int(row["offset_days"]),
                maturity_date=row["maturity_date"],
                cashflow_count=int(row["cashflow_count"]),
                dirty_price=float(row["dirty_price"]),
                central_slope_per_year=parse_optional_float(row["central_slope_per_year"]),
                second_difference=parse_optional_float(row["second_difference"]),
            )
        )
    if not points:
        raise ValueError("maturity scan produced no points")
    return points


def parse_optional_float(value: str) -> float | None:
    return float(value) if value else None


def find_focus_boundary(points: list[ScanPoint]) -> str:
    candidates = [point for point in points if point.second_difference is not None]
    if not candidates:
        raise ValueError("maturity scan produced no finite second differences")
    return max(candidates, key=lambda point: abs(point.second_difference or 0.0)).boundary_date


def render_svg(boundary_date: str, points: list[ScanPoint]) -> str:
    points = sorted(points, key=lambda point: point.offset_days)
    width = 980
    height = 610
    left = 78
    right = 34
    top = 66
    panel_gap = 64
    panel_h = 190
    plot_w = width - left - right
    price_top = top
    slope_top = top + panel_h + panel_gap
    x_values = [point.offset_days for point in points]
    price_values = [point.dirty_price for point in points]
    slope_points = [point for point in points if point.central_slope_per_year is not None]
    slope_values = [point.central_slope_per_year for point in slope_points if point.central_slope_per_year is not None]

    x_min, x_max = min(x_values), max(x_values)
    price_min, price_max = padded_range(price_values, pad_fraction=0.08)
    slope_min, slope_max = padded_range(slope_values, pad_fraction=0.15)

    def x_scale(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_scale(value: float, lo: float, hi: float, panel_top: float) -> float:
        return panel_top + panel_h - (value - lo) / (hi - lo) * panel_h

    price_polyline = polyline(
        [(x_scale(point.offset_days), y_scale(point.dirty_price, price_min, price_max, price_top)) for point in points])
    slope_polyline = polyline(
        [
            (
                x_scale(point.offset_days),
                y_scale(point.central_slope_per_year or 0.0, slope_min, slope_max, slope_top),
            )
            for point in slope_points
        ])
    boundary_x = x_scale(0)
    max_spike = max(
        (point for point in points if point.second_difference is not None),
        key=lambda point: abs(point.second_difference or 0.0),
    )
    spike_x = x_scale(max_spike.offset_days)
    spike_y = y_scale(max_spike.central_slope_per_year or 0.0, slope_min, slope_max, slope_top)

    axis_color = "#3a4658"
    grid_color = "#d5dbe5"
    price_color = "#2563eb"
    slope_color = "#dc2626"

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Fixed-rate bond maturity sensitivity near a semiannual schedule boundary</title>",
        f"<desc id=\"desc\">Dirty price and central finite-difference maturity sensitivity for maturities within seven days of {html.escape(boundary_date)}.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#172033">Maturity schedule boundary evidence</text>',
        f'<text x="{left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#526071">Boundary {html.escape(boundary_date)}; valuation date fixed; QLNet-backed baseline; direct zero-rate curve fixture.</text>',
    ]

    parts.extend(panel_axes(left, price_top, plot_w, panel_h, x_min, x_max, price_min, price_max, "Dirty PV", axis_color, grid_color, x_scale, y_scale))
    parts.append(f'<polyline points="{price_polyline}" fill="none" stroke="{price_color}" stroke-width="2.7" stroke-linejoin="round"/>')
    parts.append(f'<line x1="{boundary_x:.2f}" y1="{price_top}" x2="{boundary_x:.2f}" y2="{price_top + panel_h}" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 5"/>')

    parts.extend(panel_axes(left, slope_top, plot_w, panel_h, x_min, x_max, slope_min, slope_max, "Maturity sensitivity dPV/dT", axis_color, grid_color, x_scale, y_scale))
    parts.append(f'<polyline points="{slope_polyline}" fill="none" stroke="{slope_color}" stroke-width="2.7" stroke-linejoin="round"/>')
    parts.append(f'<line x1="{boundary_x:.2f}" y1="{slope_top}" x2="{boundary_x:.2f}" y2="{slope_top + panel_h}" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 5"/>')
    parts.append(f'<circle cx="{spike_x:.2f}" cy="{spike_y:.2f}" r="5" fill="{slope_color}" stroke="#ffffff" stroke-width="1.5"/>')
    parts.append(
        f'<text x="{min(spike_x + 12, width - 305):.2f}" y="{max(spike_y - 10, slope_top + 16):.2f}" '
        'font-family="Arial, sans-serif" font-size="12" fill="#7f1d1d">'
        f'largest local second diff {max_spike.second_difference:.3e}</text>')

    parts.append(f'<text x="{left + plot_w / 2 - 80:.2f}" y="{height - 28}" font-family="Arial, sans-serif" font-size="13" fill="#374151">offset in calendar days from semiannual boundary</text>')
    parts.append(f'<text x="{left}" y="{height - 10}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">Generated by tools/PlotFixedRateBondEvidence/plot_phase6_maturity.py from --naive-maturity-scan-csv.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def padded_range(values: list[float], pad_fraction: float) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    lo, hi = min(finite), max(finite)
    if lo == hi:
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def panel_axes(
    left: int,
    panel_top: int,
    plot_w: int,
    panel_h: int,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    label: str,
    axis_color: str,
    grid_color: str,
    x_scale,
    y_scale,
) -> list[str]:
    parts = [
        f'<rect x="{left}" y="{panel_top}" width="{plot_w}" height="{panel_h}" fill="#f8fafc" stroke="{grid_color}"/>',
        f'<text x="20" y="{panel_top + panel_h / 2:.2f}" font-family="Arial, sans-serif" font-size="13" fill="{axis_color}" transform="rotate(-90 20 {panel_top + panel_h / 2:.2f})">{html.escape(label)}</text>',
    ]

    for x in range(x_min, x_max + 1, 2):
        sx = x_scale(x)
        parts.append(f'<line x1="{sx:.2f}" y1="{panel_top}" x2="{sx:.2f}" y2="{panel_top + panel_h}" stroke="{grid_color}" stroke-width="0.8"/>')
        parts.append(f'<text x="{sx - 7:.2f}" y="{panel_top + panel_h + 18}" font-family="Arial, sans-serif" font-size="11" fill="{axis_color}">{x}</text>')

    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4.0
        sy = y_scale(value, y_min, y_max, panel_top)
        parts.append(f'<line x1="{left}" y1="{sy:.2f}" x2="{left + plot_w}" y2="{sy:.2f}" stroke="{grid_color}" stroke-width="0.8"/>')
        parts.append(f'<text x="{left - 66}" y="{sy + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="{axis_color}">{value:.3f}</text>')

    parts.append(f'<line x1="{left}" y1="{panel_top + panel_h}" x2="{left + plot_w}" y2="{panel_top + panel_h}" stroke="{axis_color}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_h}" stroke="{axis_color}" stroke-width="1.2"/>')
    return parts


if __name__ == "__main__":
    main()
