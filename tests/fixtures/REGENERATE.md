# Fixture regeneration

These three `.pcb` files are committed binary fixtures used by C# tests in
`BinaryFormatTests.cs::PcbFixtureTests`. They lock down the .pcb v1 byte
layout AND serve as cross-language byte anchors: the committed bytes are
**produced by PyChebyshev v0.18.0** (which writes the v0.14-era format), so
each fixture-load test transitively proves "ChebyshevSharp can read
PyChebyshev's bytes."

## Files

| File | Size | Source | Description |
|---|---:|---|---|
| `approx_2d_simple.pcb` | 128 B | PyChebyshev | f(x,y)=x+y on [-1,1]², n=[3,3]. The byte layout is documented byte-for-byte in `ref/PyChebyshev/docs/user-guide/binary-format.md` §"Worked example". ChebyshevSharp's writer produces byte-identical output for this fixture. |
| `approx_5d_bs.pcb` | 25116 B | PyChebyshev | 5D Black-Scholes-like shape on standard option-pricing domain. C# round-trip test (`Test_fixture_approx_5d_bs_loads_and_round_trips`) verifies that loading these bytes and re-saving produces the same bytes back. |
| `spline_1d_kink.pcb` | 100 B | PyChebyshev | abs(x) on [-1,1] with knot at 0 (the spline worked example from binary-format.md). |

### Note on tensor-value rounding

For the spline fixture, ChebyshevSharp's tensor values diverge from
PyChebyshev's by 1–4 ULPs in some entries — this is rounding noise in the
node-mapping arithmetic `(a+b)/2 + (b-a)/2 * x`, not a format-spec issue.
The mathematics is identical; the IEEE-754 rounding paths differ. We use
PyChebyshev's bytes as the source of truth so the fixture is a true
cross-language anchor; the eval-based fixture tests pass at `precision: 12`
(orders of magnitude looser than the 1–4 ULP source difference).

## Regenerating from PyChebyshev (canonical path)

Run from the repo root:

```bash
cd ref/PyChebyshev && uv sync
uv run python <<'PYEOF'
import os, math
from pychebyshev import ChebyshevApproximation, ChebyshevSpline
out = "../../tests/fixtures"

# 1) approx_2d_simple.pcb
cheb = ChebyshevApproximation(
    function=lambda pt, _: pt[0] + pt[1],
    num_dimensions=2, domain=[(-1.0, 1.0), (-1.0, 1.0)], n_nodes=[3, 3])
cheb.build()
cheb.save(f"{out}/approx_2d_simple.pcb", format="binary")

# 2) approx_5d_bs.pcb
def f5d(pt, _):
    return max(pt[0] - 100, 0) + 0.01*pt[1] - 0.5*pt[2]*pt[2] + pt[3] + pt[4]
cheb5 = ChebyshevApproximation(
    function=f5d, num_dimensions=5,
    domain=[(80.0, 120.0), (0.1, 0.4), (0.5, 1.5), (0.0, 0.05), (90.0, 110.0)],
    n_nodes=[5, 5, 5, 5, 5])
cheb5.build()
cheb5.save(f"{out}/approx_5d_bs.pcb", format="binary")

# 3) spline_1d_kink.pcb
spl = ChebyshevSpline(
    function=lambda pt, _: abs(pt[0]),
    num_dimensions=1, domain=[(-1.0, 1.0)], n_nodes=[3], knots=[[0.0]])
spl.build()
spl.save(f"{out}/spline_1d_kink.pcb", format="binary")
PYEOF
cd ../..
wc -c tests/fixtures/*.pcb
# Expected: 128, 25116, 100
```

## Regenerating from ChebyshevSharp (fallback)

If PyChebyshev is unavailable, ChebyshevSharp can produce the 2D fixture
byte-identically (verified during Phase 3 development). The 1D spline
fixture from C# diverges by 1–4 ULPs; cross-language consumers should not
treat C#-generated spline fixtures as the canonical byte form.

```bash
dotnet run --project tools/GenerateFixtures
```

## Cross-checking when bumping the submodule

When `ref/PyChebyshev` bumps to a future version, regenerate the fixtures
from Python and `cmp` against the committed bytes:

```bash
cmp tests/fixtures/approx_2d_simple.pcb /tmp/py_approx_2d_simple.pcb
# Expected: silent (bytes match exactly).
```

If `cmp` reports a difference for the 2D fixture: investigate before
proceeding — the format may have changed. The 1D spline fixture is
expected to differ slightly between languages due to FP rounding (see
"Note on tensor-value rounding" above), so use eval-based tests rather
than `cmp` for that one.
