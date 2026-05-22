---
title: Callable Bond Case Study
---

# Callable Bond Surrogates: A Full-Dimensional Chebyshev Case Study

This tutorial is a worked research case study. It asks whether a Chebyshev
surrogate can clone a callable fixed-rate bond pricer that is expensive enough
to make acceleration meaningful.

The case study is deliberately restricted and public. It uses QLNet as the
reference implementation, a pinned Federal Reserve nominal-yield-curve fixture,
a semiannual fixed-rate bullet bond, and a one-factor Hull-White tree callable
bond engine. It is not a general fixed-income engine.

## Why Callable Bonds

The earlier fixed-rate bond case study is a useful correctness exercise, but an
option-free bond is already cheap once the cashflow schedule is cached. A
callable bond adds an embedded issuer option. The standard pricing picture is:

$$
Q_{\mathrm{callable}}
  =
  Q_{\mathrm{straight}}
  -
  V_{\mathrm{issuer\ call}} .
$$

The straight-bond leg is mostly deterministic cashflow discounting. The issuer
call option is harder because the optimal call decision depends on the future
rate tree and the call schedule. That makes this example closer to the risk
workloads where Chebyshev tensors are useful: the baseline call is expensive,
but users need repeated PV and sensitivity evaluations across many scenarios.

## Public Wrapper

Every trial uses the same full-dimensional request-level wrapper:

```text
curveBumps[60], coupon, maturity, firstCall, callPrice, sigma
    -> callable dirty price per 100 notional
```

The public dimension count is 65. The 60 curve coordinates are basis-point
bumps to semiannual zero-rate pillars from 0.5Y to 30Y. The remaining
coordinates are the coupon rate, maturity in years, first-call time in years,
clean call price, and Hull-White volatility. Hull-White mean reversion is fixed
at 3% in the first harness.

Internally, a model may compress the curve or decompose the formula. The public
input contract still stays 65D so that a caller can treat the surrogate as a
request-level clone. The documentation labels each internal model honestly:

| Label | Meaning |
| --- | --- |
| Faithful full-pillar clone | Accepts arbitrary 60-pillar bump vectors without changing the risk contract. |
| Factor-risk surrogate | Projects the 60-pillar curve into a smaller factor basis, so it is only faithful for factor-like scenarios. |
| Formula-aware surrogate | Uses bond structure outside Chebyshev, then approximates only the remaining expensive component. |

## Baseline

The reference pricer is QLNet:

```text
CallableBondRequest
  -> QLNet Schedule
  -> QLNet CallableFixedRateBond
  -> QLNet InterpolatedZeroCurve<Linear>
  -> QLNet HullWhite
  -> QLNet TreeCallableFixedRateBondEngine
  -> CallableBondResult
```

Run the baseline:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj
```

The default scenario is a 30Y, 6% coupon callable bond with first call at 5Y,
call price 100, Hull-White mean reversion 3%, volatility 1%, and 80 tree steps.
The zero curve is the same pinned semiannual Federal Reserve fixture used by the
fixed-rate bond case study.

The harness checks economic sanity before fitting surrogates: the callable
dirty price should not exceed the comparable straight-bond dirty price, upward
curve bumps should reduce price, post-maturity direct curve exposure should be
negligible, tree-step convergence should be stable, and call price / volatility
effects should have the expected sign.

## Metrics

PV alone is not enough for a risk surrogate. The case study also checks first
and mixed finite-difference quantities:

| Metric | Meaning |
| --- | --- |
| PV | Dirty price per 100 notional. |
| Zero-pillar DV01 | Sensitivity to one zero-rate pillar bump. |
| Coupon derivative | Slope with respect to the coupon coordinate. |
| Sigma sensitivity | Slope with respect to Hull-White volatility. |
| Call-price sensitivity | Slope with respect to clean call price. |
| Rate-sigma mixed | Whether rate exposure changes correctly when volatility changes. |
| Call-price-sigma mixed | Whether call-price exposure changes correctly when volatility changes. |

Speed is reported with the break-even count:

$$
N_{\mathrm{break\ even}}
 =
 \frac{t_{\mathrm{build}}}
      {t_{\mathrm{baseline\ eval}} - t_{\mathrm{surrogate\ eval}}}.
$$

A surrogate is useful only if its build cost can be amortized over a realistic
number of repeated scenario or Greek evaluations.

## Trial 1: Naive Global Clone

The first trial asks the most direct question: what happens if a user points a
global high-dimensional model at the callable pricer?

$$
\widehat{Q}_{\mathrm{TT}}(x, c, T, \tau, K, \sigma)
  \approx
  Q_{\mathrm{QLNet}}(x, c, T, \tau, K, \sigma).
$$

Here \(x \in \mathbb{R}^{60}\) is the full zero-pillar bump vector, \(c\) is
coupon, \(T\) is maturity, \(\tau\) is first call, \(K\) is call price, and
\(\sigma\) is Hull-White volatility.

Run it:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --naive-surrogate-discovery
```

A dense tensor with only three nodes per dimension would need \(3^{65}\)
baseline evaluations:

```text
10,301,051,460,877,537,453,973,547,267,843
```

That is infeasible, so the naive trial uses `ChebyshevTT` and
`ChebyshevSlider` as compression probes while keeping the full 65D input.

| Model | Build evals | Max PV rel. error | Max 10Y DV01 rel. error | Max rate-sigma mixed rel. error | Max call-price-sigma mixed rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorTrain | 5,476 | 15.40% | 98.27% | 552.58% | 9,471.30% |
| Slider | 195 | 46.51% | 998.83% | 100.00% | 100.00% |

This is useful negative evidence. The global TT is too coarse to reproduce
risk quantities. The singleton Slider is worse because it is an anchored
additive model; cross-group mixed terms are zero by construction, so it cannot
learn interactions such as rate-volatility or call-price-volatility coupling.

## Trial 2: Internal Curve Compression

The next trial keeps the same 65D public wrapper but compresses the 60-pillar
curve internally into level, slope, and curvature factors:

$$
x
  \mapsto
  \left(
    \langle x, b_0 \rangle,
    \langle x, b_1 \rangle,
    \langle x, b_2 \rangle
  \right),
$$

where \(b_0\), \(b_1\), and \(b_2\) are deterministic level, slope, and
curvature basis vectors over the tenor grid. The internal approximation is then
8D:

$$
\widehat{Q}_{\mathrm{factor}}
  (f_0, f_1, f_2, c, T, \tau, K, \sigma).
$$

This is not a faithful arbitrary-pillar clone. It is a factor-risk surrogate.
That distinction is the point of the experiment: factor compression can be a
good business workflow if the risk system asks for factor scenarios, but it
cannot promise correct local key-rate risk for arbitrary pillar bumps.

Run it:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --structured-alternatives
```

| Model | Type | Internal dims | Max PV rel. error on factor scenarios | Max PV rel. error on arbitrary bumps | Surrogate eval | Break-even |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curve-factor tensor | factor-risk surrogate | 8 | 1.56% | 1.89% | 3.26 us | 6.3k |
| Curve-factor TT | factor-risk surrogate | 8 | 2.20% | 0.90% | 3.23 us | 2.5k |

This trial is the first plausible acceleration story: the QLNet baseline takes
roughly 1.6 ms per evaluation in the harness, while the factor TT evaluates in
about 3 us after construction. The limitation is equally important: local
key-rate sensitivities remain weak because the projection intentionally discards
most single-pillar directions.

## Trial 3: Embedded-Option Decomposition

The formula-aware trial tries to avoid spending Chebyshev capacity on the cheap
straight-bond component:

$$
V_{\mathrm{issuer\ call}}
  =
  Q_{\mathrm{straight}}
  -
  Q_{\mathrm{callable}}.
$$

It builds a Chebyshev surrogate for the embedded option value, then reconstructs
the callable price:

$$
\widehat{Q}_{\mathrm{callable}}
  =
  Q_{\mathrm{straight, exact}}
  -
  \widehat{V}_{\mathrm{issuer\ call}}.
$$

The rationale is sound: exact cashflow discounting should handle the easy
component, while Chebyshev focuses on the tree-driven option component. In the
current low-node probe, however, this does not yet improve the clone.

| Model | Type | Internal dims | Max PV rel. error on factor scenarios | Max PV rel. error on arbitrary bumps | Surrogate eval | Break-even |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Embedded-option curve-factor tensor | formula-aware factor-risk surrogate | 8 | 3.03% | 8.73% | 23.90 us | 6.5k |
| Embedded-option full-pillar TT | formula-aware faithful full-pillar candidate | 65 | 7.99% | 5.12% | 29.89 us | 6.1k |

The result is a useful failure: decomposition alone is not enough. The embedded
option value is smaller and more regime-sensitive than the full callable price,
so a weak low-node approximation can produce worse relative PV error even when
the formula is financially natural.

## Current Conclusion

The callable-bond harness has a clear but limited current answer.

The naive global clone does not work. It fails both price and risk metrics, and
the Slider misses important cross terms by construction. The curve-factor TT is
the best current candidate when the workload is factor-like: it preserves the
65D public wrapper, evaluates hundreds of times faster than the QLNet tree call,
and reaches about 1% PV error in the small validation bank. It is not a faithful
local key-rate risk clone, because the internal projection discards most
arbitrary 60-pillar directions.

The next research step is therefore not to claim a production replacement. It
is to improve the method that sits behind the full wrapper: richer curve bases,
PCA-style factors, active-pillar routing around cashflow and exercise dates,
regime-aware routing, and stronger validation banks that include PV, all-pillar
DV01, volatility Greeks, and mixed terms.

## Reproduce

Run the focused callable test suite:

```bash
dotnet test tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --filter "FullyQualifiedName~CallableBond"
```

Run the three public modes:

```bash
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --naive-surrogate-discovery
dotnet run --project examples/CallableBondSurrogate/CallableBondSurrogate.csproj -- --structured-alternatives
```

## Sources

Core references are listed in [Citations](citations.md), especially the sections
on Chebyshev interpolation, Tensor Train algorithms, finance applications,
callable-bond baseline libraries, and public market data.
