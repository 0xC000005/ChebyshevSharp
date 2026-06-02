---
title: American Option Dynamic Chebyshev Case Study
---

# American Option Dynamic Chebyshev Case Study

This technical blog asks a practical question: can Chebyshev interpolation solve
the continuation-value problem in an American option more accurately and faster
than simulation or reinforcement-learning baselines?

The short answer is: yes for this controlled Black-Scholes American put. The
reason is not that Chebyshev approximates the whole stopping problem blindly.
The useful split is:

```text
hard part:     exercise decision, max(payoff, continuation)
smooth part:   continuation value under the known transition law
```

The case study follows the same discipline as the fixed-rate and callable bond
examples. It starts with a trusted reference pricer, reproduces standard
statistical baselines, then applies Chebyshev interpolation only to the smooth
continuation-value objects in a finite-horizon Bellman recursion.

The runnable harness lives in `examples/AmericanOptionDynamicChebyshev`.

```bash
dotnet run --project examples/AmericanOptionDynamicChebyshev/AmericanOptionDynamicChebyshev.csproj
```

## What the example prices

The product is a one-year American put under the Black-Scholes-Merton model.
The default request uses spot `100`, strike `100`, volatility `20%`, risk-free
rate `5%`, and dividend yield `0%`.

| Item | Choice |
| --- | --- |
| Reference pricer | QLNet `VanillaOption` with `AmericanExercise` |
| Reference engine | QLNet finite-difference Black-Scholes engine |
| Cross-check engine | QLNet Cox-Ross-Rubinstein binomial tree |
| European control | QLNet analytic European engine |
| Simulation baseline | Longstaff-Schwartz least-squares Monte Carlo |
| RL baseline | Stanford-style LSPI for American-option exercise |
| Chebyshev method | Dynamic Chebyshev backward induction |
| State variable | spot price |
| Transition model | risk-neutral geometric Brownian motion |
| Exercise grid | Bermudan discretization of the American exercise window |

This is not a generic option-pricing library. It is a reproducible demonstration
of a continuation-value approximation strategy.

## The optimal-stopping problem

At each exercise date \(t_i\), the option holder compares immediate exercise
with continuation. For a put with strike \(K\), the payoff is

$$
h(S_i)=\max(K-S_i,0).
$$

The dynamic program is

$$
C_i(S_i)
  =
  e^{-r\Delta t}
  \mathbb{E}\!\left[
    V_{i+1}(S_{i+1})\mid S_i
  \right],
$$

$$
V_i(S_i)=\max\left(h(S_i), C_i(S_i)\right),
\qquad
V_N(S_N)=h(S_N).
$$

The kink is the `max`. The continuation function \(C_i\) is usually much
smoother than the final exercise decision. That is the central modeling
decision in this page: approximate continuation, not the full policy as one
opaque black box.

Under Black-Scholes-Merton dynamics,

$$
S_{i+1}
  =
  S_i
  \exp\left(
    (r-q-\tfrac{1}{2}\sigma^2)\Delta t
    + \sigma\sqrt{\Delta t}\,Z
  \right),
\qquad Z\sim N(0,1).
$$

Because this transition law is known, the conditional expectation can be
computed by quadrature. That is exactly the setting where a deterministic
numerical-analysis method has an advantage over methods that must learn the
continuation value statistically from simulated paths.

## Trusted numerical baseline

The correctness oracle is QLNet, the C# port of QuantLib-style APIs. The case
study constructs:

```text
AmericanOptionRequest
  -> QLNet PlainVanillaPayoff
  -> QLNet AmericanExercise
  -> QLNet BlackScholesMertonProcess
  -> QLNet FdBlackScholesVanillaEngine
  -> price, bumped Delta, bumped Gamma
```

The baseline is checked before any surrogate is trusted:

- an American put must be worth at least the European put;
- a non-dividend American call should match the European call within numerical
  tolerance;
- finite-difference and high-step binomial prices must agree within a small
  band;
- put price must decrease as spot rises and increase as volatility rises;
- bumped Delta and Gamma must have sensible signs away from boundary pathologies.

Those tests keep this tutorial from quietly benchmarking against a broken
home-grown pricer.

## What was parity checked

The American-option numerical oracle is QLNet, not the Chebyshev model itself.
The finite-difference engine supplies the reference price and bumped Greeks,
while the high-step binomial engine is an independent cross-check for the same
Black-Scholes product.

QuantEcon's `ContinuousDPs.jl` was used differently. It is a useful public
reference for continuous-state Bellman-equation collocation workflows, but it
solves infinite-horizon continuous-state/control examples rather than this
finite-horizon stopping problem. For that reason, this case study does not claim
direct American-option price parity against `ContinuousDPs.jl`.

Instead, the comparable layer is the Bellman expectation operator. The test
suite checks the risk-neutral first-moment identity used inside the Dynamic
Chebyshev continuation builder:

$$
e^{-r\Delta t}\mathbb{E}[S_{i+1}\mid S_i=S]
  =
S e^{-q\Delta t}.
$$

That verifies the transition-and-discount step shared by Bellman-collocation
methods before the American-option-specific stopping rule is applied.

## Baseline 1: Longstaff-Schwartz regression

Longstaff and Schwartz estimate continuation by simulation and regression. The
algorithm is intuitive:

1. Simulate many risk-neutral spot paths.
2. Work backward from maturity.
3. At each exercise date, regress discounted future cashflows on basis
   functions of current spot.
4. Exercise on a path when payoff exceeds the fitted continuation value.

Mathematically, this replaces the conditional expectation with a regression:

$$
C_i(S_i)
\approx
\beta_{i,0}
+ \beta_{i,1}\frac{S_i}{K}
+ \beta_{i,2}\left(\frac{S_i}{K}\right)^2 .
$$

This is a strong practical baseline because it is simple, path-based, and
extends to high-dimensional or path-dependent products. Its weakness in this
small one-factor example is that it spends random samples to estimate a smooth
one-dimensional continuation function whose transition law is already known.

## Baseline 2: Stanford-style LSPI

The reinforcement-learning baseline follows Ashwin Rao's Stanford American
option material. The important specialization is that exercise value is not
learned:

$$
Q(s,\mathrm{exercise})=h(s).
$$

Only continuation is represented by a linear value function:

$$
Q(s,\mathrm{continue})\approx w^\top\phi(s).
$$

The C# harness uses the source-inspired shape from the Stanford notes: Laguerre
style functions of \(S/K\) plus simple time-to-maturity features. The exact
feature details matter because LSPI is still a statistical approximation; a
different feature set can move the fitted exercise boundary and therefore the
price.

The policy compares the two:

$$
\pi(s)=
\begin{cases}
\mathrm{continue}, & w^\top\phi(s)\ge h(s),\\
\mathrm{exercise}, & \text{otherwise}.
\end{cases}
$$

This is the right RL baseline for the tutorial because it is inspectable. It
does not hide the stopping problem behind a neural network, and it uses the
known payoff exactly. It still learns from simulated transitions, so it carries
sampling error and feature-basis error.

## Dynamic Chebyshev method

Dynamic Chebyshev keeps the same Bellman recursion but changes how continuation
is computed. At every time step it builds a 1D Chebyshev interpolant of
\(C_i(S)\) on a bounded spot interval:

$$
\widehat{C}_i(S)
\approx
e^{-r\Delta t}
\mathbb{E}\!\left[
  \max\left(h(S_{i+1}),\widehat{C}_{i+1}(S_{i+1})\right)
  \mid S_i=S
\right].
$$

The expectation is evaluated with Gauss-Hermite quadrature. With quadrature
nodes \(x_m\) and weights \(w_m\),

$$
\widehat{C}_i(S)
 =
e^{-r\Delta t}
\frac{1}{\sqrt{\pi}}
\sum_m
w_m
V_{i+1}
\left(
S\exp\left(
  (r-q-\tfrac{1}{2}\sigma^2)\Delta t
  + \sqrt{2}\sigma\sqrt{\Delta t}\,x_m
\right)
\right).
$$

The value used for exercise is still exact:

$$
\widehat{V}_i(S)
 =
\max\left(h(S),\widehat{C}_i(S)\right).
$$

This is the same core lesson as other structure-aware Chebyshev examples:
preserve the non-smooth decision outside the interpolant and use Chebyshev only
where the target is smooth enough to benefit from spectral approximation.

The implementation builds one reusable model:

```csharp
var model = new DynamicChebyshevAmericanOptionPricer().Build(request, settings);
var valueAndGreeks = model.Evaluate(100.0);
```

`Build` performs the backward induction. `Evaluate` reuses the first-step
continuation interpolant to return price, Delta, and Gamma at new spot values.

## Results

One representative local run prints:

| Method | Price | Standard error | Notes |
| --- | ---: | ---: | --- |
| QLNet finite difference | `6.088238` | n/a | Reference oracle |
| QLNet analytic European | `5.573526` | n/a | European lower control |
| Longstaff-Schwartz | `6.080847` | `0.065629` | Regression simulation |
| Stanford-style LSPI | `5.745344` | `0.074933` | RL continuation baseline |
| Dynamic Chebyshev | `6.083607` | deterministic | Bellman + Chebyshev continuation |

Dynamic Chebyshev also reports:

| Metric | Value |
| --- | ---: |
| Delta | `-0.410533` |
| Gamma | `0.022946` |
| Build evaluations | `6480` |
| Build time | `0.051s` |
| Online evaluation | `6.406 us` |
| QLNet reference evaluation | `35.399 ms` |
| Online speedup | `5525.6x` |
| Spot-grid max price absolute error | `1.346E-002` |
| Spot-grid max Delta absolute error | `1.112E-002` |
| Spot-grid max Gamma absolute error | `7.993E-003` |

The speed number is an online evaluation comparison after the Chebyshev model
has been built. That is the intended use case: pay the build cost once, then
reuse the interpolants for many prices, bump-and-reprice Greeks, or scenario
evaluations. The QLNet reference timing shown here is the tutorial adapter's
full `AmericanOptionResult` path, including bump-and-reprice Delta and Gamma,
not a single raw NPV call.

## What this teaches

The baseline methods all solve the same Bellman problem, but they spend effort
in different places:

| Method | What it learns or computes | Main limitation |
| --- | --- | --- |
| QLNet finite difference | PDE/grid reference value | Correct but too expensive for repeated tutorial-level risk loops |
| Longstaff-Schwartz | Continuation by pathwise regression | Sampling noise and regression-basis error |
| LSPI | Continuation action value and exercise policy | Sampling noise, feature-basis error, policy instability near boundary |
| Dynamic Chebyshev | Continuation by deterministic collocation and quadrature | Requires known transition law and a controlled state dimension |

The case study is not saying reinforcement learning is the wrong tool in
general. RL is valuable when the transition model is unknown, the policy must
be learned from experience, or the state/action structure is too complex for a
direct numerical method. Here the transition model is known and one-dimensional,
so a deterministic Bellman-collocation method is the cleaner tool.

## Practical workflow

Use this example as a template for future optimal-stopping experiments:

1. Start with a trusted quant-library oracle.
2. Write economic sanity tests before approximation tests.
3. Reproduce standard statistical baselines, not local inventions.
4. Identify the smooth continuation value separately from the stopping rule.
5. Approximate the continuation function with Chebyshev interpolation.
6. Reuse the built model for price and Greeks.
7. Report accuracy, build cost, online speed, and method limitations together.

## Sources

Core references are listed in [Citations](citations.md), especially the
American-option, reinforcement-learning, dynamic-programming, and Dynamic
Chebyshev entries.
