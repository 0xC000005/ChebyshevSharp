---
title: Continuous-State Dynamic Programming
---

# Continuous-State Dynamic Programming

ChebyshevSharp includes a small continuous-state dynamic-programming layer for
finite-horizon Bellman problems. The current API is deliberately bounded:

- one continuous state variable;
- finite horizon solved by backward induction;
- a discrete action grid;
- Chebyshev collocation for each action-value branch;
- online evaluation of value, selected action, and derivatives.

This is not a full clone of QuantEcon `ContinuousDPs.jl`. It is the reusable
core pattern behind the American-option case study, plus general examples for
dynamic asset allocation and stochastic growth.

## Why This Belongs In ChebyshevSharp

Many dynamic programs have a continuous state and a value function that is
smooth between decision boundaries. The expensive step is repeatedly applying
the Bellman operator:

$$
V_i(x)=\max_a
\left\{
r_i(x,a)+\beta\mathbb{E}[V_{i+1}(X_{i+1})\mid x,a]
\right\}.
$$

Chebyshev collocation gives a compact way to represent the action-value
functions:

$$
Q_i(x,a)\approx \sum_{j=0}^{n-1} c_{i,a,j}T_j(x).
$$

The policy is then the action whose fitted branch is largest:

$$
\pi_i(x)=\arg\max_a Q_i(x,a).
$$

This follows the same documentation style used by numerical economics and
portfolio libraries: define problem primitives first, then show concrete
examples. QuantEcon describes dynamic-programming solvers in terms of value
functions, policies, algorithms, and solve results. CVXPortfolio separates
portfolio policies, objective terms, constraints, and simulators. PyPortfolioOpt
separates expected-return models, risk models, objectives, and optimizers. The
same separation is useful here: ChebyshevSharp should expose the Bellman
collocation machinery independently from any single finance product.

## Current API

The main types live in `ChebyshevSharp.DynamicProgramming`.

| Type | Role |
| --- | --- |
| `BellmanAction` | One discrete action, with an index, label, and optional scalar value. |
| `BellmanStepContext` | Step metadata passed into the action-value callback. |
| `FiniteHorizonBellmanProblem` | State domain, node count, action grid, terminal value, and action-value callback. |
| `FiniteHorizonBellmanSolver` | Builds Chebyshev action-value branches backward from the terminal value. |
| `FiniteHorizonBellmanModel` | Reusable solved model for online evaluation. |
| `FiniteHorizonBellmanStepSolution` | Action-value branches for one time step. |
| `BellmanEvaluation` | Value, selected action, first derivative, and second derivative. |

The action-value callback is the important user-supplied piece:

```csharp
(context, state, action, nextValue) => actionValue
```

`nextValue` is the already-solved value function for the next time step. The
solver fits one Chebyshev approximation for each action at each step, then
chooses the largest branch during evaluation.

Because `BellmanEvaluation` always reports first and second derivatives of the
active action-value branch, `FiniteHorizonBellmanProblem` requires
`MaxDerivativeOrder >= 2`.

## Example 1: Dynamic Asset Allocation

This example treats wealth as the state and the risky-asset weight as the
action. The action grid is intentionally simple:

```csharp
BellmanAction[] allocations =
[
    new(0, "cash", 0.0),
    new(1, "balanced", 0.5),
    new(2, "risky", 1.0),
];
```

The terminal utility is \(U(W)=\sqrt{W}\). At each step, the action value is the
expected next-step value under a two-state return model:

$$
Q_i(W,w)
=
\mathbb{E}\left[
V_{i+1}\left(
W\left((1-w)(1+r_f)+w(1+R)\right)
\right)
\right].
$$

The runnable code is:

```bash
dotnet run --project examples/ContinuousStateDynamicProgramming/ContinuousStateDynamicProgramming.csproj
```

Representative output:

```text
Continuous-state DP: dynamic asset allocation
Wealth | value | action | dV/dWealth
  50.0 |  7.60083 | risky    |   0.075610
 100.0 | 10.74943 | risky    |   0.053848
 150.0 | 13.16192 | risky    |   0.043399
 200.0 | 15.08648 | risky    |   0.031351
Build evaluations: 279
```

This is the general dynamic-allocation pattern: state, action grid, transition
model, terminal utility, and policy extraction. A production allocation model
would add richer return models, constraints, transaction costs, and a larger
action grid or continuous optimizer.

## Example 2: Stochastic Growth

This example follows the computational-economics pattern used in QuantEcon-style
dynamic-programming tutorials. Capital is the state, the action is a savings
rate, and productivity has two shocks.

```csharp
BellmanAction[] savingsRates =
[
    new(0, "save-20", 0.20),
    new(1, "save-40", 0.40),
    new(2, "save-60", 0.60),
];
```

With production \(y=k^\alpha\), consumption \(c=(1-s)y\), and next capital
\(k'=\epsilon sy\), the action-value function is:

$$
Q_i(k,s)
=
\log((1-s)k^\alpha)
+\beta\mathbb{E}[V_{i+1}(\epsilon s k^\alpha)].
$$

The same example command prints:

```text
Continuous-state DP: stochastic growth
Capital | value | action | dV/dCapital
    1.0 | -1.36661 | save-20  |    0.350000
    2.0 | -1.12401 | save-20  |    0.175000
    4.0 | -0.88141 | save-20  |    0.087500
    6.0 | -0.73949 | save-20  |    0.058333
Build evaluations: 396
```

The purpose is not to be a full economic model. The point is to show that the
same Chebyshev Bellman API is not tied to option exercise.

## Example 3: American Option Optimal Stopping

The American-option case study is a finance example of the same pattern. Spot
is the state and the two actions are:

```text
exercise -> receive payoff h(S)
continue -> receive discounted expected future value
```

The Bellman recursion is:

$$
C_i(S)
=
e^{-r\Delta t}
\mathbb{E}[V_{i+1}(S_{i+1})\mid S_i=S],
\qquad
V_i(S)=\max(h(S), C_i(S)).
$$

The case study keeps the stopping decision exact and applies Chebyshev
interpolation to the continuation function. See
[American Option Dynamic Chebyshev Case Study](american-option-dynamic-chebyshev.md)
for the full baseline comparison against QLNet, Longstaff-Schwartz regression,
and LSPI.

Run it with:

```bash
dotnet run --project examples/AmericanOptionDynamicChebyshev/AmericanOptionDynamicChebyshev.csproj
```

## Validation Checklist

For any continuous-state Bellman model:

1. Validate value and policy away from Chebyshev construction nodes.
2. Check monotonicity or concavity when the economics imply it.
3. Confirm policy switches are stable under node-count changes.
4. Compare against a known solution, quant-library oracle, or independent
   discretization when available.
5. Inspect derivatives only on smooth active branches; derivative jumps can
   occur where the optimal action changes.
6. Treat state-domain edges carefully. The solver clamps internal next-state
   lookups to the fitted domain, while public model evaluation rejects states
   outside the domain.

## Current Limits

This first API is intentionally small. It does not yet support:

- multi-dimensional state;
- continuous-action optimization;
- infinite-horizon fixed-point iteration;
- explicit residual-curve reporting;
- adaptive state-domain expansion;
- vector-valued rewards or constraints.

Those are natural future extensions, but the current API already gives a
general pattern for finite-horizon continuous-state problems with a discrete
action grid.

## Sources

See [Citations](citations.md) for QuantEcon, `ContinuousDPs.jl`, collocation,
CVXPortfolio, PyPortfolioOpt, and American-option references.
