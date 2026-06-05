---
title: From MDPs to Chebyshev Collocation
---

# From MDPs to Chebyshev Collocation: a Map of Methods

This page is the theory companion to the case studies. It builds the general picture first —
Markov decision processes, the Bellman equation, and how it is solved — and then specialises
to the setting ChebyshevSharp targets: a finite-horizon problem on a continuous state, solved
by Chebyshev collocation. Along the way it places four ways of representing a value function
(basis functions, regression, neural networks, and discretisation) on a single axis, and gives
a direct comparison of Chebyshev collocation against finite differences.

The worked numbers and the runnable harness live in the
[American Option Dynamic Chebyshev Case Study](american-option-dynamic-chebyshev.md); read this
page first for the *why*, then that page for the empirical study.

## Notation and conventions

A **Markov decision process** (MDP) is a tuple $(\mathcal S, \mathcal A, P, r, \gamma)$: a state
space $\mathcal S$, an action space $\mathcal A$, a transition law $P(s'\mid s,a)$, a reward
$r(s,a)$, and a discount factor $\gamma\in(0,1]$. A **policy** $\pi$ maps each state to an action.
The defining property is *Markovianity*: the next state depends on the present state and action
only, not on the path that reached it.

Two value functions recur throughout, and the rest of the page assumes the distinction:

- The **state-value** $V^\pi(s)$ is the expected discounted reward of *following policy $\pi$
  from state $s$ onward*.
- The **action-value** $Q^\pi(s,a)$ is the expected discounted reward of *taking action $a$ for
  one step and then following $\pi$*. It scores actions directly, so the greedy policy is simply
  $\pi(s)=\arg\max_a Q(s,a)$, and $V(s)=\max_a Q(s,a)$ under an optimal policy.

We write $V^\star$ and $Q^\star$ for the optimal value functions. "Prediction" means evaluating a
*fixed* policy ($V^\pi$); "control" means finding the best policy ($V^\star$). These are different
problems with different machinery, and conflating them is the first source of confusion this page
removes.

## Part 1 — The general problem: infinite-horizon MDPs on discrete states

Start with the cleanest case: finitely many states, infinitely many time steps, discount
$\gamma<1$. Consider a machine-maintenance process with condition states
$\mathcal S=\{1,2,3\}$ (new, worn, failed) and actions $\mathcal A=\{\text{operate},\text{replace}\}$.
Operating earns a reward that falls as the machine wears and moves it stochastically toward higher
wear; replacing pays a cost and resets the condition to state $1$. Everything is a finite table:
$P$ is a stack of $3\times 3$ stochastic matrices, $r$ is a $3\times 2$ array.

The value of a fixed policy $\pi$ satisfies the **Bellman expectation equation**

$$
V^\pi(s) = r(s,\pi(s)) + \gamma \sum_{s'} P(s'\mid s,\pi(s))\, V^\pi(s'),
$$

which, with finitely many states, is a linear system $V^\pi = r^\pi + \gamma P^\pi V^\pi$ — solvable
exactly by the matrix inverse $V^\pi = (I-\gamma P^\pi)^{-1} r^\pi$. The *optimal* value satisfies
the **Bellman optimality equation**

$$
V^\star(s) = \max_{a}\Bigl[\, r(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, V^\star(s') \,\Bigr],
$$

which is *not* linear: the $\max$ couples a hidden combinatorial choice of which action is active
at each state. The optimality principle — *acting greedily with respect to $V^\star$ is optimal* —
is a statement about $V^\star$, not an algorithm; the work is in obtaining $V^\star$.

The reason this discrete problem is exactly solvable is **enumerability**, not any special property
of the Bellman equation. The equation is identical in the continuous case; what changes later is the
*representation* of $V$, not the recursion.

## Part 2 — Solving it: the Bellman operator, contraction, value and policy iteration

Collect the optimality update into an operator $\mathcal T$ on value functions,

$$
(\mathcal T V)(s) = \max_a\Bigl[\, r(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,V(s')\,\Bigr],
$$

so that $V^\star$ is the **fixed point** $V^\star = \mathcal T V^\star$. With $\gamma<1$, $\mathcal T$
is a **$\gamma$-contraction** in the supremum norm,

$$
\lVert \mathcal T V_1 - \mathcal T V_2 \rVert_\infty \le \gamma\,\lVert V_1 - V_2 \rVert_\infty ,
$$

so the Banach fixed-point theorem guarantees a *unique* $V^\star$ and geometric convergence of the
iterates: $\lVert V_n - V^\star\rVert_\infty \le \gamma^{\,n}\lVert V_0 - V^\star\rVert_\infty$.

Two algorithms find the fixed point:

- **Value iteration** applies $\mathcal T$ repeatedly. It inherits the contraction's geometric but
  *asymptotic* convergence — it approaches $V^\star$ at rate $\gamma$ and never lands exactly.
- **Policy iteration** alternates two cleaner steps: *evaluate* the current policy exactly
  (the linear solve $V^\pi=(I-\gamma P^\pi)^{-1}r^\pi$) and *improve* it greedily,
  $\pi'(s)=\arg\max_a Q^\pi(s,a)$. Because there are finitely many policies and each improvement is
  strictly better until none exists, policy iteration **terminates exactly in finitely many steps**.

The two convergence guarantees are different in kind: value iteration converges by *contraction*
(Banach, asymptotic); policy iteration converges by *finite termination* (a finite policy set climbed
monotonically). Value iteration is the special case of policy iteration whose evaluation step is
truncated to a single sweep (generalised policy iteration).

Both algorithms exist because the infinite horizon makes the problem **cyclic**: a state can recur
($s_1\to s_2\to s_1$), so $V$ references itself and a fixed point is unavoidable. The next part removes
exactly this property.

## Part 3 — Finite horizon, continuous state: why the table dies

Two changes take us from Part 2 to the option setting, and they are independent.

**Finite horizon makes the problem acyclic.** With a terminal time $N$ and a known terminal value
$V_N$, augment the state with time, $x=(i,S)$. Time only moves forward, so the dependency graph is a
directed acyclic graph and the optimality recursion solves in one ordered pass — *backward induction*:

$$
V_i(S) = \max_a\Bigl[\, r + \gamma\, \mathbb{E}\!\left[V_{i+1}(S')\mid S,a\right]\Bigr],
\qquad i=N-1,\dots,0,\quad V_N \text{ given.}
$$

Backward induction *is* value iteration, specialised to a finite horizon: the terminal anchor $V_N$
replaces the role of the Banach fixed point, and the acyclic time graph collapses the
iterate-to-tolerance loop into exactly $N$ sweeps. There is no contraction argument and no convergence
check — the recursion simply terminates. This is why a finite-horizon American option, though governed
by a Bellman equation, needs neither value iteration to a fixed point nor the Banach theorem.

**Continuous state makes the table impossible.** When $S$ ranges over an interval, there are infinitely
many states and any exact value has probability zero, so the expectation
$\mathbb{E}[V_{i+1}(S')\mid S]$ is an *integral* against a density, not a finite sum over successors.
You cannot store one number per state. The only escape is to **represent** $V_i$ with finitely many
parameters and let that representation generalise across the continuum. The continuous integral — not
ignorance of the law, and not the absence of a fixed point — is what forces an approximation: under a
known model the density is available in closed form; what has no closed form is the *recursive* value,
because the $\max$ introduces an endogenous switching boundary located only by a transcendental crossing.

**Optimal stopping is the minimal instance.** An American option is the smallest non-trivial MDP:
two actions, *exercise* (absorbing) and *continue*. One action-value is free,

$$
Q_i(S,\text{exercise}) = h(S)\quad\text{(the payoff, known exactly)},
$$

and the other is the only unknown,

$$
Q_i(S,\text{continue}) = e^{-r\Delta t}\,\mathbb{E}\!\left[V_{i+1}(S')\mid S\right] =: C_i(S),
$$

with $V_i(S)=\max\bigl(h(S),C_i(S)\bigr)$. The continuation value $C_i$ is smooth; the exercise
boundary $B_i$ is the spot where the $\arg\max$ flips. Pricing the option is therefore exactly the
problem of *representing the smooth $C_i$* inside backward induction.

## Part 4 — Representing the value function: four families

All four methods below run the *same* backward recursion. They differ only in four choices: the
function family used for $C_i$, where the fit targets come from, whether the transition law is used,
and what smoothness is assumed. The data are not the teacher; the Bellman equation is the teacher —
every method is a way to carry its solution across a continuum.

### (a) Basis functions: Chebyshev collocation

Write the continuation value as a finite Chebyshev expansion on $N$ fixed nodes,

$$
C_i(S) \approx \sum_{k=0}^{N-1} a_k\,T_k(S),
$$

and fix the coefficients by forcing the polynomial through the model-computed values at the nodes —
a **square** linear solve $\Phi a = b$, *interpolation* rather than regression. ChebyshevSharp uses
Type-I Chebyshev nodes (the roots of $T_N$, no endpoints; see
[Mathematical Concepts](concepts.md)). The node locations never move across steps; only the values on
them change as time-to-maturity changes.

*Why it helps.* Forcing a high-degree polynomial through equispaced points oscillates wildly near the
ends (Runge's phenomenon); Chebyshev nodes cluster toward the endpoints and tame this (their Lebesgue
constant grows only logarithmically). For a function analytic in a Bernstein ellipse, the interpolation
error then decays *geometrically*, $O(\rho^{-N})$, where $\rho$ measures the distance to the nearest
singularity — spectral accuracy. A handful of well-placed nodes can match a fine grid, which is why the
case study resolves the continuation value with $81$ nodes. Because the representation is a polynomial,
$\Delta$ and $\Gamma$ are available analytically at any spot, with no re-differencing.

**Two node sets in two dimensions.** A persistent ambiguity is worth settling before anything else:
there are two distinct grids, in two distinct roles.

| Node set | Dimension | Role |
| --- | --- | --- |
| Chebyshev nodes ($\sim 81$) | state / spot $S$ | collocation points that *build* the interpolant $C_i(S)$ |
| Gauss–Hermite nodes ($\sim 8$) | shock $Z$ | quadrature points that *evaluate the expectation* at one Chebyshev node |

The per-step work is an outer loop over Chebyshev nodes and an inner loop over Gauss–Hermite nodes.

#### The compute-once transition kernel

The conditional expectation has structure that can be precomputed. Suppose the next-step value is
already a Chebyshev expansion, $V_{i+1}(S)=\sum_k c_k T_k(S)$. Because expectation is linear, push it
through the sum:

$$
Q_i(S_j,\text{hold})
= e^{-r\Delta t}\,\mathbb{E}\!\left[V_{i+1}(S')\mid S_j\right]
= e^{-r\Delta t}\sum_k c_k\,\mathbb{E}\!\left[T_k(S')\mid S_j\right]
= \sum_k \Gamma_{jk}\,c_k,
$$

$$
\Gamma_{jk} := e^{-r\Delta t}\,\mathbb{E}\!\left[T_k(S')\mid S_j\right].
$$

The entire expectation step collapses to one matrix–vector product, $Q_{\text{hold}}=\Gamma c$.

*Why $\Gamma$ is computed once and reused.* $\Gamma_{jk}$ depends only on the fixed node $S_j$, the
fixed basis polynomial $T_k$, and the one-step law $S_j\to S'$. Under constant $\Delta t$ and
time-homogeneous dynamics (geometric Brownian motion, local volatility $\sigma(S)$, constant-parameter
Heston, Lévy models), that law is the *same* distribution at every step — it carries no calendar-time
index and no payoff. Hence $\mathbb{E}[T_k(S')\mid S_j]$ is literally the same number at step $70\to71$
as at step $1\to2$. $\Gamma$ encodes only dynamics and grid, so one matrix serves every backward step
*and* every contract (any strike, maturity, or revaluation), amortising across the whole risk loop.

*Why it is cheap.* The expensive work — filling the $N\times N$ matrix, roughly $N^2$ Gauss–Hermite
quadratures against the known density — happens once; it is the only place the model is touched. Each
online step is then $Q_{\text{hold}}=\Gamma c$ (a dense $O(N^2)$ matrix–vector product), the exercise
decision $V=\max(h,Q_{\text{hold}})$ ($O(N)$, pointwise), and a transform back to coefficients for the
next step ($O(N\log N)$, a DCT). There is no linear solve, no PDE projection, and no re-integration.
Finite differences, by contrast, pay a (tridiagonal) solve plus an early-exercise projection at every
step and cannot reuse work across contracts.

*Coordinate systems.* Node values $v_j$ (heights at the nodes) and coefficients $c_k$ (amounts of each
basis polynomial) describe the same curve in two bases. $\Gamma$ must act on *coefficients*, because
$\mathbb{E}[T_k\mid S_j]$ multiplies "how much $T_k$ is present," which is $c_k$, not the height $v_j$.
One may keep $\Gamma$ as a moment matrix on $c$ and convert with a DCT each step, or fold the conversion
in: the fused operator $\Gamma_{\text{node}} = e^{-r\Delta t}\,\Gamma_{\text{mom}}\,\Phi^{-1}$ acts
directly on node values, with the values-to-coefficients transform $\Phi^{-1}$ baked in. These are two
factorisations of one operator.

*Why re-interpolation is unavoidable.* The kernel machinery lives in coefficient space, but the
exercise decision is a pointwise $\max$ that kinks $V$ and destroys the polynomial structure $\Gamma$
needs as input. Each step therefore applies the $\max$ in node space, then transforms back to fresh
coefficients. The linear dynamics live in coefficient space, the non-linear $\max$ in node space, and
the DCT carries between them. Non-smoothness here is a bounded cost (a reduced convergence rate and more
sensitive Greeks near the boundary), not a validity failure — interpolation through $N$ finite values
always exists.

> **Scope.** The precomputed-$\Gamma$ offline/online split above is the algorithm of Glau, Mahlstedt &
> Pötz (2019). ChebyshevSharp's case study implements the same underlying idea — Chebyshev interpolation
> of the continuation value in a backward recursion with the $\max$ applied exactly — using the more
> direct variant: it Gauss–Hermite-quadratures $V_{i+1}$ (the full value, including the $\max$) at the
> mapped next-spots every step rather than precomputing a $\Gamma$ of basis-polynomial moments. The
> derivation explains why the kernel form is the efficient target; it is not a claim that the library
> currently builds $\Gamma$.

### (b) Regression: Longstaff–Schwartz (LSM)

The same Chebyshev (or monomial) basis can be used, but with the targets generated by Monte Carlo
simulation rather than quadrature, and the fit done by *overdetermined least squares* on in-the-money
paths. Because simulated targets are noisy, the coefficients are obtained by regression, which averages
the noise out; interpolation would reproduce it faithfully, which is fatal. The rule of thumb is exact:
*clean targets are interpolated, noisy targets are regressed*. LSM is **model-free** in the operative
sense — swap geometric Brownian motion for Heston or Merton and the algorithm is unchanged; it only
needs a simulator, not the density. It is the method of choice when the law is unknown or the state is
high-dimensional and path-dependent.

### (c) Neural networks: deep optimal stopping / fitted-Q

Replace the polynomial basis with a neural network $V_\theta$ and fit $\theta$ by stochastic gradient
descent on sampled targets — LSM's regression step with a universal approximator in place of the basis.
This is the high-dimensional, unknown-model end of the spectrum: it needs no smoothness and scales to
many factors, but is data-hungry and forfeits the convergence and error guarantees the other families
carry. The reference point in this lineage is Becker, Cheridito & Jentzen (2019), "Deep optimal
stopping." ChebyshevSharp does not implement a neural representation; it is included here to complete
the structure-versus-flexibility axis.

### (d) Discretisation: finite differences

Finite differences fit a *low-order* polynomial through a few *local* neighbours — a three-point stencil
is a parabola through a point and its two neighbours — and discretise the *differential* (PDE) form of
the problem rather than the *integral* form. Each step is a (tridiagonal) linear solve plus an
early-exercise projection. This is the mature, robust choice and is exactly what the case study uses for
its trusted oracle (a $300\times300$ QLNet grid).

### Chebyshev collocation versus finite differences

Both methods fit a polynomial through grid values and read derivatives or expectations off it; they sit
at opposite extremes of that single idea, and both require a known transition law.

| | Finite differences | Chebyshev collocation |
| --- | --- | --- |
| Polynomial | many local, low-order stencils | one global, high-order interpolant |
| Coupling | local (a point hears its neighbours); sparse, banded | global (every node couples); dense |
| Convergence | algebraic, $O(h^p)$ — halve the spacing for a digit | spectral, $O(\rho^{-N})$ for smooth functions |
| Form discretised | differential (PDE), needs a per-step solve | integral (expectation), a per-step matrix–vector product |
| Greeks | re-difference the grid | analytic from the polynomial |
| Node count (this problem) | hundreds ($300$) | tens ($81$) |
| Robust to kinks | yes | no — needs smoothness, hence the smooth/non-smooth split |

The objection that collocation is "just a simplified finite-difference grid" inverts the relationship:
both share the skeleton of *discretise the state and iterate the Bellman recursion backward*, but
collocation is the higher-order, derivative-free, globally coupled method on that skeleton — the
opposite of simplified. The shared skeleton is also why trees, finite differences, and collocation are
relatives, not the same method. The smooth/non-smooth split is not a convenience but the precondition
that makes spectral accuracy attainable: the American payoff's kink lies on the interval and collapses
the Bernstein ellipse, so the method interpolates only the smooth continuation value and keeps the
non-smooth $\max$ exact and outside the polynomial.

## Part 5 — Comparing the methods

The four families answer one question — how to represent $C_i$ inside backward induction — and trade
structure for flexibility:

| Method | Representation | Targets | Needs known model? | Smoothness used | Convergence | Behaviour as dimension grows | Deterministic? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Finite differences | local stencils | PDE coefficients | yes | low-order | algebraic | grid cost $n^d$ | yes |
| Regression (LSM) | basis least-squares | simulated paths | no | none required | $O(1/\sqrt M)$ | scales well | no |
| Neural network | network weights | simulated paths | no | none required | problem-dependent | scales well | no |
| Chebyshev collocation | global polynomial | quadrature on known density | yes | exploited (spectral) | spectral for smooth | tensor cost $n^d$ | yes |

A decision rule on three axes follows directly:

- **Known model, low-dimensional, smooth continuation** — Chebyshev collocation. Spectral accuracy and
  analytic Greeks for very few parameters.
- **Known model, but a genuine PDE / non-smooth or local-in-calendar-time** — finite differences. Mature
  and robust where the smoothness precondition fails.
- **Unknown model or high-dimensional** — regression (LSM) or a neural network. Sampling buys
  model-agnosticism and dimensional reach at the cost of noise and lost guarantees.

The empirical payoff lives in the
[American Option Dynamic Chebyshev Case Study](american-option-dynamic-chebyshev.md), which prices a
one-dimensional American put and reports, on the same problem, a QLNet finite-difference oracle
($6.088238$), Longstaff–Schwartz ($6.080847$), an LSPI reinforcement-learning baseline ($5.745344$), and
the Dynamic Chebyshev solver ($6.083607$, within $0.004631$ of the oracle), the last evaluating price,
$\Delta$, and $\Gamma$ online in microseconds after an $81$-node, $80$-step build. That evidence covers
the regime where this page's argument is sharpest — known model, one dimension, smooth continuation. The
neural-network row and the high-dimensional cells above are reasoned from the methods' structure rather
than measured here; ChebyshevSharp has no neural implementation, and the dimensional-scaling claims for
collocation are addressed by the [Tensor Train](tensor-train.md) compression rather than by a dense grid.

## References

Foundational and method references are collected on the [Citations](citations.md) page; the entries most
relevant here are Bellman (1957), Bertsekas (*Dynamic Programming and Optimal Control*), Puterman (1994),
Sutton & Barto (2018) and Lagoudakis & Parr (2003) for MDPs and reinforcement learning; Judd (1998) and
Miranda & Fackler (2002) for numerical dynamic programming and collocation in economics;
Longstaff & Schwartz (2001) for LSM; Becker, Cheridito & Jentzen (2019) for deep optimal stopping;
Glau, Mahlstedt & Pötz (2019) for the Dynamic Chebyshev method; and Berrut & Trefethen (2004) and
Trefethen (2013/2019) for the barycentric and spectral-convergence background.
