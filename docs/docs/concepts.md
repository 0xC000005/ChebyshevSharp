---
title: Mathematical Concepts
---

# Mathematical Concepts

This page covers the mathematical foundations behind ChebyshevSharp. Understanding these concepts is not required to use the library, but they explain *why* Chebyshev interpolation works so well and how the key algorithms operate.

## Why Chebyshev Interpolation?

Polynomial interpolation with equally-spaced points suffers from **Runge's phenomenon** (Runge 1901) -- wild oscillations near interval endpoints that worsen as polynomial degree increases. Chebyshev nodes solve this by clustering near boundaries:

$$x_i = \cos\left(\frac{(2i-1)\pi}{2n}\right), \quad i = 1, \ldots, n$$

These are the **Type I Chebyshev nodes** (roots of the Chebyshev polynomial $T_n$), also called Gauss--Chebyshev nodes. ChebyshevSharp stores them in ascending order after mapping to the user-specified domain $[a, b]$:

$$\tilde{x}_i = \frac{a + b}{2} + \frac{b - a}{2}\, x_i$$

The Lebesgue constant for Chebyshev nodes grows only logarithmically:

$$\Lambda_n \leq \frac{2}{\pi}\log(n+1) + 1$$

(Trefethen 2013, Ch. 15), versus exponential growth for equidistant points. This means the Chebyshev interpolant is near-optimal: it approximates the function almost as well as the best polynomial of the same degree.

## Spectral Convergence

For functions analytic in a **Bernstein ellipse** with parameter $\rho > 1$, the interpolation error decays exponentially:

$$|f(x) - p_N(x)| = O(\rho^{-N})$$

Each additional node multiplies accuracy by a constant factor $\rho$. This is **spectral convergence** -- qualitatively faster than any fixed-order method (e.g., cubic splines converge as $O(h^4)$, while Chebyshev interpolation converges as $O(\rho^{-N})$ with $\rho > 1$).

### Bernstein Ellipse

A Bernstein ellipse is an ellipse in the complex plane with foci at $x = -1$ and $x = +1$. The parameter $\rho$ equals the sum of the semi-major and semi-minor axis lengths. Functions analytic inside a larger ellipse (larger $\rho$) converge faster.

**Practical implication:** The convergence rate depends on how far the function's nearest singularity (pole, branch cut, discontinuity) is from the real interval $[-1, 1]$ in the complex plane. For example:

- $f(x) = e^x$ is **entire** (no singularities) -- $\rho = \infty$, superexponential convergence. In practice, 15 nodes suffice for machine precision on $[-1, 1]$.
- $f(x) = 1/(1 + 25x^2)$ has **poles at $x = \pm i/5$** -- the Bernstein ellipse must avoid these poles, limiting $\rho$ and slowing convergence. This is the classic Runge function: equidistant interpolation diverges, but Chebyshev interpolation converges steadily.
- **Black-Scholes option prices** are analytic in all parameters over typical domains (spot, volatility, maturity, rate, strike all bounded away from zero). The effective $\rho$ is large, giving rapid convergence with 10--15 nodes per dimension.

For functions with singularities or discontinuities on the real interval itself (e.g., $|x|$ at $x = 0$), the Bernstein ellipse collapses ($\rho \to 1$) and convergence becomes algebraic rather than exponential. In such cases, use `ChebyshevSpline` to place knots at the trouble points and recover spectral convergence on each smooth piece -- see [Piecewise Chebyshev Interpolation](spline.md).

For the full theory, see Trefethen (2013), *Approximation Theory and Approximation Practice*, SIAM, Chapter 8.

## Barycentric Interpolation Formula

The interpolating polynomial is expressed in the **second-form barycentric formula** (Berrut & Trefethen 2004):

$$p(x) = \frac{\displaystyle\sum_{i=0}^{n} \frac{w_i\, f_i}{x - x_i}}{\displaystyle\sum_{i=0}^{n} \frac{w_i}{x - x_i}}$$

where the barycentric weights $w_i = 1 / \prod_{j \neq i}(x_i - x_j)$ depend **only on node positions**, not on function values. This has two important consequences:

1. **Full pre-computation.** The weights are computed once during `Build()` and reused for every evaluation. Only the function values $f_i$ change when updating the interpolant.
2. **Numerical stability.** The ratio form ensures that the interpolant reproduces constant functions exactly and avoids the catastrophic cancellation that plagues the Lagrange form.

When the evaluation point $x$ coincides with a node $x_j$ (within floating-point tolerance), the formula reduces to $p(x) = f_j$ without division by zero -- ChebyshevSharp handles this case explicitly.

The evaluation cost is $O(n)$ per dimension per query point, dominated by the weighted sum.

## Multi-Dimensional Extension

For a $d$-dimensional function, ChebyshevSharp uses **dimensional decomposition**:

1. Start with the full tensor of function values at all node combinations (shape $n_1 \times n_2 \times \cdots \times n_d$).
2. Contract one dimension at a time using barycentric interpolation.
3. Each contraction reduces dimensionality by 1 ($d$D $\to$ $(d{-}1)$D $\to$ $\cdots$ $\to$ scalar).

The key performance insight: each contraction step reshapes the tensor so that the contracting dimension becomes either the leading or trailing axis, turning the operation into a **matrix-vector multiply**. ChebyshevSharp routes these multiplies through BLAS GEMV when the problem is large enough (>256 elements), falling back to hand-rolled loops for small problems where call overhead would dominate.

This avoids the curse of dimensionality in the evaluation step -- query cost scales linearly with the number of dimensions: $O(n_1 + n_2 + \cdots + n_d)$ rather than $O(n_1 \times n_2 \times \cdots \times n_d)$. The build cost (populating the tensor) remains $O(\prod_d n_d)$, but this is a one-time upfront cost.

```csharp
// 3D evaluation: 15 x 12 x 10 = 1,800 node tensor
// Query cost: O(15 + 12 + 10) = O(37) -- not O(1,800)
double value = cheb.VectorizedEval(new[] { 100.0, 0.2, 1.0 }, new[] { 0, 0, 0 });
```

## Analytical Derivatives

Derivatives are computed using **spectral differentiation matrices** $D^{(k)}$ (Berrut & Trefethen 2004, Section 9):

$$D^{(1)}_{ij} = \frac{w_j / w_i}{x_i - x_j} \quad (i \neq j), \qquad D^{(1)}_{ii} = -\sum_{k \neq i} D^{(1)}_{ik}$$

Given function values $\mathbf{f}$ at nodes, $D^{(1)} \mathbf{f}$ gives exact derivative values at those same nodes. These derivative values are then interpolated to the query point using the barycentric formula.

Higher-order derivatives use powers of the differentiation matrix: $D^{(k)} = (D^{(1)})^k$. ChebyshevSharp pre-computes and caches these matrices during `Build()` so that requesting second or third derivatives incurs no extra matrix computation at query time.

**Advantages over finite differences:**

- **Same convergence rate as function values.** The derivative of an $n$-node Chebyshev interpolant is an $(n{-}1)$-node Chebyshev interpolant, so derivatives inherit spectral convergence.
- **No step-size tuning.** Finite differences require choosing a step size that balances truncation error against round-off error. Spectral differentiation has no such trade-off.
- **Exact for the interpolant.** The computed derivative is the exact derivative of the interpolating polynomial, not an approximation to it.

```csharp
// Price and Greeks in one call: the differentiation matrices are pre-computed
double[] results = cheb.VectorizedEvalMulti(
    new[] { 100.0, 0.2, 1.0 },
    new[] {
        new[] { 0, 0, 0 },  // price
        new[] { 1, 0, 0 },  // delta (df/dS)
        new[] { 2, 0, 0 },  // gamma (d^2f/dS^2)
        new[] { 0, 1, 0 },  // vega  (df/dsigma)
    }
);
```

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501--517.
- Runge, C. (1901). "Uber empirische Funktionen und die Interpolation zwischen aquidistanten Ordinaten." *Zeitschrift fur Mathematik und Physik* 46:224--243.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
