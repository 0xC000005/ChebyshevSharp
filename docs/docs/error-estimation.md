---
title: Error Estimation
---

# Error Estimation

## Introduction

After building a Chebyshev interpolant, you often want to know how accurate it is without comparing against the true function at thousands of test points. `ErrorEstimate()` provides an **ex ante** estimate of the interpolation error using only the Chebyshev coefficients already computed during the build step.

This is useful for:

- **Validating node counts** -- check whether your grid is fine enough before deploying.
- **Adaptive refinement** -- increase nodes in dimensions where the error is large.
- **Confidence reporting** -- attach an approximate error magnitude to interpolated values.

## Quick Start

```csharp
using ChebyshevSharp;

double f(double[] x, object? data)
{
    return Math.Sin(x[0]) * Math.Cos(x[1]);
}

var cheb = new ChebyshevApproximation(f, 2, new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } }, new[] { 15, 15 });
cheb.Build(verbose: false);
Console.WriteLine($"Error estimate: {cheb.ErrorEstimate():E2}");
```

No extra function evaluations are needed -- the estimate is computed from the tensor of function values that `Build()` already stored.

## Mathematical Background

### Chebyshev Series Expansion

Any sufficiently smooth function on $[-1, 1]$ can be expanded in Chebyshev polynomials:

$$f(x) = \sum_{k=0}^{\infty} c_k\, T_k(x)$$

where $T_k$ is the Chebyshev polynomial of the first kind of degree $k$, and $c_k$ are the expansion coefficients. When we interpolate with $n$ nodes, we compute a degree-$(n{-}1)$ polynomial that implicitly truncates this series:

$$p_n(x) = \sum_{k=0}^{n-1} \hat{c}_k\, T_k(x)$$

The interpolation error $f(x) - p_n(x)$ comes from two sources: (1) the omitted high-degree terms $c_n, c_{n+1}, \ldots$ and (2) aliasing, where these omitted terms fold back onto the computed coefficients. For well-resolved functions, both sources are small when the trailing coefficients are small.

### Why the Last Coefficient Estimates the Error

For a function analytic in a [Bernstein ellipse](concepts.md#bernstein-ellipse) with parameter $\rho > 1$, the Chebyshev coefficients satisfy $|c_k| = O(\rho^{-k})$. This means each successive coefficient is roughly $\rho$ times smaller than the previous one. The last included coefficient $|c_{n-1}|$ is therefore:

1. **An upper bound on the omitted tail.** Since $|c_k| \leq M \rho^{-k}$ and the tail sum $\sum_{k=n}^{\infty} |c_k|$ is a geometric series with ratio $\rho^{-1} < 1$, we have $\sum_{k=n}^{\infty} |c_k| \lesssim |c_{n-1}| / (\rho - 1)$. When $\rho$ is even moderately large (say, $\rho > 2$), the omitted tail is comparable in magnitude to $|c_{n-1}|$ itself.

2. **A proxy for aliasing error.** The aliased contributions (omitted terms folding onto lower coefficients) are bounded by the same geometric decay, so they are also $O(|c_{n-1}|)$ for well-resolved functions.

The practical rule: **if $|c_{n-1}|$ is small, both the truncation and aliasing errors are small, and the interpolant is well-converged.**

> **Warning: Heuristic, not a formal bound.**
> This estimate is an empirically reliable proxy, not a rigorous upper bound. Ruiz & Zeron (2021, Section 3.4) report that they have never encountered a real-world case where small trailing coefficients failed to indicate convergence. However, pathological functions (e.g., those with singularities just outside the Bernstein ellipse) could have slowly decaying coefficients that make the estimate optimistic. Always validate against known solutions when possible.

### Computing Coefficients via DCT-II

ChebyshevSharp uses **Type I Chebyshev nodes** (roots of $T_n$, also called Gauss--Chebyshev nodes; see Trefethen 2013, Ch. 3):

$$x_i = \cos\!\left(\frac{(2i - 1)\,\pi}{2n}\right), \quad i = 1, \ldots, n$$

For values sampled at these $n$ nodes, the Chebyshev expansion coefficients $c_k$ can be recovered exactly via the **Discrete Cosine Transform (DCT-II)** (Good 1961; Trefethen 2013, Ch. 3).

**Why DCT-II works.** The connection comes from the orthogonality of Chebyshev polynomials. Evaluating $T_k(x_i)$ at the Type I nodes and exploiting the identity $T_k(\cos\theta) = \cos(k\theta)$ turns the coefficient formula into a discrete cosine sum -- precisely the DCT-II. The computation runs in $O(n \log n)$ via FFT.

In practice, the implementation:

1. **Reverses** the node-order values because ChebyshevSharp stores nodes in ascending order, while the DCT expects descending (cosine) order.
2. **Divides** by $n$ (the normalization factor for the DCT-II).
3. **Halves** the zeroth coefficient ($c_0 \mathrel{/}= 2$) because the Chebyshev series convention uses $\frac{c_0}{2} T_0(x) + c_1 T_1(x) + \cdots$, while the raw DCT includes the full $c_0$.

ChebyshevSharp uses an FFT-based DCT-II (via MathNet Numerics Fourier transform) for $n > 32$ nodes, falling back to a direct $O(n^2)$ summation for small $n$ where FFT overhead would dominate.

## Multi-Dimensional Error

For a $D$-dimensional interpolant on a tensor grid with $n_d$ nodes in dimension $d$, the error estimate generalizes as follows:

1. **Extract 1-D slices.** For each dimension $d$, fix all other indices and extract every 1-D slice of the tensor along dimension $d$.
2. **Compute per-slice error.** For each slice, compute the Chebyshev coefficients via DCT-II and take the magnitude of the last coefficient $|c_{n_d - 1}|$.
3. **Maximize over slices.** Take the maximum $|c_{n_d - 1}|$ across all slices for dimension $d$. This worst-case slice represents the hardest-to-approximate 1-D cross-section of the function along that axis.
4. **Sum across dimensions.** The total error estimate is the sum of per-dimension maxima:

$$\hat{E} = \sum_{d=1}^{D} \max_{\text{slices along } d} |c_{n_d - 1}|$$

**Why sum across dimensions?** ChebyshevSharp evaluates a multi-dimensional interpolant via *dimensional decomposition* -- contracting one axis at a time (Berrut & Trefethen 2004; see also [Multi-Dimensional Extension](concepts.md#multi-dimensional-extension)). At each contraction step, the 1-D interpolation error for that dimension is injected into the remaining computation. In the worst case, these per-dimension errors add up. The summation therefore represents a **conservative heuristic**: the total error is at most the sum of the worst per-dimension errors, assuming the errors do not cancel.

> **Note: Not a tight bound.**
> In practice, $\hat{E}$ is often pessimistic because: (a) the worst-case slice rarely coincides across all other index combinations, and (b) errors from different dimensions tend to partially cancel rather than align. The estimate is best used as an order-of-magnitude indicator, not as a precise bound.

## ChebyshevSpline Error Estimation

For `ChebyshevSpline`, `ErrorEstimate()` returns the **maximum** error estimate across all pieces:

```csharp
using ChebyshevSharp;

double absFunc(double[] x, object? data) => Math.Abs(x[0]);

var spline = new ChebyshevSpline(
    function: absFunc,
    numDimensions: 1,
    domain: new[] { new[] { -1.0, 1.0 } },
    nNodes: new[] { 15 },
    knots: new[] { new[] { 0.0 } }  // knot at the kink
);
spline.Build(verbose: false);
Console.WriteLine($"Spline error estimate: {spline.ErrorEstimate():E2}");
```

Each piece is an independent `ChebyshevApproximation` on a smooth sub-interval. The piece with the largest error estimate determines the overall estimate, since that piece limits the accuracy of the full spline.

## ChebyshevSlider Error Estimation

`ChebyshevSlider` also supports `ErrorEstimate()`. The slider error is the **sum** of the error estimates from each individual slide:

```csharp
using ChebyshevSharp;

double f(double[] x, object? data)
{
    return Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);
}

var slider = new ChebyshevSlider(
    function: f,
    numDimensions: 3,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11, 11 },
    partition: new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
    pivotPoint: new[] { 0.0, 0.0, 0.0 }
);
slider.Build();
Console.WriteLine($"Slider error estimate: {slider.ErrorEstimate():E2}");
```

> **Warning: Cross-group interaction error.**
> The slider error estimate captures **per-slide interpolation error only** -- the error from approximating each slide's low-dimensional function with Chebyshev polynomials. Error from the additive sliding *decomposition itself* (i.e., the cross-group coupling that the sliding formula ignores) is **not** included. For example, if $f(x_1, x_2) = x_1 \cdot x_2$ and the partition is `[[0], [1]]`, the sliding approximation is structurally unable to represent the product term, regardless of node count. The error estimate will report near-zero (each 1-D slide is well-resolved), but the true error can be large. For strongly coupled functions, always validate with test points.

## Convergence Example

As the number of nodes increases, the error estimate should decrease -- rapidly for smooth functions. This example demonstrates spectral convergence for a 1-D function:

```csharp
using ChebyshevSharp;

double f(double[] x, object? data) => Math.Sin(x[0]);

for (int n = 5; n <= 30; n += 5)
{
    var cheb = new ChebyshevApproximation(f, 1, new[] { new[] { -1.0, 1.0 } }, new[] { n });
    cheb.Build(verbose: false);
    Console.WriteLine($"n={n,2}: error_estimate = {cheb.ErrorEstimate():E2}");
}
```

For $\sin(x)$, which is entire (analytic everywhere in the complex plane), the coefficients decay exponentially, so you should see the estimate drop by several orders of magnitude as $n$ grows.

## Class Support

| Class | How `ErrorEstimate()` Works |
|-------|---------------------------|
| `ChebyshevApproximation` | Per-dimension DCT-II of 1-D slices; maximize over slices per dimension; sum across dimensions. |
| `ChebyshevSpline` | Computes `ErrorEstimate()` for each piece; returns the **maximum** across all pieces. |
| `ChebyshevSlider` | Computes `ErrorEstimate()` for each slide; returns the **sum** across all slides. Does not capture decomposition error. |

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501--517.
- Good, I. J. (1961). "The Colleague Matrix, a Chebyshev Analogue of the Companion Matrix." *The Quarterly Journal of Mathematics* 12(1):61--68.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.* Wiley Finance. Section 3.4.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM. Chapters 3--4.
