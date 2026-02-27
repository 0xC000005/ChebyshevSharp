---
title: Sliding Technique
---

# Sliding Technique

The **Sliding Technique** enables Chebyshev approximation of high-dimensional functions by decomposing them into a sum of low-dimensional interpolants. This sidesteps the curse of dimensionality at the cost of losing cross-group interactions.

## Motivation

A full tensor Chebyshev interpolant on $n$ dimensions with $m$ nodes per dimension requires $m^n$ function evaluations. For $n = 10$ and $m = 11$, that is over 25 billion evaluations — clearly infeasible.

The sliding technique partitions the dimensions into small groups and builds a separate Chebyshev interpolant (a **slide**) for each group, with all other dimensions fixed at a **pivot point**. The total cost becomes the *sum* of the group grid sizes rather than their *product*.

## Algorithm

Given a function $f: \mathbb{R}^n \to \mathbb{R}$, a pivot point $\mathbf{z} = (z_1, \ldots, z_n)$, and a partition of dimensions into $k$ groups [3, Ch. 7]:

1. Evaluate the **pivot value** $v = f(\mathbf{z})$.
2. For each group $i$, build a **slide** $s_i$ — a `ChebyshevApproximation` on the group's dimensions, with all other dimensions fixed at their pivot values.
3. Evaluate using the additive formula:

$$
f(\mathbf{x}) \approx v + \sum_{i=1}^{k} \bigl[ s_i(\mathbf{x}_{G_i}) - v \bigr]
$$

where $\mathbf{x}_{G_i}$ denotes the components of $\mathbf{x}$ belonging to group $i$.

## When to Use Sliding

Sliding works well when:

- The function is **additively separable** or nearly so (e.g., $\sin(x_1) + \sin(x_2) + \sin(x_3)$).
- Cross-group interactions are **weak** relative to within-group effects.
- The number of dimensions is **too large** for full tensor interpolation (say, $n > 6$).

Sliding does **not** work well when:

- Variables in different groups are **strongly coupled** (e.g., Black-Scholes where $S$, $T$, and $\sigma$ interact multiplicatively).
- High accuracy is required far from the pivot point.

> **Alternative: Tensor Train.**
> For general (non-separable) high-dimensional functions, consider `ChebyshevTT` (not yet implemented in ChebyshevSharp -- Phase 4). TT-Cross captures cross-variable coupling that the sliding decomposition misses, at the cost of using finite differences for derivatives instead of analytical spectral differentiation.

> **Choosing the partition.**
> Group variables that have strong non-linear interactions together. For example, if $f = x_1^3 x_2^2 + x_3$, group $(x_1, x_2)$ in one slide and $x_3$ in another.

## Usage

### Basic construction

```csharp
using ChebyshevSharp;

// Additively separable function
double MyFunc(double[] x, object? _)
    => Math.Sin(x[0]) + Math.Sin(x[1]) + Math.Sin(x[2]);

var slider = new ChebyshevSlider(
    function: MyFunc,
    numDimensions: 3,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11, 11 },
    partition: new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },  // each dim is its own slide
    pivotPoint: new[] { 0.0, 0.0, 0.0 }
);
slider.Build();

// Evaluate function value
double val = slider.Eval(new[] { 0.5, 0.3, -0.2 }, new[] { 0, 0, 0 });

// Evaluate derivative w.r.t. x0
double dfdx0 = slider.Eval(new[] { 0.5, 0.3, -0.2 }, new[] { 1, 0, 0 });
```

### Multi-dimensional slides

For functions with within-group coupling, use larger groups:

```csharp
double G(double[] x, object? _)
    => Math.Pow(x[0], 3) * Math.Pow(x[1], 2) + Math.Sin(x[2]) + Math.Sin(x[3]);

var slider = new ChebyshevSlider(
    function: G,
    numDimensions: 4,
    domain: new[] { new[] { -2.0, 2.0 }, new[] { -2.0, 2.0 },
                    new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 12, 12, 8, 8 },
    partition: new[] { new[] { 0, 1 }, new[] { 2 }, new[] { 3 } },  // 2D + 1D + 1D
    pivotPoint: new[] { 0.0, 0.0, 0.0, 0.0 }
);
slider.Build();
```

### Build cost comparison

```csharp
// Full tensor: 12 * 12 * 8 * 8 = 9,216 evaluations
// Sliding:     12*12 + 8 + 8   = 160 evaluations  (57x fewer)
Console.WriteLine($"Slider build evaluations: {slider.TotalBuildEvals}");
```

### Multi-output evaluation

Compute function value and derivatives at the same point:

```csharp
double[] results = slider.EvalMulti(
    new[] { 0.5, 0.3, -0.2, 0.1 },
    new[] {
        new[] { 0, 0, 0, 0 },  // function value
        new[] { 1, 0, 0, 0 },  // d/dx0
        new[] { 0, 0, 1, 0 },  // d/dx2
    }
);
```

## Derivatives

The slider supports analytical derivatives through its slides. Only the slide containing the differentiated dimension contributes:

$$
\frac{\partial}{\partial x_j} f(\mathbf{x}) \approx \frac{\partial}{\partial x_j} s_i(\mathbf{x}_{G_i})
$$

where $j \in G_i$. The pivot value $v$ is constant and drops out.

### Cross-group derivatives are zero

Because slides are independent functions of disjoint variable groups, **mixed partial derivatives across groups are exactly zero**. For example, with partition `[[0, 1], [2]]`:

- $\frac{\partial^2 f}{\partial x_0 \partial x_1}$ — computed within the `[0, 1]` slide (correct)
- $\frac{\partial^2 f}{\partial x_0 \partial x_2}$ — returns 0 ($x_0$ and $x_2$ are in different slides)

This is mathematically correct for the sliding approximation, but may differ from the true function's cross-derivatives. If cross-group sensitivities matter, group those variables together or use full tensor interpolation.

## Error Estimation

`ErrorEstimate()` returns the **sum** of per-slide Chebyshev error estimates. This measures the interpolation error within each slide but does **not** capture the cross-group interaction error inherent in the sliding decomposition.

```csharp
double err = slider.ErrorEstimate();
```

For additively separable functions, the error estimate accurately reflects the total error. For coupled functions, the actual error may be larger due to the sliding approximation itself.

### Accuracy near and far from pivot

The sliding approximation is most accurate near the pivot point. As the evaluation point moves away from the pivot in multiple dimensions simultaneously, cross-coupling errors accumulate. For strongly coupled functions like Black-Scholes, errors of 20--50% at domain boundaries have been observed.

## Serialization

`ChebyshevSlider` supports `Save` and `Load` with the same JSON format used by the other classes:

```csharp
slider.Save("slider.json");
var loaded = ChebyshevSlider.Load("slider.json");
```

The loaded slider is fully functional for evaluation, derivatives, error estimation, extrusion, slicing, and arithmetic. The original function reference is not retained (`Function` is `null` after load).

## Extrusion and Slicing

`ChebyshevSlider` supports the same `Extrude` and `Slice` operations as `ChebyshevApproximation` and `ChebyshevSpline`.

**Extrude** adds new dimensions as singleton slide groups with constant tensor values equal to `PivotValue`. The new dimension contributes nothing to the sliding sum:

```csharp
var slider4d = slider3d.Extrude((3, new[] { 0.0, 5.0 }, 9));
```

**Slice** fixes a dimension at a specific value, reducing dimensionality. For single-dimension groups, the slide is evaluated and absorbed into the remaining slides. For multi-dimension groups, the underlying `ChebyshevApproximation` is sliced:

```csharp
var slider2d = slider3d.Slice((0, 0.5));
```

See [Advanced Usage](advanced-usage.md) for more on extrusion and slicing.

## Arithmetic Operators

Sliders on the same grid support pointwise arithmetic:

```csharp
var sum    = slider1 + slider2;
var diff   = slider1 - slider2;
var neg    = -slider1;
var scaled = slider1 * 2.0;
var halved = slider1 / 2.0;
```

Both operands must have the same dimensions, domain, node counts, **partition**, and **pivot point**. The result's `PivotValue` is computed from the operands' pivot values (e.g., addition adds them).

## Choosing the Right Class

| Scenario | Class | Build Cost |
|----------|-------|-----------|
| Smooth function, low dimensions (1--5D) | `ChebyshevApproximation` | $\prod_i n_i$ |
| Function with discontinuities/singularities | `ChebyshevSpline` | $\text{pieces} \times \prod_i n_i$ |
| High dimensions, additively separable | `ChebyshevSlider` | $\sum_g \prod_{i \in g} n_i$ |
| High dimensions, general coupling | `ChebyshevTT` (planned) | $O(d \cdot n \cdot r^2)$ |

> **Method availability.**
> `ChebyshevSlider` does **not** support `Nodes()`, `FromValues()`, `EvalBatch()`,
> `Integrate()`, `Roots()`, `Minimize()`, or `Maximize()`. These operations require the
> full tensor grid, which the sliding decomposition avoids by design. Use
> `ChebyshevApproximation` or `ChebyshevSpline` for these features.

## References

1. Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM.
2. Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." *SIAM Review* 46(3):501-517.
3. Ruiz, I. & Zeron, M. (2022). *Machine Learning for Risk Calculations: A Practitioner's View.* Wiley Finance.
