---
_layout: landing
---

# ChebyshevSharp

Fast Chebyshev interpolation, derivatives, splines, Slider, Tensor Train, and
finite-horizon dynamic programming tools for .NET.

Use ChebyshevSharp when direct model evaluations are expensive but repeated
values, sensitivities, integrals, roots, or surrogate evaluations need to run
quickly inside a .NET application.

<p>
  <a class="btn btn-primary" href="docs/getting-started.md">Get Started</a>
  <a class="btn btn-default" href="docs/which-class.md">Choose a Class</a>
  <a class="btn btn-default" href="api/ChebyshevSharp.yml">API</a>
</p>

```bash
dotnet add package ChebyshevSharp
```

## What Do You Want to Do?

<div class="row">
  <div class="col-md-6">
    <h3><a href="docs/tutorials.md">Learn by following</a></h3>
    <p>
      Start with guided introductions and worked case studies. Use this path
      when you are new to the library or want to see a complete modelling
      workflow before adapting it.
    </p>
    <ul>
      <li><a href="docs/introduction.md">Read the introduction</a></li>
      <li><a href="docs/getting-started.md">Build your first interpolant</a></li>
      <li><a href="docs/examples.md">Run the example projects</a></li>
    </ul>
  </div>
  <div class="col-md-6">
    <h3><a href="docs/how-to.md">Solve a task</a></h3>
    <p>
      Jump to concrete API patterns for construction, refinement,
      differentiation, serialization, high-dimensional models, and dynamic
      programming.
    </p>
    <ul>
      <li><a href="docs/error-driven-construction.md">Control approximation error</a></li>
      <li><a href="docs/greeks.md">Compute sensitivities and Greeks</a></li>
      <li><a href="docs/tensor-train.md">Build a Tensor Train interpolant</a></li>
    </ul>
  </div>
</div>

<div class="row">
  <div class="col-md-6">
    <h3><a href="docs/concepts.md">Understand the math</a></h3>
    <p>
      Use the concept pages when you need the numerical assumptions behind
      Chebyshev nodes, spectral convergence, interpolation stability,
      piecewise smoothness, Sobol indices, and performance.
    </p>
    <ul>
      <li><a href="docs/concepts.md">Review the mathematical concepts</a></li>
      <li><a href="docs/error-estimation.md">Understand error estimates</a></li>
      <li><a href="docs/performance.md">Read the performance notes</a></li>
    </ul>
  </div>
  <div class="col-md-6">
    <h3><a href="docs/reference.md">Look up details</a></h3>
    <p>
      Use reference pages when you need exact API documentation, validation
      commands, file formats, citations, release history, or contribution
      rules.
    </p>
    <ul>
      <li><a href="api/ChebyshevSharp.yml">Open the API reference</a></li>
      <li><a href="docs/testing-and-validation.md">Check validation commands</a></li>
      <li><a href="docs/changelog.md">Read the release notes</a></li>
    </ul>
  </div>
</div>

## Common Workflows

| If you want to... | Start here |
|-------------------|------------|
| Choose between dense, spline, Slider, and TT models | [Which Class Should I Use?](docs/which-class.md) |
| Build a smooth dense approximation | [Getting Started](docs/getting-started.md) |
| Build from a precomputed Chebyshev grid | [Pre-computed Values](docs/from-values.md) |
| Handle kinks, singularities, or known breakpoints | [Piecewise Chebyshev Interpolation](docs/spline.md) and [Special Points](docs/special-points.md) |
| Work with high-dimensional functions | [Sliding Technique](docs/slider.md) and [Tensor Train Interpolation](docs/tensor-train.md) |
| Compute derivatives, Greeks, integrals, roots, or algebraic combinations | [Computing Greeks](docs/greeks.md), [Calculus](docs/calculus.md), and [Chebyshev Algebra](docs/algebra.md) |
| Solve a finite-horizon Bellman problem | [Continuous-State Dynamic Programming](docs/continuous-state-dynamic-programming.md) |
| Save, load, validate, or benchmark a model | [Serialization](docs/serialization.md), [Testing & Validation](docs/testing-and-validation.md), and [Performance](docs/performance.md) |

## Case Studies

These tutorials are public, reproducible examples of how Chebyshev methods behave
in applied numerical workflows.

- [Fixed-Rate Bond Case Study](docs/fixed-rate-bond-surrogate.md) shows why a
  request-level fixed-rate bond surface should be decomposed around the smooth
  discounting pieces instead of cloned as one global high-dimensional object.
- [Callable Bond Case Study](docs/callable-bond-surrogate.md) studies callable
  bond risk acceleration while preserving the non-smooth exercise decision.
- [American Option Case Study](docs/american-option-dynamic-chebyshev.md)
  compares regression, reinforcement-learning-style simulation, and dynamic
  Chebyshev continuation approximation.

## Minimal Example

```csharp
using ChebyshevSharp;

double Function(double[] x, object? data) =>
    Math.Sin(x[0]) + Math.Cos(x[1]);

var cheb = new ChebyshevApproximation(
    function: Function,
    numDimensions: 2,
    domain: new[] { new[] { -1.0, 1.0 }, new[] { -1.0, 1.0 } },
    nNodes: new[] { 11, 11 });

cheb.Build();

double value = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 0, 0 });
double dx0 = cheb.VectorizedEval(new[] { 0.5, 0.3 }, new[] { 1, 0 });
```

## Project Links

- [NuGet package](https://www.nuget.org/packages/ChebyshevSharp)
- [GitHub repository](https://github.com/0xC000005/ChebyshevSharp)
- [Support & Reporting](docs/support.md)
- [Contributing](docs/contributing.md)
- [Citations](docs/citations.md)
