---
title: Support & Reporting
---

# Support & Reporting

ChebyshevSharp is a numerical library, so reports are most useful when they are
small, reproducible, and explicit about the mathematical expectation.

## Questions

For usage questions, start a [GitHub
Discussion](https://github.com/0xC000005/ChebyshevSharp/discussions) and
include:

- What you are trying to approximate.
- Which class you are using: `ChebyshevApproximation`, `ChebyshevSpline`,
  `ChebyshevSlider`, or `ChebyshevTT`.
- A small code sample with domain and node counts.
- The result you expected and the result you observed.

## Bugs

Use the [issue template
chooser](https://github.com/0xC000005/ChebyshevSharp/issues/new/choose). Bug
reports should include:

- ChebyshevSharp version and .NET SDK version.
- Operating system and CPU architecture.
- Minimal runnable code.
- Expected behavior, actual behavior, and exception text if any.

## Numerical Accuracy

For accuracy reports, include the function, domain, node counts, construction
method, tolerance, and reference value. If possible, say whether the same case
matches PyChebyshev, NumPy Chebyshev routines, MoCaX, or another independent
calculation.

## Security

Do not open public issues for security problems. Follow the repository
[`SECURITY.md`](https://github.com/0xC000005/ChebyshevSharp/security/policy)
file.
