# Documentation Audit Workflow

> Required workflow: keep this file committed and update it on each
> documentation-audit branch. Do not rely on chat history as the only source of
> scope, decisions, open issues, or audited pages.

## Objective

Review the entire public documentation set for ChebyshevSharp and improve it one
bounded surface at a time. Each cycle must make the documentation clearer,
better cited, more accurate against the C# implementation, and cleaner about
what belongs in public user documentation versus contributor or maintainer
documentation.

This workflow runs alongside the comprehensive code audit. Use the same rigor:
issue first, narrow scope, verify against references and implementation, make a
focused PR, wait for CI, record evidence here, then continue with the next
surface.

## Documentation Principles

- Organize public docs around Diataxis: tutorials teach, how-to guides solve a
  task, concept pages explain, and reference pages describe exact behavior.
- Follow project-specific style first, then Google developer documentation
  guidance for clarity, procedures, links, and examples.
- Use NumPy as the main open-source documentation model: clear landing pages,
  concepts/user guide, API reference, contributor guide, and issue-driven docs
  improvements.
- Use Microsoft developer-content guidance for the API/reference layer:
  reference docs and runnable code examples are the foundation.
- Use The Good Docs Project templates as sanity checks for README, concepts,
  how-to guides, reference pages, and contribution material.

## Source Hierarchy

When reviewing claims, prefer sources in this order:

1. ChebyshevSharp implementation and tests in `src/`, `tests/`, `examples/`,
   and CI workflows.
2. Primary mathematical references such as Trefethen, Berrut and Trefethen,
   Ruiz and Zeron, and cited tensor-train references.
3. Upstream/reference implementations, including `ref/PyChebyshev/` and MoCaX,
   only as validation provenance or compatibility evidence.
4. External documentation style references, including Diataxis, NumPy, Google
   developer docs, Microsoft developer docs, Write the Docs, and The Good Docs
   Project.

Do not use PyChebyshev or MoCaX language as public product positioning unless
the page is explicitly about validation provenance, fixture regeneration, or
contributor workflow.

## Public vs. Contributor Content

Public user-facing pages should answer:

- What problem does this solve?
- When should a user choose this class or method?
- What are the important mathematical and numerical assumptions?
- What is the smallest correct C# example?
- What errors or limitations should a user expect?

Contributor or maintainer pages may discuss:

- parity tracking
- source-port history
- fixture provenance
- MoCaX/PyChebyshev comparison details
- audit mechanics
- mutation testing expectations
- release and validation workflow

If a public page contains maintainer language, move or reframe it. For example,
write "validated against reference implementations" instead of making the
library sound like a feature-copying project.

## Cycle Checklist

For each documentation surface:

1. Open or select one GitHub issue with a clear page set, user problem, and
   acceptance criteria.
2. Read the current page, linked pages, generated API docs if relevant, and the
   corresponding C# implementation/tests.
3. Search online for current best practices or primary references when the page
   makes a style, citation, or mathematical claim.
4. Check PyChebyshev and MoCaX only when validating formulas, behavior, fixture
   provenance, or documented compatibility.
5. Classify every paragraph as public-user, contributor, maintainer, or release
   history. Move or rewrite misplaced content.
6. Verify citations: every cited paper/book/link must exist, support the claim,
   and be used with consistent author-year prose.
7. Verify examples: examples must compile mentally from public API names and
   should be runnable when feasible.
8. Run documentation validation:
   - `docfx docs/docfx.json`
   - targeted `dotnet test` when examples or behavior claims depend on tests
   - `dotnet run --project examples/...` when changing runnable examples
   - link checks when adding or changing external links
9. Open a focused PR. Include issue link, pages changed, references checked, and
   commands run.
10. After merge, update this file's ledger with issue, PR, status, evidence, and
    any follow-up issue.

## Issue Template for Audit Cycles

Use this body shape for each documentation-audit issue:

```md
## Scope
- Pages:
- Audience:

## Problem

## Audit Tasks
- [ ] Check public/private framing.
- [ ] Verify implementation and tests behind behavior claims.
- [ ] Verify citations and external links.
- [ ] Check examples and commands.
- [ ] Improve flow, headings, and cross-links.
- [ ] Run DocFX and any relevant code/test commands.

## Acceptance Criteria
- Public pages explain user value before provenance.
- Maintainer/provenance details are moved or clearly scoped.
- Citations and links support the claims they appear near.
- Examples use current C# API names and are copyable.
- Verification commands are listed in the PR.
```

## Current Audit Queue

| Area | Pages | Status | Issue | PR | Notes |
| --- | --- | --- | --- | --- | --- |
| Public identity and provenance framing | `README.md`, `docs/index.md`, `docs/docs/introduction.md`, `docs/docs/getting-started.md`, package metadata | Complete | [#149](https://github.com/0xC000005/ChebyshevSharp/issues/149) | [#151](https://github.com/0xC000005/ChebyshevSharp/pull/151) | Reframed public identity, package description, and validation provenance language. |
| Citation style and mathematical source support | `docs/docs/citations.md`, `docs/docs/concepts.md`, math-heavy pages | Complete | [#152](https://github.com/0xC000005/ChebyshevSharp/issues/152) | [#153](https://github.com/0xC000005/ChebyshevSharp/pull/153) | Verified DOI metadata, link reachability, node-convention source evidence, and math-heavy wording. |
| Navigation and learning flow | `docs/toc.yml`, `docs/docs/toc.yml`, landing pages | Complete | [#154](https://github.com/0xC000005/ChebyshevSharp/issues/154) | [#155](https://github.com/0xC000005/ChebyshevSharp/pull/155) | Concepts now precede advanced how-tos; orphan pages are represented in the TOC. |
| Class-selection journey | `docs/docs/which-class.md`, class pages | In PR | [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156) | [#157](https://github.com/0xC000005/ChebyshevSharp/pull/157) | Choice rules are now cost-aware, implementation-checked, and linked to validation guides. |
| Dense approximation docs | `getting-started.md`, `adaptive-refinement.md`, `error-driven-construction.md`, `from-values.md` | Not started | TBD | TBD | Check examples, validation, and error language. |
| Spline docs | `spline.md`, calculus interactions | Not started | TBD | TBD | Clarify piecewise behavior, knots, discontinuities, Sobol limits. |
| Slider docs | `slider.md`, `greeks.md`, `performance.md` | Not started | TBD | TBD | Clarify workflow and benchmark framing. |
| Tensor Train docs | `tensor-train.md`, TT sections in related pages | Not started | TBD | TBD | Explain TT intuition, rank tradeoffs, dense-materialization limits. |
| Algebra/calculus/special operations | `algebra.md`, `calculus.md`, `extrude-slice.md`, `special-points.md` | Not started | TBD | TBD | Separate mathematical intuition from exact API reference. |
| Serialization and binary format | `serialization.md`, `binary-format.md`, fixture docs | Not started | TBD | TBD | Public persistence docs vs contributor fixture provenance. |
| Validation and contributing docs | `testing-and-validation.md`, `.github/`, contribution docs | Not started | TBD | TBD | Keep public expectations clear without leaking internal notes. |
| API reference surface | XML docs in `src/`, generated `docs/api/` | Not started | TBD | TBD | Ensure public XML docs are accurate and not implementation-history heavy. |

## Session Log

- 2026-05-08: Created this documentation-audit workflow after the v0.13.0
  release. Initial pilot target is public identity and provenance framing.
- 2026-05-08: Opened pilot issue
  [#149](https://github.com/0xC000005/ChebyshevSharp/issues/149) for public
  identity and provenance framing.
- 2026-05-08: Opened workflow PR
  [#150](https://github.com/0xC000005/ChebyshevSharp/pull/150) to make this
  file durable in the repository.
- 2026-05-08: Merged workflow PR
  [#150](https://github.com/0xC000005/ChebyshevSharp/pull/150), then started
  pilot implementation branch `docs/audit-public-identity` for
  [#149](https://github.com/0xC000005/ChebyshevSharp/issues/149).
- 2026-05-08: Opened pilot implementation PR
  [#151](https://github.com/0xC000005/ChebyshevSharp/pull/151). Local
  verification: `git diff --check`, `docfx docs/docfx.json`, `dotnet pack
  src/ChebyshevSharp --configuration Release --output
  /tmp/chebsharp-doc-audit-public-identity-pack --verbosity minimal`, and
  nuspec description inspection.
- 2026-05-08: PR
  [#151](https://github.com/0xC000005/ChebyshevSharp/pull/151) passed remote
  `Format, Pack, and Docs`, `.NET 8 library build`, `.NET 10 tests`, `All
  Tests Passed`, and `codecov/patch`.
- 2026-05-08: Merged pilot PR
  [#151](https://github.com/0xC000005/ChebyshevSharp/pull/151), closing
  [#149](https://github.com/0xC000005/ChebyshevSharp/issues/149). Opened next
  audit issue [#152](https://github.com/0xC000005/ChebyshevSharp/issues/152)
  for citations, formulas, and source provenance.
- 2026-05-08: Implemented local citation/provenance audit for
  [#152](https://github.com/0xC000005/ChebyshevSharp/issues/152). Evidence:
  C# Type-I/DCT-II and TT finite-difference paths checked against source;
  PyChebyshev and MoCaX node conventions checked in local upstream source;
  all DOI records verified through Crossref metadata; scoped external links
  checked; `git diff --check` passed; `docfx docs/docfx.json` succeeded with
  restore vulnerability-data warnings only.
- 2026-05-08: Merged citation/provenance PR
  [#153](https://github.com/0xC000005/ChebyshevSharp/pull/153), closing
  [#152](https://github.com/0xC000005/ChebyshevSharp/issues/152). Opened
  [#154](https://github.com/0xC000005/ChebyshevSharp/issues/154) for
  documentation navigation and learning flow.
- 2026-05-08: Implemented local navigation audit for
  [#154](https://github.com/0xC000005/ChebyshevSharp/issues/154). Evidence:
  compared `docs/docs/toc.yml` against all `docs/docs/*.md` source pages;
  found `advanced-usage.md` missing from the TOC; moved concept pages before
  how-to guides; clarified the landing-page and getting-started next-step
  paths.
- 2026-05-08: Merged navigation PR
  [#155](https://github.com/0xC000005/ChebyshevSharp/pull/155), closing
  [#154](https://github.com/0xC000005/ChebyshevSharp/issues/154). Opened
  [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156) for the
  class-selection journey.
- 2026-05-08: Implemented local class-selection audit for
  [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156). Evidence:
  checked class-specific method claims against `ChebyshevApproximation`,
  `ChebyshevSpline`, `ChebyshevSlider`, and `ChebyshevTT` source; clarified
  dense-grid, spline-piece, slider-group, and TT-Cross cost rules; replaced a
  remaining TT-SVD "optimal" phrase with deterministic-reference wording;
  `git diff --check`, `docfx docs/docfx.json`, and the explicit
  `tests/ChebyshevSharp.Tests` project test command passed.
- 2026-05-08: Opened class-selection PR
  [#157](https://github.com/0xC000005/ChebyshevSharp/pull/157) for
  [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156).

## External Workflow References

- Diataxis: <https://diataxis.fr/>
- NumPy documentation landing page: <https://numpy.org/doc/stable/>
- NumPy documentation contribution guide: <https://numpy.org/devdocs/dev/howto-docs.html>
- Google developer documentation style guide: <https://developers.google.com/style/>
- Google cross-reference guidance: <https://developers.google.com/style/cross-references>
- Microsoft developer-content guidance: <https://learn.microsoft.com/en-us/style-guide/developer-content/>
- Write the Docs documentation principles: <https://www.writethedocs.org/guide/writing/docs-principles/>
- The Good Docs Project templates: <https://www.thegooddocsproject.dev/template>
