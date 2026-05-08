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
- Treat examples as first-class documentation. A page is not complete just
  because it is correct; each audit must ask whether a small, current C#
  example would help the reader choose or use the feature.

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
- Does the page need a derivation, proof sketch, formula breakdown, or citation
  to make the mathematical claim defensible?
- What is the smallest correct C# example?
- Would an additional practical example prevent a likely user mistake?
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
7. Verify mathematical explanations: for formulas, convergence claims,
   interpolation rules, tensor-rank statements, quadrature, derivatives, roots,
   or error estimates, decide whether the page needs a derivation, proof sketch,
   formula breakdown, or stronger citation. Add it when it helps users trust or
   correctly apply the feature; keep full proofs in concept/reference pages
   instead of crowding quickstarts.
8. Verify and improve examples:
   - examples must compile mentally from public API names and should be
     runnable when feasible
   - concept/how-to pages should include the smallest useful C# example for the
     main workflow
   - add an example when the API has common traps, such as node conventions,
     knot placement, tensor-train dense materialization, `FromValues()` rebuild
     limits, or domain validation
   - prefer a concrete numerical example over prose when it explains behavior
     more clearly
   - avoid redundant examples when a nearby linked page already covers the same
     workflow well
9. Run documentation validation:
   - `docfx docs/docfx.json`
   - targeted `dotnet test` when examples or behavior claims depend on tests
   - `dotnet run --project examples/...` when changing runnable examples
   - link checks when adding or changing external links
10. Open a focused PR. Include issue link, pages changed, references checked, and
   commands run.
11. After merge, update this file's ledger with issue, PR, status, evidence, and
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
- [ ] Decide whether formulas need explanation, proof sketches, or citations.
- [ ] Add or improve examples where they would materially reduce user mistakes.
- [ ] Improve flow, headings, and cross-links.
- [ ] Run DocFX and any relevant code/test commands.

## Acceptance Criteria
- Public pages explain user value before provenance.
- Maintainer/provenance details are moved or clearly scoped.
- Citations and links support the claims they appear near.
- Mathematical claims have enough explanation, citation support, or proof
  sketch for the target page type.
- Examples use current C# API names and are copyable.
- Important workflows have practical examples unless a nearby linked page
  already provides the right one.
- Verification commands are listed in the PR.
```

## Current Audit Queue

| Area | Pages | Status | Issue | PR | Notes |
| --- | --- | --- | --- | --- | --- |
| Public identity and provenance framing | `README.md`, `docs/index.md`, `docs/docs/introduction.md`, `docs/docs/getting-started.md`, package metadata | Complete | [#149](https://github.com/0xC000005/ChebyshevSharp/issues/149) | [#151](https://github.com/0xC000005/ChebyshevSharp/pull/151) | Reframed public identity, package description, and validation provenance language. |
| Citation style and mathematical source support | `docs/docs/citations.md`, `docs/docs/concepts.md`, math-heavy pages | Complete | [#152](https://github.com/0xC000005/ChebyshevSharp/issues/152) | [#153](https://github.com/0xC000005/ChebyshevSharp/pull/153) | Verified DOI metadata, link reachability, node-convention source evidence, and math-heavy wording. |
| Navigation and learning flow | `docs/toc.yml`, `docs/docs/toc.yml`, landing pages | Complete | [#154](https://github.com/0xC000005/ChebyshevSharp/issues/154) | [#155](https://github.com/0xC000005/ChebyshevSharp/pull/155) | Concepts now precede advanced how-tos; orphan pages are represented in the TOC. |
| Class-selection journey | `docs/docs/which-class.md`, class pages | Complete | [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156) | [#157](https://github.com/0xC000005/ChebyshevSharp/pull/157) | Choice rules are now cost-aware, implementation-checked, and linked to validation guides. |
| Dense approximation docs | `getting-started.md`, `adaptive-refinement.md`, `error-driven-construction.md`, `from-values.md`, `error-estimation.md` | Complete | [#158](https://github.com/0xC000005/ChebyshevSharp/issues/158) | [#159](https://github.com/0xC000005/ChebyshevSharp/pull/159) | Dense auto-N, FromValues, and error-estimation wording now matches source and tests. |
| Spline docs | `spline.md`, `special-points.md`, `adaptive-refinement.md`, calculus interactions | Complete | [#160](https://github.com/0xC000005/ChebyshevSharp/issues/160) | [#161](https://github.com/0xC000005/ChebyshevSharp/pull/161) | Public wording now distinguishes explicit knots from heuristic AutoKnots; examples and spline XML docs were checked against source/tests. |
| Slider docs | `slider.md`, `greeks.md`, `performance.md` | Complete | [#162](https://github.com/0xC000005/ChebyshevSharp/issues/162) | [#163](https://github.com/0xC000005/ChebyshevSharp/pull/163) | Clarified pivot cost, per-slide error diagnostics, cross-group derivative limits, and public benchmark framing. |
| Tensor Train docs | `tensor-train.md`, TT sections in related pages | Complete | [#164](https://github.com/0xC000005/ChebyshevSharp/issues/164) | [#165](https://github.com/0xC000005/ChebyshevSharp/pull/165) | Explain TT intuition, rank tradeoffs, dense-materialization limits. |
| Algebra/calculus/special operations | `algebra.md`, `calculus.md`, `extrude-slice.md`, `special-points.md` | Complete | [#166](https://github.com/0xC000005/ChebyshevSharp/issues/166) | [#167](https://github.com/0xC000005/ChebyshevSharp/pull/167) | Qualified exactness, cost, calculus, and edge-case claims. |
| Serialization and binary format | `serialization.md`, `binary-format.md`, fixture docs | Complete | [#168](https://github.com/0xC000005/ChebyshevSharp/issues/168) | [#169](https://github.com/0xC000005/ChebyshevSharp/pull/169) | Public persistence docs now separate full .NET JSON state from `.pcb` limits and fixture provenance. |
| Validation and contributing docs | `testing-and-validation.md`, `.github/`, contribution docs | In progress | [#170](https://github.com/0xC000005/ChebyshevSharp/issues/170) | TBD | Local audit implemented; PR pending. |
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
- 2026-05-08: Merged class-selection PR
  [#157](https://github.com/0xC000005/ChebyshevSharp/pull/157), closing
  [#156](https://github.com/0xC000005/ChebyshevSharp/issues/156). Opened
  [#158](https://github.com/0xC000005/ChebyshevSharp/issues/158) for the dense
  approximation workflow.
- 2026-05-08: Implemented local dense approximation audit for
  [#158](https://github.com/0xC000005/ChebyshevSharp/issues/158). Evidence:
  checked `ChebyshevApproximation` auto-N, `GetOptimalN1`, `Nodes`,
  `FromValues`, `ErrorEstimate`, and `SetOriginalFunctionValues` source; checked
  `ErrorThresholdTests`, `FromValuesTests`, `CoverageGapTests`, and
  `ErrorEstimateConsumerTests`; verified Type-I node convention against current
  NumPy `chebpts1` docs and public XML documentation structure against
  Microsoft C# XML docs. Local gates: `git diff --check`,
  `docfx docs/docfx.json`, CI-style targeted `dotnet test` filter
  (`120` tests passed), and `dotnet format --verify-no-changes`.
- 2026-05-08: Opened dense approximation PR
  [#159](https://github.com/0xC000005/ChebyshevSharp/pull/159) for
  [#158](https://github.com/0xC000005/ChebyshevSharp/issues/158).
- 2026-05-08: Merged dense approximation PR
  [#159](https://github.com/0xC000005/ChebyshevSharp/pull/159), closing
  [#158](https://github.com/0xC000005/ChebyshevSharp/issues/158). Opened
  [#160](https://github.com/0xC000005/ChebyshevSharp/issues/160) for the spline
  workflow.
- 2026-05-08: Strengthened the audit workflow so every documentation pass must
  explicitly decide whether the page needs more mathematical explanation, a
  proof sketch, citation support, or practical examples. This follows the
  NumPy documentation-maintenance pattern of prioritizing technical
  inaccuracies while also filling usage-example and broader tutorial/how-to
  gaps.
- 2026-05-08: Implemented local spline workflow audit for
  [#160](https://github.com/0xC000005/ChebyshevSharp/issues/160). Evidence:
  checked `ChebyshevSpline` routing, knot-boundary derivative rejection,
  `WithSpecialPoints`, `AutoKnots`, `Nodes`, `FromValues`, serialization,
  `ErrorEstimate`, integration, roots, and optimization against source and
  spline-related tests; removed public PyChebyshev positioning from
  special-points docs and XML comments; corrected an `O(1/n^2)` explanation
  and per-piece bound wording; made examples more copyable. Local gates:
  `git diff --check`, `docfx docs/docfx.json`, `dotnet restore
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj -p:TargetFramework=net10.0`,
  `dotnet build tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj
  --framework net10.0 --no-restore`, spline-focused `dotnet test` filter
  (`400` tests passed), `dotnet restore src/ChebyshevSharp/ChebyshevSharp.csproj`,
  and `dotnet format src/ChebyshevSharp/ChebyshevSharp.csproj
  --verify-no-changes --no-restore`.
- 2026-05-08: Opened spline workflow PR
  [#161](https://github.com/0xC000005/ChebyshevSharp/pull/161) for
  [#160](https://github.com/0xC000005/ChebyshevSharp/issues/160).
- 2026-05-08: Merged spline workflow PR
  [#161](https://github.com/0xC000005/ChebyshevSharp/pull/161), closing
  [#160](https://github.com/0xC000005/ChebyshevSharp/issues/160). Opened
  [#162](https://github.com/0xC000005/ChebyshevSharp/issues/162) for the slider
  workflow.
- 2026-05-08: Implemented local slider workflow audit for
  [#162](https://github.com/0xC000005/ChebyshevSharp/issues/162). Evidence:
  checked `ChebyshevSlider.Build`, `TotalBuildEvals`, `Eval`, `EvalMulti`,
  `ErrorEstimate`, integration, roots, optimization, serialization, algebra,
  and validation against source and slider-related tests; verified anchored
  decomposition/pivot sensitivity against Zhang, Choi, and Karniadakis (2011);
  removed public Python-source provenance from slider XML comments; reframed
  performance wording away from public reference-implementation positioning;
  and clarified that slider error estimates are per-slide diagnostics, not
  decomposition-error bounds. Local gates: `git diff --check`,
  `docfx docs/docfx.json`, `dotnet restore
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj -p:TargetFramework=net10.0
  -p:NuGetAudit=false`, `dotnet build
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0
  --no-restore`, slider/Greeks/calculus-focused `dotnet test` filter (`277`
  tests passed), `dotnet restore src/ChebyshevSharp/ChebyshevSharp.csproj
  -p:NuGetAudit=false`, and `dotnet format
  src/ChebyshevSharp/ChebyshevSharp.csproj --verify-no-changes`.
- 2026-05-08: Opened slider workflow PR
  [#163](https://github.com/0xC000005/ChebyshevSharp/pull/163) for
  [#162](https://github.com/0xC000005/ChebyshevSharp/issues/162).
- 2026-05-08: Merged slider workflow PR
  [#163](https://github.com/0xC000005/ChebyshevSharp/pull/163), closing
  [#162](https://github.com/0xC000005/ChebyshevSharp/issues/162). Opened
  [#164](https://github.com/0xC000005/ChebyshevSharp/issues/164) for the Tensor
  Train workflow.
- 2026-05-08: Implemented local Tensor Train documentation audit for
  [#164](https://github.com/0xC000005/ChebyshevSharp/issues/164). Evidence:
  checked `ChebyshevTT.Build`, TT-Cross convergence checks, TT-SVD/ALS dense
  paths, `FromValues`, `ToDense`, `EvalBatch`, finite-difference `EvalMulti`,
  `ErrorEstimate`, progress reporting, and tensor-shape overflow tests against
  source and TT-related tests; verified TT references against Oseledets (2011),
  Oseledets and Tyrtyshnikov (2010), Bigoni, Engsig-Karup, and Marzouk (2016),
  and Glau, Kressner, and Statti (2019); softened unbacked performance and
  derivative-accuracy claims; added held-out TT-Cross validation and
  `ChebyshevTT.FromValues` examples. Local gates: `git diff --check`,
  `docfx docs/docfx.json`, `dotnet restore
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj -p:TargetFramework=net10.0
  -p:NuGetAudit=false`, `dotnet build
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0
  --no-restore`, TT/from-values/evaluation-focused `dotnet test` filter (`427`
  tests passed), `dotnet restore src/ChebyshevSharp/ChebyshevSharp.csproj
  -p:NuGetAudit=false`, and `dotnet format
  src/ChebyshevSharp/ChebyshevSharp.csproj --verify-no-changes`.
- 2026-05-08: Opened Tensor Train workflow PR
  [#165](https://github.com/0xC000005/ChebyshevSharp/pull/165) for
  [#164](https://github.com/0xC000005/ChebyshevSharp/issues/164).
- 2026-05-08: Merged Tensor Train workflow PR
  [#165](https://github.com/0xC000005/ChebyshevSharp/pull/165), closing
  [#164](https://github.com/0xC000005/ChebyshevSharp/issues/164). Opened
  [#166](https://github.com/0xC000005/ChebyshevSharp/issues/166) for algebra,
  calculus, and special operations.
- 2026-05-08: Implemented local algebra/calculus/special-operations audit for
  [#166](https://github.com/0xC000005/ChebyshevSharp/issues/166). Evidence:
  checked algebra compatibility, scalar validation, TT rounded binary algebra,
  Fejer-1 integration, colleague-matrix roots, extrema, extrusion/slicing,
  special-point validation, and related Greeks/advanced-usage wording against
  source and focused tests; verified public references for Berrut and Trefethen
  (2004), Good (1961), Waldvogel (2006), and the Chebfun root/extrema guide;
  qualified exactness, cost, and root/optimization guarantees; removed
  release-history/provenance wording from user docs. Local gates:
  `git diff --check`, `docfx docs/docfx.json`, `dotnet restore
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj -p:TargetFramework=net10.0
  -p:NuGetAudit=false`, `dotnet build
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0
  --no-restore`, algebra/calculus/extrude/slice/special-points/Greeks-focused
  `dotnet test` filter (`350` tests passed), `dotnet restore
  src/ChebyshevSharp/ChebyshevSharp.csproj -p:NuGetAudit=false`, and
  `dotnet format src/ChebyshevSharp/ChebyshevSharp.csproj
  --verify-no-changes`.
- 2026-05-08: Merged algebra/calculus/special-operations PR
  [#167](https://github.com/0xC000005/ChebyshevSharp/pull/167), closing
  [#166](https://github.com/0xC000005/ChebyshevSharp/issues/166). Opened
  [#168](https://github.com/0xC000005/ChebyshevSharp/issues/168) for
  serialization and binary-format docs.
- 2026-05-08: Implemented local serialization/binary-format documentation audit
  for [#168](https://github.com/0xC000005/ChebyshevSharp/issues/168).
  Evidence: checked JSON and .pcb save/load paths for approximation, spline,
  slider, and TT against source and persistence tests; verified .pcb security
  framing against Microsoft's BinaryFormatter security guidance; reframed
  public docs around supported objects, dropped state, class-tag loading,
  validation behavior, and contributor fixture provenance; removed a public
  PyChebyshev-only bad-magic error message. Local gates: `git diff --check`,
  `docfx docs/docfx.json`, `dotnet restore
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj -p:TargetFramework=net10.0
  -p:NuGetAudit=false`, `dotnet build
  tests/ChebyshevSharp.Tests/ChebyshevSharp.Tests.csproj --framework net10.0
  --no-restore`, serialization/load/save-focused `dotnet test` filter (`314`
  tests passed), `dotnet restore src/ChebyshevSharp/ChebyshevSharp.csproj
  -p:NuGetAudit=false`, and `dotnet format
  src/ChebyshevSharp/ChebyshevSharp.csproj --verify-no-changes`.
- 2026-05-08: Merged serialization/binary-format PR
  [#169](https://github.com/0xC000005/ChebyshevSharp/pull/169), closing
  [#168](https://github.com/0xC000005/ChebyshevSharp/issues/168). Opened
  [#170](https://github.com/0xC000005/ChebyshevSharp/issues/170) for
  validation and contributing docs.
- 2026-05-08: Implemented local validation/contributing documentation audit for
  [#170](https://github.com/0xC000005/ChebyshevSharp/issues/170). Evidence:
  checked `CONTRIBUTING.md`, `SUPPORT.md`, README contribution links, docs-site
  contributor/support pages, PR and issue templates, `test.yml`, `docs.yml`,
  `mutation.yml`, `publish.yml`, `.codecov.yml`, and `stryker-config.json`;
  verified GitHub Discussions and Issues are enabled; checked GitHub community
  guidance for contributor guidelines, issue/PR templates, community profiles,
  and support resources; aligned DocFX/Stryker versions, Codecov patch-gate
  wording and no-report behavior, mutation workflow expectations, and
  release-gate language. Local gates: `git diff --check`, `docfx docs/docfx.json`,
  `dotnet restore
  src/ChebyshevSharp/ChebyshevSharp.csproj -p:NuGetAudit=false`, `dotnet build
  src/ChebyshevSharp/ChebyshevSharp.csproj --framework net10.0 --no-restore
  --verbosity minimal`, and `dotnet format
  src/ChebyshevSharp/ChebyshevSharp.csproj --verify-no-changes`.

## External Workflow References

- Diataxis: <https://diataxis.fr/>
- NumPy documentation landing page: <https://numpy.org/doc/stable/>
- NumPy documentation contribution guide: <https://numpy.org/devdocs/dev/howto-docs.html>
- Google developer documentation style guide: <https://developers.google.com/style/>
- Google cross-reference guidance: <https://developers.google.com/style/cross-references>
- Microsoft developer-content guidance: <https://learn.microsoft.com/en-us/style-guide/developer-content/>
- Write the Docs documentation principles: <https://www.writethedocs.org/guide/writing/docs-principles/>
- The Good Docs Project templates: <https://www.thegooddocsproject.dev/template>
