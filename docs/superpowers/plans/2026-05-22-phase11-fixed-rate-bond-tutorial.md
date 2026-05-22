# Phase 11 Fixed-Rate Bond Tutorial Implementation Plan

> **For agentic workers:** Follow this plan task-by-task and update the
> checkboxes as work is completed. Keep Phase 11 to documentation and tutorial
> work unless the user explicitly changes scope.

**Goal:** Publish a clear public tutorial/case study that explains the fixed-rate
bond surrogate research, including the baseline pricer, data fixture, naive
global TT/Slider failure evidence, structured alternatives, analytic coupon
identity, and schedule-aware routing limits.

**Branch:** `phase11-fixed-rate-bond-tutorial`

**Tracking issue:** #191

---

## File Map

- Modify or create documentation case-study page under `docs/`.
- Modify existing docs index or examples navigation only where needed for
  discoverability.
- Modify `docs/research/fixed-rate-bond-surrogate/status.md` with progress,
  verification, PR, and tracking-issue links.
- Do not change C# source, tests, fixture data, or public APIs unless a
  documentation build requires a mechanical link/path correction.

## Task 1: Inventory Existing Public Surfaces

- [x] Identify the current documentation pages that mention the fixed-rate bond
      example.
- [x] Identify the best destination for a tutorial page and the minimum links
      needed from examples/research pages.
- [x] Confirm no proprietary wording exists in the pages to be edited.

## Task 2: Draft Tutorial Narrative

- [x] Write the reference-problem section with QLNet and pinned Federal Reserve
      fixture assumptions.
- [x] Write the naive global surrogate section using Phase 6 evidence.
- [x] Write the common-practice alternatives section using Phase 7 evidence.
- [x] Write the coupon-linearity and maturity-schedule sections using Phase 8
      and Phase 9 evidence.
- [x] Write the router outcome and practical conclusion using Phase 10 evidence.

## Task 3: Verify Claims and Links

- [x] Check every command in the tutorial against the current example CLI.
- [x] Check every numerical claim against the committed phase reports.
- [x] Check every public source link used by the tutorial.
- [x] Keep citations as source references only; do not teach citation style.

## Task 4: Documentation Build and Local Checks

- [x] Run `docfx docs/docfx.json`.
- [x] Run `git diff --check`.
- [x] Run any focused markdown/link checks already present in the repo.
- [x] Update `status.md` with verification commands and results.

## Task 5: Tracking and PR

- [ ] Comment on issue #191 with the Phase 11 branch and scope.
- [ ] Open one coherent Phase 11 PR after local verification passes.
- [ ] Keep all review fixes in that PR.
- [ ] Do not start Phase 12 until the Phase 11 PR is merged or explicitly
      closed without merge.

## Exit Criteria

- [ ] Public tutorial is readable without private context.
- [ ] Claims are evidence-backed by the phase reports.
- [ ] Limitations are stated clearly.
- [ ] DocFX and diff checks pass.
- [ ] Tracking issue and PR are synchronized.
