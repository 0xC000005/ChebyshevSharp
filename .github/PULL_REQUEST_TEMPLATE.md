## Summary

-

## Linked Issues

Fixes #

## Type of Change

- [ ] Bug fix
- [ ] Numerical behavior or algorithm change
- [ ] Documentation-only change
- [ ] Test, CI, or infrastructure change
- [ ] Release/version metadata change

## Validation

- [ ] `dotnet format --verify-no-changes --verbosity minimal`
- [ ] `dotnet build --configuration Release --no-restore`
- [ ] `dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage"`
- [ ] `dotnet pack src/ChebyshevSharp --configuration Release --no-build --output artifacts/packages`
- [ ] `docfx docs/docfx.json`
- [ ] `codecov/patch` is passing, or the PR explains why there are no coverable changes
- [ ] Stryker.NET run or explanation for skipping when numerical core paths changed

## Numerical/Docs Checklist

- [ ] Added or updated focused regression tests
- [ ] Documented mathematical conventions, formulas, or citations when behavior changed
- [ ] Updated README/docs/changelog for public API or user-visible behavior
- [ ] Explained Stryker skips, surviving mutants, or coverage gaps when numerical core paths changed
- [ ] Included benchmark evidence for performance-sensitive changes
