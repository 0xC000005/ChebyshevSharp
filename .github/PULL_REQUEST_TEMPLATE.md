## Summary

-

## Linked Issues

Fixes #

## Validation

- [ ] `dotnet format --verify-no-changes --verbosity minimal`
- [ ] `dotnet build --configuration Release --no-restore`
- [ ] `dotnet test --configuration Release --no-build --verbosity minimal --collect:"XPlat Code Coverage"`
- [ ] `dotnet pack src/ChebyshevSharp --configuration Release --no-build --output artifacts/packages`
- [ ] `docfx docs/docfx.json`
- [ ] Stryker.NET run or explanation for skipping when numerical core paths changed

## Numerical/Docs Checklist

- [ ] Added or updated focused regression tests
- [ ] Documented mathematical conventions, formulas, or citations when behavior changed
- [ ] Updated README/docs/changelog for public API or user-visible behavior
- [ ] Included benchmark evidence for performance-sensitive changes
