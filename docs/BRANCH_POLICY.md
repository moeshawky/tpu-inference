# Branch Policy and Governance

To prevent confusion between default branches, active serving checkouts, and exact runtime SHAs, the following policies strictly apply to repository state and branch roles.

## Core Concepts

### `main`
- **Role**: DEFAULT BRANCH (GitHub default/upstream-reconciliation line).
- **Policy**: This branch is used for consuming or comparing upstream development. **It must not be assumed to be the active serving line.**

### `i-j-port`
- **Role**: CANONICAL SERVING BRANCH.
- **Policy**: This is the canonical integration line for TPU inference serving. Accepted serving fixes and model integration changes are promoted here. This is deliberately distinct from the GitHub default branch (`main`).

### Experiment and Work Branches
- **Role**: EXPERIMENT BRANCH / WORK BRANCH.
- **Policy**: These are temporary branches (e.g., PR, development, profiler, reconciliation, or fix branches). **Success while running them does not promote them automatically.**
  - A temporary branch does NOT become the serving branch merely because:
    - an agent checks it out
    - a script runs from it
    - a benchmark succeeds from it
    - its SHA is temporarily pinned.

### Runtime Specimen (Exact SHA)
- **Role**: RUNTIME SPECIMEN.
- **Policy**: An exact runtime SHA represents a specific executed state. **It does not redefine branch ownership or role.** The currently pinned serving SHA does not automatically mean the current branch becomes the canonical serving branch.

## PR Workflows

For serving changes:
- Base your work on `origin/i-j-port`
- Target your PR to the `i-j-port` branch.
- **Never choose your base merely from the GitHub default (`main`).** PR branches must be based on their intended target branch.

## State Census Helper

To reliably identify your current repository checkout state and its relationship to the canonical branch without modifying the repository, run the census helper:

```bash
./scripts/repo_state_census.sh
```

This read-only helper reports your current HEAD, upstream tracking, branch roles, and whether you are running an experimental/non-canonical checkout.
