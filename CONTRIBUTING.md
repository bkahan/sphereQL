# Contributing to sphereQL

Thanks for your interest in contributing to sphereQL. This document covers the
workflows and expectations for contributing code, reporting bugs, and
participating in the project.

## Code of Conduct

Be respectful and constructive. We're all here to build good software. Harassment,
personal attacks, and bad-faith behavior will not be tolerated.

## Reporting Bugs

Open a [GitHub issue](https://github.com/bkahan/sphereQL/issues) with:

- **Title**: a short, specific summary (not "it doesn't work")
- **Environment**: OS, Rust version (`rustc --version`), Python version if
  applicable, and which crate(s) are involved
- **Steps to reproduce**: minimal code or commands that trigger the bug
- **Expected behavior**: what should happen
- **Actual behavior**: what happens instead, including full error output
- **Severity**: does it crash, produce wrong results, or just look odd?

If you can write a failing test case, even better -- include it in the issue or
reference it in your PR.

## Suggesting Features

Open an issue with the `enhancement` label. Describe:

- The problem you're trying to solve (not just the solution you want)
- Who benefits and how
- Any alternative approaches you've considered

Feature discussions happen in issues before code is written. This avoids wasted
effort on PRs that don't align with the project direction.

## Development Setup

### Prerequisites

- Rust stable (2024 edition) -- install via [rustup](https://rustup.rs)
- Python 3.12+ and [maturin](https://www.maturin.rs) for Python binding work
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) for WASM binding work

### Clone and verify

```bash
git clone https://github.com/bkahan/sphereQL.git
cd sphereQL

# Verify everything builds and passes
cargo test --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
```

### Python development

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv/Scripts/activate on Windows
pip install maturin pytest numpy

cd sphereql-python
maturin develop
pytest -v
```

## Branching and Pull Requests

**Never commit directly to `main`.** All changes go through pull requests.

### Workflow

1. **Create a branch** from `main` with a descriptive name:
   - `fix/qdrant-timeout-handling`
   - `feature/incremental-pca`
   - `docs/python-visualization-guide`
   - `refactor/pipeline-error-types`

2. **Keep commits focused.** Each commit should be a single logical change. Write
   commit messages that explain *why*, not just *what*:
   ```
   fix: handle Lagged error in subscription broadcast

   Subscribers that fall behind now skip missed events instead of
   disconnecting. This prevents slow consumers from losing their
   subscription entirely.
   ```

3. **Open a PR** against `main` when your work is ready for review. In the PR
   description:
   - Summarize what changed and why
   - Link any related issues (`Closes #42`, `Fixes #17`)
   - Note anything reviewers should pay attention to
   - Include a test plan if the changes aren't covered by automated tests

4. **All CI checks must pass** before merge. The CI pipeline runs:
   - `cargo test --workspace --all-features` (including doc-tests)
   - `cargo clippy --workspace --all-features --all-targets` with `-D warnings`
   - `cargo fmt --all -- --check`
   - Per-feature compilation matrix (core, index, layout, embed, graphql,
     vectordb, full, no-default-features)
   - Python build + pytest on Python 3.12

5. **Address review feedback** with new commits (don't force-push during review
   unless asked). Once approved, the maintainer will merge.

### PR size

Keep PRs small enough to review in one sitting. If a change touches more than
~10 files or involves both a refactor and new functionality, split it into
separate PRs:

1. Refactor PR (no behavior change)
2. Feature PR (built on the clean foundation)

This makes review faster and reverts safer.

## Issue Tracking

- Reference issues in commit messages and PR descriptions (`Fixes #N`,
  `Part of #N`)
- When closing an issue via PR, use GitHub's
  [closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)
  so the issue auto-closes on merge
- If a PR addresses an issue partially, note what remains and update the issue
- Keep issue discussions on-topic -- open new issues for tangential problems

## Code Standards

### Rust

- Follow the existing code style. Run `cargo fmt` before committing.
- All public APIs need doc comments. Internal code should only have comments
  where the *why* is non-obvious.
- No `unwrap()` or `expect()` in library code that handles user input. Use
  proper error types. Panics are acceptable only for invariant violations that
  indicate a bug.
- Validate at system boundaries (user input, FFI, deserialization). Trust
  internal types.
- New features need tests. Bug fixes need a regression test that fails without
  the fix.

### Python (PyO3)

- Match the ergonomics of idiomatic Python. Use keyword arguments with defaults
  where it makes sense.
- Type stubs in `sphereql-python/python/sphereql/sphereql.pyi` must stay in
  sync with Rust changes.
- Test with pytest, not just Rust unit tests.

### WASM

- JSON in, JSON out. Keep the WASM API surface minimal.
- Test with `wasm-pack test` when adding new bindings.

## Testing

### Running tests

```bash
# Full workspace (excludes Python/WASM which need special toolchains)
cargo test --workspace --exclude sphereql-python --exclude sphereql-wasm

# Full workspace with all features (requires Python headers)
cargo test --workspace --all-features

# Single crate
cargo test -p sphereql-core

# Single test
cargo test -p sphereql-embed -- kernel_pca::tests::kernel_pca_fit_default

# Python
cd sphereql-python && maturin develop && pytest -v
```

### Writing tests

- Unit tests go in a `#[cfg(test)] mod tests` block in the same file as the
  code they test
- Integration tests go in the crate's `tests/` directory
- Test edge cases: empty input, single item, boundary values, error paths
- Use descriptive test names that read as sentences:
  `two_items_pushed_apart_by_repulsion`, not `test2`

## Documentation

- Update doc comments when changing public API signatures
- If you add a new crate feature, update the feature flags table in README.md
- Example scripts go in the relevant crate's `examples/` directory
- Run `cargo doc --workspace --all-features --no-deps` to verify docs build

## Licensing

sphereQL is [MIT licensed](LICENSE). By submitting a pull request, you agree
that your contribution will be licensed under the same terms. If your
contribution includes code from another project, note the original license and
ensure it is MIT-compatible.

## Getting Help

- Open an issue for bugs or feature discussions
- For questions about the codebase or how to approach a contribution, open an
  issue with the `question` label

Thank you for contributing.
