# Testing

The canonical list of local test/lint commands and the full CI checklist
lives in **[CONTRIBUTING.md](../CONTRIBUTING.md)** — see
[Testing](../CONTRIBUTING.md#testing) for the commands and
[Branching and Pull Requests](../CONTRIBUTING.md#branching-and-pull-requests)
for what each CI job runs. Keeping a single copy avoids the drift this
page used to accumulate.

## Release pipeline

Separate release workflows publish to
[crates.io](../.github/workflows/crates-publish.yml) and
[PyPI](../.github/workflows/python-publish.yml) automatically when a
GitHub Release is created. PyPI wheels are built for Linux
x86_64/aarch64, macOS x86_64/aarch64, and Windows x86_64.
