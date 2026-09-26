# Contributing

`develop` is the default branch. Use topic branches from `develop` and target
pull requests there. Pull requests use rebase-merge only; linear history is
required. Install development dependencies with `pip install -e ".[dev]"`.
Stage the host library as described below before running `pytest`; also run
`ruff check src tests scripts setup.py` before submitting changes.

## Host library and wheels

The default build produces a source distribution and a `py3-none-any` wheel
without a native library, even if one is staged in the source checkout:

```bash
python -m build
```

For a native wheel, build and test a runtime source checkout, then select the
native variant explicitly:

```bash
python scripts/runtime_bundle.py --source RUNTIME_SOURCE
TIGRIS_WHEEL_VARIANT=native python -m build --wheel
```

An existing archive can be staged with `--archive ARCHIVE --sha256 SHA256`.
The staging command checks archive and library hashes, ABI, and platform.
Editable installations can use the staged library too. Execution never downloads
native code. `run` requires the bundled library; inspection, compilation,
analysis, code generation, and zoo downloads also work in the pure wheel.

CI builds runtime source from the same branch when available, otherwise from
`develop`. Tag builds test against the matching runtime tag. Pull requests build
and install one native wheel plus the pure wheel; `develop`, `main`, and release
builds test the full platform and Python matrix outside the checkout. Only
release-wheel builds download pinned archives. `runtime-host.json` records their
release and per-platform SHA-256 values; an unset pin blocks release wheels,
not source CI. Publishing uploads the tested wheels and source distribution.

## Releasing

`main` only fast-forwards to a tested `develop` commit through the "Promote to
main" workflow (`.github/workflows/promote.yml`), with no release or back-merge
pull requests. The release tag is created with the GitHub release on that commit.

Compiler and runtime release as a pair with the same version, starting with
`v0.11.0`. The number identifies the tested combination. The plan schema,
accepted schema range, host ABI, `compatibility.json`, and zoo runtime ranges
retain their separate compatibility meanings; matching release numbers do not
replace those checks.

1. Prepare both components with the same release number and test them together.
   The runtime's `scripts/check_version_sources.py --expect X.Y.Z` must pass.
2. Promote the runtime's tested `develop` commit through "Promote to main"
   (`.github/workflows/promote.yml`), with a dry run first. Then create the GitHub
   release `vX.Y.Z` with its tag on that commit. Publish the runtime first and
   wait for `host.yml` to attach all native archives and SHA-256 files. For the
   first pair this is `v0.11.0`; `v0.10.2` has no host archives.
3. Pin `runtime-host.json` to that tag and its per-platform archive hashes.
   Review the downloaded manifests and run the wheel tests.
4. Promote the compiler's tested `develop` commit through "Promote to main"
   (`.github/workflows/promote.yml`), with a dry run first. Then create the GitHub
   release `vX.Y.Z` with its tag on that commit. Its tag and setuptools-scm
   version must agree with the runtime pin; release-wheel builds enforce this
   equality.

`scripts/release.sh X.Y.Z --notes NOTES.md` in each repository runs its steps up to
a draft GitHub release: it lands the version change (runtime) or the runtime pin
(compiler) through a pull request once CI passes, promotes the merged commit
with a dry run first, and creates the draft on it. Publishing the draft tags
the release and starts the release workflows. Run it in the runtime first, and
in the compiler after the runtime release is published.

A compiler-only change still releases both components, including a new ESP
component version. Its runtime release note says: "No runtime implementation
changes; released with the matching compiler version."
