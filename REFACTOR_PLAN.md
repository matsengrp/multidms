# Release Plan: Merge jaxmodels branch to main

## Overview
Prepare the `jaxmodels` branch for merge to main and release. The `multidms.jaxmodels` module will be added to the package as a new module alongside the existing `biophysical` module.

## Quick Summary
1. **Code Quality** (Steps 1-4): Format code, lint, install package, build docs
2. **Testing** (Steps 5-6): Run full test suite, verify functionality
3. **Verify on Branch** (Steps 7-9): Push to remote, verify CI passes, clean notebooks
4. **Merge** (Step 10): Merge jaxmodels branch to main
5. **Prepare Release** (Steps 11-14): Verify on main, update CHANGELOG, bump version to 1.2.0
6. **Release** (Steps 15-17): Push to GitHub, monitor CI/PyPI, verify installation

## Current State
- **Branch**: `jaxmodels`
- **Module location**: `multidms/jaxmodels.py` (will remain at this location)
- **Test coverage**: `tests/test_jaxmodels.py`
- **Notebooks**:
  - `notebooks/jaxmodels/jaxmodels.ipynb`
  - `notebooks/jaxmodels/jaxmodels_simulation_fits.ipynb`
  - `notebooks/jaxmodels/jaxmodels_empirical_fits.ipynb`

## Implementation Steps

### 1. Code quality and formatting
- [ ] Run `pixi run black .` to format all code
- [ ] Run `pixi run ruff check .` to check for linting issues
- [ ] Fix any linting issues that appear
- [ ] Verify code follows project conventions

### 2. Installation and dependency check
- [ ] Run `pixi run pip install -e ".[dev]"` to install in development mode
- [ ] Verify `import multidms.jaxmodels` works
- [ ] Check that all dependencies are properly installed (especially equinox, jaxtyping)

### 3. Documentation
- [ ] Run `pixi run make -C docs clean` to clean docs
- [ ] Run `pixi run make -C docs html` to build documentation
- [ ] Verify documentation builds without errors
- [ ] Check if jaxmodels docstrings are properly formatted

### 4. Testing
- [ ] Run `pixi run pytest tests/test_jaxmodels.py` to run jaxmodels tests
- [ ] Run `pixi run pytest tests/` to run all unit tests
- [ ] Run `pixi run pytest --doctest-modules multidms/` to run doctests
- [ ] Run full test suite: `pixi run pytest --doctest-modules multidms tests -vv`
- [ ] Verify all tests pass

### 5. Verification
- [ ] Test importing in Python REPL:
  ```python
  import multidms
  import multidms.jaxmodels as jaxmodels
  # Verify Data, Latent, Model, fit, etc. are accessible
  ```
- [ ] Verify existing `multidms.biophysical` module still works
- [ ] Run a simple fit test from test suite manually

### 6. CI verification (on jaxmodels branch)
- [ ] Push jaxmodels branch to remote: `git push origin jaxmodels`
- [ ] Check GitHub Actions to ensure CI passes on the branch
- [ ] Fix any CI failures before proceeding to merge
- [ ] Verify all jobs pass (tests, linting, docs build)

### 7. Notebook cleanup
- [ ] Clear outputs from notebooks to avoid committing stale outputs:
  - `notebooks/jaxmodels/jaxmodels.ipynb`
  - `notebooks/jaxmodels/jaxmodels_simulation_fits.ipynb`
  - `notebooks/jaxmodels/jaxmodels_empirical_fits.ipynb`
- [ ] Test that notebooks run correctly:
  - Run at minimum the first few cells of each notebook
  - Verify imports work: `import multidms.jaxmodels`
  - If time permits, run full notebooks to ensure end-to-end functionality

### 8. Git operations (on jaxmodels branch)
- [ ] Review all changes with `git status` and `git diff`
- [ ] Stage any modified files if needed:
  - `git add multidms/jaxmodels.py` (if modified)
  - `git add tests/test_jaxmodels.py` (if modified)
  - `git add notebooks/jaxmodels/` (if modified)
- [ ] Commit any remaining changes with descriptive message
- [ ] Ensure no unintended files are committed (check git status)
- [ ] Push to remote if needed: `git push origin jaxmodels`

### 9. Pre-merge checklist
- [ ] All tests passing on jaxmodels branch
- [ ] Black formatting applied
- [ ] Ruff linting passing
- [ ] Documentation builds successfully
- [ ] Package installs correctly
- [ ] Notebooks run correctly
- [ ] No breaking changes to existing `multidms` API
- [ ] CI passing on GitHub Actions

### 10. Merge to main
- [ ] Switch to main branch: `git checkout main`
- [ ] Ensure main is up to date: `git pull origin main`
- [ ] Merge jaxmodels branch: `git merge jaxmodels`
- [ ] Resolve any merge conflicts if they arise
- [ ] Verify merge was successful with `git log`

### 11. Post-merge verification on main
- [ ] Run full test suite on main: `pixi run pytest --doctest-modules multidms tests -vv`
- [ ] Rebuild docs: `pixi run make -C docs clean && pixi run make -C docs html`
- [ ] Test imports in Python REPL to confirm everything still works
- [ ] Check that no unexpected changes were introduced during merge

### 12. Update CHANGELOG
- [ ] Edit `CHANGELOG.rst` to document the new version
- [ ] Add entry describing the addition of `multidms.jaxmodels` module
- [ ] Example entry:
  ```
  1.2.0
  -----
  * Added `multidms.jaxmodels` module - new API for global epistasis modeling using JAX and Equinox. This provides an alternative implementation to the existing `multidms.biophysical` module with improved performance and a simplified interface.
  ```
- [ ] Stage the changelog: `git add CHANGELOG.rst`

### 13. Version bump (on main branch)
- [ ] Run `pixi run bumpver update --minor` to bump version (1.1.0 → 1.2.0)
  - This updates version in `pyproject.toml`, `multidms/__init__.py`, and `docs/conf.py`
  - Creates a git commit and tag automatically
  - The commit will include the CHANGELOG.rst update if staged
- [ ] Verify the version was bumped correctly in all files
- [ ] Check git log to confirm bumpver created the version commit and tag

### 14. Pre-release checklist
- [ ] All tests still passing on main
- [ ] Version bumped correctly in all files
- [ ] CHANGELOG.rst updated
- [ ] Git tag created (check with `git tag -l`)
- [ ] All changes committed

## Expected File Structure After Merge
```
multidms/
├── __init__.py              # Unchanged
├── jaxmodels.py             # New module being added
├── biophysical.py           # Unchanged
├── data.py                  # Unchanged
├── model.py                 # Unchanged
├── model_collection.py      # Unchanged
├── plot.py                  # Unchanged
└── utils.py                 # Unchanged

tests/
├── test_jaxmodels.py        # New test file
└── test_data.py             # Existing

notebooks/jaxmodels/
├── jaxmodels.ipynb                     # New notebook
├── jaxmodels_simulation_fits.ipynb     # New notebook
└── jaxmodels_empirical_fits.ipynb      # New notebook
```

## Import Pattern
```python
import multidms.jaxmodels as jaxmodels
# Access classes: jaxmodels.Data, jaxmodels.Latent, jaxmodels.Model, jaxmodels.fit
```

## Notes

### Design Decisions
- The `multidms.jaxmodels` module is added alongside `multidms.biophysical`
- The `biophysical` module remains untouched, ensuring no breaking changes
- Users can choose which API to use based on their needs
- The jaxmodels API provides a simplified interface with JAX/Equinox
- All existing unit tests should continue to pass

### Potential Issues to Watch For
- **Import errors**: Ensure the jaxmodels module is properly packaged
- **Notebook kernel restarts**: May need to restart Jupyter kernels after install
- **Cache invalidation**: Python's `__pycache__` might need clearing if imports act strangely
- **Ruff/Black**: Ensure jaxmodels.py has proper docstrings and formatting
- **Test discovery**: pytest should find tests in `test_jaxmodels.py`
- **Documentation build**: Sphinx should auto-document the jaxmodels module
- **Dependencies**: Ensure equinox and jaxtyping are properly specified in requirements

## Release Steps

### 15. Push to GitHub
- [ ] Push main branch and tags: `git push --tags origin main`
- [ ] Verify the push was successful on GitHub
- [ ] Check that the new version tag appears in the GitHub releases/tags

### 16. Monitor release
- [ ] Monitor GitHub Actions for CI/CD pipeline
- [ ] Verify all tests pass on main after push
- [ ] Check if PyPI publishing workflow triggers (based on tag)
- [ ] Verify the package is published correctly to PyPI (if auto-publishing is enabled)
- [ ] Check package version on PyPI matches the bumped version (1.2.0)

### 17. Post-release verification
- [ ] Test installing the new version: `pip install multidms==1.2.0` (in a fresh environment)
- [ ] Verify the jaxmodels module is accessible: `import multidms.jaxmodels`
- [ ] Test basic functionality with the new module
- [ ] Create GitHub release notes (optional but recommended)
- [ ] Announce the release (if applicable)
