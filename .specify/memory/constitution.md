# multidms Constitution

## Core Principles

### I. Correctness First (NON-NEGOTIABLE)
Scientific software analyzing laboratory data demands absolute correctness. Every change must prioritize accuracy over performance, convenience, or developer preferences.

**Requirements:**
- All biophysical model equations must match published theory
- Numerical computations must be validated against known results
- Breaking changes to core algorithms require simulation validation
- When in doubt, default to the mathematically rigorous approach
- Document any approximations or simplifications explicitly

### II. Documentation-Driven Development
Code without documentation is incomplete. Documentation is not optional—it's part of the implementation.

**Requirements:**
- NumPy-style docstrings required for all public functions, classes, and methods
- Doctests are the primary testing mechanism—include runnable examples in docstrings
- Complex algorithms must reference equations in the biophysical model documentation
- Changes to public APIs require documentation updates before PR approval
- Examples in notebooks/ demonstrate real-world usage patterns

### III. Simple, Focused Interface
Users are scientists analyzing data, not software engineers. The API must be minimal, intuitive, and hard to misuse.

**Design Constraints:**
- Core workflow: `Data` → `Model` → `ModelCollection` → results
- Each class has a single, clear responsibility
- Parameter names reflect scientific concepts, not implementation details
- Sensible defaults for common use cases
- Advanced features exposed through optional parameters, not new classes
- If a feature requires extensive documentation to use correctly, reconsider the design

### IV. Code Quality Standards (NON-NEGOTIABLE)
Consistency enables collaboration. Style enforcement is automated and non-negotiable.

**Enforcement:**
- Black formatting with line length 89 (run `black .` before every commit)
- Ruff linting must pass (`ruff check .`)
- GitHub Actions will reject non-compliant PRs automatically
- No exceptions without explicit approval and documentation

### V. Reproducibility and Stability
Scientific results must be reproducible across environments and time.

**Requirements:**
- Version pinning for dependencies with known breaking changes (e.g., numpy, altair)
- CI testing on Python 3.9, 3.10, 3.11 on Ubuntu and macOS
- Bumpver for coordinated version updates across all files
- Changes to numerical algorithms require validation that results remain consistent
- Random number generation must use explicit seeds in examples and tests

### VI. JAX-First Performance
JAX enables high-performance automatic differentiation. Use it appropriately, but don't sacrifice clarity.

**Guidelines:**
- Core computational kernels in `multidms.biophysical` use JAX transformations (jit, grad, vmap)
- Pure functions preferred for JAX compatibility
- Profile before optimizing—premature optimization causes maintenance burden
- Document any JAX-specific constraints (e.g., static shapes, pure functions)
- Provide clear error messages when JAX constraints are violated

### VII. Scientific Rigor
This package supports peer-reviewed research. Scientific integrity is paramount.

**Standards:**
- Biophysical models grounded in published theory with citations
- Simulation validation for new model features
- Cross-validation workflows built into Model class
- Statistical methods must cite primary sources
- Results must be reproducible from documented parameters

## Development Workflow

### Testing Hierarchy
2. **Unit tests** (primary): Isolated tests in `tests/` for complex edge cases
1. **Doctests**: Inline examples in docstrings, run via `pytest --doctest-modules`
3. **Notebook validation**: Jupyter notebooks in `notebooks/` demonstrate real workflows
4. **CI validation**: Automated testing on multiple Python versions and platforms

### Pull Request Requirements
- Black formatting applied (`black .`)
- Ruff linting passes (`ruff check .`)
- All doctests and unit tests pass (`pytest --doctest-modules -vv`)
- Documentation updated for API changes
- One conceptual change per PR (see CONTRIBUTING.rst)
- Clear commit messages describing the "why" not just the "what"

### Versioning and Releases
- MAJOR.MINOR.PATCH semantic versioning via bumpver
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, documentation improvements
- Release process: `bumpver update --patch/minor/major` → `git push --tags origin main`

## Constraints and Boundaries

### What This Package Is
- A tool for joint modeling of multiple DMS experiments
- A library for global-epistasis model fitting with JAX optimization
- A platform for estimating mutation effects across experimental conditions

### What This Package Is Not
- A general-purpose bioinformatics toolkit
- A visualization library (we provide plotting utilities, not a framework)
- A data preprocessing pipeline (minimal data cleaning; users prepare their data)

### Technology Commitments
- Python 3.9+ required
- JAX ecosystem for optimization (jax, jaxopt)
- pandas for data structures (DataFrames with specific column conventions)
- Multiple visualization backends (matplotlib, seaborn, altair) for different needs

## Governance

This constitution reflects the values and practices of the multidms project as developed by the Matsen group, William DeWitt, and the Bloom Lab.

**Amendment Process:**
1. Propose changes via GitHub issue with rationale
2. Discuss with maintainers and stakeholders
3. Document impact on existing code
4. Update constitution with version bump
5. Communicate changes to contributors

**Enforcement:**
- Pull requests must comply with all NON-NEGOTIABLE principles
- CI automatically enforces code quality standards
- Maintainers verify scientific rigor and documentation completeness
- Non-compliant changes will not be merged, even if functionally correct

**Questions and Clarifications:**
- Raise issues on GitHub: https://github.com/matsengrp/multidms/issues
- Contact: jgallowa@fredhutch.org

---

**Version**: 1.0.0 | **Ratified**: 2025-01-28 | **Last Amended**: 2025-01-28
