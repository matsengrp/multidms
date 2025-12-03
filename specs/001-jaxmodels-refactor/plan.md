# Implementation Plan: JAX Models Refactoring

**Branch**: `001-jaxmodels-refactor` | **Date**: 2025-11-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-jaxmodels-refactor/spec.md`
**Status**: In Progress (Updated 2025-01-28)

**Update (2025-01-28)**: Phase 1 infrastructure complete. See spec.md "Completed Work" section for details. Architecture decision: `jaxmodels.Data` class kept as internal implementation detail.

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Refactor the multidms package to replace the legacy `biophysical.py` modeling code with the new `jaxmodels.py` implementation, releasing version 2.0.0. The refactoring maintains API compatibility for the core `Data`, `Model`, and `ModelCollection` classes while completely deprecating `biophysical.py`. All functionality will be tested through comprehensive unit tests and doctests, code duplication will be eliminated, and error handling will provide actionable user guidance. The new backend uses JAX with equinox for high-performance automatic differentiation while preserving the familiar interface scientists rely on.

## Technical Context

**Language/Version**: Python 3.9, 3.10, 3.11 (CI tested on all three versions)
**Primary Dependencies**: JAX ≥0.4.29, equinox, jaxopt, pandas ≥2.2.0, numpy (no version constraint)
**Storage**: N/A (in-memory DataFrame processing)
**Testing**: pytest with doctests (`pytest --doctest-modules -vv`)
**Target Platform**: Ubuntu and macOS (Linux and macOS CI validation)
**Project Type**: Single Python package (scientific library)
**Performance Goals**: Model fitting within 2x of v1.x performance for standard datasets (>1000 variants, 3 conditions)
**Constraints**: >90% code coverage, Black formatting (line 89), Ruff linting, JAX JIT compilation compatible
**Scale/Scope**: Datasets up to millions of variants (memory permitting), 10 public classes, comprehensive doctests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Correctness First ✓ **PASS**
- All jaxmodels.py biophysical equations will be validated against published theory
- FR-043: Tests MUST validate numerical correctness against expected biophysical equations
- Simulation validation included in User Story 5 acceptance criteria
- Breaking changes to algorithms documented with before/after validation

### II. Documentation-Driven Development ✓ **PASS**
- FR-041: NumPy-style docstrings required for all public functions/classes
- FR-042: Doctests required in docstrings and must pass with pytest
- FR-052: Migration guide from v1.x to v2.0 required
- FR-054: Breaking changes documented with before/after examples
- Notebook examples must run successfully (SC-001)

### III. Simple, Focused Interface ✓ **PASS**
- Core workflow preserved: `Data` → `Model` → `ModelCollection`
- FR-053: Maintain compatibility for core API where possible
- FR-008: Model class wraps jaxmodels while maintaining backward-compatible API
- No new top-level classes introduced; refactoring is internal

### IV. Code Quality Standards ✓ **PASS**
- FR-040: Black formatting (line length 89) and Ruff linting required
- FR-038: >90% code coverage required
- SC-003: All tests must pass on Python 3.9, 3.10, 3.11 on Ubuntu and macOS
- GitHub Actions will enforce automatically

### V. Reproducibility and Stability ✓ **PASS**
- Dependency versions pinned (numpy ≤1.26.0, altair ==5.1.2, pandas ≥2.2.0)
- FR-055: Version updated to 2.0.0 following semantic versioning
- CI testing on multiple Python versions and platforms (SC-003)
- SC-006: Documentation includes working examples with clear API reference

### VI. JAX-First Performance ✓ **PASS**
- Core computational refactoring to jaxmodels.py using JAX transformations (jit, grad, vmap)
- FR-013: Global epistasis options from jaxmodels (Identity, Sigmoid)
- FR-014: Loss function selection using jaxmodels implementations
- SC-004: Performance within 2x of v1.x (acceptable for correctness-first refactoring)

### VII. Scientific Rigor ✓ **PASS**
- FR-043: Numerical validation confirms jaxmodels correctness against biophysical equations
- SC-001: Existing example notebooks must produce scientifically valid results
- Cross-validation workflows preserved in Model class
- User Story 5 ensures comprehensive testing and validation

**Overall Status**: ✅ **ALL GATES PASS** - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/001-jaxmodels-refactor/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
multidms/                # Main package directory
├── __init__.py          # Package exports (Data, Model, ModelCollection)
├── data.py              # Data class - REFACTOR to use jaxmodels
├── model.py             # Model class - REFACTOR to use jaxmodels
├── model_collection.py  # ModelCollection class - REFACTOR to use jaxmodels
├── jaxmodels.py         # NEW backend (already exists, core implementation)
├── biophysical.py       # DEPRECATE - raise ImportError on import
├── plot.py              # Visualization utilities - UPDATE for jaxmodels
└── utils.py             # Helper functions - UPDATE as needed

tests/                   # Test suite
├── test_data.py         # Data class tests - EXPAND coverage
├── test_model.py        # Model class tests - CREATE/EXPAND
├── test_model_collection.py  # ModelCollection tests - CREATE/EXPAND
└── test_jaxmodels.py    # jaxmodels numerical validation - CREATE

notebooks/               # Example workflows
├── fit_delta_BA1_example.ipynb  # UPDATE to use v2.0
└── [other notebooks]    # UPDATE all to validate v2.0

docs/                    # Sphinx documentation
├── [various .rst files] # UPDATE API references and migration guide
└── conf.py              # Documentation configuration
```

**Structure Decision**: This is a single Python package refactoring. The existing flat module structure in `multidms/` is preserved. The refactoring is internal - replacing the `biophysical.py` backend with `jaxmodels.py` while maintaining the public API surface of `Data`, `Model`, and `ModelCollection` classes. No new directories or major structural changes are needed.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations - all constitution checks pass.

---

## Post-Design Constitution Re-Check

**Date**: 2025-11-10
**Phase**: After Phase 1 (Design & Contracts)

### Re-Evaluation Results

All constitution principles continue to be satisfied after design phase:

✅ **I. Correctness First**: Data model includes comprehensive validation contracts. Error handling specified for all edge cases (invalid data, unseen mutations, convergence failures). Numerical validation tests defined in contracts.

✅ **II. Documentation-Driven Development**: Contracts define complete API documentation including examples, error messages, and post-conditions. Quickstart provides runnable examples. All public methods have specifications.

✅ **III. Simple, Focused Interface**: Design preserves Data → Model → ModelCollection workflow. No new top-level classes. API compatibility maintained. User-facing changes minimal.

✅ **IV. Code Quality Standards**: Contracts specify Black/Ruff requirements. Testing strategy includes doctests, unit tests, and coverage targets (>90%). All formatting rules preserved.

✅ **V. Reproducibility and Stability**: Version pinning documented. Multi-platform testing specified. Parameter schemas defined for reproducible results.

✅ **VI. JAX-First Performance**: Design uses JAX transformations (jit, grad, vmap) through jaxmodels backend. Wrapper pattern isolates JAX details from users while maintaining performance. Target: ≤2x v1.x time.

✅ **VII. Scientific Rigor**: Three-tier testing strategy (unit/integration/regression). Simulation validation included. Cross-validation patterns documented. Biophysical model equations validated.

**Final Status**: ✅ **ALL GATES PASS** - Ready for implementation (Phase 2)

No design changes needed to satisfy constitution.
