# Implementation Tasks: JAX Models Refactoring

**Feature**: 001-jaxmodels-refactor
**Branch**: `001-jaxmodels-refactor`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Overview

This document provides implementation tasks for refactoring multidms to use jaxmodels backend, organized by user story priority for incremental delivery.

**Total Tasks**: 47
**User Stories**: 5 (P1: US1, US5 | P2: US2, US4 | P3: US3)
**MVP Scope**: User Story 1 + User Story 5 (Basic fitting + Testing)

## Implementation Strategy

1. **MVP First**: Complete User Story 1 (Basic Model Fitting) and User Story 5 (Testing) for core functionality
2. **Incremental Delivery**: Each user story is independently testable and can be completed in isolation
3. **Parallel Opportunities**: Tasks marked [P] can be developed concurrently
4. **Test-Driven**: User Story 5 tasks create test infrastructure used throughout

---

## Phase 1: Setup & Infrastructure

**Goal**: Initialize project dependencies and deprecate biophysical.py

### Tasks

- [ ] T001 Deprecate biophysical.py module by replacing content with ImportError in multidms/biophysical.py
- [ ] T002 Verify JAX dependencies in pyproject.toml are correct (JAX ≥0.4.29, equinox, jaxopt, pandas ≥2.2.0)
- [ ] T003 [P] Create migration guide documentation in docs/migration_v2.rst
- [ ] T004 [P] Start CHANGELOG.rst section for v2.0.0 (track breaking changes as development progresses)

**Acceptance**: biophysical.py raises ImportError with helpful message, dependencies validated, migration guide and changelog started

**Note**: Version remains at 1.x during development. Intermediate functional versions can use remaining 1.x numbers (e.g., 1.3.0, 1.4.0). Version 2.0.0 will be set in Phase 7 when all work is complete.

---

## Phase 2: Foundational Infrastructure (User Story 5 - Testing Foundation)

**Goal**: Create comprehensive test infrastructure for validating jaxmodels integration

**User Story**: US5 - Testing and Validation (P1)

**Independent Test**: Run `pytest tests/` and verify test infrastructure is working with >90% coverage capability

### Tasks

- [ ] T005 [US5] Create tests/test_jaxmodels.py for numerical validation of jaxmodels computations
- [ ] T006 [US5] Expand tests/test_data.py to include new Data properties (binary_map, targets, condition_indices, weights)
- [ ] T007 [US5] Create tests/test_model.py for Model class with jaxmodels backend integration
- [ ] T008 [US5] Create tests/test_model_collection.py for ModelCollection with jaxmodels models
- [ ] T009 [P] [US5] Set up pytest configuration for doctest support in pyproject.toml or pytest.ini
- [ ] T010 [P] [US5] Configure coverage reporting with >90% target in pyproject.toml

**Acceptance**: Test files created, pytest runs successfully, coverage infrastructure configured

---

## Phase 3: User Story 1 - Basic Model Fitting (P1)

**Goal**: Enable basic DMS model fitting with jaxmodels backend

**User Story**: US1 - Basic Model Fitting with New Backend

**Independent Test**: Create Data object, fit Model, verify predictions match jaxmodels computation

### 3.1 Data Class Refactoring

- [ ] T011 [US1] Add binary_map property to Data class in multidms/data.py returning (n_variants × n_mutations) numpy array
- [ ] T012 [US1] Add targets property to Data class returning dict[str, np.ndarray] of functional scores per condition
- [ ] T013 [US1] Add condition_indices property to Data class returning np.ndarray mapping variants to condition integers
- [ ] T014 [US1] Add weights property to Data class returning optional np.ndarray from count data
- [ ] T015 [US1] Remove collapse_identical_variants parameter from Data.__init__() method
- [ ] T016 [US1] Update Data class docstring with NumPy-style docs and doctests for new properties
- [ ] T017 [US1] Add validation in Data.__init__() to check for NaN/inf in func_score column (FR-001a)
- [ ] T018 [US1] Add validation to check count data presence when needed (will be used by Model)

### 3.2 Model Class Refactoring

- [ ] T019 [US1] Refactor Model.__init__() in multidms/model.py to accept Data and store reference
- [ ] T020 [US1] Remove internal conversion to jaxmodels.Data (Data now provides properties directly)
- [ ] T021 [US1] Update Model.fit() to pass data.binary_map, data.targets, data.condition_indices, data.weights to jaxmodels.Model
- [ ] T022 [US1] Implement warmstart Ridge regression without requiring count data (use data.weights if available, else None)
- [ ] T023 [US1] Update Model.params property to return jaxmodels parameters (beta, beta0, shift, theta)
- [ ] T024 [US1] Implement Model.get_mutations_df() to extract parameters from jaxmodels fitted model
- [ ] T025 [US1] Implement Model.get_variants_df() to extract variant predictions from jaxmodels
- [ ] T026 [US1] Implement Model.add_phenotypes_to_df() for predictions on new data
- [ ] T027 [US1] Add validation to detect unseen mutations and raise clear error in add_phenotypes_to_df()
- [ ] T028 [US1] Update Model.convergence_trajectory_df to track loss and error from jaxmodels optimization
- [ ] T029 [US1] Update Model class docstring with complete NumPy-style docs and doctests

### 3.3 Error Handling for User Story 1

- [ ] T030 [US1] Add error message for missing count data when loss_type='count_loss' (FR-044a)
- [ ] T031 [US1] Add error message for warmstart parameter shape mismatches (FR-047, FR-048)
- [ ] T032 [US1] Add convergence failure warning with actionable guidance (FR-045)
- [ ] T033 [US1] Add error handling for numerical errors in jaxmodels optimization (FR-049)

### 3.4 Testing for User Story 1

- [ ] T034 [US1] Write unit tests in tests/test_data.py for new Data properties (binary_map, targets, etc.)
- [ ] T035 [US1] Write integration tests in tests/test_model.py for Model.fit() with jaxmodels backend
- [ ] T036 [US1] Write tests for Model.get_mutations_df() and Model.get_variants_df() output format
- [ ] T037 [US1] Write tests for error handling (missing columns, invalid data, unseen mutations)
- [ ] T038 [US1] Write regression tests comparing v1.x and v2.0 outputs on same dataset
- [ ] T039 [US1] Write doctest examples in Data and Model docstrings and verify they pass

**US1 Acceptance**: Data object created, Model fits successfully, get_mutations_df() and get_variants_df() return correct data, all tests pass

---

## Phase 4: User Story 2 - Visualization (P2)

**Goal**: Ensure all visualization methods correctly display jaxmodels-computed parameters

**User Story**: US2 - Visualization and Exploration

**Independent Test**: Fit model, call visualization methods, verify plots display jaxmodels data

**Dependencies**: Requires US1 (basic fitting must work first)

### 4.1 Visualization Method Updates

- [ ] T040 [P] [US2] Update Model.plot_epistasis() in multidms/model.py to extract data from jaxmodels params
- [ ] T041 [P] [US2] Update Model.plot_pred_accuracy() to use jaxmodels predictions
- [ ] T042 [P] [US2] Update Model.plot_param_hist() to work with jaxmodels params structure
- [ ] T043 [P] [US2] Update Model.mut_param_heatmap() to extract jaxmodels beta/shift parameters

### 4.2 Plot Module Updates

- [ ] T044 [P] [US2] Review and update multidms/plot.py helper functions to work with jaxmodels data structures
- [ ] T045 [P] [US2] Add docstrings and doctests to all updated visualization methods

### 4.3 Testing for User Story 2

- [ ] T046 [US2] Write tests in tests/test_model.py verifying plot methods execute without error
- [ ] T047 [US2] Write visual regression tests comparing v1.x and v2.0 plot outputs (optional, can be manual)

**US2 Acceptance**: All plot methods work correctly, display jaxmodels-computed values, tests pass

---

## Phase 5: User Story 4 - Enhanced Error Handling (P2)

**Goal**: Provide clear, actionable error messages for all failure scenarios

**User Story**: US4 - Error Handling and Diagnostics

**Independent Test**: Intentionally trigger error conditions, verify messages are helpful

**Dependencies**: Can be developed in parallel with US2

### 5.1 Data Validation Enhancements

- [ ] T048 [P] [US4] Enhance Data class error messages for missing columns (FR-044)
- [ ] T049 [P] [US4] Add detailed error message for invalid alphabet specification (FR-046)
- [ ] T050 [P] [US4] Add error message for invalid reference condition (FR-050)

### 5.2 Model Validation Enhancements

- [ ] T051 [P] [US4] Improve convergence failure warning message with specific guidance (FR-045)
- [ ] T052 [P] [US4] Add helpful error for unseen mutations with list of problematic mutations (FR-024)
- [ ] T053 [P] [US4] Add validation error messages for incompatible model parameters

### 5.3 Testing for User Story 4

- [ ] T054 [US4] Write comprehensive error message tests in tests/test_data.py and tests/test_model.py
- [ ] T055 [US4] Verify all error messages are actionable (manually or via user testing)

**US4 Acceptance**: All error scenarios produce clear, actionable messages with guidance for resolution

---

## Phase 6: User Story 3 - ModelCollection (P3)

**Goal**: Enable parallel model fitting and comparison across hyperparameters

**User Story**: US3 - Multiple Model Fitting and Comparison

**Independent Test**: Create parameter grid, call fit_models(), verify ModelCollection works with jaxmodels

**Dependencies**: Requires US1 (basic fitting)

### 6.1 ModelCollection Refactoring

- [ ] T056 [US3] Update fit_models() function in multidms/model_collection.py to use jaxmodels backend
- [ ] T057 [US3] Verify ModelCollection class works with Model objects containing jaxmodels params
- [ ] T058 [US3] Update ModelCollection.split_apply_combine_muts() for jaxmodels param structure
- [ ] T059 [US3] Update ModelCollection.mut_param_dataset_correlation() for jaxmodels
- [ ] T060 [US3] Update ModelCollection.shift_sparsity() for jaxmodels shift parameters
- [ ] T061 [US3] Update ModelCollection.mut_param_heatmap() for jaxmodels params
- [ ] T062 [US3] Update ModelCollection.mut_param_traceplot() for jaxmodels params

### 6.2 Documentation and Testing

- [ ] T063 [P] [US3] Update ModelCollection docstrings with NumPy-style docs and doctests
- [ ] T064 [US3] Write tests in tests/test_model_collection.py for parallel fitting with jaxmodels
- [ ] T065 [US3] Write tests for aggregation methods with jaxmodels-fitted models

**US3 Acceptance**: fit_models() works with parameter grids, ModelCollection aggregation methods produce correct output

---

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Finalize documentation, performance validation, and release readiness

### 7.1 Documentation

- [ ] T066 [P] Update README.md with v2.0 quick start example
- [ ] T067 [P] Update all existing notebooks to use v2.0 API (notebooks/*.ipynb)
- [ ] T068 [P] Verify all example notebooks run successfully (SC-001)
- [ ] T069 [P] Generate updated API documentation with Sphinx (make -C docs html)
- [ ] T070 [P] Complete migration guide with all breaking changes documented (FR-052, FR-054)

### 7.2 Code Quality

- [ ] T071 [P] Run Black formatter on all modified files (black multidms/ tests/)
- [ ] T072 [P] Run Ruff linter and fix all issues (ruff check multidms/ tests/)
- [ ] T073 Verify code coverage >90% for Data, Model, ModelCollection (SC-002)
- [ ] T074 Remove all code duplication - extract to utility functions (FR-039, SC-010)

### 7.3 Performance & Validation

- [ ] T075 Benchmark v2.0 fitting time vs v1.x on standard dataset (>1000 variants, 3 conditions)
- [ ] T076 Verify performance within 2x of v1.x (SC-004)
- [ ] T077 Profile and optimize if performance target not met
- [ ] T078 Run full test suite on Python 3.9, 3.10, 3.11 (SC-003)
- [ ] T079 Run full test suite on Ubuntu and macOS (SC-003)

### 7.4 Final Validation & Release

- [ ] T080 Verify production code has zero references to biophysical.py (SC-007)
- [ ] T081 Count and document all breaking API changes (target <10) (SC-008)
- [ ] T082 Manually test error messages for common mistakes (SC-009)
- [ ] T083 Final review of all docstrings and doctests
- [ ] T084 Finalize CHANGELOG.rst with complete v2.0.0 release notes
- [ ] T085 Update version to 2.0.0 in pyproject.toml (FR-055)
- [ ] T086 Create git tag v2.0.0 and push to trigger release workflow
- [ ] T087 Create GitHub release v2.0.0 with changelog and migration guide

**Phase 7 Acceptance**: All documentation complete, performance validated, code quality standards met, version 2.0.0 set, ready for release

---

## Dependencies & Execution Order

### Story Completion Order

```
Phase 1 (Setup)
    ↓
Phase 2 (US5 Foundation)
    ↓
Phase 3 (US1 - Basic Fitting) ← MVP CORE
    ↓
Phase 4 (US2 - Visualization) \
                                → Phase 7 (Polish)
Phase 5 (US4 - Error Handling) /
    ↓
Phase 6 (US3 - ModelCollection)
    ↓
Phase 7 (Polish & Release)
```

### Critical Path

1. **MVP (Phases 1-3)**: Setup → Testing Foundation → Basic Fitting
2. **Full Feature Set**: Add US2 (Visualization), US4 (Error Handling), US3 (ModelCollection)
3. **Release Ready**: Polish phase

### Blocking Dependencies

- **US2, US4, US3**: All require US1 (basic fitting) to be complete
- **US2 and US4**: Can be developed in parallel
- **US3**: Can start after US1, independent of US2/US4
- **Phase 7**: Requires all user stories complete

---

## Parallel Execution Opportunities

### Phase 1 (Setup)
- T003 (migration guide) || T004 (CHANGELOG) can run in parallel with T001, T002

### Phase 2 (US5 Foundation)
- T005-T008 (test file creation) can all run in parallel
- T009, T010 (config) can run in parallel

### Phase 3 (US1)
- After T011-T014 complete: T016 (docs) can run parallel with T015 (removal)
- T017, T018 (validation) can run in parallel
- T034-T039 (tests) can run in parallel after implementation tasks complete

### Phase 4 (US2)
- T040-T043 (visualization updates) are fully parallel
- T044-T045 (plot module) are fully parallel
- Can run entire US2 in parallel with US4

### Phase 5 (US4)
- T048-T050 (Data validation) all parallel
- T051-T053 (Model validation) all parallel
- Can run entire US4 in parallel with US2

### Phase 6 (US3)
- T063 (docs) can run parallel with T056-T062 implementation

### Phase 7 (Polish)
- T066-T070 (documentation) all parallel
- T071-T072 (code quality) all parallel
- T075-T077 (performance) can run after implementation complete

---

## Testing Strategy

### Test Levels (per User Story 5)

1. **Unit Tests**: Isolated component testing
   - tests/test_data.py: Data class properties and validation
   - tests/test_model.py: Model class methods
   - tests/test_model_collection.py: ModelCollection methods
   - tests/test_jaxmodels.py: Numerical validation

2. **Integration Tests**: Component interaction
   - Data → Model fitting pipeline
   - Model → visualization pipeline
   - ModelCollection aggregation pipeline

3. **Regression Tests**: v1.x vs v2.0 comparison
   - Same dataset, compare parameter estimates
   - Verify results within acceptable tolerance

4. **Doctests**: Inline examples in docstrings
   - Every public method must have doctest
   - Run with `pytest --doctest-modules`

### Coverage Target

- **Minimum**: >90% code coverage (FR-038, SC-002)
- **Focus**: Data, Model, ModelCollection classes
- **Tool**: pytest-cov configured in Phase 2

---

## MVP Scope Recommendation

**Minimum Viable Product**: Phases 1-3 (Setup + US5 Foundation + US1 Basic Fitting)

**Rationale**:
- Delivers core functionality (data processing + model fitting)
- Includes comprehensive testing infrastructure
- Allows early validation of jaxmodels integration
- Provides foundation for incremental feature addition

**MVP Deliverable**:
- Working Data class with new properties
- Working Model class with jaxmodels backend
- Basic fitting and prediction functionality
- Comprehensive test suite with >90% coverage
- biophysical.py deprecated

**Post-MVP Increments**:
- Increment 1: Add US2 (Visualization) and US4 (Error Handling) in parallel
- Increment 2: Add US3 (ModelCollection)
- Increment 3: Phase 7 (Polish and Release)

---

## Task Summary

| Phase | User Story | Priority | Task Count | Can Parallelize |
|-------|------------|----------|------------|-----------------|
| 1 | Setup | - | 4 | 2 tasks (T003, T004) |
| 2 | US5 (Testing) | P1 | 6 | 6 tasks (T005-T010) |
| 3 | US1 (Fitting) | P1 | 29 | ~15 tasks |
| 4 | US2 (Visualization) | P2 | 8 | 6 tasks (T040-T045) |
| 5 | US4 (Error Handling) | P2 | 8 | 6 tasks (T048-T053) |
| 6 | US3 (ModelCollection) | P3 | 10 | 1 task (T063) |
| 7 | Polish | - | 19 | 7 tasks (T066-T072) |
| **Total** | | | **84** | **~43 parallelizable** |

---

## Validation Checklist

Before marking phase complete, verify:

- [ ] **Phase 1**: biophysical.py raises ImportError, version 2.0.0, migration guide started
- [ ] **Phase 2**: Test infrastructure working, pytest runs, coverage configured
- [ ] **Phase 3 (MVP Core)**: Data object created, Model fits, predictions work, tests pass
- [ ] **Phase 4**: All visualization methods work with jaxmodels data
- [ ] **Phase 5**: All error scenarios produce helpful messages
- [ ] **Phase 6**: ModelCollection works with jaxmodels models, parallel fitting succeeds
- [ ] **Phase 7**: >90% coverage, performance <2x v1.x, all docs updated, tests pass on all platforms

---

## Notes for Implementation

1. **JAX Compatibility**: Ensure all functions passed to JAX are pure (no side effects)
2. **Breaking Changes**: Document all API changes in migration guide as discovered
3. **Numerical Validation**: Critical - validate jaxmodels computations against biophysical equations
4. **Performance**: Profile before optimizing; 2x v1.x is acceptable
5. **Error Messages**: Must be actionable - include what went wrong and how to fix it
6. **Doctests**: Every public method needs working examples in docstring

---

**Generated**: 2025-11-10
**Status**: Ready for implementation
**Next Step**: Begin Phase 1 (Setup) or start MVP (Phases 1-3)
