# Research: JAX Models Refactoring

**Feature**: 001-jaxmodels-refactor
**Date**: 2025-11-10
**Status**: Complete

## Overview

This document captures research findings and technical decisions for refactoring multidms from `biophysical.py` to `jaxmodels.py` backend. Since `jaxmodels.py` already exists and implements the core modeling logic, research focuses on integration patterns, API preservation strategies, and migration approaches.

## Key Research Questions & Findings

### 1. JAX and Equinox Integration Patterns

**Question**: What are the best practices for wrapping JAX/equinox models in a user-friendly Python API?

**Research Findings**:
- **equinox** provides `eqx.Module` base class for PyTree-compatible models with automatic parameter management
- JAX functions (`jit`, `grad`, `vmap`) require pure functions; state must be passed explicitly
- Common pattern: Wrapper class holds JAX model internally, exposes high-level methods that handle state transformation
- Parameter initialization and updates should use functional style (return new state rather than mutate)

**Decision**:
- `Model` class will wrap `jaxmodels.Model` internally as a private attribute
- Public API methods (`fit()`, `get_mutations_df()`, etc.) will handle conversion between pandas DataFrames and JAX arrays
- Parameter access through `model.params` property returns the current JAX PyTree

**Rationale**:
- Maintains familiar object-oriented API for scientists
- Isolates JAX-specific details from users
- Enables gradual refactoring without breaking existing code

**Alternatives Considered**:
- Expose JAX model directly → Rejected: Too low-level, breaks existing API
- Create parallel v2 API → Rejected: Fragments user base, doubles maintenance burden

---

### 2. Data Architecture and jaxmodels Integration

**Question**: How should `multidms.data.Data` integrate with the jaxmodels backend?

**Research Findings**:
- `jaxmodels.py` needs: per-condition binary maps (X), model targets (y), and optional count data
- Current `Data` class already provides this via `arrays["X"]` and `arrays["y"]` dictionaries
- `jaxmodels.Data` exists as a type-annotated wrapper providing clean interface to jaxmodels
- `jaxmodels.Data.from_multidms()` conversion method works well

**Decision (UPDATED 2025-01-28)**:
- **KEEP `jaxmodels.Data` class as internal implementation detail**
- `multidms.data.Data` provides per-condition data via existing dictionaries:
  - `arrays["X"]` - dictionary of binary maps per condition
  - `arrays["y"]` - dictionary of functional scores per condition
  - `arrays["pre_count"]`, `arrays["post_count"]` - optional count data
- `jaxmodels.Data.from_multidms(data, condition)` converts for each condition
- User-facing `Model` class wraps this conversion internally
- **IMPORTANT**: Data class will NOT aggregate identical variants automatically. Users must handle barcode collapsing/aggregation themselves before creating Data object.

**Rationale**:
- `jaxmodels.Data` provides clean type-annotated interface for jaxmodels backend
- Existing conversion via `.from_multidms()` works well
- Removing it would require significant refactoring of `jaxmodels.fit()`
- Since it's internal-only (not user-facing), keeping it simplifies implementation
- No memory overhead since conversion happens on-demand per condition

**Alternatives Considered**:
- Remove `jaxmodels.Data` entirely → Rejected: Requires extensive jaxmodels refactoring
- Add properties to `multidms.Data` for concatenated arrays → Rejected: Loses per-condition structure
- Store JAX arrays natively in Data → Rejected: Breaks pandas-based API

---

### 3. Warmstart Ridge Regression Without Counts

**Question**: How can warmstart Ridge regression work without count data (as specified in clarifications)?

**Research Findings**:
- Current `jaxmodels.py` warmstart uses counts as sample weights in Ridge solver
- Ridge regression mathematically doesn't require weights - they're optional for heteroscedastic data
- scikit-learn and JAX-based solvers support unweighted Ridge regression

**Decision**:
- Modify `jaxmodels.py` warmstart logic to make `sample_weights` optional
- When counts unavailable: Run Ridge regression with equal weights (or no weights)
- Validate this produces reasonable initialization through unit tests

**Rationale**:
- Makes warmstart more flexible and accessible
- Equal-weight Ridge is still a principled initialization strategy
- Aligns with clarification decision

**Alternatives Considered**:
- Skip warmstart entirely without counts → Rejected: Loses valuable initialization signal
- Require counts for warmstart → Rejected: Contradicts clarification decision

**Implementation Notes**:
- Check `jaxmodels.py` Ridge solver for weight parameter
- If necessary, add conditional logic: `weights = sample_weights if available else None`
- Test convergence quality with and without weights

---

### 4. Deprecation Strategy for biophysical.py

**Question**: What's the best way to deprecate a core module while providing clear migration guidance?

**Research Findings**:
- Python standard: `warnings.warn(DeprecationWarning)` for soft deprecation
- Hard deprecation: Raise `ImportError` immediately
- Clarification specifies hard deprecation (ImportError) for v2.0

**Decision**:
- Replace `biophysical.py` content with single import guard that raises `ImportError`
- Error message includes:
  - Clear statement that module is deprecated in v2.0
  - Link to migration guide documentation
  - Suggestion to use multidms v1.x if biophysical.py is required

**Implementation**:
```python
# biophysical.py (v2.0)
raise ImportError(
    "The multidms.biophysical module has been deprecated in version 2.0.0. "
    "All modeling functionality now uses the jaxmodels backend. "
    "Please see the migration guide at https://multidms.readthedocs.io/en/latest/migration_v2.html "
    "If you require the legacy biophysical module, please use multidms v1.x."
)
```

**Rationale**:
- Clear, immediate feedback prevents silent failures
- Provides actionable next steps
- Aligns with clarification decision (Option A)

**Alternatives Considered**:
- Soft deprecation with warnings → Rejected: Clarification specified hard error
- Remove file entirely → Rejected: Less informative error message

---

### 5. Test Strategy for Numerical Correctness

**Question**: How do we validate that jaxmodels produces mathematically correct results?

**Research Findings**:
- Scientific computing best practice: Compare against analytical solutions or validated reference implementations
- JAX computation graphs can be tested for correct gradient flow
- Simulation validation: Generate synthetic data with known parameters, verify recovery

**Decision**:
Three-tier testing approach:

1. **Unit tests**: Test individual jaxmodels components (latent phenotype calculation, global epistasis functions, loss functions)
2. **Integration tests**: Fit models to synthetic data with known ground truth, verify parameter recovery
3. **Regression tests**: Compare v2.0 results to v1.x results on real datasets (should be nearly identical)

**Test Implementation**:
- `tests/test_jaxmodels.py`: Numerical validation against biophysical equations
- `tests/test_model.py`: Integration tests with synthetic data
- `tests/test_migration.py`: Regression tests comparing v1.x and v2.0 outputs

**Rationale**:
- Multi-level validation catches errors at different abstractions
- Synthetic data tests prove correctness independent of legacy code
- Regression tests ensure continuity for existing users

---

### 6. Error Handling and Validation Patterns

**Question**: What's the best way to validate inputs and provide actionable error messages?

**Research Findings**:
- Python standard: Raise specific exception types (`ValueError`, `TypeError`, `KeyError`)
- Scientific libraries (numpy, pandas, scikit-learn) use detailed error messages with examples
- JAX errors can be cryptic; wrapper should catch and translate them

**Decision**:
- Input validation in `Data` and `Model` constructors (fail fast)
- JAX errors caught and re-raised with user-friendly messages
- Error messages follow pattern: "What went wrong" + "How to fix it"

**Example Error Messages**:
```python
# Missing columns
raise ValueError(
    f"DataFrame missing required columns: {missing_cols}. "
    f"Expected columns: {required_cols}. "
    f"Please ensure your DataFrame has 'condition', 'aa_substitutions', and 'func_score' columns."
)

# Invalid functional scores
raise ValueError(
    f"Found {n_invalid} rows with NaN or infinite functional scores at indices: {invalid_rows[:10]}... "
    f"Please clean your data before creating a Data object. "
    f"You can use df.dropna() or df[np.isfinite(df.func_score)] to filter invalid values."
)

# Unseen mutations
raise ValueError(
    f"Cannot predict on variants with unseen mutations: {unseen_muts}. "
    f"These mutations were not present in the training data. "
    f"Please retrain the model with a dataset that includes these mutations."
)
```

**Rationale**:
- Scientists may not be experienced programmers; errors must be educational
- Actionable guidance reduces support burden
- Aligns with FR-044 through FR-050 error handling requirements

---

### 7. Performance Optimization Strategy

**Question**: How can we ensure v2.0 meets the "within 2x of v1.x" performance target?

**Research Findings**:
- JAX JIT compilation can be slower on first call but much faster on subsequent calls
- JAX compilation overhead depends on input shapes; static shapes perform better
- Profiling tools: `jax.profiler`, Python `cProfile`

**Decision**:
- Profile v1.x baseline on standard datasets (>1000 variants, 3 conditions)
- Profile v2.0 and identify bottlenecks
- Optimization priorities:
  1. Ensure JAX functions are JIT-compiled where appropriate
  2. Minimize DataFrame ↔ JAX array conversions
  3. Avoid redundant computations in visualization methods

**Acceptance Criteria**:
- Model fitting time ≤ 2x v1.x for standard datasets
- If >2x, profile and optimize hot paths before release

**Rationale**:
- Performance targets are relative, not absolute (2x is acceptable for correctness-first refactoring)
- Premature optimization avoided; optimize only if benchmarks fail
- JAX should be faster for large datasets due to better vectorization

---

## Summary of Key Decisions

| Decision Area | Choice | Rationale |
|---------------|--------|-----------|
| Wrapper Pattern | Model wraps jaxmodels internally | Preserves API, isolates JAX details |
| Data Architecture | Keep jaxmodels.Data as internal, Data provides per-condition dicts | Clean type interface, minimal refactoring needed |
| Variant Aggregation | NO automatic aggregation in Data class | Users handle preprocessing, full control, simpler Data class |
| Warmstart Strategy | Make sample_weights optional | Enables warmstart without counts |
| Deprecation | Hard error (ImportError) on biophysical.py | Clear feedback, aligns with clarifications |
| Testing | Three-tier (unit/integration/regression) | Comprehensive validation of correctness |
| Error Messages | Descriptive + actionable guidance | User-friendly for scientists |
| Performance | Profile-driven optimization | Meet 2x target without premature optimization |

## Dependencies and Constraints

### Confirmed Dependencies
- JAX ≥0.4.29: Core computational framework
- equinox: PyTree-compatible model abstraction
- jaxopt: Optimization algorithms
- pandas ≥2.2.0: DataFrame operations
- numpy ≤1.26.0: Array operations (version pinned for stability)

### Technical Constraints
- JAX JIT requirements: Pure functions, static shapes where possible
- Python 3.9-3.11 compatibility: CI tested on all versions
- Black formatting (line 89) and Ruff linting: Non-negotiable
- >90% code coverage: Required for release

### Integration Points
- `jaxmodels.py`: Core modeling backend (already implemented)
- `Data` class: Preprocessing and one-hot encoding
- `Model` class: Fitting, prediction, visualization
- `ModelCollection`: Parallel model fitting and aggregation
- Visualization: matplotlib, seaborn, altair plotting functions

## Next Steps

Phase 1 will use these research findings to:
1. Design data model showing relationship between Data, Model, jaxmodels.Data, and jaxmodels.Model
2. Define API contracts for public methods with input/output specifications
3. Create quickstart guide demonstrating the refactored workflow
4. Update agent context with jaxmodels-specific technical details

---

**Research Complete**: 2025-11-10
**Ready for**: Phase 1 (Design & Contracts)
