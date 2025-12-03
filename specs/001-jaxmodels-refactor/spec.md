# Feature Specification: JAX Models Refactoring

**Feature Branch**: `001-jaxmodels-refactor`
**Created**: 2025-01-28
**Status**: In Progress
**Input**: User description: "I am refactoring the multidms package to replace the core modeling code (found primarily in multidms/biophysical.py) with the new modeling code in multidms/jaxmodels.py. Once finished, will have version 2.0.0 of the package, which will have a very similar (to a reasonable extent) interface to the one we have now except the Data, Model, and ModelCollection interface will be calling out the multidms/jaxmodels.py modeling code, and the multidms/biophysical.py module will be completely deprecated. The final product will have integrated the new approach to modeling, with the interface and functionality of existing package. The final product will test every functionality in a unit test, will not have repeated code, will be easy to understand, will be easy to install and use, and have correct error handling with useful error messages when there are user, or model fitting errors."

## Completed Work (Session 2025-01-28)

### Phase 1: Core Infrastructure ✅
1. **biophysical.py deprecated** - Replaced with ImportError directing users to migration guide and v1.x versions
2. **Model class refactored** - New simplified wrapper around jaxmodels backend:
   - `__init__(data, loss_type, ge_type, l2reg, fusionreg, beta0_ridge)`
   - `fit(warmstart, maxiter, tol, ...)`
   - `get_mutations_df(phenotype_as_effect)`
   - `get_variants_df(phenotype_as_effect)`
   - Properties: `data`, `params`, `convergence_trajectory_df`
3. **Data class verified** - Existing structure confirmed correct; no changes needed
4. **jaxmodels.Data kept as internal** - Decision to keep as implementation detail rather than remove

### Architecture Decision: jaxmodels.Data
**Decision**: Keep `jaxmodels.Data` class as internal implementation detail.

**Rationale**: The `jaxmodels.Data` class provides a clean, type-annotated interface for the jaxmodels backend. Removing it would require significant refactoring of `jaxmodels.fit()` and related functions. Since it's not user-facing and works well, we keep it as internal plumbing.

**Updated Flow**:
```
multidms.Data → jaxmodels.Data.from_multidms() → jaxmodels.fit() → multidms.Model
```

The user-facing API remains: `Data` → `Model.fit()` → results

## Clarifications

### Session 2025-11-10

- Q: Should count data (pre_counts, post_counts) be optional or conditionally required based on the selected loss function? → A: Count data is required only when using count_loss. It is optional for functional_score_loss and optional for warmstart (warmstart should work without counts data by using Ridge regression without sample_weights)
- Q: How should the system respond when users try to import or use functions from the deprecated biophysical.py module? → A: Raise ImportError immediately when biophysical.py is imported with message directing to migration guide
- Q: When making predictions on variants with mutations not seen during training, how should the system respond? → A: Raise clear error listing the unseen mutations and suggesting retraining with expanded data
- Q: How should the system handle NaN or infinite values in the functional score column during Data object creation? → A: Raise error during Data creation listing rows with invalid values and suggesting data cleaning
- Q: How should the system handle very large datasets that may exceed available memory? → A: Document practical dataset size limits and raise informative error if memory insufficient
- Q: Should the Data class aggregate identical variants/barcodes automatically? → A: No. Users should collapse barcodes or aggregate variants themselves before creating Data object. Data class accepts data as-is and fits to either collapsed variants or barcode replicates based on user's preprocessing choice.
- Q: Should we have a separate jaxmodels.Data class for JAX-compatible data representation? → A: **UPDATED (2025-01-28)**: Keep jaxmodels.Data as internal implementation detail. The multidms.Data class already provides per-condition data via arrays["X"] and arrays["y"] dictionaries. The jaxmodels.Data.from_multidms() conversion works well and requires no changes.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Model Fitting with New Backend (Priority: P1)

A researcher using multidms imports the package and fits a DMS model to their experimental data using the existing `Data`, `Model`, and `fit()` API, but the computation now uses the new jaxmodels implementation internally rather than the legacy biophysical module.

**Why this priority**: This is the core functionality of the package - researchers must be able to fit models to their data. Without this, the package has no value. All other features depend on this working correctly.

**Independent Test**: Can be fully tested by creating a Data object, initializing a Model, calling fit(), and verifying predictions match expected values from the new jaxmodels backend.

**Acceptance Scenarios**:

1. **Given** a researcher has DMS experimental data in the required format, **When** they create a `multidms.Data` object and `multidms.Model` object and call `model.fit()`, **Then** the model fits successfully using jaxmodels backend and produces valid parameter estimates
2. **Given** a fitted model, **When** the researcher calls `model.get_mutations_df()`, **Then** they receive mutation effect parameters (beta and shift values) consistent with the jaxmodels computation
3. **Given** a fitted model, **When** the researcher calls `model.get_variants_df()`, **Then** they receive variant predictions with latent phenotype and functional score values computed by jaxmodels
4. **Given** a researcher has existing code using multidms v1.x, **When** they upgrade to v2.0 and re-run with minimal API changes, **Then** their workflow completes successfully with results from the new backend

---

### User Story 2 - Visualization and Exploration (Priority: P2)

A researcher uses visualization methods to explore fitted model parameters and assess model quality, with all plots correctly reflecting data from the jaxmodels backend.

**Why this priority**: Visualizations are critical for scientific interpretation and model validation. Researchers need to trust that plots accurately represent the new model's parameters. This is P2 because basic fitting (P1) must work first.

**Independent Test**: Can be tested by fitting a model, calling visualization methods (`plot_epistasis()`, `plot_pred_accuracy()`, `mut_param_heatmap()`), and verifying plots correctly display jaxmodels-computed values.

**Acceptance Scenarios**:

1. **Given** a fitted model, **When** researcher calls `model.plot_epistasis()`, **Then** the global epistasis curve reflects the jaxmodels-computed transformation function
2. **Given** a fitted model, **When** researcher calls `model.plot_pred_accuracy()`, **Then** the scatter plot shows observed vs. predicted functional scores from jaxmodels predictions
3. **Given** a fitted model, **When** researcher calls `model.mut_param_heatmap()`, **Then** the interactive heatmap displays beta and shift parameters computed by jaxmodels
4. **Given** a fitted model, **When** researcher calls `model.plot_param_hist()`, **Then** the histogram shows the distribution of the specified parameter from jaxmodels

---

### User Story 3 - Multiple Model Fitting and Comparison (Priority: P3)

A researcher fits multiple models with different hyperparameters or cross-validation splits using `ModelCollection` and `fit_models()`, with all parallel fitting operations using the jaxmodels backend.

**Why this priority**: Model selection and validation require fitting many models. This is P3 because it builds on basic fitting (P1) and is an advanced workflow used for hyperparameter tuning.

**Independent Test**: Can be tested by setting up a parameter grid, calling `fit_models()`, creating a `ModelCollection`, and verifying all models in the collection used jaxmodels and produced valid results.

**Acceptance Scenarios**:

1. **Given** a researcher has defined a hyperparameter grid for lasso coefficients, **When** they call `fit_models()` with multiple datasets and lasso values, **Then** all models fit in parallel using jaxmodels and complete successfully
2. **Given** a ModelCollection with multiple fitted models, **When** researcher calls `mc.mut_param_dataset_correlation()`, **Then** the correlation plot reflects parameter values from all jaxmodels-fitted models
3. **Given** a ModelCollection, **When** researcher calls `mc.shift_sparsity()`, **Then** the sparsity plot shows accurate shift parameter statistics from jaxmodels
4. **Given** a ModelCollection, **When** researcher calls `mc.split_apply_combine_muts()`, **Then** the aggregated mutations dataframe contains parameters from all jaxmodels-fitted models

---

### User Story 4 - Error Handling and Diagnostics (Priority: P2)

A researcher encounters various error conditions (malformed data, convergence failures, invalid parameters) and receives clear, actionable error messages that help them resolve the issue.

**Why this priority**: Good error handling is essential for scientific software correctness and user experience. This is P2 because it spans all workflows and prevents silent failures.

**Independent Test**: Can be tested by intentionally providing invalid inputs, forcing convergence failures, and verifying error messages are informative and guide users to solutions.

**Acceptance Scenarios**:

1. **Given** a researcher provides data with missing required columns, **When** they try to create a Data object, **Then** they receive an error message specifying which columns are missing and what format is expected
2. **Given** a model fails to converge within iteration limit, **When** fitting completes, **Then** the user receives a warning with the final error value, tolerance threshold, and suggestions to increase iterations or adjust hyperparameters
3. **Given** a researcher provides incompatible warmstart parameters, **When** they initialize a model, **Then** they receive an error explaining the parameter incompatibility and how to fix it
4. **Given** a researcher tries to predict on data with mutations not seen during training, **When** they call prediction methods, **Then** they receive a clear error listing the unseen mutations and suggesting retraining with expanded data

---

### User Story 5 - Testing and Validation (Priority: P1)

A developer or researcher runs the test suite to verify all functionality works correctly with the jaxmodels backend, with comprehensive unit tests covering data processing, model fitting, predictions, and visualizations.

**Why this priority**: This is P1 because comprehensive testing is required to ensure scientific correctness. The specification states "test every functionality in a unit test" as a core requirement.

**Independent Test**: Can be tested by running the test suite (`pytest`) and verifying all tests pass, with coverage reports showing comprehensive coverage of Data, Model, and ModelCollection functionality.

**Acceptance Scenarios**:

1. **Given** the refactored codebase, **When** a developer runs `pytest tests/`, **Then** all unit tests pass including tests for Data class, Model class, and ModelCollection class
2. **Given** the test suite, **When** examining test coverage, **Then** coverage includes data loading, model initialization, warmstart, fitting, prediction, and all visualization methods
3. **Given** the test suite, **When** tests are run, **Then** numerical validation confirms jaxmodels produces mathematically correct results matching expected biophysical model equations
4. **Given** the test suite, **When** tests run, **Then** edge cases are covered including empty data, single-mutation variants, convergence failures, and boundary conditions

---

### Edge Cases

- What happens when a dataset contains only wildtype sequences (no mutations)?
- How does the system handle conditions where all variants have identical functional scores?
- How does the system handle numerical instabilities in global epistasis transformations (e.g., extreme latent phenotype values)?
- What happens when shift regularization drives all shifts exactly to zero?
- How does fitting behave when two conditions have completely non-overlapping mutation sets?
- How does the system handle invalid alphabet specifications?

## Requirements *(mandatory)*

### Functional Requirements

#### Data Processing

- **FR-001**: System MUST provide a Data class that accepts pandas DataFrames with required columns (condition, aa_substitutions, func_score) and processes them for model fitting
- **FR-001a**: Data class MUST validate that functional scores do not contain NaN or infinite values, raising error with list of invalid rows and suggesting data cleaning
- **FR-002**: System MUST support multiple experimental conditions with different wildtype sequences in a single Data object
- **FR-003**: Data class MUST convert substitution strings to one-hot encoded matrices compatible with jaxmodels sparse array formats
- **FR-004**: Data class MUST handle optional count data (pre_counts, post_counts, pre_count_wt, post_count_wt). Count data is required only when using count_loss and optional for functional_score_loss and warmstart
- **FR-005**: Data class MUST identify and convert mutations at non-identical sites between conditions to reference wildtype coordinates
- **FR-006**: Data class MUST provide site_map, mutations_df, variants_df, and non_identical_sites attributes matching current API

#### Model Fitting

- **FR-008**: System MUST provide a Model class that wraps jaxmodels fitting functionality while maintaining backward-compatible API
- **FR-009**: Model class MUST support initialization from Data objects and convert to jaxmodels.Data format internally
- **FR-010**: Model class MUST provide fit() method that calls jaxmodels.fit() with appropriate parameters
- **FR-011**: System MUST support warmstart initialization using Ridge regression (should work with or without count data; when count data is absent, Ridge regression proceeds without sample_weights)
- **FR-012**: System MUST support manual parameter initialization (beta0_init, beta_init, alpha_init) to override defaults
- **FR-013**: Model class MUST support global epistasis options (Identity, Sigmoid) from jaxmodels
- **FR-014**: System MUST support loss function selection (functional_score_loss with Huber, count_loss with negative binomial)
- **FR-015**: Model class MUST support regularization parameters (l2reg, fusionreg, beta0_ridge)
- **FR-016**: System MUST support beta parameter clipping with configurable ranges to prevent extreme values
- **FR-017**: System MUST track convergence via loss trajectory and optimizer state
- **FR-018**: Model class MUST provide convergence_trajectory_df attribute showing loss and error over iterations

#### Parameter Access and Prediction

- **FR-019**: Model class MUST provide get_mutations_df() method returning beta, shift, and predicted functional scores per mutation
- **FR-020**: Model class MUST provide get_variants_df() method returning predicted latent phenotypes and functional scores per variant
- **FR-021**: System MUST support phenotype_as_effect parameter to report predictions as differences from wildtype
- **FR-022**: Model class MUST expose params attribute with beta, beta0, shift, and theta parameters from jaxmodels
- **FR-023**: Model class MUST provide add_phenotypes_to_df() method for making predictions on new data
- **FR-024**: System MUST detect mutations not seen during training when making predictions on new data and raise clear error listing the unseen mutations with suggestion to retrain model with expanded data

#### Visualization

- **FR-025**: Model class MUST provide plot_epistasis() method showing global epistasis transformation curves
- **FR-026**: Model class MUST provide plot_pred_accuracy() method showing observed vs. predicted functional scores
- **FR-027**: Model class MUST provide plot_param_hist() method showing distribution of any parameter
- **FR-028**: Model class MUST provide mut_param_heatmap() method showing interactive heatmaps of mutation effects
- **FR-029**: All visualization methods MUST correctly extract and display data from jaxmodels-fitted models
- **FR-030**: Visualization methods MUST support customization via matplotlib axes or generate standalone figures

#### Model Collections

- **FR-031**: System MUST provide ModelCollection class accepting dataframe of fitted models with hyperparameters
- **FR-032**: System MUST provide fit_models() function for parallel fitting of multiple models across parameter grids
- **FR-033**: ModelCollection MUST provide split_apply_combine_muts() for aggregating mutation parameters across fits
- **FR-034**: ModelCollection MUST provide mut_param_dataset_correlation() for comparing parameters between replicates
- **FR-035**: ModelCollection MUST provide shift_sparsity() for visualizing regularization effects
- **FR-036**: ModelCollection MUST provide mut_param_heatmap() for aggregated parameter visualizations
- **FR-037**: ModelCollection MUST provide mut_param_traceplot() for tracking specific mutations across fits

#### Code Quality and Testing

- **FR-038**: All functionality MUST have corresponding unit tests with >90% code coverage
- **FR-039**: System MUST avoid code duplication - shared logic extracted to utility functions
- **FR-040**: Code MUST follow Black formatting (line length 89) and pass Ruff linting
- **FR-041**: All public functions and classes MUST have NumPy-style docstrings with examples
- **FR-042**: Doctests MUST be included in docstrings and pass when run with pytest
- **FR-043**: Tests MUST validate numerical correctness of jaxmodels computations against expected biophysical equations

#### Error Handling

- **FR-044**: System MUST validate required DataFrame columns and provide informative error messages for missing columns
- **FR-044a**: System MUST validate that count data is present when count_loss is selected and provide clear error message if missing
- **FR-044b**: System MUST document practical dataset size limits in documentation and provide informative error if memory allocation fails, suggesting dataset size reduction or hardware upgrade
- **FR-045**: System MUST check for convergence and warn users with final error value and tolerance when convergence fails
- **FR-046**: System MUST validate alphabet specifications and provide clear error for invalid alphabet choices
- **FR-047**: System MUST detect and report warmstart failures with actionable guidance (e.g., parameter dimension mismatches, Ridge solver errors)
- **FR-048**: System MUST validate parameter shapes match data dimensions and report mismatches clearly
- **FR-049**: System MUST handle numerical errors in optimization gracefully with informative error messages
- **FR-050**: System MUST validate reference condition exists in data and report clear error if missing

#### Deprecation and Migration

- **FR-051**: System MUST deprecate biophysical.py module completely - biophysical.py MUST raise ImportError on import with clear message directing users to migration guide and v2.0 documentation
- **FR-052**: Documentation MUST provide migration guide from v1.x to v2.0 highlighting API changes
- **FR-053**: System MUST maintain compatibility for core API (Data, Model, ModelCollection) where possible
- **FR-054**: Breaking changes MUST be documented in changelog with before/after examples
- **FR-055**: System version MUST be updated to 2.0.0 following semantic versioning

### Key Entities

- **Data**: Represents processed DMS experimental data from one or more conditions, including one-hot encoded mutation matrices, functional scores, optional count data, site maps, and metadata about mutations and variants. Wraps jaxmodels.Data internally.

- **Model**: Represents a fitted global epistasis model with parameters (beta, shift, theta), convergence state, and methods for prediction and visualization. Wraps jaxmodels.Model and fitting logic internally.

- **ModelCollection**: Represents a collection of fitted models with different hyperparameters or datasets, enabling comparison, aggregation, and batch visualization of results across models.

- **Latent Phenotype (φ)**: The additive combination of mutation effects (beta parameters) before global epistasis transformation. Computed by jaxmodels.Latent.

- **Global Epistasis Function (g)**: Non-linear transformation mapping latent phenotype to fitness. Implemented by jaxmodels.GlobalEpistasis subclasses (Identity, Sigmoid).

- **Mutation Effects (β)**: Per-mutation parameters representing the contribution to latent phenotype in the reference condition.

- **Shift Parameters (Δ)**: Per-mutation, per-condition parameters representing how mutation effects differ from the reference in non-reference conditions.

- **Functional Score**: Observed or predicted measurement from DMS experiment, representing log-enrichment relative to wildtype.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All existing example notebooks (e.g., fit_delta_BA1_example.ipynb) run successfully with v2.0 and produce scientifically valid results from jaxmodels backend
- **SC-002**: Test suite achieves greater than 90% code coverage for Data, Model, and ModelCollection classes
- **SC-003**: All unit tests pass on Python 3.9, 3.10, and 3.11 on both Ubuntu and macOS
- **SC-004**: Model fitting on standard datasets (>1000 variants, 3 conditions) completes within 2x the time of v1.x performance
- **SC-005**: Package installs successfully without dependency conflicts
- **SC-006**: Documentation includes working examples using jaxmodels backend with clear API reference
- **SC-007**: Production code has zero references to biophysical.py module
- **SC-008**: Breaking API changes number fewer than 10 and all documented in migration guide
- **SC-009**: Error messages for common mistakes (missing columns, convergence failures) are actionable and user-tested
- **SC-010**: No code duplication exists - all shared logic factored into utility functions or base classes

## Assumptions

- Users are familiar with Python scientific computing stack (numpy, pandas, jax)
- Users have DMS data in DataFrame format matching current multidms conventions
- The jaxmodels.py module is mathematically correct and implements the intended biophysical model
- Current visualization methods using matplotlib/seaborn/altair will continue to be supported
- Parallel fitting using multiprocessing will continue to be the parallelization strategy
- JAX will continue to be the computational backend (no plans to support other backends)
- Version 2.0.0 is a major version allowing breaking changes where necessary for jaxmodels integration
- The equinox library used by jaxmodels is stable and suitable for production use
- Users upgrading from v1.x to v2.0 are willing to make minor code adjustments for a cleaner API
- Dataset sizes are expected to fit in available memory; practical limits will be documented (typical expectation: up to millions of variants on systems with sufficient RAM)

## Out of Scope

- Adding new biophysical model types beyond what jaxmodels currently supports
- GPU optimization or distributed computing beyond current multiprocessing approach
- Backwards compatibility with biophysical.py module (it will be deprecated)
- Support for non-JAX computational backends
- Automated model selection or hyperparameter optimization
- Interactive web-based interfaces or dashboards
- Integration with workflow management systems (Snakemake, Nextflow)
- Real-time or streaming data processing
- Automatic aggregation/collapsing of identical variants or barcodes (users handle preprocessing)

## Dependencies

- Successful completion depends on jaxmodels.py being feature-complete and tested
- Requires coordination with documentation updates to reflect v2.0 API
- Requires changelog updates documenting all breaking changes
- Depends on maintaining compatibility with dependencies: JAX ≥0.4.29, equinox, jaxopt, pandas ≥2.2.0
- Assumes CI/CD infrastructure supports running full test suite on multiple platforms
