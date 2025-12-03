# Contract: Data Class

**Class**: `multidms.Data`
**Purpose**: Preprocess and validate DMS experimental data for model fitting
**Module**: `multidms/data.py`

## Constructor

### `Data.__init__(data, *, reference=None, alphabet=None)`

**Purpose**: Create a Data object from a pandas DataFrame with DMS experimental results.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `data` | `pd.DataFrame` | Yes | - | DataFrame with experimental results |
| `reference` | `str` | No | First condition | Name of reference condition for computing shifts |
| `alphabet` | `list[str]` | No | Standard AA | List of allowed amino acids |

**Required DataFrame Columns**:
- `condition` (str): Experimental condition name
- `aa_substitutions` (str): Mutation string (e.g., "A123B" or "A123B,C456D")
- `func_score` (float): Functional score measurement

**Optional DataFrame Columns**:
- `pre_counts` (int): Pre-selection counts
- `post_counts` (int): Post-selection counts
- `pre_count_wt` (int): Wildtype pre-selection count
- `post_count_wt` (int): Wildtype post-selection count

**Returns**: `Data` object

**Raises**:
- `ValueError`: If required columns missing
- `ValueError`: If `func_score` contains NaN or infinite values
- `ValueError`: If substitution format invalid
- `ValueError`: If `alphabet` contains invalid values
- `ValueError`: If `reference` not in data conditions

**Example**:
```python
import pandas as pd
from multidms import Data

df = pd.DataFrame({
    'condition': ['A', 'A', 'B', 'B'],
    'aa_substitutions': ['', 'A123B', '', 'A123B'],
    'func_score': [0.0, 1.5, 0.0, 2.0]
})

data = Data(df, reference='A')
```

**Post-conditions**:
- `data.variants_df` contains one-hot encoded mutations
- `data.mutations_df` contains mutation catalog
- `data.site_map` maps sites to mutation indices
- `data.non_identical_sites` lists sites with different wildtypes across conditions

---

## Properties

### `data.variants_df`

**Type**: `pd.DataFrame`

**Description**: Per-variant data with one-hot encoded mutation features.

**Columns**:
- `condition`: Condition name
- `aa_substitutions`: Original substitution string
- `func_score`: Functional score
- `[mutation_columns]`: Binary indicators for each mutation

**Example**:
```python
print(data.variants_df.head())
#   condition aa_substitutions  func_score  mut_A123B  mut_C456D
# 0         A                          0.0          0          0
# 1         A           A123B          1.5          1          0
```

---

### `data.mutations_df`

**Type**: `pd.DataFrame`

**Description**: Catalog of all mutations observed across conditions.

**Columns**:
- `mutation`: Mutation string (e.g., "A123B")
- `site`: Position in sequence
- `wildtype`: Wildtype amino acid
- `mutant`: Mutant amino acid

**Example**:
```python
print(data.mutations_df.head())
#   mutation  site wildtype mutant
# 0   A123B   123        A      B
# 1   C456D   456        C      D
```

---

### `data.site_map`

**Type**: `dict[int, list[int]]`

**Description**: Mapping from sequence position to mutation indices.

**Example**:
```python
print(data.site_map)
# {123: [0], 456: [1]}  # Site 123 has mutation index 0, site 456 has index 1
```

---

### `data.non_identical_sites`

**Type**: `list[int]`

**Description**: Sites where conditions have different wildtype amino acids.

**Example**:
```python
print(data.non_identical_sites)
# [100, 250]  # Conditions differ at positions 100 and 250
```

---

## Properties for jaxmodels (new in v2.0)

### `data.binary_map`

**Type**: `np.ndarray`

**Description**: One-hot encoded mutation matrix (binary map).

**Shape**: `(n_variants, n_mutations)`

**Purpose**: Provides the X matrix for jaxmodels optimization.

**Example**:
```python
print(data.binary_map.shape)
# (4, 2)  # 4 variants, 2 mutations observed

print(data.binary_map)
# [[0, 0],  # wildtype: no mutations
#  [1, 0],  # has mutation 0
#  [0, 0],  # wildtype
#  [1, 0]]  # has mutation 0
```

---

### `data.targets`

**Type**: `dict[str, np.ndarray]`

**Description**: Target values (functional scores or counts) per condition.

**Format**: Keys are condition names, values are arrays of targets

**Purpose**: Provides y values for jaxmodels loss calculation.

**Example**:
```python
print(data.targets)
# {
#   'reference': array([0.0, 1.2, 0.0, 1.2]),
#   'variant_A': array([0.1, 1.5, 0.1, 1.5])
# }
```

---

### `data.condition_indices`

**Type**: `np.ndarray`

**Description**: Integer indices mapping each variant to its condition.

**Shape**: `(n_variants,)`

**Purpose**: Allows jaxmodels to group variants by condition.

**Example**:
```python
print(data.condition_indices)
# [0, 0, 1, 1]  # First two variants in condition 0, next two in condition 1
```

---

### `data.weights`

**Type**: `np.ndarray` or `None`

**Description**: Sample weights derived from count data (if available).

**Shape**: `(n_variants,)` or `None`

**Purpose**: Provides sample weights for weighted loss functions.

**Returns**: `None` if count data not available, otherwise array of weights.

**Example**:
```python
# If count data available
print(data.weights)
# [0.95, 1.10, 0.98, 1.05]  # Weights based on count reliability

# If no count data
print(data.weights)
# None
```

---

## Validation Contract

### Input Validation

**On Construction**:
1. Check required columns present in DataFrame
2. Validate `func_score` contains no NaN or inf values
3. Validate substitution strings match format `[A-Z]\d+[A-Z]`
4. Validate `reference` exists in `data.condition` values
5. Validate `alphabet` contains only valid amino acid codes

**Error Message Format**:
```
ValueError: DataFrame missing required columns: {missing}.
Expected: condition, aa_substitutions, func_score.
Found: {present}.
```

### Data Processing

**No Automatic Aggregation**:
- Data is accepted as-is; no automatic collapsing of identical variants
- Users must aggregate/collapse barcodes or variants themselves before creating Data object
- Model fits to the data provided (either collapsed variants or barcode replicates)

**Substitution Encoding**:
- Split comma-separated mutations: "A123B,C456D" → ["A123B", "C456D"]
- Wildtype (empty string) encoded as all-zero vector
- Single mutation "A123B" encoded as single 1 in corresponding column

---

## Invariants

**Post-Construction Invariants**:
1. `len(data.variants_df)` ≥ 1 (at least one variant)
2. `len(data.mutations_df)` ≥ 0 (zero mutations allowed if only wildtype)
3. All mutations in `mutations_df` appear in at least one variant in `variants_df`
4. `site_map` keys match unique sites in `mutations_df`
5. `reference` in `data.variants_df.condition.unique()`

**Data Integrity**:
- `variants_df.func_score` contains no NaN or inf
- One-hot encoding columns sum to mutation count per variant
- `mutations_df.mutation.unique()` matches column names in `variants_df`

---

## Performance Characteristics

**Time Complexity**:
- Construction: O(n × m) where n = variants, m = mutations per variant
- One-hot encoding: O(n × total_mutations)
- Aggregation: O(n log n) for sorting/grouping

**Space Complexity**:
- `variants_df`: O(n × total_mutations) - sparse in practice
- `mutations_df`: O(total_mutations)
- `site_map`: O(number_of_sites)

**Scalability**:
- Tested on datasets with >100,000 variants
- Memory usage scales linearly with n_variants × n_mutations
- Millions of variants may require >16GB RAM

---

## Backward Compatibility

**v1.x → v2.0 Compatibility**: ⚠️ **MOSTLY COMPATIBLE** (minor breaking changes)

**Compatible**:
- Constructor signature (except removed `collapse_identical_variants` parameter)
- All existing properties (`variants_df`, `mutations_df`, `site_map`, etc.) return same format
- Validation rules for required columns identical
- Error messages enhanced but backward compatible

**Breaking Changes**:
- `collapse_identical_variants` parameter removed from constructor
- New properties added: `binary_map`, `targets`, `condition_indices`, `weights` (won't affect v1.x code)

**Migration Notes**:
```python
# v1.x
data = Data(df, collapse_identical_variants='mean')  # ❌ Parameter removed

# v2.0
df_collapsed = df.groupby(['condition', 'aa_substitutions']).agg({'func_score': 'mean'})
data = Data(df_collapsed)  # ✓ User handles aggregation
```
