from timeit import default_timer as timer

import multidms
import pandas as pd
import numpy as np

func_score_df = pd.read_csv("test_func_score.csv")
"""
   condition aa_substitutions  func_score
   0          1              M1E         2.0
   1          1              G3R        -7.0
   2          1              G3P        -0.5
   3          1              M1W         2.3
   4          2              M1E         1.0
   5          2              P3R        -5.0
   6          2              P3G         0.4
   7          2          M1E P3G         2.7
   8          2          M1E P3R        -2.7
   9          2              P2T         0.3

      1  2
      1  M  M
      3  G  P
"""

data = multidms.MultiDmsData(
    func_score_df,
    alphabet = multidms.AAS_WITHSTOP,
    reference = 1,
    assert_site_integrity=True, # TODO testthis
)

model = multidms.MultiDmsModel(
    data, 
    PRNGKey=23
)

def test_bmap_mut_df_order():
    """ 
    Assert that the binarymap rows and columns match
    mutations_df indicies exactly.
    """
    
    # test the mutations order for both
    mut_df = data.mutations_df
    for condition in data.conditions:
        bmap = data.binarymaps[condition]
        assert np.all(mut_df.mutation == bmap.all_subs)

    # make sure the indices into the bmap are ordered 0-n
    for i, sub in enumerate(mut_df.mutation):
        assert sub == bmap.i_to_sub(i)


# TODO test the other reference, as well
def test_non_identical_conversion():
    """ 
    Test the conversion to with respect reference wt sequence. 
    """
    data = multidms.MultiDmsData(
        func_score_df,
        alphabet = multidms.AAS_WITHSTOP,
        collapse_identical_variants = "mean",
        reference = 1,
        assert_site_integrity=True, # TODO testthis
    )

    assert np.all(data.binarymaps[1].binary_variants.todense() == [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    assert np.all(data.binarymaps[2].binary_variants.todense() == [
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])

def test_model_PRNGKey():
    """
    Simply test the instatiation of a model with different PNRG keys
    to make sure the seed structure truly ensures the same parameter
    initialization values
    """

    model_1 = multidms.MultiDmsModel(data, PRNGKey=23)
    model_2 = multidms.MultiDmsModel(data, PRNGKey=23)
    for param, values in model_1.params.items():
        assert np.all(values == model_2.params[param])

def test_model_phenotype_predictions():
    """
    Make sure that the substitution conversion and binary
    encoding are correct in `MultiDmsModel.add_phenotype_to_df`
    by comparing the training data internal predictions
    match those of external predictions on that same data.
    """
    internal_pred = model.variants_df
    external_pred = model.add_phenotypes_to_df(
            func_score_df, unknown_as_nan=True
    ).dropna()
    assert np.all(internal_pred.predicted_latent.values == external_pred.predicted_latent.values)
    assert np.all(internal_pred.predicted_func_score.values == external_pred.predicted_func_score.values)
