from timeit import default_timer as timer

import multidms
import pandas as pd
import numpy as np


#    1  2
# 1  M  M
# 3  G  P


func_score_df = pd.DataFrame({
    'condition' : [
        "1","1","1","1", 
        "2","2","2","2","2","2"
    ],
    'aa_substitutions' : [
        'M1E', 'G3R', 'G3P', 'M1W', 
        'M1E', 'P3R', 'P3G', 'M1E P3G', 'M1E P3R', 'P2T'],
    'func_score' : [
        2, -7, -0.5, 2.3, 
        1, -5, 0.4, 2.7, -2.7, 0.3
    ]
})

def test_non_identical_conversion():
    """ Test the conversion to with respect reference wt sequence. """
    data = multidms.MultiDmsData(
        func_score_df,
        alphabet = multidms.AAS_WITHSTOP,
        reference = "1",
        assert_site_integrity=True,
    )
    #print(data.site_map)
    #assert False

    assert np.all(data.binarymaps['1'].binary_variants.todense() == [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    assert np.all(data.binarymaps['2'].binary_variants.todense() == [
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    #>>> data.variants_df
    #  condition aa_substitutions  weight  func_score var_wrt_ref
    #  0         1              G3P       1        -0.5         G3P
    #  1         1              G3R       1        -7.0         G3R
    #  2         1              M1E       1         2.0         M1E
    #  3         1              M1W       1         2.3         M1W
    #  4         2              M1E       1         1.0     G3P M1E
    #  5         2          M1E P3G       1         2.7         M1E
    #  6         2          M1E P3R       1        -2.7     G3R M1E
    #  8         2              P3G       1         0.4
    #  9         2              P3R       1        -5.0         G3R
    #  >>> data.site_map
    #     1  2
    #     3  G  P
    #     1  M  M


