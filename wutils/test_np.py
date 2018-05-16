from wutils import np as utils_np
import numpy as np

from ddf import DDF


def test_get_str_columns():
    df = DDF({
        'MSZoning': ['FV'],
        'MiscFeature': [np.nan],
        'MoSold': [2]
        })
    result = utils_np.get_str_columns(df)
    for col in result:
        assert col in ['MSZoning', 'MiscFeature']
