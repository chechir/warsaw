import numpy as np


def fillna(array, na_value):
    array = array.copy()
    ix = np.isnan(array) | np.isinf(array)
    if np.isscalar(na_value):
        array[ix] = na_value
    else:
        array[ix] = na_value[ix]
    return array


def get_str_columns(df):
    str_columns = [col for col in df.columns
                   if not np.issubdtype(df[col].dtype, np.number)]
    return str_columns
