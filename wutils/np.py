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


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def ensure_is_list(obj):
    return obj if isinstance(obj, list) else [obj]


def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask


def concatenate(data, fillna=None):
    all_keys = [d.keys() for d in data]
    flat_keys = [k for keys in all_keys for k in keys]
    keys = set(flat_keys)
    _data = {k: [] for k in keys}
    for row in data:
        for k in keys:
            _data[k].append(row.get(k, fillna))
    return _data


def is_nptimedelta(v):
    try:
        answer = 'timedelta' in v.dtype.name
    except:
        answer = False
    return answer


def is_datetime(v):
    return 'datetime' in str(v.dtype)


def get_group_ixs(*group_ids, **kwargs):
    """ Returns a dictionary {groupby_id: group_ix}.

    group_ids:
        List of IDs to groupbyy
    kwargs:
        bools = True or False, if True returns a boolean array
    """
    group_ids = _ensure_group_ids_hashable(group_ids)
    grouped_ixs = _get_group_ixs(group_ids)
    grouped_ixs = _convert_int_indices_to_bool_indices_if_necessary(grouped_ixs, kwargs)
    return grouped_ixs


def _ensure_group_ids_hashable(group_ids):
    if len(group_ids) == 1:
        combined_group_ids = group_ids[0]
    else:
        combined_group_ids = zip(*group_ids)
    is_list_of_list = lambda ids: isinstance(ids[0], list)
    is_matrix = lambda ids: isinstance(ids, np.ndarray) and ids.ndim == 2
    if is_list_of_list(combined_group_ids) or is_matrix(combined_group_ids):
        hashable_group_ids = [tuple(group_id) for group_id in combined_group_ids]
    else:
        hashable_group_ids = combined_group_ids
    return hashable_group_ids


def _convert_int_indices_to_bool_indices_if_necessary(ixs, kwargs):
    bools = kwargs.get('bools', False)
    if bools:
        length = np.sum([len(v) for v in ixs.itervalues()])
        ix_to_bool = lambda v, length: ss.np.ix_to_bool(v, length)
        ixs = {k: ix_to_bool(v, length) for k, v in ixs.iteritems()}
    return ixs


def _get_group_ixs(ids):
    id_hash = defaultdict(list)
    for j, key in enumerate(ids):
        id_hash[key].append(j)
    id_hash = {k: np.array(v) for k, v in id_hash.iteritems()}
    return id_hash


def get_ordered_group_ixs(group_ids):
    od_ixs = OrderedDict()
    for i, val in enumerate(group_ids):
        if val in od_ixs:
            od_ixs[val].append(i)
        else:
            od_ixs[val] = [i]
    return od_ixs
