from collections import OrderedDict, defaultdict
from functools import wraps, partial
import copy
import json
import warnings

import numpy as np
import pandas as pd

from utils import np as utilsnp

default_config = {
            'repr': {
                'style': 'horizontal',
                'max_length': 10
                }
            }


class CoreDF(object):
    def __init__(self, data=None):
        """
        Tabular data sctructure, essentially a dictionary with added functionality.

        Parameters
        ----------
        data (dict, list, other): if data type is not dict of list, it must be compatible
            with the construction of a OrderedDict

        Examples
        --------
        >>> df = DDF()
        >>> df
            Empty DDF
            -----
            Shape: (0, 0)
        >>> df = DDF({'a': [1,2,3], 'b': [2,3,4]})
        >>> df
                a   b
            0   1   2
            1   2   3
            2   3   4

            -----
            Shape: (3, 2)
        >>> df = DDF([{'a':1}, {'a':2}])
        >>> df
                a
            0   1
            1   2

            -----
            Shape: (2, 1)
        """
        data = copy.copy(data)
        if data is None:
            data = OrderedDict()
        elif isinstance(data, dict):
            data = self._parse_dict(data)
        elif isinstance(data, list):
            data = self._convert_list_to_dict(data)
            data = self._parse_dict(data)
        else:
            try:
                data = OrderedDict(data)
            except Exception:
                raise TypeError('cant instantiate from {}'.format(type(data)))
        self._data = data

        self._config = default_config

    def colapply(self, cols, func, inplace=True):
        def _colapply(cols, funcs, data):
            for col in cols:
                data[col] = func(data[col])

        cols = utilsnp.ensure_is_list(cols)
        if inplace:
            _colapply(cols, func, self._data)
        else:
            new_data = copy.deepcopy({k: self[k] for k in cols})
            new_data.update(copy.copy({k: self[k] for k in self if k not in cols}))
            _colapply(cols, func, new_data)
            result = type(self)(new_data)
            return result

    def rename(self, renamer, inplace=False):
        """
        Rename columns of df.

        Arguments
        ---------
            renamer (dict, func): transformations to apply to columns

        Returns
        -------
            renamed_df (df): new object

        Examples
        --------
        >>> df = DDF({'a': range(10)})
        >>> df.rename(lambda col: col + '_suffix', inplace=True)
        >>> df.rename({'a_suffix': 'b'}, inplace=True)
        """
        if not isinstance(renamer, dict):
            renamer = {c: renamer(c) for c in self}

        def _rename_columns(dct):
            for old, new in renamer.iteritems():
                dct[new] = dct.pop(old)

        if inplace:
            _rename_columns(self._data)
        else:
            new_data = self._data.copy()
            _rename_columns(new_data)
            new_df = type(self)(new_data)
            return new_df

    def fillna(self, value, inplace=False):
        """
        Replacing null values in df.

        Arguments
        ---------
            value (scalar, dictionary): if scalar, the same value will be used to fill
                all null values in the df. If a dictionary, the holes will be filled with
                the values specified for each column. If the filling value are an
                `np.ndarray`, the array will be filled with the fill value at the
                corresponding index.

        Returns
        -------
            filled (df)

        Example
        -------
        >>> df = DDF({'a': np.array([1, 2, 3, 4, None, np.nan, 7, 8, 9, 10])})
        >>> df.fillna(999)
        >>> df.fillna({0:np.arange(len(df))})
        >>> df['b'] = np.array([1, 2, 3, 4, 5, 6, np.nan, np.nan, np.nan, np.nan])
        >>> df.fillna({'a':101, 'b':999})
        """
        warnings.warn('Fillna is not longer in place by default!')

        def _fillna(dct, value):
            if isinstance(value, dict):
                for colname, values in dct.items():
                    if colname in value:
                        fillvalue = value[colname]
                        fillix = pd.isnull(values)
                        if isinstance(fillvalue, np.ndarray):
                            values[fillix] = fillvalue[fillix]
                        else:
                            values[fillix] = fillvalue
            else:
                for v in dct.itervalues():
                    v[pd.isnull(v)] = value

        if inplace:
            _fillna(self._data, value)
        else:
            cols_to_copy = [k for k, v in self._data.iteritems() if any(pd.isnull(v))]
            new_data = copy.deepcopy({k: self[k] for k in cols_to_copy})
            new_data.update(copy.copy({k: self[k] for k in self if k not in cols_to_copy}))
            _fillna(new_data, value)
            filled = type(self)(new_data)
            return filled

    def iterrows(self):
        n_rows = len(self)
        for row in xrange(n_rows):
            yield self.rowslice(row)

    def pop(self, column):
        return self._data.pop(column)

    def drop_rows(self, row_numbers):
        ix = utilsnp.ix_to_bool(row_numbers, len(self))
        return self[~ix]

    def head(self, N=5):
        return self.rowslice(np.arange(N))

    def tail(self, N=5):
        tail_ix = np.arange(len(self) - N, len(self))
        return self.rowslice(tail_ix)

    @classmethod
    def from_pandas(cls, df):
        data = OrderedDict((column, df[column].values) for column in df)
        return cls(data)

    @classmethod
    def from_hdf(cls, path, key='df'):
        df = pd.read_hdf(path, key)
        return cls.from_pandas(df)

    # @classmethod
    # def load_latest(cls, path_pattern):
    #     path = ss.paths.get_latest_path(path_pattern)
    #     assert '.h5' in path, 'Method only works with h5 files'
    #     return cls.from_hdf(path)

    @classmethod
    def from_csv(cls, path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        return cls.from_pandas(df)

    def to_pandas(self):
        return pd.DataFrame(self._data)

    def to_hdf(self, path, key='df'):
        df = self.to_pandas()
        df.to_hdf(path, key)

    def to_csv(self, path, **kwargs):
        df = self.to_pandas()
        if 'index' not in kwargs:
            kwargs['index'] = False
        df.to_csv(path, **kwargs)

    def to_json(self):
        assert all(map(lambda c: isinstance(c, (str, unicode)), self.columns))
        df = self.to_pandas()
        raw_json = df.to_json(date_format='iso')
        dtypes = {c: str(self[c].dtype) for c in df}
        js = {'values': raw_json, 'dtypes': dtypes}
        js = json.dumps(js)
        return js

    @classmethod
    def from_json(cls, jsons):
        js = json.loads(jsons)
        df = pd.read_json(js['values'], dtype=js['dtypes'])
        df = df.sort_index()
        df = cls.from_pandas(df)
        return df

    def rowslice(self, index):
        row_slices = OrderedDict((k, v[index]) for k, v in self._data.iteritems())
        return type(self)(row_slices)

    def colslice(self, columns_or_filter):
        """
        Creating a new df with selected columns

        Arguments
        ---------
            slicer (list, filter_function): if `list`, the new ddf will have the columns
                in the list. If filter_function, a list will be created according to
                the columns selected by the filter.

        Examples
        --------
        >>> df = DDF({'a': [1,2,3], 'b': [1,2,3]})
        >>> df.colslice(lambda col: col == 'a')
                a
            0   1
            1   2
            2   3

            -----
            Shape: (3, 1)
        >>> df.colslice(['b'])
                b
            0   1
            1   2
            2   3

            -----
            Shape: (3, 1)
        """
        def is_iterator(obj):
            return hasattr(obj, 'next')
        if callable(columns_or_filter) and (not is_iterator(columns_or_filter)):
            selected = filter(columns_or_filter, self.columns)
        elif np.isscalar(columns_or_filter):
            selected = utilsnp.ensure_is_list(columns_or_filter)
        else:
            selected = columns_or_filter
        col_slice = OrderedDict((k, self.data[k]) for k in selected)
        return self.__class__(col_slice)

    def sort(self, keys, axis=0, ascending=True):
        if axis == 0:
            return self.rowsort(keys, ascending)
        else:
            raise NotImplementedError

    def rowsort(self, keys, ascending=True):
        #TODO: inplace?
        if np.isscalar(keys):
            keys = [keys]
        sort_ix = np.lexsort([self.data[k] for k in keys[::-1]])
        if not ascending:
            sort_ix = sort_ix[::-1]
        return self.rowslice(sort_ix)

    @property
    def columns(self):
        return self._data.keys()

    @property
    def values(self):
        return np.array(self._data.values()).T

    def copy(self, deep=True):
        if deep:
            copied = copy.deepcopy(self)
        else:
            copied = copy.copy(self)
            data = copy.copy(self._data)
            self._data = data
        return copied

    def append(self, other_df, axis=None):
        """
        Appending other DDFs to self. It is not always necessary to provide the axis
        for appending - often this method will understand which axis is supposed to be
        used.

        Arguments
        ---------
            other_df (ddf): df to append
            axis (int): axis over which to append
        """
        if len(self) == 0:
            appended = other_df
            return appended
        if len(other_df) == 0:
            return self

        if axis is None:
            has_strictly_new_cols = len(set(other_df).difference(self)) == len(set(other_df))
            if sorted(other_df) == sorted(self):
                appended = self._append_rows(other_df)
            elif len(other_df) == len(self) and has_strictly_new_cols:
                appended = self._append_columns(other_df)
            else:
                raise RuntimeError('appending is ambiguous; please provide axis.')
        else:
            if axis == 0:
                appended = self._append_rows(other_df)
            elif axis == 1:
                appended = self._append_columns(other_df)
            else:
                raise NotImplementedError('DDF only has two axes: 0 and 1.')
        return appended

    @property
    def shape(self):
        return (len(self), len(self._data))

    @wraps(pd.merge)
    def merge(self, other_df, **kwargs):
        def is_string(dtype):
            return (dtype.type is np.str_) or (dtype.type is np.unicode_)

        def get_stringtypes(df, stringtypes=None):
            if stringtypes is None:
                stringtypes = defaultdict(set)
            for col in df:
                dtype = df[col].dtype
                if is_string(dtype):
                    stringtypes[col].add(dtype)
            return stringtypes

        strings = get_stringtypes(self)
        strings = get_stringtypes(other_df, strings)

        self_df = self.to_pandas()
        if hasattr(other_df, 'to_pandas'):
            other_df = other_df.to_pandas()

        merged = self_df.merge(other_df, **kwargs)
        merged = DDF.from_pandas(merged)
        for col, dtypes in strings.items():
            if col not in merged:
                continue
            newdtype = dtypes.pop() if len(dtypes) == 1 else np.result_type(*dtypes)
            if newdtype.type is np.str_:
                merged[col] = merged[col].astype(str)
            elif newdtype.type is np.unicode_:
                merged[col] = merged[col].astype(unicode)
        return merged

    def drop_duplicates(self, *args, **kwargs):
        string_cols = [f for f in self if self[f].dtype.type is np.str_]
        self_df = self.to_pandas()
        new_df = self_df.drop_duplicates(*args, **kwargs)
        df = self.from_pandas(new_df)
        df.colapply(string_cols, lambda v: v.astype(str))
        return df

    def to_dict(self):
        return dict(self._data)

    def _parse_dict(self, data):
        if len(data) == 0:
            data = OrderedDict()
        else:
            if isinstance(data, dict):
                for k, v in data.iteritems():
                    if not isinstance(v, np.ndarray):
                        if isinstance(v, list) and np.any([isinstance(_, np.datetime64) for _ in v]):
                            v = pd.to_datetime(v).values
                        else:
                            v = np.atleast_1d(np.array(v))
                    if v.ndim == 2 and v.shape[1] == 1:
                        v = v.flatten()
                    data[k] = copy.copy(v)
            data = OrderedDict(data)
            self._check_data_format(data)
        return data

    def _convert_list_to_dict(self, data):
        def is_list_of_dicts(data):
            return isinstance(data, list) and all([isinstance(x, dict) for x in data])

        if is_list_of_dicts:
            data = utilsnp.concatenate(data, np.nan)
        return data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        # TODO: this order of operations is retarded
        self._data = new_data
        self._check_internal_data_dict_format()

    def __getitem__(self, indexer):
        def _is_boolean_array(v):
            bools = (bool, np.bool_)
            return isinstance(v, np.ndarray) and all([isinstance(x, bools) for x in v])

        if isinstance(indexer, (list, tuple)):
            columns = indexer
            return np.array([self._data[column] for column in columns]).T
        elif isinstance(indexer, slice):
            index = indexer
            return self.rowslice(index)
        elif _is_boolean_array(indexer):
            assert len(indexer) == len(self)
            return self.rowslice(indexer)
        else:
            key = indexer
            return self._data[key]

    def __setitem__(self, columns, values):
        columns, values = self._ensure_setitem_args_are_valid(columns, values)
        for column, value in zip(columns, values.T):
            self._data[column] = copy.copy(value)
        self._check_internal_data_dict_format()

    def _ensure_setitem_args_are_valid(self, columns, values):
        columns = utilsnp.ensure_is_list(columns)
        values = self._ensure_values_is_matrix(values, columns)
        n_rows, n_cols = values.shape
        if len(self) > 0:
            assert len(self) == n_rows, 'Array does not have the correct number of rows.'
        assert len(columns) == n_cols, 'Array does have the correct number of columns.'
        return columns, values

    def _ensure_values_is_matrix(self, values, columns):
        if isinstance(values, list):
            values = np.array(values)
        elif np.isscalar(values):
            target_shape = (len(self), len(columns))
            values = np.repeat(values, np.prod(target_shape)).reshape(target_shape)
        if values.ndim == 1:
            if len(values) == len(columns):
                values = np.broadcast_to(values, (len(self), len(columns)))
            else:
                values = values.reshape(-1, 1)
        return values

    def __len__(self):
        if len(self._data) == 0:
            return 0
        else:
            first_value = self._data.itervalues().next()
            return 1 if np.isscalar(first_value) else len(first_value)

    def __repr__(self):
        max_length = self.get_option('repr.max_length')
        if self.get_option('repr.style') == 'horizontal':
            formatter = HorizontalFormatter(max_length=max_length)
        elif self.get_option('repr.style') == 'vertical':
            formatter = VerticalFormatter(max_length=max_length)
        return formatter.get_repr(self)

    def __iter__(self):
        return self._data.iterkeys()

    def _append_rows(self, other_df):
        all_columns = set(other_df.columns + self.columns)
        len_df, len_self = len(other_df), len(self)
        data = self._data.copy()
        for column in all_columns:
            u = data.get(column, np.repeat(np.nan, len_self))
            u = np.atleast_1d(u)
            try:
                v = other_df[column]
            except KeyError:
                if utilsnp.is_nptimedelta(u):
                    v = np.repeat(np.nan, len_df).astype(u.dtype)
                else:
                    v = np.repeat(np.nan, len_df)
            v = np.atleast_1d(v)
            u, v = self._reconcile_dtypes(u, v)
            uv = np.concatenate([u, v])
            data[column] = uv
        new_df = type(self)(data)
        return new_df

    def _reconcile_dtypes(self, u, v):
        # TODO: Not seamless, but how to concatenate arbitrary vectors? May set
        # priority: you always try to cast the other DF's array into your own
        # type?
        if utilsnp.is_datetime(u) or utilsnp.is_datetime(v):
            dtype = u.dtype if utilsnp.is_datetime(u) else v.dtype
            return u.astype(dtype), v.astype(dtype)
        else:
            return u, v

    def _append_columns(self, other_df):
        if len(other_df) != len(self):
            raise ValueError('dfs need to be of the same length')
        data = self._data.copy()
        for column, array in other_df._data.iteritems():
            data[column] = array
        new_df = type(self)(data)
        return new_df

    def equals(self, other_df):
        def get_comparison_func(value):
            if value.dtype in [np.float, np.float32]:
                return np.allclose
            else:
                return np.array_equal

        if len(self) != len(other_df):
            return False
        if sorted(self.columns) != sorted(other_df.columns):
            return False
        for col, value in self._data.iteritems():
            other_value = other_df[col]
            if 'datetime' in str(value.dtype):
                all_equal = (other_value == value).all()
            else:
                skip_ix = pd.isnull(value) & pd.isnull(other_value)
                compare_ix = ~skip_ix
                comparison_func = get_comparison_func(value)
                all_equal = comparison_func(value[compare_ix], other_value[compare_ix])
            if not all_equal:
                return False
        return True

    def _check_internal_data_dict_format(self):
        self._check_data_format(self._data)

    def _check_data_format(self, dct):
        if not all(type(v) == np.ndarray for v in dct.itervalues()):
            raise RuntimeWarning('Not all values are NumPy array. ')

        unique_ndims = np.unique([v.ndim for v in dct.itervalues()])
        if len(unique_ndims) > 1:
            raise RuntimeWarning('Arrays do not have the same dimensions. ')
        if unique_ndims[0] != 1:
            raise RuntimeWarning('Arrays are not one dimensional. ')

        lengths = np.unique([len(v) for v in dct.itervalues()])
        if len(lengths) > 1:
            raise RuntimeWarning('Arrays do not have the same lengths. ')

    def _ipython_key_completions_(self):
        return self.columns


class SeamlessMixin(object):
    def get_odds(self, market):
        return self['{}_odds'.format(market)]

    def get_flags(self, market):
        return self['{}_flag'.format(market)]

    def describe(self, *args, **kwargs):
        self.to_pandas().describe(*args, **kwargs)

    def is_equal_to(self, *args, **kwargs):
        return self.equals(*args, **kwargs)

    def group_filter(self, groupby, func):
        group_ixs = self.get_group_ixs(groupby)
        keep_rows = []
        for ix in group_ixs.itervalues():
            if isinstance(func, slice):
                keep_rows.append(ix[func])
            else:
                sliced_df = self.rowslice(ix)
                keep_ix = func(sliced_df)
                keep_rows.append(ix[keep_ix])
        keep_rows = utilsnp.flatten(keep_rows)
        return self.rowslice(keep_rows)

    def get_group_ixs(self, group_name, bools=False):
        if not hasattr(self, '_ixs_cache'):
            self._ixs_cache = {}

        group_str = group_name if isinstance(group_name, str) else ''.join(group_name)
        ix_name = '{}_bools:{}'.format(group_str, bools)
        group_name = tuple(group_name) if isinstance(group_name, list) else group_name
        ixs, array_copy = self._ixs_cache.get(ix_name, (None, None))
        current_groups = self[group_name]
        is_array_equal = np.array_equal(array_copy, current_groups)
        if (not is_array_equal or ixs is None):
            group_ids = self._amend_groups(current_groups)
            ixs = utilsnp.get_group_ixs(group_ids, bools=bools)
            self._ixs_cache[ix_name] = (ixs, current_groups.copy())
        return ixs

    def _amend_groups(self, groups):
        if groups.ndim == 2:
            return [tuple(row) for row in groups]
        else:
            return groups

    def get_cols(self, *keywords):
        def is_string(x):
            return isinstance(x, (str, unicode))

        def filter_func(column, keywords):
            def selector(column, keyword):
                if is_string(column) and is_string(keyword):
                        return keyword.lower() in column.lower()
                elif (not is_string(column)) and (not is_string(keyword)):
                    return column == keyword
                else:
                    return False
            return all(selector(column, keyword) for keyword in keywords)
        filter_func_ = partial(filter_func, keywords=keywords)
        return filter(filter_func_, self.columns)

    def _get_group_ixs(self, group_ids):
        if isinstance(group_ids, (list, tuple)):
            groups = [self._data[group_id] for group_id in group_ids]
        else:
            groups = [self._data[group_ids]]
        return utilsnp.get_group_ixs(*groups)

    def group_reduce(self, group_ids, func):
        """ func: group_ddf -> scalar. """
        data = OrderedDict()
        group_ixs = self._get_group_ixs(group_ids)
        for group, group_ix in group_ixs.iteritems():
            group_ddf = self.rowslice(group_ix)
            data[group] = func(group_ddf)
        return self.__class__(data)

    def group_apply(self, columns, func):
        group_ixs = self._get_group_ixs(columns)
        return self._group_apply(group_ixs, func)

    def _group_apply(self, group_ixs, func):
        """ func: group_ddf -> group_vector. """
        applied_arr = None
        for group_ix in group_ixs.itervalues():
            group_ddf = self.rowslice(group_ix)
            group_output = func(group_ddf)
            if applied_arr is None:
                applied_arr = self._initialise_group_apply_output(group_output)
            applied_arr[group_ix] = group_output
        return applied_arr

    def _initialise_group_apply_output(self, dummy_data):
        if np.isscalar(dummy_data) or getattr(dummy_data, 'ndim', None) == 1:
            container_shape = (len(self), )
        else:
            container_shape = (len(self), dummy_data.shape[1])
        return np.nan * np.zeros(container_shape)

    def drop_columns(self, columns):
        for column in columns:
            del self._data[column]
        return self

    def set_option(self, pattern, value):
        """ Example: to change the repr: `set_option('repr.style', 'vertical')"""
        context, key = pattern.split('.')
        self._config[context][key] = value

    def get_option(self, pattern):
        context, key = pattern.split('.')
        return self._config[context][key]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_config')
        return d

    def __setstate__(self, d):
        d['_config'] = default_config
        self.__dict__ = d
        return d


class DDF(CoreDF, SeamlessMixin):
    pass


class VerticalFormatter(object):
    max_cols = 20
    max_width = 90
    horspacing = ' '*3

    def __init__(self, max_length=60):
        self.max_length = max_length

    def get_formatter(self, values):
        def format_dates(value):
            return str(value.astype('datetime64[D]'))

        def default_formatter(value):
            return str(value)

        formatters = [
                (lambda value: value.dtype == '<m8[ns]', format_dates),
                ]
        for selector, formatter in formatters:
            if selector(values):
                return formatter
        return default_formatter

    def select_colums(self, df):
        cols = df.columns
        ncols = len(cols)
        cutoff = self.max_cols / 2
        if ncols > self.max_cols:
            cols.pop(cols.index('...'))
            cols = cols[:cutoff] + ['...'] + cols[-cutoff:]
        return cols

    def format_columns(self, df):
        is_long = len(df) > self.max_length
        is_wide = df.shape[1] > self.max_cols
        data = OrderedDict()
        cutoff = self.max_length / 2
        for k, v in df.data.items():
            formatter = self.get_formatter(v)
            if is_long:
                head = [formatter(x) for x in v[:cutoff]]
                tail = [formatter(x) for x in v[-cutoff:]]
                vstr = head + ['..'] + tail
            else:
                vstr = [formatter(x) for x in v]
            max_entry_len = max([len(str(x)) for x in vstr])
            max_len = max([len(str(k)), max_entry_len])
            formatted_values = [x.ljust(max_len) for x in vstr]
            formatted_name = str(k).ljust(max_len)
            data[formatted_name] = formatted_values
        if is_wide:
            data['...'] = np.repeat('...', len(vstr))
        new_df = DDF(data)
        return new_df

    def get_blocks(self, cols):
        def get_rowwidth(lst):
            return len(self.horspacing.join(lst))

        cols = cols[:]
        blocks = []
        row = []
        while True:
            try:
                col = cols.pop(0)
            except IndexError:
                blocks.append(row)
                break
            if get_rowwidth(row) + len(col) < self.max_width:
                row.append(col)
            else:
                blocks.append(row)
                row = [col]
        return blocks

    def get_rownums(self, df):
        nums = np.arange(len(df))
        nums = nums.astype(str)
        maxlen = max([len(x) for x in nums])
        nums = [num.ljust(maxlen) for num in nums]
        cutoff = self.max_length / 2
        if len(df) > self.max_length:
            nums = nums[:cutoff] + ['..'.ljust(maxlen)] + nums[-cutoff:]
        return nums

    def _get_repr_body(self, df):
        rownums = self.get_rownums(df)
        df = self.format_columns(df)
        cols = self.select_colums(df)
        blocks = self.get_blocks(cols)

        for block in blocks:
            rownum_header = ' ' * len(rownums[0])
            yield self.horspacing.join([rownum_header] + block)
            yield '\n'
            for i in range(len(df)):
                yield self.horspacing.join([rownums[i]] + [df[col][i] for col in block])
                yield '\n'
            yield '\n'

    def get_repr(self, df):
        if len(df) == 0:
            rep = 'Empty DDF\n'
        else:
            rep = ''.join(self._get_repr_body(df))
        shape = 'Shape: {}'.format(df.shape)
        rep += '-----\n{}'.format(shape)
        rep += '\n\033[93m<<<this is a beta repr, please give MBK feedback>>>\033[0m'
        return rep


class HorizontalFormatter(object):
    def __init__(self, max_length):
        self.max_length = max_length

    def get_repr(self, df):
        items = self._get_repr_item_generator(df)
        table = '\n'.join([self._convert_col_to_str(c, v) for c, v in items])
        shape = 'DDF Shape: {}'.format(df.shape)
        return '{}\n-----\n{}'.format(table, shape)

    def _get_repr_item_generator(self, df):
        def is_first_or_last(j):
            return (j < self.max_length) or (j >= (num_columns - self.max_length))

        num_columns = len(df.columns)
        for j, (column, value) in enumerate(df.data.iteritems()):
            if is_first_or_last(j):
                yield column, value
            elif (j == self.max_length):
                yield '\n.....', '.....\n'

    def _convert_col_to_str(self, column, value):
        if isinstance(value, np.ndarray) and value.dtype == '<M8[ns]':
            value = value.astype('datetime64[D]')
        column = str(column)
        if len(column) > 80:
            column = column[:77] + '...'
        value = str(value).replace('\n', '\n\t')
        return '{}:\n\t{}'.format(column, value)

