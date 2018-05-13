import numpy as np

from ddf import DDF

from housePriceDone.data import TRAIN_PATH, load_df
from housePriceDone import eval_model as ev


df = DDF({
    'MSZoning': ['FV', 'RH', 'RM'],
    'MiscFeature': [np.nan, 'Gar2', 'Othr'],
    'MoSold': [2, 5, 9],
    'SalePrice': [1000, 20000, 400000]
    })


def test_eval_model_on_cv(tmpdir):
    df = load_df(TRAIN_PATH, nrows=10)
    ev.eval_model_on_cv(df=df, log_file=tmpdir)
    results_df = DDF.from_csv(tmpdir)
    assert len(tmpdir) == 1
    assert np.isclose(results_df['rmse'][0] == 10)


def test_clean_data():
    result_df = ev.clean_data(df)
    assert result_df['MSZoning'].dtype == float
    assert np.all(df['MoSold'] == result_df['MoSold'])


def test_add_features():
    result_df = ev.add_features(df)
    assert 'f:MoSold**2' in result_df


def test_get_mm():
    result_df = ev.get_mm(df)
    assert 'SalesPrice' not in result_df.columns


def test_get_targets():
    result = ev.get_targets(df)
    assert np.allclose(result, np.array([1000, 20000, 400000]))


def test_get_cv_ixs():
    ixs = ev.get_cv_ixs(df)
    expected_ixs = {
            0: {'train': np.array([True, True, False]),
                'val': np.array([False, False, True])},
            1: {'train': np.array([False, False, True]),
                'val': np.array([True, True, False])}
            }
    for fold in expected_ixs:
        assert np.all(ixs[fold]['train'] == expected_ixs[fold]['train'])
        assert np.all(ixs[fold]['val'] == expected_ixs[fold]['val'])






