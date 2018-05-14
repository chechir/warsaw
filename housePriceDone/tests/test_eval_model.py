import numpy as np

from utils.ddf import DDF

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
    model_params = {'min_data': 1, 'min_data_in_bin': 1}
    log_path = str(tmpdir) + 'file.csv'
    ev.eval_model_on_cv(
            df=df, log_file=log_path,
            extra_model_params=model_params)
    results_df = DDF.from_csv(log_path)
    assert len(results_df) == 1
    assert np.isclose(results_df['rmse'][0], 0.28874433)


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
    expected = np.log(np.array([1000, 20000, 400000]))
    assert np.allclose(result, expected)


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


def test_evaluate_preds():
    preds = np.log(np.array([1000, 20000, 400000]))
    targets = preds[:]
    result = ev.evaluate_preds(preds, targets)
    for metric in result:
        assert result[metric] == 0
