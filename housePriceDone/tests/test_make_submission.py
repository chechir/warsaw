import numpy as np

from wutils.ddf import DDF
from housePriceDone import make_submission as ms


df = DDF({
    'Id': np.arange(3),
    'MiscFeature': [np.nan, 'Gar2', 'Othr'],
    'MoSold': [2, 5, 9],
    'SalePrice': [1000, 20000, 400000]
    })

train_df = df.colslice(df.columns)
test_df = df.drop_columns(['SalePrice'])


def test_make_submission_file(tmpdir):
    sub_path = str(tmpdir) + 'sub.csv'
    model_params = {'min_data': 1, 'min_data_in_bin': 1}
    ms.make_submission_file(train_df, test_df, sub_path, extra_model_params=model_params)
    result_df = DDF.from_csv(sub_path)
    assert 'SalePrice' in result_df.columns
    assert 'Id' in result_df.columns
    assert len(result_df.columns) == 2


def test_append_dfs():
    df_train = df.colslice(df.columns)
    df_test = df.drop_columns(['SalePrice'])
    result_df = ms.append_dfs(df_train, df_test)
    assert len(result_df) == len(df_train) + len(df_test)


def test_get_lb_ixs():
    df = DDF({
        'col1': np.arange(5),
        'SalePrice': np.array([100.]*3 + [np.nan]*2)
        })
    ixs = ms.get_lb_ixs(df)
    expected = {
            0: {'train': np.array([True]*3 + [False]*2),
                'val': np.array([False]*3 + [True]*2)}
            }
    assert np.all(expected[0]['train'] == ixs[0]['train'])
    assert np.all(expected[0]['val'] == ixs[0]['val'])


def test_generate_sub_file(tmpdir):
    preds = np.arange(10)
    id_sub = np.arange(10)
    sub_file_path = tmpdir + 'file.csv'
    ms.generate_sub_file(preds, id_sub, sub_file_path)
    result_df = DDF.from_csv(sub_file_path)
    assert result_df.shape[1] == 2
    assert result_df.shape[0] == 10
