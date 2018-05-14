from collections import OrderedDict

import numpy as np

from utils.ddf import DDF
from housePriceDone.data import TRAIN_PATH, TEST_PATH, load_df
from housePriceDone import eval_model as ev
from utils.modelling import CVFitter
import utils


def make_submission_file(train_df, test_df, sub_path, extra_model_params={}):
    df = append_dfs(train_df, test_df)
    df = ev.clean_data(df)
    df = ev.add_features(df)
    mm, targets = ev.get_mm(df), ev.get_targets(df)
    ixs = get_lb_ixs(df)
    params = {'n_estimators': 1100, 'learning_rate': 0.01, 'silent': 0}
    params.update(extra_model_params)
    model = ev.get_model(**params)
    fitter = CVFitter(model)
    results = fitter.fit(mm, targets, ixs)
    preds = results['preds'][0]
    generate_sub_file(preds, test_df['Id'], sub_path)


def append_dfs(df_train, df_test):
    df_test['SalePrice'] = np.nan
    result_df = df_train.append(df_test, axis=0)
    return result_df


def get_lb_ixs(df):
    train_ixs = np.isfinite(df[ev.target_name])
    result = OrderedDict({0: {'train': train_ixs, 'val': ~train_ixs}})
    return result


def generate_sub_file(preds, sub_id, sub_file_path):
    df = DDF({
        'Id': sub_id,
        ev.target_name: np.exp(preds)
        })
    df.to_csv(sub_file_path)


if __name__ == '__main__':
    train_df = load_df(TRAIN_PATH)
    test_df = load_df(TEST_PATH)
    sub_path = utils.paths.dropbox() + '/sub_lgb.csv'
    make_submission_file(train_df, test_df, sub_path)
