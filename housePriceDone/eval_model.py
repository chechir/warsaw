import utils
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

from housePriceDone.data import TRAIN_PATH, load_df
from utils.modelling import CVFitter

target_name = 'SalePrice'


def eval_model_on_cv(df, log_file, extra_model_params={}):
    df = clean_data(df)
    df = add_features(df)
    mm, targets = get_mm(df), get_targets(df)
    ixs = get_cv_ixs(df)
    model = get_model(n_jobs=10, n_estimators=1, **extra_model_params)
    fitter = CVFitter(model)
    results = fitter.fit(mm, targets, ixs, early_stopping_rounds=50)
    preds = results['combined_preds']
    evals = evaluate_preds(preds, df)
    utils.io.append_csv(evals, log_file)


def clean_data(df):
    str_columns = utils.np.get_str_columns(df)
    for col in str_columns:
        df[col] = utils.features.categorical_to_numeric(df, col)
    for col in df.columns:
        df[col] = utils.np.fillna(df[col], -99)
    return df


def add_features(df):
    df['f:MoSold**2'] = df['MoSold'] ** 2
    return df


def get_mm(df):
    columns_to_drop = [target_name]
    mm_cols = [col for col in df.columns if col not in columns_to_drop]
    return df.colslice(mm_cols)


def get_targets(df):
    return df[target_name]


def get_cv_ixs(df, split_ratio=0.5):
    order = np.arange(len(df))
    split_point = round(len(df) * split_ratio, 0) - 1
    train_ixs = order <= split_point
    result = OrderedDict({
        0: {'train': train_ixs, 'val': ~train_ixs},
        1: {'train': ~train_ixs, 'val': train_ixs}
        })
    return result


def get_model(**kwargs):
    params = {
            'n_estimators': 9999,
            'n_jobs': 10,
            'max_depth': 20
            }
    params.update(kwargs)
    model = utils.models.LGBRModel(params)
    return model


def evaluate_preds(preds, df):
    evals = {
            'mse': mean_squared_error(df[target_name], preds),
            'mae': mean_absolute_error(df[target_name], preds)
            }
    return evals


if __name__ == '__main__':
    df = load_df(TRAIN_PATH)


