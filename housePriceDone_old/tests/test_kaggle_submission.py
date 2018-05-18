import os
import pandas as pd

from housePriceDone.data import TRAIN_PATH, TEST_PATH
from housePriceDone.make_submission import generate_kaggle_submission


def test_generate_kaggle_submission(tmpdir):
    submission_path = os.path.join(tmpdir)
    df_train = load_small_df(TRAIN_PATH)
    df_test = load_small_df(TEST_PATH)
    generate_kaggle_submission(df_train, df_test, submission_path)
    resulting_df = pd.read_csv(submission_path)
    assert len(resulting_df) == len(df_test)
    assert 'salePrice' in resulting_df.columns


def load_small_df(data_path):
    return pd.read_csv(data_path, nrows=10)

