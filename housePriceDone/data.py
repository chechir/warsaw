import wutils
import os
from wutils.ddf import DDF

TRAIN_PATH = os.path.abspath('input/train.csv')
TEST_PATH = os.path.abspath('input/test.csv')


def load_df(data_path, nrows=None):
    return DDF.from_csv(data_path, nrows=nrows)

