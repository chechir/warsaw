import utils
import os
from ddf import DDF

TRAIN_PATH = os.path.join(utils.paths.dropbox(), 'HousePricesData/train.csv')


def load_df(data_path, nrows=None):
    return DDF.from_csv(data_path, nrows=nrows)

