from repo.data import test_data_path
from ddf import DDF


def get_test_set():
    DDF.from_hd5(test_data_path)


def create_house_prizes_sub():
    pass
