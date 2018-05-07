import os
from house_prizes import sub_kaggle as hp
from ddf import DDF
import seamless as ss

small_test_set_path = os.path.join(ss.paths.dropbox(), 'sample_sub_file.hd5')


def test_end_to_end():
    n_expected_rows = 30000
    sub_name = 10
    test_set = _get_small_test_set()
    hp.create_house_prizes_sub(sub_name)
    sub = load_sub(sub_path)
    assert len(sub) == n_expected_rows


def _get_small_test_set():
    df = DDF.from_hd5(small_test_set_path)
    return df


def _generate_small_test_set():
    df = hp.get_test_set()
    small_df = df.head()
    small_df.to_hd5(small_test_set_path)



