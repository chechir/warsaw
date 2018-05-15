
def get_them(a_list, x):
    y = []
    for _ in a_list:
        if x in _:
            y.append(x)
    return y


def get_feats_containing_pattern_(features, pattern):
    result = []
    for feat in features:
        if pattern in feat:
            result.append(feat)
    return result


def get_feats_containing_pattern(features, pattern):
    return [feat for feat in features if pattern in feat]



if __name__ == '__main__':
    pattern = 'cat_'
    feats = ['cat_1', 'num_2', 'num_3', 'num10', 'cat_33']
    new_feats = get_feats_containing_pattern(feats, pattern)

