
def get_them(text, a_list):
    y = []
    for x in a_list:
        if text in [x]:
            y.append(x)
    return y


def get_feats_containing_text(text, features):
    result = []
    for feat in features:
        if text in [feat]:
            result.append(feat)
    return result
