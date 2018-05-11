import re
url = 'https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings'


def get_wordlist_from_text(
        text, remove_stopwords=False, stem_words=False):
    text = convert_words_to_lower(text)
    if remove_stopwords:
        text = apply_remove_stopwords(text)
    text = clean_text(text)
    if stem_words:
        text = apply_stem_words(text)
    return text


def convert_words_to_lower(text):
    result = _split_words(text)
    result = result.lower(result)
    result = _join_words(result)
    return result.lower()


def clean_text(text):
    result = text[:]
    replaces = {
            "[^A-Za-z0-9^,!.\/'+-=]": ' ',
            ' e g ': ' eg ', "\'s": " ",
            "\'ve": " have ", "can't": "cannot ",
            "n't": " not ",
            "i'm": "i am ",
            "\'re": " are ",
            "\'d": " would ",
            "\'ll": " will "
            }
    for pattern, replace_value in replaces.iteritems():
        result = re.sub(pattern, replace_value, result)
    pass


def remove_stop_words():
    pass


def apply_stem_words():
    pass


def apply_remove_stopwords():
    pass


def _split_words(text):
    return [word for word in text]


def _join_words(text):
    return ' '.join(text)

