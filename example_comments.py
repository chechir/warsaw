import re
from nltk.corpus import stopwords
url = 'https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings'


def get_wordlist_from_text(
        text, remove_stopwords=False, stem_words=False):
    result = text.lower()
    if remove_stopwords:
        result = apply_remove_stopwords(result)
    result = clean_text(result)
    if stem_words:
        result = apply_stem_words(result)
    return result


def apply_remove_stopwords(text):
    stops = set(stopwords.words("english"))
    result = text.split()
    result = [word for word in result if word not in stops]
    result = _join_words(result)
    result = ' '.join(text)
    return result


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
    return result


def _split_words(text):
    return [word for word in text]


def _join_words(text):
    return ' '.join(text)


def remove_stop_words():
    pass


def apply_stem_words():
    pass




def _split_words(text):
    return [word for word in text]


def _join_words(text):
    return ' '.join(text)

