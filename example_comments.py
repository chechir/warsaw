url = 'https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings'


def get_wordlist_from_text(
        text, remove_stopwords=False, stem_words=False):
    text = convert_words_to_lower(text)
    text = split_words(text)
    if remove_stopwords:
        text = apply_remove_stopwords(text)
    text = clean_text(text)
    if stem_words:
        text = apply_stem_words(text)
    return text


def convert_words_to_lower():
    pass


def split_words():
    pass


def clean_text():
    pass


def remove_stop_words():
    pass


def apply_stem_words():
    pass


def apply_remove_stopwords():
    pass



