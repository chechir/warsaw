
url = 'https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings'


if __name__ == '__main__':

    def get_wordlist_from_text(text, remove_stopwords=False, stem_words=False):
        text = convert_words_to_lower(text)
        text = split_words(text)
        if remove_stopwords:
            text = remove_stop_words(text)

        text = clean_text(text)


def convert_words_to_lower():
    pass


def split_words():
    pass


def clean_text():
    pass


def remove_stop_words():
    def





