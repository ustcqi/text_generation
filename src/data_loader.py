import sys
import string

import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')


def string_module_test():
    print(string.punctuation)
    print(type(string.punctuation))
    print(string.ascii_letters)
    print(string.printable)
#string_module_test()

# remove all the words that its charactor not in string.printable
def check(word):
    chars = list(set(word))
    ascii = set(string.printable)
    for char in chars:
        if char not in ascii:
            return False
    return True


def get_raw_text(filename):
    raw_text = open(filename, 'r').read().split('\n')
    raw_text = ' '.join(raw_text)
    raw_text = [word.strip(string.punctuation) for word in raw_text.split() if check(word)]
    raw_text = ' '.join(raw_text)
    raw_text = raw_text.replace('-', ' ')
    raw_text = ' '.join(raw_text.split())
    return raw_text


def get_all_words(raw_text):
    all_words = raw_text.split()
    return all_words


def get_word2idx(all_words):
    unique_words = sorted(list(set(all_words)))
    word_to_index = dict((w, i) for i, w in enumerate(unique_words))
    return word_to_index


def get_idx2word(filename):
    raw_text = get_raw_text(filename)
    all_words = raw_text.split()
    unique_words = sorted(list(set(all_words)))
    index_to_word = dict((i, w) for i, w in enumerate(unique_words))
    return index_to_word


def get_nb_words(all_words):
    nb_words = len(all_words)
    return nb_words


def get_nb_vocab(filename):
    raw_text = get_raw_text(filename)
    all_words = raw_text.split()
    unique_words = sorted(list(set(all_words)))
    nb_vocab = len(unique_words)
    return nb_vocab


def get_train_data(filename, step=100):
    raw_text = get_raw_text(filename)
    all_words = get_all_words(raw_text)
    nb_words = get_nb_words(all_words)
    nb_vocab = get_nb_vocab(filename)
    print('Total words', nb_words)
    print('Total Vocab', nb_vocab)
    word2idx = get_word2idx(all_words)
    #idx2word = get_idx2word(all_words)

    #step = 100
    X = []
    Y = []
    for i in xrange(0, nb_words - step):
        seq_in = all_words[i: i+step]
        seq_out = all_words[i+step]
        X.append([word2idx[word] for word in seq_in])
        Y.append(word2idx[seq_out])

    #nb_patterns = len(X)
    #X = np.array(X)
    #X = np.reshape(X, (nb_patterns, step, 1)) / float(nb_vocab)
    #Y = np.array(Y)
    #from keras.utils import np_utils
    #Y = np_utils.to_categorical(Y)
    return X, Y


def transform(X, Y, nb_vocab, step=100):
    X = np.reshape(X, (len(X), step, 1)) / float(nb_vocab)
    from keras.utils import np_utils
    Y = np_utils.to_categorical(Y)
    return X, Y
