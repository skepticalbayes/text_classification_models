from __future__ import division
from __future__ import print_function
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
import os as os
import config as cfg
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# can be sentence or word
input_mask_mode = "sentence"
w_sep = re.compile("(?u)\\b[a-z'A-Z0-9_-]+\\b")


# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_data(fname, test=False):
    print("==> Loading test from %s" % fname)
    if test:
        df = pd.read_csv(fname)
    else:
        df = pd.read_csv(fname)
    return df


def get_raw_data(fname, test=False):
    train_raw = init_data(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       fname), test=test)
    return train_raw


def load_glove(dim):
    word2vec = {}

    print("==> loading glove")
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])

    print("==> glove is loaded")

    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True, test=False):
    if word2vec:
        if not word in word2vec:
            create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        if not test:
            next_index = len(vocab)
            vocab[word] = next_index
            ivocab[next_index] = word
        else:
            return 0
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def process_context(context, word2vec, vocab, ivocab, embed_size, test=False):
    result_temp = []
    if w_sep.findall(context):
        for w in w_sep.findall(context):
            if w:
                result_temp.append(process_word(word=w,
                                                word2vec=word2vec,
                                                vocab=vocab,
                                                ivocab=ivocab,
                                                word_vector_size=embed_size,
                                                to_return="index", test=test))
    return result_temp


def process_input(data_raw, word2vec, vocab, ivocab, embed_size, cw, test=False):
    if test:
        q_obj = {'q': list(data_raw['input']), 'pred': None}

    inputs = data_raw['input'].apply(lambda x: process_context(x, word2vec, vocab, ivocab, embed_size, test=test))
    # input_sen_lens = data_raw['input_sen_len'].apply(lambda x: [int(i) for i in x.split(' <eos> ') if int(i)])
    input_len = inputs.apply(lambda x: len(x))

    # data_raw.drop(['input', 'input_sen_len'], axis=1, inplace=True)

    if not test:
        mlb = MultiLabelBinarizer()
        answers = mlb.fit_transform(data_raw['keyword_id'].apply(
            lambda x: [i for i in x.split(' <eos> ')]))
        del data_raw
        if cw:
            class_weights = calculating_class_weights(answers)
        else:
            class_weights = None
        joblib.dump(mlb, os.path.join(cfg.DATA_DIR, 'cc_mlb_train.pkl'))
    else:
        mlb = joblib.load(os.path.join(cfg.DATA_DIR, 'cc_mlb_train.pkl'))
        answers = mlb.transform(data_raw['keyword_id'].apply(
            lambda x: [i for i in x.split(' <eos> ')]))
        del data_raw
    del mlb
    if test:
        return inputs, input_len, answers, q_obj
    else:
        return inputs, answers, input_len, class_weights


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


# def get_lens(inputs, split_sentences=False):
#     lens = np.zeros((len(inputs)), dtype=int)
#     for i, t in enumerate(inputs):
#         lens[i] = t.shape[0]
#     return lens

# def get_sentence_lens(inputs):
#     lens = np.zeros((len(inputs)), dtype=int)
#     sen_lens = []
#     max_sen_lens = []
#     for i, t in enumerate(inputs):
#         sentence_lens = np.zeros((len(t)), dtype=int)
#         for j, s in enumerate(t):
#             sentence_lens[j] = len(s)
#         lens[i] = len(t)
#         sen_lens.append(sentence_lens)
#         max_sen_lens.append(np.max(sentence_lens))
#     return lens, sen_lens, max(max_sen_lens)


def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            inp = inp[:max_len]
            padded_sentences = [
                np.pad(s[:max_sen_len], (0, max_sen_len - len(s[:max_sen_len])), 'constant', constant_values=0) for j, s
                in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences) - max_len):]
                lens[i] = max_len
            if padded_sentences:
                padded_sentences = np.vstack(padded_sentences)
                padded_sentences = np.pad(padded_sentences, ((0, max_len - len(inp)), (0, 0)), 'constant',
                                          constant_values=0)
                padded[i] = padded_sentences
        return padded

    padded = [np.pad(inp[:max_len], (0, max_len - len(inp[:max_len])), 'constant', constant_values=0) for i, inp in
              enumerate(inputs)]
    return np.vstack(padded)


def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding


def load_input_data(config, fname, cw, test=False):
    if test:
        vocab = joblib.load(os.path.join(cfg.DATA_DIR, 'data', 'cc_vocabulary.pkl'))
        ivocab = joblib.load(os.path.join(cfg.DATA_DIR, 'data', 'cc_inverse_vocabulary.pkl'))
        if config.word2vec_init:
            pass
        else:
            word2vec = {}
    else:
        vocab = {}
        ivocab = {}

        if config.word2vec_init:
            assert config.embed_size == 100
            word2vec = load_glove(config.embed_size)
        else:
            word2vec = {}
    train_raw = get_raw_data(fname, test=test)
    # set word at index zero to be end of sentence token so padding with zeros is consistent
    if not test:
        process_word(word="",
                     word2vec=word2vec,
                     vocab=vocab,
                     ivocab=ivocab,
                     word_vector_size=config.embed_size,
                     to_return="index")
        process_word(word="eos",
                     word2vec=word2vec,
                     vocab=vocab,
                     ivocab=ivocab,
                     word_vector_size=config.embed_size,
                     to_return="index")

    print('==> get train inputs')
    if test:
        test_data = process_input(train_raw, word2vec, vocab, ivocab, config.embed_size, test=test, cw=cw)
    else:
        train_data = process_input(train_raw, word2vec, vocab, ivocab, config.embed_size, test=test, cw=cw)
        joblib.dump(vocab, os.path.join(cfg.DATA_DIR, 'cc_vocabulary.pkl'))
        joblib.dump(ivocab, os.path.join(cfg.DATA_DIR, 'cc_inverse_vocabulary.pkl'))
        if word2vec:
            pass
    # print('==> get test inputs')
    if test:
        inputs, input_lens, answers, q_obj = test_data  # if config.train_mode #else test_data
        # max_sen_len = config.max_sen_len
        # max_mask_len = max_sen_len
        max_input_len = int(min(config.sequence_length, config.max_allowed_inputs))
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        # input_masks = np.zeros(len(inputs))
        answer_classes = config.num_classes
        # answers = np.stack(answers)

        # print('max input len: {}'.format(max_sen_len))
    else:
        inputs, answers, input_lens, class_weights = train_data  # if config.train_mode #else test_data

        # max_sen_len = int(np.percentile(list(itertools.chain(*[l for l in sen_lens if l])), 90))
        # max_mask_len = max_sen_len

        max_input_len = int(min(np.percentile(input_lens, 90), config.max_allowed_inputs))

        inputs = pad_inputs(inputs, input_lens, max_input_len)

        print(inputs.shape)
        # input_masks = np.zeros(len(inputs))

        # answers = np.stack(answers)
        answer_classes = answers.shape[-1]
        print('max len sentences: {}'.format(max_input_len))
        print ('mean len sentences: {}'.format(input_lens.mean()))
        print('mode len sentences: {}'.format(input_lens.mode()))
        print('median len sentences: {}'.format(input_lens.median()))
        # print('mean sentence len: {}'.format(int(np.mean(list(itertools.chain(*[l for l in sen_lens if l]))))))
        # print('max sentence len: {}'.format(max_sen_len))
        train_volume = int(round(answers.shape[0] * config.num_train))

    if not test:
        train = inputs[:train_volume], answers[:train_volume]  # input_masks[:train_volume],

        valid = inputs[train_volume:], answers[train_volume:]  # input_masks[train_volume:],
        return train, valid, max_input_len, len(vocab), answer_classes, class_weights
    else:
        test = inputs, answers  # input_masks,
        return test, max_input_len, len(vocab), answer_classes, q_obj
