from __future__ import division
from __future__ import print_function
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
import os as os
import itertools
import numpy as np
import pandas as pd
import config as cfg
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight


# can be sentence or word
input_mask_mode = "sentence"
w_sep = re.compile("(?u)\\b[a-z'A-Z0-9_-]+\\b")


# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_data(fname, test=False):
    print("==> Loading test from %s" % fname)
    if test:
        df = pd.read_csv(fname, nrows=100)
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


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if word2vec:
        if not word in word2vec:
            create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def process_context(context, word2vec, vocab, ivocab, embed_size):
    result = []
    for ws in context.split(' <eos> '):
        if w_sep.findall(ws):
            result_temp = []
            for w in w_sep.findall(ws):
                if w:
                    result_temp.append(process_word(word=w,
                                                    word2vec=word2vec,
                                                    vocab=vocab,
                                                    ivocab=ivocab,
                                                    word_vector_size=embed_size,
                                                    to_return="index"))
            result.append(result_temp)
    return result


def process_input(data_raw, cw, word2vec, vocab, ivocab, embed_size, test=False):
    if test:
        q_obj = {'q': list(data_raw['input']), 'pred': None}

    inputs = data_raw['input'].apply(lambda x: process_context(x, word2vec, vocab, ivocab, embed_size))

    data_raw.drop(['input'], axis=1, inplace=True)
    if not test:
        mlb = MultiLabelBinarizer()
        answers = mlb.fit_transform(data_raw['keyword_id'].apply(lambda x: [i for i in x.split(' <eos> ')]))
        del data_raw
        joblib.dump(mlb, os.path.join(cfg.DATA_DIR, 'mlb_cc_dmn_train.pkl'))
        del mlb
        if cw:
            class_weights = calculating_class_weights(answers)
        else:
            class_weights = None
    if test:
        return inputs, q_obj
    else:
        return inputs, answers, class_weights


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


def batch(inputs, config):
    batch_size = len(inputs)
    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = int(min(np.percentile(document_sizes, 90), config.max_allowed_inputs))
    document_sizes = document_sizes.clip(max=document_size)
    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = int(np.percentile(list(itertools.chain(*[l for l in sentence_sizes_ if l])), 90))
    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD
    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document[:document_size]):
            sentence_sizes[i, j] = min(sentence_sizes_[i][j], sentence_size)
            for k, word in enumerate(sentence[:sentence_size]):
                b[i, j, k] = word
    return b, document_sizes, sentence_sizes, document_size, sentence_size

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding


def load_input_data(config, fname, cw, test=False):
    if test:
        vocab = joblib.load(os.path.join(cfg.DATA_DIR, 'vocabulary.pkl'))
        ivocab = joblib.load(os.path.join(cfg.DATA_DIR, 'inverse_vocabulary.pkl'))
        if config.word2vec_init:
            word2vec = joblib.load('word2vec.pkl')
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
        joblib.dump(vocab, os.path.join(cfg.DATA_DIR, 'vocabulary.pkl'))
        joblib.dump(ivocab, os.path.join(cfg.DATA_DIR, 'inverse_vocabulary.pkl'))
        if word2vec:
            joblib.dump(word2vec, os.path.join(cfg.DATA_DIR, 'word2vec.pkl'))
    # print('==> get test inputs')
    if test:
        inputs, q_obj = test_data  # if config.train_mode #else test_data
        max_sen_len = config.max_sen_len
        max_mask_len = max_sen_len
        max_input_len = config.max_input_len
        inputs, input_lens, sen_lens, max_input_len, max_sen_len = batch(inputs, config)
        # input_masks = np.zeros(len(inputs))
        answer_classes = config.num_classes
        # answers = np.stack(answers)

        print('max input len: {}'.format(max_sen_len))
    else:
        inputs, answers, class_weights = train_data  # if config.train_mode #else test_data
        joblib.dump(class_weights, os.path.join(cfg.DATA_DIR, 'class_weights.pkl'))
        inputs, input_lens, sen_lens, max_input_len, max_sen_len = batch(inputs, config)

        print(inputs.shape)
        # input_masks = np.zeros(len(inputs))

        # answers = np.stack(answers)
        answer_classes = answers.shape[-1]
        print('max num sentences: {}'.format(max_input_len))
        print ('mean num sentences: {}'.format(input_lens.mean()))
        print ('mode num sentences: {}'.format(stats.mode(input_lens)))
        print ('median num sentences: {}'.format(np.median(input_lens)))
        print('mean sentence len: {}'.format(int(np.mean(list(itertools.chain(*[l for l in sen_lens]))))))
        print('median sentence len: {}'.format(int(np.median(list(itertools.chain(*[l for l in sen_lens]))))))
        print('mode sentence len: {}'.format(int(stats.mode(list(itertools.chain(*[l for l in sen_lens])))[0][0])))
        print('mode sentence count: {}'.format(int(stats.mode(list(itertools.chain(*[l for l in sen_lens])))[1][0])))
        print('max sentence len: {}'.format(max_sen_len))
        print ('number of classes: {}'.format(answer_classes))
        train_volume = int(round(answers.shape[0] * config.num_train))

    if not test:
        train = inputs[:train_volume], answers[:train_volume], sen_lens[:train_volume], input_lens[:train_volume]  # input_masks[:train_volume],

        valid = inputs[train_volume:], answers[train_volume:], sen_lens[train_volume:], input_lens[train_volume:]  # input_masks[train_volume:],
        return train, valid, len(vocab), answer_classes, class_weights
    else:
        test = inputs  # input_masks,
        return test, len(vocab), answer_classes, q_obj


if __name__ == '__main__':
    from collections import namedtuple
    config =dict(
    batch_size = 100,
    embed_size = 100,
    hidden_cell_size = 100,
    hidden_attention_word_size = 100,
    hidden_attention_sentence_size = 100,

    max_epochs = 100,
    early_stopping = 20,

    max_sen_len = 9,
    max_input_len = 5,
    num_hidden_layers = 2,

    dropout = 0.9,
    lr = 0.001,
    l2 = 0.0001,

    cap_grads = False,
    max_grad_val = 10,
    noisy_grads = False,

    word2vec_init = False,
    embedding_init = np.sqrt(3),

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000,
    anneal_by = 1.5,

    num_hops = 3,
    num_attention_features = 4,

    max_allowed_inputs = 130,
    num_train = .80,

    floatX = np.float32,

    train_mode = True
    )
    MyTuple = namedtuple('MyTuple', sorted(config))
    my_tuple = MyTuple(**config)
    print (my_tuple.word2vec_init)
    load_input_data(my_tuple, False)
