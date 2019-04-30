# Created by Bhaskar at 8/9/18
import sys

from bilstm_training_data import process_context, pad_inputs, get_raw_data

reload(sys)
import tensorflow as tf
import numpy as np
import os
from sklearn.externals import joblib
from bilstm_model import Config, TextRNN
import config as cnfg


cfg = Config()
vocab = joblib.load(os.path.join(cnfg.DATA_DIR, 'cc_vocabulary.pkl'))
ivocab = joblib.load(os.path.join(cnfg.DATA_DIR, 'cc_inverse_vocabulary.pkl'))
answer_labels = joblib.load(os.path.join(cnfg.DATA_DIR, 'cc_mlb_train.pkl')).classes_


with tf.variable_scope('BILSTMDCH') as scope:
    config_ = tf.ConfigProto()
    # config_.gpu_options.allow_growth = True
    config_.intra_op_parallelism_threads = 3
    config_.inter_op_parallelism_threads = 3
    # 4.Instantiate Model
    textRNN = TextRNN(num_classes=cfg.num_classes, class_weights=cfg.class_weights,
                      learning_rate=cfg.lr,
                      batch_size=cfg.batch_size, decay_steps=1000, decay_rate=0.9,
                      sequence_length=cfg.sequence_length, vocab_size=len(vocab),
                      embed_size=cfg.embed_size, is_training=True,
                      initializer=tf.random_normal_initializer(stddev=0.1),
                      multiple_layers=True, multi_label_flag=cfg.multilabel)


    sess = tf.Session(config=config_)
    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    # if os.path.exists('weights/checkpoint'):
    # try:
    print('==> restoring weights')
    if cfg.class_weights:
        print "cc1"
        saver.restore(sess, os.path.join(cnfg.BASE_DIR, 'weights/clf_weighted.weights'))
    else:
        print "cc"
        saver.restore(sess, os.path.join(cnfg.BASE_DIR, 'weights/clf.weights'))
    # except:
    #     print("Can't find the checkpoint.going to stop")
    # sess.graph.finalize()


def parse_input(x, vcb=vocab, ivcb=ivocab):
    inputs = np.array([process_context(i, word2vec=None, vocab=vcb, ivocab=ivcb, embed_size=cfg.embed_size, test=True) for i in x])
    input_len = [len(i) for i in inputs]
    inputs = pad_inputs(inputs, input_len, cfg.sequence_length)
    return inputs


def predict_prob(text):
    inputs = parse_input(text)
    feed = {textRNN.input_x: inputs,
            textRNN.dropout_keep_prob: 1.0}
    return sess.run([textRNN.possibility], feed_dict=feed)[0]


def predict_top_k(probs, k=1):
    labels = np.array([answer_labels[np.argpartition(i, -k)[-k:]] for i in probs])
    probs = np.array([i[np.argpartition(i, -k)[-k:]] for i in probs])
    # print results.shape
    return labels, probs


def predict_kwd(probs, th):
    return np.array([answer_labels[i >= th] for i in probs])


if __name__ == '__main__':
    pass