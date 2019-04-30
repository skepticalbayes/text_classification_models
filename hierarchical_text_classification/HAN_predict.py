# Created by Bhaskar at 31/8/18
import sys

from self_attention_network_training_data import process_context, batch, get_raw_data

reload(sys)
import tensorflow as tf
import numpy as np
import os
from sklearn.externals import joblib
from HAN_model import Config, HANClassifierModel
from HAN_train import cell_maker
import config as cnfg

cfg = Config()
vocab = joblib.load(os.path.join(cnfg.DATA_DIR, 'vocabulary.pkl'))
ivocab = joblib.load(os.path.join(cnfg.DATA_DIR, 'inverse_vocabulary.pkl'))
answer_labels = joblib.load(os.path.join(cnfg.DATA_DIR, 'mlb_cc_dmn_train.pkl')).classes_

with tf.variable_scope('HAN') as scope:
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    config_.intra_op_parallelism_threads = 16
    config_.inter_op_parallelism_threads = 16
    # 4.Instantiate Model
    textHAN = HANClassifierModel(vocab_size=len(vocab), embedding_size=cfg.embed_size,
                                 classes=cfg.num_classes, word_cell=cell_maker,
                                 sentence_cell=cell_maker,
                                 word_output_size=cfg.hidden_attention_word_size,
                                 sentence_output_size=cfg.hidden_attention_sentence_size,
                                 num_classes=cfg.num_classes, class_weights=cfg.class_weights,
                                 max_grad_norm=5.0,
                                 dropout_keep_proba=1, is_training=False, multi_label_flag=cfg.multilabel
                                 )


    sess = tf.Session(config=config_)
    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    # if os.path.exists('weights/checkpoint'):
    # try:
    print('==> restoring weights')
    if cfg.class_weights:
        saver.restore(sess, 'weights/clf1.weights')
    else:
        saver.restore(sess, 'weights/clf.weights')
    # except:
    #     print("Can't find the checkpoint.going to stop")


def parse_input(x, vcb=vocab, ivcb=ivocab):
    inputs = np.array([process_context(i, word2vec=None, vocab=vcb, ivocab=ivcb, embed_size=cfg.embed_size) for i in x])
    inputs, input_lens, sen_lens, _, _ = batch(inputs, cfg)
    return inputs, input_lens, sen_lens


def predict_prob(ip, il, sl):
    feed = {textHAN.inputs: ip,
            textHAN.sentence_lengths: il,
            textHAN.word_lengths: sl,
            textHAN.is_training: False}
    return tf.squeeze(np.array(sess.run([textHAN.possibility], feed_dict=feed)), axis=0)


def predict_top_k(probs, k=5):
    results = tf.nn.top_k(probs, k=k)
    # print results
    results = sess.run(results)
    labels = np.array([answer_labels[j] for i in results.indices for j in i])
    probs = results.values
    # print results.shape
    return labels, probs


def predict_cc(probs, th):
    return np.array([answer_labels[i >= th] for i in probs])


if __name__ == '__main__':
    df = get_raw_data(test=True)
    ip, il, sl = parse_input(x=df['input'])
    result = predict_prob(ip, il, sl)
    df['result'], df['probs'] = predict_top_k(result, k=1)
    df.to_csv('test.csv', index=False)
    # Created by Bhaskar at 1/9/18