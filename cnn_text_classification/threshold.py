import argparse
import sys

import time

from classification_data import load_input_data

reload(sys)
import tensorflow as tf
import numpy as np
from cnn_model import Config, TextCNN
from sklearn.metrics import accuracy_score, precision_score, jaccard_similarity_score,precision_recall_fscore_support, roc_auc_score

cfg = Config()
ts = np.arange(0, 1, 0.01, dtype=float)
ts = sorted(ts, reverse=True)
parser = argparse.ArgumentParser()
parser.add_argument("-F", "--file_name", help="file name for data")
args = parser.parse_args()
fname = args.file_name

def repredict(Y_pred, t):
    ypred = (Y_pred >= t)*1
    return ypred


with tf.variable_scope('CNN') as scope:
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    config_.intra_op_parallelism_threads = 16
    config_.inter_op_parallelism_threads = 16
    # 4.Instantiate Model
    test, max_sentences, vocab_size, num_classes, _ = load_input_data(cfg, cw=cfg.class_weights,
                                                                      test=True, )
    textCNN = TextCNN(filter_sizes=cfg.filter_sizes, num_filters=cfg.num_filters,
                      num_classes=cfg.num_classes, class_weights=cfg.class_weights, learning_rate=cfg.lr,
                      batch_size=cfg.batch_size, decay_steps=1000,
                      decay_rate=0.9, sequence_length=cfg.sequence_length, vocab_size=vocab_size,
                      embed_size=cfg.embed_size, is_training=False,
                      initializer=tf.random_normal_initializer(stddev=0.1),
                      multi_label_flag=True, clip_gradients=5.0, decay_rate_big=0.50)

    sess = tf.Session(config=config_)
    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    # if os.path.exists('weights/checkpoint'):
    # try:
    print('==> restoring weights')
    if cfg.class_weights:
        print ('weights/clf_weighted.weights')
        saver.restore(sess, 'weights/weights/clf_weighted.weights')
    else:
        saver.restore(sess, 'weights/clf.weights')
    # except:
    #     print("Can't find the checkpoint.going to stop")
    total_steps = len(test[0]) // 1000

    # shuffle data
    ip, a = test
    b1_Ts = []
    b2_Ts = []
    b3_Ts = []
    jc_Ts = []
    p_Ts = []
    f_Ts = []
    p = []
    f = []
    j = []
    for step in range(total_steps):
        print ('\r Step: {}'.format(step))
        start = time.time()
        index = range(step * 1000, (step + 1) * 1000)
        feed = {textCNN.input_x: ip[index],
                textCNN.dropout_keep_prob: 1,
                textCNN.tst: not cfg.train_mode}
        probs = sess.run([textCNN.possibility], feed_dict=feed)
        probs = sess.run(tf.squeeze(np.array(probs), axis=0))
        prec = np.array([precision_score(a[index].reshape(-1, ), repredict(probs.reshape(-1, ), t)) for t in ts])
        if prec[prec >= .95].shape[0]>0:
            b1_Ts.append(ts[np.argmin(prec[prec >= .95])])
        if prec[prec >= .9].shape[0]>0:
            b2_Ts.append(ts[np.argmin(prec[prec >= .90])])
        if prec[prec >= .8].shape[0] > 0:
            b3_Ts.append(ts[np.argmin(prec[prec >= .80])])
        jc = [jaccard_similarity_score(a[index], repredict(probs, t)) for t in ts]
        jc_Ts.append(ts[np.argmax(jc)])
        scores = [precision_recall_fscore_support(a[index], repredict(probs, t), average='samples') for t in ts]
        precision = [i[0] for i in scores]
        f_score = [i[2] for i in scores]
        p_Ts.append(ts[np.argmax(precision)])
        f_Ts.append(ts[np.argmax(f_score)])
        p.append(np.max(precision))
        f.append(np.max(np.max(f_score)))
        j.append(np.max(np.max(jc)))
        end = time.time()
        sys.stdout.write('\r{} / {} : b1_T = {} b2_T = {} b3_T = {} Jc_T = {} jc = {} p_T = {} p = {} f_T = {} f = {} epoch_run_time = {}'.format(
            step, total_steps, np.mean(b1_Ts), np.mean(b2_Ts),
            np.mean(b3_Ts), np.mean(jc_Ts), np.mean(j), np.mean(p_Ts), np.mean(p), np.mean(f_Ts), np.mean(f_score), end - start))
        sys.stdout.flush()
        print('\n')