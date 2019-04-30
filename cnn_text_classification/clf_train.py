from __future__ import print_function
from __future__ import division

import tensorflow as tf

import time
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-l", "--l2_loss", type=float, default=0.0001, help="specify l2 loss constant")
parser.add_argument("-n", "--num_runs", type=int, help="specify the number of model runs")
parser.add_argument("-F", "--file_name", help="file name for data")
parser.add_argument("-f", "--final_train", help="final train after cross validation")

args = parser.parse_args()
from cnn_model import Config

config = Config()


config.l2 = args.l2_loss if args.l2_loss is not None else 0.000003
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1
final_train = True if args.final_train else False
fname = args.file_name
if final_train:
    best_overall_train_loss = float('inf')
else:
    best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('CNN') as scope:
    from cnn_model import TextCNN
    from classification_data import load_input_data

    if not final_train:
        train, valid, max_sentences, vocab_size, num_classes, class_wts = load_input_data(config,
                                                                                          cw=config.class_weights,
                                                                                          fname=fname)
    else:
        train, max_sentences, vocab_size, num_classes, _ = load_input_data(config,
                                                                                  cw=config.class_weights,
                                                                                  ft=final_train,
                                                                           fname=fname)
        class_wts = None

    model = TextCNN(filter_sizes=config.filter_sizes,num_filters=config.num_filters,
                    num_classes=num_classes, class_weights=class_wts,
                    learning_rate=config.lr, batch_size=config.batch_size, decay_steps=1000,
                    decay_rate=0.9, sequence_length=max_sentences, vocab_size=vocab_size,
                    embed_size=config.embed_size, is_training=True,
                    initializer=tf.random_normal_initializer(stddev=0.1),
                    multi_label_flag=True, clip_gradients=5.0, decay_rate_big=0.50)

for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    config_.intra_op_parallelism_threads = 16
    config_.inter_op_parallelism_threads = 16

    with tf.Session(config=config_) as session:
        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)
        if not final_train:
            best_val_epoch = 0
            prev_epoch_loss = float('inf')
            best_val_loss = float('inf')
            # best_val_accuracy = 0.0
        else:
            best_train_epoch = 0
            prev_epoch_loss = float('inf')
            best_train_loss = float('inf')
            # best_val_accuracy = 0.0

        if args.restore:
            print('==> restoring weights')
            if config.class_weights:
                saver.restore(session, 'weights/clf.weights')
            else:
                saver.restore(session, 'weights/clf_weighted.weights')
        if not final_train:
            total_train_steps = len(train[0]) // config.batch_size
            total_valid_steps = len(valid[0]) // config.batch_size
        else:
            total_train_steps = len(train[0]) // config.batch_size
        print('==> starting training')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            if not final_train:
                train_loss = model.run_epoch(config,
                    session, train, epoch, train_writer,
                    train_op=model.train_op, train=True)
                valid_loss = model.run_epoch(config, session, valid)
                print('Training loss: {}'.format(train_loss))
                print('Validation loss: {}'.format(valid_loss))
                # print('Training accuracy: {}'.format(train_accuracy))
                # print('Vaildation accuracy: {}'.format(valid_accuracy))

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    if best_val_loss < best_overall_val_loss:
                        print ('saving weights')
                        if config.class_weights:
                            saver.save(session, 'weights/clf.weights')
                        else:
                            saver.save(session, 'weights/cclf_weighted.weights')
                        best_overall_val_loss = best_val_loss
                        # best_val_accuracy = valid_accuracy

                # anneal
                if train_loss > prev_epoch_loss * config.anneal_threshold:
                    config.lr /= config.anneal_by
                    print('annealed lr to %f' % config.lr)

                prev_epoch_loss = train_loss

                if epoch - best_val_epoch > config.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))
            else:
                train_loss = model.run_epoch(config,
                                             session, train, epoch, train_writer,
                                             train_op=model.train_op, train=True, device=0)
                print('Training loss: {}'.format(train_loss))
                # print('Training accuracy: {}'.format(train_accuracy))
                # print('Vaildation accuracy: {}'.format(valid_accuracy))

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_epoch = epoch
                    if best_train_loss < best_overall_train_loss:
                        print('saving weights')
                        if config.class_weights:
                            saver.save(session, 'weights/clf.weights')
                        else:
                            saver.save(session, 'weights/clf_weighted.weights')
                        best_overall_train_loss = best_train_loss
                        # best_val_accuracy = valid_accuracy

                # anneal
                if train_loss > prev_epoch_loss * config.anneal_threshold:
                    config.lr /= config.anneal_by
                    print('annealed lr to %f' % config.lr)

                prev_epoch_loss = train_loss

                if epoch - best_train_epoch > config.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))


            # print('Best validation accuracy:', best_val_accuracy)
