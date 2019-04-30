from __future__ import print_function
from __future__ import division
import tensorflow as tf
import time
import argparse
import os
import config as cfg


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-l", "--l2_loss", type=float, default=0.0001, help="specify l2 loss constant")
parser.add_argument("-n", "--num_runs", type=int, help="specify the number of model runs")
parser.add_argument("-F", "--file_name", help="file name for data")
parser.add_argument("--max-grad-norm", type=float, default=5.0)


args = parser.parse_args()

from HAN_model import Config

config = Config()


config.l2 = args.l2_loss if args.l2_loss is not None else 0.0001
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1
fname = args.file_name
best_overall_val_loss = float('inf')
if config.class_weights:
    checkpoint_name = 'clf1.weights'
else:
    checkpoint_name = 'clf.weights'
checkpoint_dir = os.path.join(cfg.BASE_DIR, 'weights')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

# create model
with tf.variable_scope('HAN') as scope:
    from HAN_model import HANClassifierModel
    from self_attention_network_training_data import load_input_data
    from bn_lstm import BNLSTMCell

    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
    GRUCell = tf.nn.rnn_cell.GRUCell
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    def cell_maker():
        cell = BNLSTMCell(config.hidden_cell_size, is_training)  # h-h batchnorm LSTMCell
        # cell = GRUCell(30)
        return MultiRNNCell([cell] * config.num_hidden_layers)

    def HAN_model_1(session, restore_only=False):
        """Hierarhical Attention Network"""

        if restore_only:
            test=True
            test, vocab_size, num_classes, class_weights = load_input_data(config, test=test,
                                                                           fname=fname, cw=config.class_weights)
        else:
            test=False
            train, valid, vocab_size, num_classes, class_weights = load_input_data(config, test=test,
                                                                                   cw=config.class_weights,
                                                                                   fname=fname)
        model = HANClassifierModel(vocab_size=vocab_size, embedding_size=config.embed_size,
                                   classes=num_classes, word_cell=cell_maker, sentence_cell=cell_maker,
                                   word_output_size=config.hidden_attention_word_size,
                                   sentence_output_size=config.hidden_attention_sentence_size,
                                   num_classes=num_classes, class_weights=class_weights,
                                   max_grad_norm=args.max_grad_norm,
                                   dropout_keep_proba=config.dropout, is_training=is_training,
                                   multi_label_flag=config.multilabel)

        saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint:
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            saver.restore(session, checkpoint.model_checkpoint_path)
        elif restore_only:
            print('==> restoring weights')
            saver.restore(session, checkpoint.model_checkpoint_path)
        else:
            print("Created model with fresh parameters")
            session.run(tf.global_variables_initializer())
        # tf.get_default_graph().finalize()
        if restore_only:
            return model, saver, test
        else:
            return model, saver, train, valid


for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
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

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        # best_val_accuracy = 0.0

        if args.restore:
            model, saver, test = HAN_model_1(session, restore_only=True)
        else:
            model, saver, train, valid = HAN_model_1(session, restore_only=False)
        total_train_steps = len(train[0]) // config.batch_size
        total_valid_steps = len(valid[0]) // config.batch_size
        print('==> starting training')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
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
                    best_overall_val_loss = best_val_loss
                    # best_val_accuracy = valid_accuracy
            if train_loss < prev_epoch_loss:
                print('Saving weights')
                if model.class_weights:
                    saver.save(session, 'weights/clf1.weights')
                else:
                    saver.save(session, 'weights/clf.weights')
            # anneal
            if train_loss > prev_epoch_loss * config.anneal_threshold:
                config.lr /= config.anneal_by
                print('annealed lr to %f' % config.lr)

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))