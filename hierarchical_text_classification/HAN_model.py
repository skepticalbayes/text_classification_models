import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import time
from model_components import task_specific_attention, bidirectional_rnn

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 1000
    embed_size = 100
    hidden_cell_size = 100
    hidden_attention_word_size = 100
    hidden_attention_sentence_size = 100
    class_weights = None
    max_epochs = 100
    early_stopping = 20
    num_classes = 2081
    max_sen_len = 9
    max_input_len = 5
    num_hidden_layers = 5

    dropout = 0.9
    lr = 0.001
    l2 = 0.0001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3)
    multilabel = True
    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = .90

    floatX = np.float32

    train_mode = True


class HANClassifierModel():
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 num_classes,
                 class_weights,
                 max_grad_norm,
                 dropout_keep_proba,
                 is_training=None,
                 learning_rate=1e-4,
                 device='/cpu:0',
                 scope=None,
                 multi_label_flag=False):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.num_classes = num_classes
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba
        self.multi_label_flag = multi_label_flag
        with tf.variable_scope(scope or 'tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            self.class_weights = class_weights

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None, self.num_classes), dtype=tf.float32, name='labels')

            (self.document_size,
             self.sentence_size,
             self.word_size) = tf.unstack(tf.shape(self.inputs))

            self._init_embedding(scope)

            # embeddings cannot be placed on GPU
            with tf.device(device):
                self._init_body(scope)

        with tf.variable_scope('train'):
            self.loss_val = self.get_weighted_loss(weights=self.class_weights, y_true=self.labels, y_pred=self.logits)

            # self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            # tf.summary.scalar('loss', self.loss)

            # self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            # tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss_val, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def get_weighted_loss(self, weights, y_true, y_pred, l2_lambda=0.0001):
        if self.class_weights:
            loss = tf.reduce_mean((weights[:, 0] ** (1 - y_true)) *
                              (weights[:, 1] ** y_true) *
                              tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        else:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss+l2_losses
        return loss


    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell(), self.word_cell(),
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            # sentence_level

            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence') as scope:
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell(), self.sentence_cell(), sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, self.sentence_output_size, scope=scope)

                with tf.variable_scope('dropout'):
                    sentence_level_output = layers.dropout(
                        sentence_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sentence_level_output, self.classes, activation_fn=None)
                if self.multi_label_flag:
                    self.possibility = tf.nn.sigmoid(self.logits)
                else:
                    self.possibility = tf.nn.softmax(self.logits)

                self.prediction = tf.argmax(self.logits, axis=-1)

    def run_epoch(self, config, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2,
                  train=False):
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0
        if train:
            is_training = True
        else:
            is_training = False

        # shuffle data
        p = np.random.permutation(len(data[0]))
        ip, a, sl, il = data
        ip, a, sl, il = ip[p], a[p], sl[p], il[p]

        for step in range(total_steps):
            start = time.time()
            index = range(step * config.batch_size, (step + 1) * config.batch_size)
            feed = {self.inputs: ip[index],
                    self.sentence_lengths: il[index],
                    self.word_lengths: sl[index],
                    self.labels: a[index],
                    self.is_training: is_training}
            step, summaries, loss, _ = session.run([self.global_step,
                                                    self.summary_op,
                                                    self.loss_val,
                                                    train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summaries, num_epoch * total_steps + step)

            # answers = a[step*config.batch_size:(step+1)*config.batch_size]
            # accuracy += np.sum(pred == answers)/float(len(answers))


            total_loss.append(loss)
            if verbose and step % verbose == 0:
                end = time.time()
                sys.stdout.write('\r{} / {} : loss = {} epoch_run_time = {}'.format(
                    step, total_steps, np.mean(total_loss), end - start))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss)  # , accuracy/float(total_steps)


if __name__ == '__main__':
    try:
        from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
    except ImportError:
        LSTMCell = tf.nn.rnn_cell.LSTMCell
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        GRUCell = tf.nn.rnn_cell.GRUCell

    tf.reset_default_graph()
    with tf.Session() as session:
        model = HANClassifierModel(
            vocab_size=10,
            embedding_size=5,
            classes=2,
            word_cell=GRUCell(10),
            sentence_cell=GRUCell(10),
            word_output_size=10,
            sentence_output_size=10,
            max_grad_norm=5.0,
            dropout_keep_proba=0.5,
        )
        session.run(tf.global_variables_initializer())

        fd = {
            model.is_training: False,
            model.inputs: [[
                [5, 4, 1, 0],
                [3, 3, 6, 7],
                [6, 7, 0, 0]
            ],
                [
                    [2, 2, 1, 0],
                    [3, 3, 6, 7],
                    [0, 0, 0, 0]
                ]],
            model.word_lengths: [
                [3, 4, 2],
                [3, 4, 0],
            ],
            model.sentence_lengths: [3, 2],
            model.labels: [0, 1],
        }

        print(session.run(model.logits, fd))
        session.run(model.train_op, fd)
