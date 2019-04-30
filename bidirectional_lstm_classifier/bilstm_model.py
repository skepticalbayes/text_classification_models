# -*- coding: utf-8 -*-
#TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import sys
import tensorflow as tf
import time
from tensorflow.contrib import rnn
import numpy as np

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 1000
    embed_size = 100
    hidden_size = 100
    max_epochs = 100
    early_stopping = 20
    class_weights = None
    dropout = 0.9
    lr = 0.001
    l2 = 0.0001
    num_classes = 4
    max_allowed_inputs = 130
    sequence_length = 28
    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    word2vec_init = False
    embedding_init = np.sqrt(3)

    num_train = .90

    floatX = np.float32
    mutiple_layers = True
    train_mode = True
    multilabel = True

class TextRNN:
    def __init__(self,num_classes, class_weights, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1),
                 multi_label_flag=False, multiple_layers=True):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.num_sampled = 20
        self.multiple_layers = multiple_layers
        self.multi_label_flag = multi_label_flag
        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0,trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if self.multi_label_flag:
            self.possibility = tf.nn.sigmoid(self.logits)
        else:
            self.possibility = tf.nn.softmax(self.logits)
        if not is_training:
            return
        self.loss_val = self.loss() #-->self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) #shape:[None,sentence_length,embed_size]
        #2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words,
                                                     dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>",
              outputs)  # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]

        # 3. Second LSTM layer
        rnn_cell = rnn.BasicLSTMCell(self.hidden_size * 2)
        if self.dropout_keep_prob is not None:
            rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
        _, final_state_c_h = tf.nn.dynamic_rnn(rnn_cell, output_rnn, dtype=tf.float32)
        final_state = final_state_c_h[1]

        # 4 .FC layer
        output = tf.layers.dense(final_state, self.hidden_size * 2, activation=tf.nn.tanh)

        # 5. logits(use linear layer)
        with tf.name_scope(
                "output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(output, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self, l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            if self.class_weights:
                weights = tf.gather(self.class_weights, self.input_y)
                #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
                #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
                losses = tf.losses.sparse_softmax_cross_entropy(labels=self.input_y, logits=self.logits, weights=weights);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            else:
                losses = tf.losses.sparse_softmax_cross_entropy(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss+l2_losses
        return loss


    def loss_nce(self,l2_lambda=0.0001): #0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training: #training
            #labels=tf.reshape(self.input_y,[-1])               #[batch_size,1]------>[batch_size,]
            labels=tf.expand_dims(self.input_y,1)                   #[batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),#[hidden_size*2, num_classes]--->[num_classes,hidden_size*2]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b_projection,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                               inputs=self.output_rnn_last,# [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                               num_sampled=self.num_sampled,  #scalar. 100
                               num_classes=self.num_classes,partition_strategy="div"))  #scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def run_epoch(self, config, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2,
                  train=False):
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(len(data[0]))
        ip, a = data
        ip, a = ip[p], a[p]

        for step in range(total_steps):
            start = time.time()
            index = range(step * config.batch_size, (step + 1) * config.batch_size)
            feed = {self.input_x: ip[index],
                    self.input_y: a[index],
                    self.dropout_keep_prob: dp
                    }
            loss, _ = session.run(
                [self.loss_val, train_op], feed_dict=feed)

            # if train_writer is not None:
            #     train_writer.add_summary(summary, num_epoch * total_steps + step)

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

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=10
    learning_rate=0.01
    batch_size=8
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRNN(num_classes, class_weights=None, learning_rate=learning_rate,
                    batch_size=batch_size, decay_steps=decay_steps, decay_rate=decay_rate,
                    sequence_length=sequence_length, vocab_size=vocab_size,
                    embed_size=embed_size, is_training=True,initializer=tf.random_normal_initializer(stddev=0.1),
                    multiple_layers=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,0,1,2,1,2]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
test()
