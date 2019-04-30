Various models implemented as per their research papers which gave state of the art results for various natural language tasks like classification.

Text Classification
-------------------------------------------------------------------------
the purpose of this repository is to explore text classification methods in NLP with deep learning.

UPDATE: 

it has all kinds of baseline models for text classificaiton.

it also support for multi-label classification where multi label associate with an sentence or document.

although many of these models are simple, and may not get you to top level of the task.but some of these models are very classic, so they may be good to serve as baseline models.

each model has a test function under model class. you can run it to performance toy task first. the model is indenpendent from dataset.

Models:
-------------------------------------------------------------------------

1) TextCNN   
2) TextRNN    
3) Hierarchical Attention Network    
Performance
-------------------------------------------------------------------------

(mulit-label label prediction task,ask to prediction top5, 3 million training data,full score:0.5)

Model   |TextCNN|TextRNN| HierAtteNet
---     | ---   | ---   |---
Score   | 0.405 | 0.358 |0.398
Training|  2h   | 10h   | 2h  
--------------------------------------------------------------------------------------------------

 
`HierAtteNet` means Hierarchical Attention Networkk;

Usage:
-------------------------------------------------------------------------------------------------------
1) model is in `xxx_model.py`
2) run python `xxx_train.py` to train the model
3) run python `xxx_predict.py` to do inference(test).

-------------------------------------------------------------------------

Environment:
-------------------------------------------------------------------------------------------------------
python 2.7+ tensorflow 1.1

(tensorflow 1.2,1.3,1.4 also works; most of models should also work fine in other tensorflow version, since we use very few features bond to certain version; if you use python 3.5, it will be fine as long as you change print/try catch function)

-------------------------------------------------------------------------

Models Detail:
-------------------------------------------------------------------------

1.TextCNN:
-------------
Implementation of <a href="http://www.aclweb.org/anthology/D14-1181"> Convolutional Neural Networks for Sentence Classification </a>

Structure:embedding--->conv--->max pooling--->fully connected layer-------->softmax

Check: cnn_model.py

In order to get very good result with TextCNN, you also need to read carefully about this paper <a href="https://arxiv.org/abs/1510.03820">A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification</a>: it give you some insights of things that can affect performance. although you need to  change some settings according to your specific task.

Convolutional Neural Network is main building box for solve problems of computer vision. Now we will show how CNN can be used for NLP, in in particular, text classification. Sentence length will be different from one to another. So we will use pad to get fixed length, n. For each token in the sentence, we will use word embedding to get a fixed dimension vector, d. So our input is a 2-dimension matrix:(n,d). This is similar with image for CNN. 

Firstly, we will do convolutional operation to our input. It is a element-wise multiply between filter and part of input. We use k number of filters, each filter size is a 2-dimension matrix (f,d). Now the output will be k number of lists. Each list has a length of n-f+1. each element is a scalar. Notice that the second dimension will be always the dimension of word embedding. We are using different size of filters to get rich features from text inputs. And this is something similar with n-gram features. 

Secondly, we will do max pooling for the output of convolutional operation. For k number of lists, we will get k number of scalars. 

Thirdly, we will concatenate scalars to form final features. It is a fixed-size vector. And it is independent from the size of filters we use.

Finally, we will use linear layer to project these features to per-defined labels.

![alt text](https://github.com/brightmart/text_classification/blob/master/images/TextCNN.JPG)

-------------------------------------------------------------------------


2.TextRNN
-------------
Structure v1:embedding--->bi-directional lstm--->concat output--->average----->softmax layer

check: bilstm_model.py

![alt text](https://github.com/brightmart/text_classification/blob/master/images/bi-directionalRNN.JPG)

-------------------------------------------------------------------------

3.Hierarchical Attention Network:
-------------
Implementation of <a href="https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf">Hierarchical Attention Networks for Document Classification</a>

Structure:

1) embedding 

2) Word Encoder: word level bi-directional GRU to get rich representation of words

3) Word Attention:word level attention to get important information in a sentence

4) Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

5) Sentence Attetion: sentence level attention to get important sentence among sentences

5) FC+Softmax

![alt text](https://github.com/brightmart/text_classification/blob/master/images/HAN.JPG)

In NLP, text classification can be done for single sentence, but it can also be used for multiple sentences. we may call it document classification. Words are form to sentence. And sentence are form to document. In this circumstance, there may exists a intrinsic structure. So how can we model this kinds of task? Does all parts of document are equally relevant? And how we determine which part are more important than another?

It has two unique features: 

1)it has a hierarchical structure that reflect the hierarchical structure of documents; 

2)it has two levels of attention mechanisms used at the word and sentence-level. it enable the model to capture important information in different levels.

Word Encoder:
For each words in a sentence, it is embedded into word vector in distribution vector space. It use a bidirectional GRU to encode the sentence. By concatenate vector from two direction, it now can form a representation of the sentence, which also capture contextual information.

Word Attention:
Same words are more important than another for the sentence. So attention mechanism is used. It first use one layer MLP to get uit hidden representation of the sentence, then measure the importance of the word as the similarity of uit with a word level context vector uw and get a normalized importance through a softmax function. 

Sentence Encoder: 
for sentence vectors, bidirectional GRU is used to encode it. Similarly to word encoder.

Sentence Attention: 
sentence level vector is used to measure importance among sentences. Similarly to word attention.

Input of data: 

Generally speaking, input of this model should have serveral sentences instead of sinle sentence. shape is:[None,sentence_lenght]. where None means the batch_size.

In my training data, for each example, i have four parts. each part has same length. i concat four parts to form one single sentence. the model will split the sentence into four parts, to form a tensor with shape:[None,num_sentence,sentence_length]. where num_sentence is number of sentences(equal to 4, in my setting).

check:p1_HierarchicalAttention_model.py

for attentive attention you can check <a href='https://github.com/brightmart/text_classification/issues/55'>attentive attention</a>
