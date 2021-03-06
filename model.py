"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For 'CNN-non-static' gets to 82.1% after 61 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.7, 0.8)
hidden_dims = 100

For 'CNN-rand' gets to 78-79% after 7-8 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

For 'CNN-static' gets to 75.4% after 7 epochs with following settings:
embedding_dim = 100
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

* it turns out that such a small data set as "Movie reviews with one
sentence per review"  (Pang and Lee, 2005) requires much smaller network
than the one introduced in the original article:
- embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
- 2 filter sizes (instead of 3)
- higher dropout probabilities and
- 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
- embedding initialization does not require prebuilt Google Word2Vec data.
Training Word2Vec on the same "Movie reviews" data set is enough to
achieve performance reported in the article (81.6%)

** Another distinct difference is slidind MaxPooling window of length=2
instead of MaxPooling over whole feature map as in the article
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
# from keras.utils import plot_model

np.random.seed(2)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for
# Sentence Classification, Section 3 for detail.

parser = argparse.ArgumentParser(
    description="CNN for Sentence Classification.")
parser.add_argument('model_variation', metavar='Model-Variation', nargs=1,
                    help='The model you want to use.',
                    choices=['CNN-non-static', 'CNN-static', 'CNN-rand'])
parser.add_argument('datsets', metavar='Datasets', nargs=1,
                    help='The datasets you want to use.',
                    choices=['mr', 'sst'])
parser.add_argument('-e', dest='embedding_dim', metavar='Embedding-Dim',
                    help='Size of embedding.',
                    default=300)
parser.add_argument('-f', dest='num_filters', metavar='Num-Filters',
                    help='Number of filters.',
                    default=150)
parser.add_argument('-d', dest='hidden_dims', metavar='Hidden-Dims',
                    help='Size of Hidden.',
                    default=150)

args = parser.parse_args()
model_variation = args.model_variation[0]
print(('Model variation is %s' % model_variation))

now_date = datetime.now().strftime("%Y%m%d%H%M%S")

# Model Hyperparameters
loss_function = 'binary_crossentropy'
act_2 = 'sigmoid'
if args.datsets[0] == 'mr':
    sequence_length = 56
    label_num = 1
else:
    sequence_length = 53
    label_num = 5
embedding_dim = int(args.embedding_dim)
filter_sizes = (3, 4)
num_filters = int(args.num_filters)
dropout_prob = (0.25, 0.5)
hidden_dims = int(args.hidden_dims)

# Training parameters
batch_size = 50
num_epochs = 100
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count
context = 10        # Context window size

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data(args.datsets[0])

if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    embedding_weights = train_word2vec(
        x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation == 'CNN-static':
        x = embedding_weights[0][x]
elif model_variation == 'CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

# Shuffle data
if args.datsets[0] == 'mr':
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
else:
    split = pd.read_csv('data/sst-split.txt', header=0)
    tr_index = np.array(split['splitset_label'] == 1, dtype=bool)
    vl_index = np.array(split['splitset_label'] == 2, dtype=bool)
    x_tr = x[tr_index]
    y_tr = y[tr_index]
    x_vl = x[vl_index]
    y_vl = y[vl_index]
    y_tr = to_categorical(y_tr)
    y_vl = to_categorical(y_vl)

print(("Vocabulary Size: {:d}".format(len(vocabulary))))

# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
graph_in = Input(shape=(sequence_length, embedding_dim))
convs = []
for fsz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=fsz,
                  padding='valid',
                  activation='relu',
                  strides=1)(graph_in)
    pool = MaxPooling1D(pool_size=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = concatenate(convs)
else:
    out = convs[0]

graph = Model(inputs=graph_in, outputs=out)
# plot_model(graph, to_file='result/Result/' +
#      model_variation + '-' + now_date + '-CNN.png')

# main sequential model
model = Sequential()
if not model_variation == 'CNN-static':
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(
    sequence_length, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(label_num))
model.add(Activation(act_2))
model.compile(loss=loss_function,
              optimizer='rmsprop', metrics=['accuracy'])
# plot_model(model, to_file='result/Result/' + model_variation + '-' + now_date + '.png')

# Training model
# ==================================================
if args.datsets[0] == 'mr':
    model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
              epochs=num_epochs, validation_split=val_split, verbose=2)
else:
    model.fit(x_tr, y_tr, batch_size=batch_size,
              nb_epoch=num_epochs, validation_data=(x_vl, y_vl), verbose=2)
