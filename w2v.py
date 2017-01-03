from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import datetime


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    embedding_model = word2vec.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                   else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                                   for w in vocabulary_inv])]
    return embedding_weights

if __name__ == '__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)
