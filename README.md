# Convolutional Neural Networks for Sentence Classification

Train convolutional network for sentiment analysis. Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim, [link](http://arxiv.org/pdf/1408.5882v2.pdf). Inspired by Denny Britz article "Implementing a CNN for Text Classification in TensorFlow", [link](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).

Forking from [alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)
* added SST datasets
* added Google News Word2Vec
* added parameters as arguments

## Some thoughts

It turns out that such a small data set as "Movie reviews with one sentence per review" (Pang and Lee, 2005) requires much smaller network than the one introduced in the original article:
* embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100 dimensions as it has much fewer trainable weights)
* 2 filter sizes (instead of 3)
* higher dropout probabilities and
* 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
* embedding initialization does require prebuilt Google Word2Vec data.

Another distinct difference is slidind MaxPooling window of length=2 instead of MaxPooling over whole feature map as in the article

## Dependencies

* The [Keras](http://keras.io/) Deep Learning library and most recent [Theano](http://deeplearning.net/software/theano/install.html#install) backend should be installed. You can use pip for that.
Not tested with TensorFlow, but should work.
