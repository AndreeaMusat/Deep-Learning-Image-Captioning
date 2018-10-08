from __future__ import print_function

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Bidirectional
from keras.layers import Masking, Flatten, Dropout, Merge, Activation
from keras.layers.wrappers import Wrapper, TimeDistributed
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from utils import exit

import tensorflow.contrib.keras as tck
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

EMBED_DIR = os.path.join('..', 'embeddings')

def get_embed_matrix(voc_size, embed_size, embed_type, tok2idx, idx2tok):

		embedding_matrix = np.random.random((voc_size + 1, embed_size))

		if embed_type == 'glove':
			if embed_size != 300:
				exit('For glove initialization, embed size should be 300', -1)

			with open(os.path.join(EMBED_DIR, 'glove.6B.300d.txt')) as file:
				for line in file:
					values = line.split()
					word = values[0]
					if word not in tok2idx: continue
					coefs = np.asarray(values[1:], dtype=np.float)
					idx = tok2idx[word] + 1
					embedding_matrix[idx] = coefs

		elif embed_type == 'universal_sentence_encoder':
			module_url = 'https://tfhub.dev/google/universal-sentence-encoder/1'
			embed = hub.Module(module_url)
			tf.logging.set_verbosity(tf.logging.ERROR)

			toks = []
			for i in range(voc_size):
				toks.append(idx2tok[i])

			with tf.Session() as sess:
				sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
				toks_embeddings = sess.run(embed(toks))

				for i, tok_embed in enumerate(np.array(toks_embeddings).tolist()):
					embedding_matrix[i + 1] = tok_embed
		else:
			return None

		print('Read embedding matrix ', embed_type)

		return embedding_matrix


class SoftmaxWithTemperature(Activation):

	def __init__(self, activation, **kwargs):
		super(SoftmaxWithTemperature, self).__init__(activation, **kwargs)
		self.__name__ = 'temperature_softmax'


# This function is adapted from here: 
# https://github.com/keras-team/keras/blob/master/keras/activations.py
def softmax_with_temperature(x, temp=1.0, axis=1):
	x = x / temp
	ndim = K.ndim(x)
	if ndim == 2:
		return K.softmax(x)
	elif ndim > 2:
		e = K.exp(x - K.max(x, axis=axis, keepdims=True))
		s = K.sum(e, axis=axis, keepdims=True)
		return e / s
	else:
		raise ValueError('Cannot apply softmax to a tensor that is 1D. '
	                     'Received input: %s' % x)


def pre_inject_model(model_name, batch_size, learning_rate, maxlen, voc_size, embed_size,
			cnn_feats_size, dropout_rate, embed_matrix=None):
	# create input layer for the cnn features
	cnn_feats_input = Input(shape=(cnn_feats_size,))

	# normalize CNN features 
	normalized_cnn_feats = BatchNormalization(axis=-1)(cnn_feats_input)

	# embed CNN features to have same dimension with word embeddings
	embedded_cnn_feats = Dense(units=embed_size,
				   activation='relu')(normalized_cnn_feats)

	# add time dimension so that this layer output shape is (None, 1, embed_size)
	final_cnn_feats = RepeatVector(1)(embedded_cnn_feats)

	# mask the cnn feats so that it matches the word embeddings mask when they are concatenated
	masked_cnn_feats = Masking()(final_cnn_feats)
	
	# create input layer for the captions (each caption has max maxlen words)
	caption_input = Input(shape=(maxlen,))

	# mask the input: returns tensor of shape: (batch_size, maxlen, 1)
	# masked_caption = Masking(mask_value=-1, input_shape=(maxlen,))(caption_input)

	# embed the captions (voc_size + 1 because we have 0 where the value shouldnt be used)
	if embed_matrix is None:
		embedded_caption = Embedding(input_dim=voc_size + 1,
					     output_dim=embed_size,
					     input_length=maxlen, 
					     mask_zero=True)(caption_input)
	else:
		embedded_caption = Embedding(input_dim=voc_size + 1,
					     output_dim=embed_size,
		        		     input_length=maxlen,
					     weights=[embed_matrix],
					     trainable=True,
					     mask_zero=True)(caption_input)

	# concatenate CNN features and the captions.
	# Ouput shape should be (None, maxlen + 1, embed_size)
	img_caption_concat = concatenate([masked_cnn_feats, embedded_caption], 
								   axis=1)
	
	# now feed the concatenation into a LSTM layer (many-to-many)
	lstm_layer = LSTM(units=embed_size,
			  input_shape=(maxlen + 1, embed_size),
			  return_sequences=True,
			  dropout=dropout_rate, 
			  recurrent_dropout=0.15)(img_caption_concat)

	# # create a custom object for our 'temperature_softmax' activation
	temperature = 1.1    # TODO: make this a parameter
	temp_softmax = lambda x : softmax_with_temperature(x, temperature)
	get_custom_objects().update({'temperature_softmax': SoftmaxWithTemperature(temp_softmax)})

	# create a fully connected layer to make the predictions
	pred_layer = TimeDistributed(Dense(units=voc_size,
					   activation='temperature_softmax'))(lstm_layer)

	# build the model with CNN features and captions as input and 
	# predictions output
	model = Model(inputs=[cnn_feats_input, caption_input], 
		      outputs=pred_layer)

	model.compile(loss='categorical_crossentropy',
		      optimizer='adam', 
		      metrics=['accuracy'])
	model.summary()
	
	return model


def get_model(model_name, batch_size, learning_rate, maxlen, voc_size, embed_size,
			cnn_feats_size, dropout_rate, embed_matrix=None):

	if model_name == 'pre-inject':
		model = pre_inject_model(model_name, batch_size, learning_rate, maxlen, voc_size, embed_size, 
					cnn_feats_size, dropout_rate, embed_matrix)

	return model
