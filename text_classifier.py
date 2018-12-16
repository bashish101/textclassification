from keras.layers import Input
from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dense, Permute, GRU, Conv1D, Concatenate, Add, Activation

import tensorflow as tf
import keras.backend as K

class Classifier(object):
	def __init__(self,
		     max_feature_value,
		     max_sequence_len,
		     num_classes,
		     embedding_dim = 64):
		self.embedding_dim = embedding_dim
		self.max_feature_value = max_feature_value
		self.max_sequence_len = max_sequence_len
		self.num_classes = num_classes
		self.input_shape = (self.max_sequence_len,)

		self.weights = {0 : 0.025, 1 : 0.975}

	def __call__(self):
		input_tensor = Input(shape = self.input_shape)
		return self.build_model(input_tensor)

	def build_model(self, x):
		
		input_tensor = x

		x = Embedding(self.max_feature_value,
			      self.embedding_dim, 
			      input_length = self.max_sequence_len)(x)

		x = Conv1D(64,
			   3,
			   padding = 'valid',
			   strides = 1)(x)
		x = Conv1D(64,
			   3,
			   padding = 'valid',
			   dilation_rate = 2)(x)

		x = GlobalMaxPooling1D()(x)

		x = Dense(self.num_classes, activation = 'sigmoid')(x)

		output_tensor = x
		model = Model([input_tensor], [output_tensor])

		model.compile(loss = self.focal_loss,
			      optimizer = 'adam',
			      metrics = ['accuracy'])
		return model

	def focal_loss(self, y_true, y_pred, alpha = 0.75, gamma = 2):
		alpha_factor = K.ones_like(y_true) * alpha
		alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
		focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
		focal_weight = alpha_factor * focal_weight ** gamma
		loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
		return loss
