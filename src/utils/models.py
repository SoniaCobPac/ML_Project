# data manipulation
import numpy as np
import pandas as pd 
from sqlalchemy import create_engine

# route files
import os
import sys

# ml model
import pickle

import tensorflow as tf
from tensorflow import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization 
from keras.layers import Bidirectional


sep = os.sep

def get_route(steps):
    """
    This function appends the route of the file to the sys path
    to be able to import files from/to other foders.

    Param: Steps (int) to go up to the required folder
    """
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route 

get_route(2)
import utils.mining_data_tb as md


def lstm_model (num_units, num_dense, input_shape):
    """
    Builds and compiles a simple RNN model
    Param:
              num_units: Number of units of a the simple RNN layer
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              input_shape: input_shape
    """
    model = Sequential()
    model.add(LSTM(num_units, input_shape=input_shape))
    model.add(Dense(num_dense))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc", "RootMeanSquaredError"])

    return model


def bi_lstm_model(num_units, num_dense, input_shape):
    """
    Builds and compiles a Bidirectional LSTM model
    Param:
              num_units: Number of units of a the LSTM layer
              num_dense: Number of neurons in the dense layer followed by the RNN layer
              input_shape: input_shape
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(num_units, input_shape=input_shape, return_sequences=True)))
    model.add(Dense(num_dense))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(num_units)))
    model.add(Dense(num_dense))
    model.add(Dropout(0.3))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc", "RootMeanSquaredError"])

    return model

# LSTM Generator
def generator_model(num_units= x.shape[0], n_inputs=(x.shape[1], x.shape[2])):
    """
    Model to generate new samples.
    """
    model = Sequential()
    model.add(LSTM(num_units, input_shape=n_inputs, return_sequences=True))
    model.add(Dense(n_inputs[1]))
    model.add(Activation("softmax"))

    return model
    
# LSTM Discriminator
def discriminator_model(num_units= x.shape[0], n_inputs=(x.shape[1], x.shape[2])):
    """
    Model to classify samples as real (from the domain) or fake (generated).
    """
    model = Sequential()
    model.add(LSTM(num_units, input_shape=n_inputs))
    model.add(Dense(n_inputs[1]))    
    model.add(Dense(1, activation="sigmoid"))
    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def gan_model(g_model, d_model):
    """
    
    """
	# make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    model.add(BatchNormalization())
    # add the discriminator
    model.add(d_model)
    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


# train the generator and discriminator
def train(x, g_model, d_model, gan_model, y_gan, n_epochs, n_batch):
	batch_per_epoch = int(x.shape[0]/n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(batch_per_epoch):
			# get randomly selected 'real' samples
			x_real, y_real = md.generate_real_samples(x, n_samples=half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(x_real, y_real)
			# generate 'fake' examples 
			x_fake, y_fake = md.generate_fake_data(x, g_model, n_samples=half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)
			# prepare points in latent space as input for the generator
			x_latent = md.generate_latent_points(x, n_samples=n_batch)
			# create inverted labels for the fake samples
			y_fake_1 = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(x_latent, y_fake_1)
			# summarize loss on this batch
			print('>%d, %d/%d, loss_real=%.3f, loss_fake=%.3f loss_latent=%.3f' %
				(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))
		# evaluate the model performance, sometimes
		if (n_epochs+1) % 10 == 0:
			check_performance(n_epochs, g_model, d_model, x, n_batch)


# evaluate the discriminator, plot generated images, save generator model
def check_performance(epoch, g_model, d_model, x, n_samples=100):
	# prepare real samples
	x_real, y_real = md.generate_real_samples(x, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = md.generate_fake_data(x, g_model, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	return('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))


