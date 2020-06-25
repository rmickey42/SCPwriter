import tensorflow as tf
from tf import keras

EPOCHS = 100

def getData():

	return data

def loadModel():


def train(dataset):
	model = Sequential()
	model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])
	model.add(Dense(vocab_size, activation='softmax'))
	model.compile(loss = 'categorical_crossentropy'
	              optimizer = 'adam',
	              metrics = ['accuracy'])
	model.fit(X, y, epochs = EPOCHS, verbose = 2)