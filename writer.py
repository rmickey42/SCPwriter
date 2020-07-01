import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
import random
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

VOCAB_LEN = None
PATTERN_LEN = 50

EPOCHS = 20
BATCH_SIZE = 128

DATATEST = False

def prepData():
	text = ""
	for i in range(200):
		with open('scps/scp-2.txt', 'r', encoding='utf-8') as f:
			ft = f.read()
			if ft.strip():
				text = text+ft
	tokenized = list(set(text))
	tokenInt = dict((tok, i) for i, tok in enumerate(tokenized))
	tokFromInt = dict(zip(tokenInt.values(), tokenInt.keys()))
	VOCAB_LEN = len(tokenized)
	X = []
	y = []
	for i in range(len(text)-PATTERN_LEN):
		patt = text[i:i+PATTERN_LEN]
		outPatt = text[i+PATTERN_LEN]
		X.append([tokenInt[w] for w in patt])
		y.append(tokenInt[outPatt])
	X = np.reshape(X, (len(X), PATTERN_LEN, 1))
	X = X / float(len(tokenized))
	y = utils.to_categorical(y)

	if(DATATEST):
		print(X)
		print(y)
	return (X, y), (tokenInt, tokFromInt)

def trainModel(X, y):
	model = keras.Sequential()
	model.add(layers.LSTM(100, input_shape=(X.shape[1], X.shape[2])))
	model.add(layers.Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
	model.save('latest-model.h5')

if(len(sys.argv) <= 1):
	DATATEST = True
	print("DATA TEST MODE")
(X, y), (tokenInt, tokFromInt) = prepData()

if(len(sys.argv) > 1 and sys.argv[1] == '-t'): 
	trainModel(X, y)
#elif(len(sys.argv) > 1 and sys.argv[1] == '-g'): generation code