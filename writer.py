import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
import random
import sys
from nltk.corpus import stopwords
from collections import Counter
import numpy as np

#how many words in a predictable pattern
PATTERN_LEN = 3

EPOCHS = 20
BATCH_SIZE = 32

#whether to only load data or not
DATATEST = False

TRAIN_SIZE = 150

def prepData():
	#adds all non-empty documents from 1-TRAIN_SIZE to a string in order
	text = ""
	for i in range(1, TRAIN_SIZE+1):
		with open('scps/scp-{}.txt'.format(i), 'r', encoding='utf-8') as f:
			ft = f.read()
			if ft.strip():
				text = text+ft

	#grabs all words in a list
	words = text.replace('\n', ' ').split(' ')

	#tokenize list of words (only one of each)
	tokenized = set(words)

	#dictionaries to convert from each token to its id and each id to its token
	tokenInt = dict((tok, i) for i, tok in enumerate(tokenized))
	tokFromInt = dict(zip(tokenInt.values(), tokenInt.keys()))
	VOCAB_LEN = len(tokenized)

	#unformatted x and y
	unfX = []
	unfY = []

	#adds pattern to training input and what comes after the pattern to training output
	for i in range(len(words)-PATTERN_LEN):
		patt = words[i:i+PATTERN_LEN]
		outPatt = words[i+PATTERN_LEN]
		unfX.append([tokenInt[w] for w in patt])
		unfY.append(tokenInt[outPatt])

	#reshapes for neural net input, normalizes, one-hot encodes output
	X = np.reshape(unfX, (len(unfX), PATTERN_LEN, 1))
	X = X / float(len(tokenized))
	y = utils.to_categorical(unfY)

	#prints prepped data if datatest enabled
	if(DATATEST):
		print(X)
		print(y)
	return (X, y), (tokenInt, tokFromInt), (unfX, unfY), VOCAB_LEN

def trainModel(X, y):
	model = keras.Sequential()
	model.add(layers.LSTM(128, input_shape=(X.shape[1], X.shape[2]), dropout=0.1))
	model.add(layers.Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
	model.save('latest-model.h5')


def generateText(unfX, VOCAB_LEN, tokFromInt, numWords=100):
	model = models.load_model('latest-model.h5')
	seed = random.randrange(0, len(unfX)-1)
	pattern = unfX[seed]
	text = ""
	#predicts what will come after pattern, adds that to text, shifts pattern up to fit the predicted word
	for i in range(numWords):
		#shapes pattern into input shape for model
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(VOCAB_LEN)

		#predict next word and converts it to string
		prediction = model.predict(x, verbose=0)
		wordid = np.argmax(prediction)
		result = tokFromInt[wordid]

		#add word to existing text
		text = text + result + " "

		#add next word and shift pattern so first word is not in anymore
		pattern.append(wordid)
		pattern = pattern[1:len(pattern)]
	return text

#if no arguments, enable datatest
if(len(sys.argv) <= 1):
	DATATEST = True
	print("DATA TEST MODE")
(X, y), (tokenInt, tokFromInt), (unfX, unfY), VOCAB_LEN = prepData()

#if argument -t, train model, if argument -g, generate text, second argument is number of words
if(len(sys.argv) > 1 and sys.argv[1] == '-t'): 
	trainModel(X, y)
elif(len(sys.argv) > 1 and sys.argv[1] == '-g'):
	words = 100
	if(len(sys.argv) > 2 and int(sys.argv[2])):
		words = int(sys.argv[2])
	text = generateText(unfX, VOCAB_LEN, tokFromInt, words)
	print(text)