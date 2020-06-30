import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.probability import FreqDist
import random
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

num_samples = 0

def prepData():

	for i in range(1, 1000):
		with open('scps/scp-{}.txt'.format(i), 'r') as f:
			text = f.read()
			if text.strip():
				tokenized = word_tokenize(text)
				freq = Counter(tokenized)
				num_samples = num_samples+1


def trainModel(x, y):
	model = keras.Sequential()
	model.add(LSTM(3, input_shape=(num_samples)))

prepData()

