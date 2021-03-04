# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Embedding,\
    Conv1D, MaxPooling1D, GlobalMaxPool1D, Conv2D, Flatten, GlobalMaxPool2D, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model, load_model
import os
import nltk
from gensim.models import Word2Vec, KeyedVectors
import re
# nltk.download('punkt')
import tensorflow.keras.callbacks as ES
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.utils import to_categorical
import os, sys

# style conifg
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')

# import Trump tweets (dataset from Kaggle.com, scrapped from twitter)
df = pd.read_csv('realdonaldtrump.csv', sep=',')
# shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# first look at the data
df.head()
df = df[['content', 'id']]

# remove at least some of the links/unwanted strings
df['content'] = df['content'].apply(lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'pic.twitter', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'.com', '', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'@', '', x))

text = ''

for i in df['content']:
        for j in i.split():
            text += ' ' + j.lower()

chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)

length = 30
X_lst = []
Y = []

for i in range(0, n_chars - length, 1):
    x = text[i:i + length]
    y = text[i+length]
    X_lst.append([char_to_int[char] for char in x])
    Y.append(char_to_int[y])

print(len(X_lst))
X = np.reshape(X_lst, (len(X_lst), length, 1))

# normalize data
X = X / float(n_vocab)
#Y = to_categorical(Y)
Y = np.array(Y)

#######################################

X_train = X[:int(len(X)/4*3)]
Y_train = Y[:int(len(X)/4*3)]
X_test = X[int(len(X)/4*3):]
Y_test = Y[int(len(X)/4*3):]

i = Input(shape=(length, 1,))
x = LSTM(256, return_sequences=True)(i)
x = Dropout(0.3)(x)
x = LSTM(256)(x)
x = Dropout(0.3)(x)
x = Dense(n_vocab, activation='softmax')(x)

cb = ES.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ES.ModelCheckpoint(filepath=os.getcwd(), monitor='val_loss', verbose=1, save_best_only=True,
                                save_freq='epoch', save_weights_only=True)

model1 = Model(i, x)
model1.load_weights(filepath=os.getcwd())

model1.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy')
r1 = model1.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test), batch_size=64,
                callbacks=[cb, checkpoint])

plt.plot(r1.history['loss'], label='loss')
plt.plot(r1.history['val_loss'], label=['val_loss'])
plt.legend()

###########################

start = np.random.randint(0, len(X_lst)-1)
seed_input_text = X_lst[start]
seed_input = X[start].reshape(1, length, 1)
generated_text_idx = []


for i in range(200):
    yhat_prob = model1.predict(seed_input)
    yhat = np.argmax(yhat_prob)
    seed_input = np.roll(seed_input, -1)
    seed_input[:, -1] = yhat / n_vocab
    generated_text_idx.append(yhat)

generated_text = [int_to_char[word] for word in generated_text_idx]
print(''.join(generated_text))

generated_text_seed = [int_to_char[word] for word in seed_input_text]
print(''.join(generated_text_seed))





