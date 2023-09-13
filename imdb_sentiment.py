import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import nltk

nltk.download("punkt")

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")
sns.set()

file_path = "~/Desktop/IMDB Dataset.csv"

imdb = pd.read_csv(file_path)

imdb = imdb.sample(frac=1).reset_index(drop=True)

num_words = 10000
max_sequence_length = 128

tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(imdb["review"])

X = tokenizer.texts_to_sequences(imdb["review"])
X = pad_sequences(X, maxlen=max_sequence_length, padding="post", truncating="post")

y = imdb["sentiment"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

train_size = int(0.8 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

model = Sequential()
model.add(
    Embedding(input_dim=num_words, output_dim=128, input_length=max_sequence_length)
)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(
    X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test)
)


plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
