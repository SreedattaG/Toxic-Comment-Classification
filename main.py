import os
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dropout, Dense #type: ignore
from tensorflow.keras.models import Sequential, load_model #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy #type: ignore
from tensorflow.keras.initializers import Constant #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv(os.path.join('train.csv'))
toxicity_labels = df.columns[2:]
df['binary_labels'] = df[toxicity_labels].apply(lambda x: 1 if x.any() else 0, axis=1)
y = df['binary_labels'].values
X = df['comment_text']

MAX_FEATURES = 200000
MAX_SEQUENCE_LENGTH = 1800
EMBEDDING_DIM = 100

vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache().shuffle(200).batch(16).prefetch(8)

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

embedding_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

vocab_size = len(vectorizer.get_vocabulary()) + 1
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in enumerate(vectorizer.get_vocabulary()):
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model_path = 'toxicity.h5'

if os.path.exists(model_path):
    print("Using existing model/version of 'toxicity.h5'")
    model = load_model(model_path, custom_objects={'TextVectorization': vectorizer})
    model.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics=['accuracy'])
else:
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), trainable=False),
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
        Dropout(0.5),
        Bidirectional(LSTM(32, activation='tanh')),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)

    history = model.fit(train, epochs=1, validation_data=val, callbacks=[early_stopping, reduce_lr])
    model.save(model_path)

y_true = []
y_pred = []

for x_batch, y_batch in test:
    y_true.extend(y_batch.numpy()) 
    y_pred.extend((model.predict(x_batch) > 0.5).astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-toxic', 'Toxic'], yticklabels=['Non-toxic', 'Toxic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Toxic Comment Classification')
plt.show()
