# -*- coding: utf-8 -*-
"""lab5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mybKXIB8KIBdubN02m9rU9Io1C69K6od
"""

import numpy as np
import tensorflow as tf

# np.random.seed(42)

def generate_regression_data(n=30):
  # Generate regression dataset
  X = np.linspace(-5, 5, n).reshape(-1, 1)
  y = np.sin(2 * X) + np.cos(X) + 5
  # simulate noise
  data_noise = np.random.normal(0, 0.05, n).reshape(-1, 1)
  # Generate training data
  Y = y + data_noise
  return X.reshape(-1, 1), Y.reshape(-1, 1)

# X, Y = generate_regression_data()

def generate_classification_data(n=30):
  # Class 1 - samples generation
  X1_1 = 1 + 4 * np.random.rand(n, 1)
  X1_2 = 1 + 4 * np.random.rand(n, 1)
  class1 = np.concatenate((X1_1, X1_2), axis=1)
  Y1 = np.ones(n)
  # Class 0 - samples generation
  X0_1 = 3 + 4 * np.random.rand(n, 1)
  X0_2 = 3 + 4 * np.random.rand(n, 1)
  class0 = np.concatenate((X0_1, X0_2), axis=1)
  Y0 = np.zeros(n)
  X = np.concatenate((class1, class0))
  Y = np.concatenate((Y1, Y0))
  idx0 = [i for i, v in enumerate(Y) if v == 0]
  idx1 = [i for i, v in enumerate(Y) if v == 1]
  return X, Y, idx0, idx1

# X, Y, idx0, idx1 = generate_classification_data()

# Zadanie 1 A
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X, Y = generate_regression_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu z użyciem SGD jako optymalizatora
model1.compile(optimizer='sgd', loss='mean_squared_error')

# Trenowanie modelu
history = model1.fit(X_train, Y_train, epochs=500)

# Przewidywanie na całym zbiorze treningowym
Y_pred = model1.predict(X_train)

print(f'last loss: {history.history["loss"][-1]}')

# ZADANIE 1B
X, Y = generate_regression_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu z użyciem SGD jako optymalizatora
model2.compile(optimizer='sgd', loss='mean_squared_error')

# Trenowanie modelu
history = model2.fit(X_train, Y_train, epochs=500)

# Przewidywanie na całym zbiorze treningowym
y_pred = model2.predict(X_train)
final_loss = history.history['loss'][-1]

print(f'Final loss: {final_loss}')

# ZADANIE 1C
X, Y = generate_regression_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu z użyciem SGD jako optymalizatora
model3.compile(optimizer='sgd', loss='mean_squared_error')

# Trenowanie modelu
history = model3.fit(X_train, Y_train, epochs=500)

# Przewidywanie na całym zbiorze treningowym
y_pred = model3.predict(X_train)
final_loss = history.history['loss'][-1]

print(f'Final loss: {final_loss}')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, Y = generate_regression_data()

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Tworzenie modelu
model3 = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)  # Warstwa wyjściowa dla problemu regresji
])

# Kompilacja modelu z użyciem SGD jako optymalizatora
model3.compile(optimizer='sgd', loss='mean_squared_error')

# Trenowanie modelu
history = model3.fit(X_train, Y_train, epochs=1000)

# Przewidywanie na całym zbiorze treningowym
y_pred = model3.predict(X_train)
final_loss = history.history['loss'][-1]

d.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
print(f'Final loss: {final_loss}')

#zad 2
import pandas as pd


X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=500)

y_pred = model.predict(X_train)
final_loss = history.history['loss'][-1]
final_acc = history.history['accuracy'][-1]

print(f'Final loss: {final_loss}')
print(f'Final accuracy: {final_acc}')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=500, validation_data=(X_test, Y_test))

y_pred = model.predict(X_train)
final_loss = history.history['loss'][-1]
final_acc = history.history['accuracy'][-1]
val_loss = history.history['val_loss'][-1]

print(f'Final loss: {final_loss}')
print(f'Final accuracy: {final_acc}')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

#zad3
X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model1.fit(X_train, Y_train, epochs=1000, validation_data=(X_test, Y_test))

y_pred = model1.predict(X_train)
final_loss = history.history['loss'][-1]
final_acc = history.history['accuracy'][-1]
val_loss = history.history['val_loss'][-1]
print(f'Final loss: {final_loss}')
print(f'Final accuracy: {final_acc}')
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
history = model2.fit(X_train, Y_train, epochs=2000,  validation_data=(X_test, Y_test))

y_pred = model2.predict(X_train)
final_loss = history.history['loss'][-1]
final_acc = history.history['accuracy'][-1]
val_loss = history.history['val_loss'][-1]
print(f'Final loss: {final_loss}')
print(f'Final accuracy: {final_acc}')

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model3.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model3.fit(X_train, Y_train, epochs=1000, validation_data=(X_val, Y_val))

test_loss, test_acc = model1.evaluate(X_test, Y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

# Wyświetlenie wykresu
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

#zad 4
from tensorflow.keras.optimizers import Adam


X, Y, idx0, idx1 = generate_classification_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rates = [0.01, 0.1, 1]

for lr in learning_rates:
    # Kompilacja modelu
    model3.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['accuracy'])

    # Ustawienie funkcji wczesnego zatrzymania
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=50)

    # Trenowanie modelu
    history = model3.fit(X_train, Y_train, epochs=1000, validation_data=(X_test, Y_test), callbacks=[earlystopper])

    # Wyświetlenie wyników dla różnych szybkości uczenia
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b', label=f'Training loss (lr={lr})')
    plt.plot(epochs, val_loss, 'r', label=f'Validation loss (lr={lr})')
    plt.title(f'Training and Validation Loss (lr={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
