# -*- coding: utf-8 -*-
"""
### Mengimpor library yang dibutuhkan
"""

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

"""### Mengubah dataset menjadi dataframe dan membuang data yang tidak relevan
Dataset diunduh melalui Kaggle dengan link sebagai berikut:
[link text](https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017)
"""

df = pd.read_csv('seattleprec.csv')
df = df.drop(columns='tempmax')
df = df.drop(columns='tempmin')
df = df.drop(columns='rain')
df = df.dropna()
df.info()

df.head()

df.isnull().sum()

"""### Menerapkan Normalisasi pada dataset dengan StandardScaler()
### dan melakukan pemeriksaan ambang batas MAE
"""

standard_scaler = StandardScaler()
df['precipitation'] = standard_scaler.fit_transform(df[['precipitation']].values).flatten()

threshold_mae = (df['precipitation'].max() - df['precipitation'].min()) * 10/100
print(threshold_mae)

"""### Membuat plot dari data"""

dates = df['date'].values
level = df['precipitation'].values

plt.figure(figsize=(15,5))
plt.plot(dates, level)
plt.title('Seattle Precipitation 1948-2017',
          fontsize=20);

"""### Membagi data training dan data testing"""

dates_latih, dates_test, level_latih, level_test = train_test_split(dates, level, test_size=0.2)

"""### Mengubah data menjadi format yang dapat diterima oleh model"""

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda w: (w[:-1], w[-1:]))
  return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(level_latih, window_size=60, batch_size=100, shuffle_buffer=1000)
val_set = windowed_dataset(level_test, window_size=60, batch_size=100, shuffle_buffer=5000)

"""### Arsitektur menggunakan 2 buah layer LSTM

"""

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
])

"""### Menggunakan Optimizers"""

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])

"""### Menggunakan fungsi Callback agar pelatihan berhenti ketika MAE terlalu tinggi.

"""

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae')>threshold_mae):
      print('\nMAE terlalu tinggi. Menghentikan pelatihan')
      self.model.stop_training = True
callbacks = myCallback()

"""### Melatih model dengan fungsi fit()"""

history = model.fit(train_set, validation_data=val_set, callbacks=[callbacks], epochs=10)

"""### Melakukan plotting pada hasil pelatihan"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='center right')
plt.show()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='center right')
plt.show()
