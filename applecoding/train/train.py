import tensorflow as tf
import numpy as np
import pandas as pd

path = './'
fileName = 'weight_storage.npz'

data = pd.read_csv('../model/gpascore.csv')
data = data.dropna()  # 빈값이 있는걸 날려줌

yData = data['admit'].values

xData = []
for i, rows in data.iterrows():
    xData.append([rows['gre'], rows['gpa'], rows['rank']])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='tanh'))
model.add(tf.keras.layers.Dense(128, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(xData), np.array(yData), epochs=500)
model.save_weights('../weight/weight')

exit()
