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

model.load_weights('../weight/weight')

loss, acc = model.evaluate(np.array(xData), np.array(yData), verbose=2)
print("복원된 모델의 정확도:{:5.2f}%".format(100 * acc))

# 예측
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)

exit()
