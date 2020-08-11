import tensorflow as tf
from tensorflow import keras
import numpy as np

# 模型 = keras.Sequential 定义神经网络([keras.layers.Dense(units=神经网络数, input_shape=[输入执行为1])]) 预测 X对应的Y值是多少
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# 激励函数 （优化函数，损失函数）
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))