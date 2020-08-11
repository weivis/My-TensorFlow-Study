import tensorflow as tf

from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)), # 64个过滤器, 输入值的尺寸 28X28，
    keras.layers.MaxPooling2D(2,2),                         # 最大值化
    keras.layers.Conv2D(64, (3,3), activation='relu'), # 64个过滤器, 输入值的尺寸 28X28，
    keras.layers.MaxPooling2D(2,2),                         # 最大值化
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),         # 128个神经单元
    keras.layers.Dense(10, activation='softmax')        # 输出10项
])

# tf.nn.relu 激励函数 线性整流函数 返回大于0的数值过滤小于0
# tf.nn.softmax 返回这个集合中最大的数

model.fit(test_image, test_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
# prediction = model.predict(my_images)