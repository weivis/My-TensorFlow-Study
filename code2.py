import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 训练集和测试集
TRAINING_DIR = "./mnist/rps/"
VALIDATION_DIR  = "./mnist/rps-test-set/"

# https://keras.io/zh/preprocessing/image/
training_datagen = ImageDataGenerator(rescale = 1./255) # keras 图像预处理 rescale: 重缩放因子
validation_datagen = ImageDataGenerator(rescale = 1./255) # keras 图像预处理 rescale: 重缩放因子
'''
    flow_from_directory(directory): 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    # target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸

    # class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, 
    "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 
    生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.

'''

# 图像预处理
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(150,150), class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(150,150), class_mode='categorical'
)

# 构建网络
model = tf.keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),

    keras.layers.MaxPooling2D(2,2),


    keras.layers.Flatten(),

    keras.layers.Dropout(0.5),


    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(3, activation='softmax') # 三个神经元输出
])

# 训练方法
model.compile(loss = 'categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

# 训练深度
history = model.fit_generator(
    train_generator, epochs=10, validation_data=validation_generator, verbose=1
)

# 保存模型
model.save("rps_model.h5")