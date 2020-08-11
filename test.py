from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# 训练集和测试集
TRAINING_DIR = "./mnist/rps/"

training_datagen = ImageDataGenerator(rescale = 1./255) # keras 图像预处理 rescale: 重缩放因子
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(150,150), class_mode='categorical'
)

images = './bu.jpg'
modelpath = 'rps_model.h5'

model = load_model(modelpath)


img = image.load_img(images, target_size=(150, 150))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)

class_indices = train_generator.class_indices
print(classes)

id = ["paper(布)", "rock(石头)", "scissors(剪刀)"]

for i in range(3):
    print(id[i],"的可能性: ", classes[0][i])

im = Image.open(images)
im.show()