import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from keras.models import Sequential,load_model
import keras

train_images = pd.read_csv("dataset/emnist-balanced-train.csv",header=None)
test_images = pd.read_csv("dataset/emnist-balanced-test.csv",header=None)
map_images = pd.read_csv("dataset/emnist-balanced-mapping.txt",header=None) 
#The 1st row would be treated as header if not set header to none.

# Seperating labels from features in training and test data.
train_x = train_images.iloc[:,1:]  
train_y = train_images.iloc[:,0]  
train_x = train_x.values

test_x = test_images.iloc[:,1:]
test_y = test_images.iloc[:,0]
test_x = test_x.values

# ascii_map just for the convenince, i've removed the first column in map_images.
ascii_map = []
for i in map_images.values:
    ascii_map.append(i[0].split()[1])

# The images in train_images are not in a proper orientation,hence to make them appropriate for training & testing data.

def rot_flip(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.apply_along_axis(rot_flip,1,train_x)
test_x = np.apply_along_axis(rot_flip,1,test_x)
plt.imshow(train_x[2])
train_x.shape

train_x = train_x.astype('float32')
train_x = train_x/255.0

test_x = test_x.astype('float32')
test_x = test_x/255.0

train_x = train_x.reshape(-1, 28,28, 1)   #Equivalent to (112800,28,28,1)
test_x = test_x.reshape(-1, 28,28, 1)   #Equivalent to (18800,28,28,1)

model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(47,activation='softmax'))

model.compile(optimizer = 'adam',loss= "sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(
    train_x,
    train_y,
    validation_data = (test_x,test_y),
    epochs = 2,
    batch_size=64
)

y_pred = model.predict(test_x)
y_pred

from sklearn.metrics import accuracy_score

y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(test_y, y_pred_labels)
print("Accuracy: %.2f%%" % (accuracy * 100))

ascii_map = []
for i in map_images.values:
    ascii_map.append(i[0].split()[1])


# Adding character to associated ASCII Value
character = []
for i in ascii_map:
    character.append(chr(int(i)))
# plt.imshow(np.rot90(np.fliplr(train_x[1].reshape(28,28))))
character = pd.DataFrame(character)

ascii_map = pd.DataFrame(ascii_map)
ascii_map["Character"] = character
ascii_map.to_csv("mapping.csv",index=False,header=True)

model.save('emnist.keras')

from tensorflow.keras.models import load_model
model_loaded = load_model('emnist.keras')
model_loaded.summary()

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model_loaded, 'emnist_model.tfjs')