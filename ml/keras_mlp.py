#参考《机器学习之路》p182的代码

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

np.random.seed(42)

nb_classes = 10       # 类别
img_size = 28 * 28    # 输入图片大小
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32') / 255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32') / 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential([
    Dense(512, input_shape=(img_size,)),
    Activation('relu'),
    Dropout(0.2),
    Dense(512, input_shape=(512,)),
    Activation('relu'),
    Dropout(0.2),
    Dense(10, input_shape=(512,), activation='softmax'),
])

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 128
nb_epoch = 20

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('accuracy: {}'.format(score[1]))





