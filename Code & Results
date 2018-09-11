import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np
from sklearn.metrics import confusion_matrix

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test = np.array(x_test).reshape(-1, 28, 28, 1)

model = Sequential()
ZeroPadding2D(padding=(1, 1), data_format="channels_last")
model.add(Conv2D(64, (3,3), input_shape = x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(32))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 3)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Loss = ',test_loss,', Acc = ', test_acc*100,'%')

'''Results for Loss & Accuracy
Epoch 1/3
60000/60000 [==============================] - 108s 2ms/step - loss: 0.1774 - acc: 0.9471
Epoch 2/3
60000/60000 [==============================] - 105s 2ms/step - loss: 0.0616 - acc: 0.9812
Epoch 3/3
60000/60000 [==============================] - 106s 2ms/step - loss: 0.0448 - acc: 0.9860
10000/10000 [==============================] - 6s 551us/step
Loss =  0.04054443840603344 , Acc =  98.74000000000001 %'''

y_pred = model.predict([x_test])
pred = np.zeros(len(y_pred))
for i in range(len(y_pred)):
  pred[i]=(np.argmax(y_pred[i]))
C = confusion_matrix(y_test, pred)
print(C)

'''Confusion Matrix
[[ 973    0    0    0    0    0    3    2    2    0]
 [   0 1124    4    1    1    1    0    0    4    0]
 [   1    0 1017    1    0    0    0    7    6    0]
 [   0    0    1 1004    0    2    0    2    1    0]
 [   0    0    0    0  973    0    2    0    3    4]
 [   0    0    1    6    0  881    1    1    1    1]
 [   2    2    0    0    6    5  941    0    2    0]
 [   0    1    7    0    1    0    0 1016    1    2]
 [   2    0    1    3    2    0    0    2  963    1]
 [   1    1    1    3    3    7    0    3    8  982]]'''
 
