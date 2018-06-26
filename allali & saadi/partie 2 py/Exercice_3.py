#######"#  exercice 3 ##############################
x_train=X_train
x_test=X_test
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(32,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
pool = MaxPooling2D(pool_size=(2, 2))

model.add(Conv2D(64,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
pool = MaxPooling2D(pool_size=(2, 2))

model.add(Flatten())

model.add(Dense(100,  input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
sgd = SGD(learning_rate)
#model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.add(Dense(10,  input_dim=784, name='fc2'))
model.add(Activation('softmax'))
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
scores = model.evaluate(x_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
