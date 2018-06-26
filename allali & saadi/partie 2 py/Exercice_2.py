#######"#  exercice 2 ##############################
model = Sequential()
model.add(Dense(100,  input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
model.add(Dense(10,  input_dim=784, name='fc1'))
model.add(Activation('softmax'))
#model.summary()

#######"#  exercice 2   SUITE ##############################
import h5py
model = Sequential()
model.add(Dense(100,  input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
sgd = SGD(learning_rate)
#model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.add(Dense(10,  input_dim=784, name='fc2'))
model.add(Activation('softmax'))
learning_rate = 0.3
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    print("Yaml Model ",savename,".yaml saved to disk")
  # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print("Weights ",savename,".h5 saved to disk")
saveModel(model, "Perceptron")