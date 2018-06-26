#################  Exercice 3 ###############
from keras.models import model_from_yaml
def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(savename+".h5")
    print("Weights ",savename,".h5 loaded ")
    return model

model=loadModel("Perceptron")
model.summary()

from keras.optimizers import SGD
import keras.utils as np_utils
X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255
Y_test = np_utils.to_categorical(y_test, 10)

learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.pop()
model.pop()
X=model.predict(X_test)

convex_hulls= convexHulls(X, y_test)
ellipses = best_ellipses(X,y_test)
go=neighboring_hit(X, y_test)
visualization(X, y_test, convex_hulls, ellipses ,'MLP', go)