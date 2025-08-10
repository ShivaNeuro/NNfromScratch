#Loading Packaging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test)= fashion_mnist.load_data()
print(X_train_full.shape)
print(X_train_full.dtype)

# Creating Validation data
# Splitting test to valiation and test
X_valid , X_train = X_train_full[:5000]/255. , X_train_full[5000:]/255.
y_valid, y_train = y_train_full[:5000],y_train_full[5000:]
X_test = X_test/255.

# Plot to check
plt.imshow(X_train[0],cmap="binary")
plt.axis("off")
plt.show()
# To check how labels are represented
print(y_train)
print(len(y_train))
print(y_train[0])
# To represent with class names
class_names = ["T_shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankleboot"]
print(class_names[y_train[0]])
print(X_valid.shape)
print(X_train.shape)

# Plot number of images
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2 , n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows,n_cols,index+1)
        plt.imshow(X_train[index],cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    plt.show()

## Using Keras API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

## Seed at the start- to have same values of weights and biases
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

print(model.layers)
print(model.summary())
#keras.utils.plot_model(model,"my_fashion_mnist_model.png",show_shapes=True)
hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer(hidden1.name) is hidden1)
weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=10,validation_data=(X_valid,y_valid))

print(history.params)
#print(history.epochs)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test,y_test)

## Testing samples
X_new = X_test[:3]
plt.figure(figsize=(7.2,2.4))
for index, image in enumerate(X_new):
    plt.subplot(1,3,index+1)
    plt.imshow(image,cmap="binary",interpolation="nearest")
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


y_predict = np.argmax(model.predict(X_new),axis=-1)
print(np.array(class_names)[y_predict])

