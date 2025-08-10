import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full , X_test, y_train_full, y_test = train_test_split(housing.data,housing.target,random_state=42)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full,random_state=42)
print(len(X_train_full))
print(len(X_test))
print(len(X_valid))

### Convert X_train into a dataframe with attribute names
df_train = pd.DataFrame(X_train, columns = housing.feature_names)
print(df_train.head())

## Standarizing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

print(X_train.shape[1:])

## Model Architeture
#model = keras.models.Sequential([keras.layers.Dense(30,activation="relu",input_shape = X_train.shape[1:]),
#                                 keras.layers.Dense(10,activation="relu"),keras.layers.Dense(1)])
model = keras.models.Sequential([keras.layers.Dense(20,activation="relu",kernel_regularizer=keras.regularizers.l1_l2(0.001),input_shape = X_train.shape[1:]),
                                keras.layers.Dense(1)])
model.compile(loss="mean_squared_error",optimizer=keras.optimizers.SGD(learning_rate=0.0005))
history = model.fit(X_train,y_train,epochs=80,validation_data=(X_valid,y_valid))

##Plt of training and validation loss
plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label="Validation loss")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Testing the neural networks.
mse_test = model.evaluate(X_test,y_test)

# Make predictions on new data
X_new = X_test[:3]
y_pred = model.predict(X_new)

for i in range(len(X_new)):
    print(f"Sample {i+1}")
    print(f"Predicted: {y_pred[i][0]}")
    print(f"Actual: {y_test[i]}")