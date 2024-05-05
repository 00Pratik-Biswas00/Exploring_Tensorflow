# Relationship - y = 3x+1.

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


# The process of training the neural network, where it learns the relationship between the X's and Y's, is in the model.fit call. That's where it will go through the loop before making a guess, measuring how good or bad it is (the loss), or using the optimizer to make another guess. It will do that for the number of epochs that you specify.
model.fit(xs, ys, epochs=110)


# You have a model that has been trained to learn the relationship between X and Y. You can use the model.predict method to have it figure out the Y for a previously unknown X.
print(model.predict([10.0]))