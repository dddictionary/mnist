import os
from typing import Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# print(tf.sysconfig.get_build_info())

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def generate_model(x_train: np.ndarray, y_train: np.ndarray):
    # x is the actual image, y is the classification. more descriptive
    # names for them would be image_test, classification_test and image_train, classification_train.

    # normalizing the data to be between 0-1. The values are usually between 0-255 for a grayscale image.

    model = tf.keras.models.Sequential()

    # adding layers
    # it's 28 by 28 becuase each input node is going to be corresponding to one pixel of the input image
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # i do not know what these layers are for im ngl
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # output layer. 10 for the 10 digits
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save("mnist.model")


def evaluate_model(x_test: np.ndarray, y_test: np.ndarray, model: tf.keras.Model = None):
    if model is None:
        model = tf.keras.models.load_model("mnist.model")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")


def test_model_on_input(model: tf.keras.Model):
    img_num = 1
    while os.path.isfile(f"./digits/digit{img_num}.png"):
        try:
            img = cv2.imread(f"./digits/digit{img_num}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This number is most likely a {np.argmax(prediction)}!")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print(f"No image for digit {img_num}")
        finally:
            img_num += 1


def main():
    # generate_model(x_train, y_train)
    # evaluate_model(x_test, y_test)
    test_model_on_input(model=tf.keras.models.load_model("mnist.model"))


if __name__ == "__main__":
    main()
