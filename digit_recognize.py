import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


# mnist = tf.keras.datasets.mnist # mnist is a dataset of 28x28 images of handwritten digits and their labels
# (x_train, y_train),(x_test, y_test) = mnist.load_data() # unpacks images to x_train/x_test and labels to y_train/y_test

# x_train = tf.keras.utils.normalize(x_train, axis=1) # scales data between 0 and 1
# x_test = tf.keras.utils.normalize(x_test, axis=1) # scales data between 0 and 1

# model = tf.keras.models.Sequential() # a basic feed-forward model
# model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # takes our 28x28 and makes it 1x784
# model.add(tf.keras.layers.Dense(128, activation='relu')) # a simple fully-connected layer
# model.add(tf.keras.layers.Dense(128, activation='relu')) # a simple fully-connected layer
# model.add(tf.keras.layers.Dense(10, activation='softmax')) # our output layer. 10 units for 10 classes. Softmax for probability distribution

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])# Good default optimizer to start with

# model.fit(x_train,y_train , epochs=3) # train the model

# model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0] # read the image
        img = np.invert(np.array([img])) # make it the right size
        prediction = model.predict(img)
        print(f"The result is probably: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary) # show the image
        plt.show()
    except:
        print("An exception occurred")
    finally:
        image_number += 1
