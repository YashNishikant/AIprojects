import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras import datasets, layers, models

(trainingimages, traininglables), (testingimages, testinglables), = datasets.cifar10.load_data()
trainingimages, testingimages = trainingimages/255, testingimages/255

classnames = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.xticks([])
    plt.imshow(trainingimages[i], cmap=plt.cm.binary)
    plt.xlabel(classnames[traininglables[i][0]])
plt.show()

trainingimages = trainingimages[:20000]
traininglabels = traininglables[:20000]
testingimages = testingimages[:4000]
testinglabels = testinglables[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainingimages, traininglabels, epochs=100, validation_data=(testingimages, testinglabels))

loss, accuracy = model.evaluate(testingimages, testinglabels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')