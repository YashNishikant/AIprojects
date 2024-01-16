import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras import datasets, layers, models

(trainingimages, traininglables), (testingimages, testinglables), = datasets.cifar10.load_data()
trainingimages, testingimages = trainingimages/255, testingimages/255

classnames = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

trainingimages = trainingimages[:20000]
traininglabels = traininglables[:20000]
testingimages = testingimages[:4000]
testinglabels = testinglables[:4000]

model = models.load_model('classification/image_classifier.model')

img = cv.imread('classification/cat.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

print(f"{classnames[np.argmax(model.predict(np.array([img]) / 255))]}")

plt.show()