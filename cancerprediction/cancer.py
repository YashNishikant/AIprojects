import pandas

dataset = pandas.read_csv('cancerprediction\cancer.csv')
x = dataset.drop(columns=['diagnosis(1=m, 0=b)']) #information for each tumor   (DATA)
y = dataset['diagnosis(1=m, 0=b)'] #malignant vs benign diagnosis               (ACTUAL/RESULT)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)#             20% of data and result goes for testing 

import tensorflow as tf #                                                       Create Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=xtrain.shape[1:], activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=1000)