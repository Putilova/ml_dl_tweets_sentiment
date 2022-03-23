# -*- coding: utf-8 -*-
"""tf3_keras_age_salary_tfjs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KJ9qUZeEmV023kAB7BMw-ra5CKvcUDTr
"""

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('..\_05_20_save_export_reload_pytorch_models\storepurchasedata_large.csv')

dataset.head()

dataset.describe()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

loss, accuracy =model.evaluate(X_test, y_test)

loss

accuracy

model.summary()

prediction1 = model.predict(sc.transform(np.array([[42,50000]])))[:,1]
print("prediction 1")
print(prediction1)

predict2 = model.predict(sc.transform(np.array([[20,40000]])))[:,1]
print("prediction 2")
print(predict2)

#model.save('customer_behavior_model/1')

#!ls

#!ls customer_behavior_model/1


"""from tensorflow.keras.models import load_model

cust_model = load_model('customer_behavior_model/1/')

prediction3 = cust_model.predict(sc.transform(np.array([[42,50000]])))[:,1]
 prediction3

predict4 = model.predict(sc.transform(np.array([[20,40000]])))[:,1]
predict4

sc.transform(np.array([[20,40000]]))

sc.transform(np.array([[42,50000]]))"""

#!ls

#!zip -r customermodel.zip customer_behavior_model

#!ls

#from google.colab import files

#!pip install tensorflowjs

import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, "/content/")

#!ls

#!cat model.json

#from google.colab import files

#files.download('model.json')

#!ls

#files.download('group1-shard1of1.bin')

