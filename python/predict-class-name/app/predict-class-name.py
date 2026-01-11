import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import csv

from sklearn.utils import Bunch

def load_dataset():    
    with open(r'dataset.csv') as csv_file:
        data_reader = csv.reader(csv_file)

        feature_names = next(data_reader)[:-1]
        data = []
        target = []

        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append([float(num) for num in features])
            target.append(int(label))
        
        data = np.array(data)
        target = np.array(target)

    return Bunch(data=data, target=target, feature_names=feature_names)

ds = load_dataset()

X = ds.data        
y = ds.target      

y_encoded = to_categorical(y)

training = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

(
    X_train, 
    X_test, 
    y_train, 
    y_test
) = training

# functions to be applied
relu = Dense(8, input_shape=(5,), activation='relu')   
softmax = Dense(3, activation='softmax')                  

model = Sequential\
([
    relu,
    softmax
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

sample = np.array\
(
    [
        [
        
            0.1502795572743873,    #speaking
            0.2502795572743873,    #formal
            0.2502795572743873,    #clever
            0.3502795572743873,    #helpful
            0.4502795572743873,    #creative

        ]
    ]   
)

print("")
print("")

prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

print("")
print("Target   Class")
print("0        Graphical Designer")
print("1        Software Engineer")
print("2        Lead Team")

print("")
print("Dataset Sample [ without target value ]")
print(sample)

print("")
print("Predicted Probabilities [ Softmax Output ]:", prediction)

print("")
print("Predicted Class:", ["Graphical Designer","Software Engineer","Lead Team"][predicted_class])



