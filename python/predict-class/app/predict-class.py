import numpy as np
import math as mt
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #avoid info and warning messages

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

def load_dataset():    
    with open(r'data/dataset.csv') as csv_file:
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

    return Bunch\
    (
        data=data,
        target=target, 
        feature_names=feature_names
    )

ds = load_dataset()

X = ds.data        
y = ds.target      

y_encoded = to_categorical(y)

training = train_test_split\
(
    X, 
    y_encoded, 
    test_size=0.2, 
    random_state=42
)

(
    X_train, 
    X_test, 
    y_train, 
    y_test
) = training

# functions to be applied
relu = Dense\
(
    8, 
    input_shape=(5,), 
    activation='relu'
)   

softmax = Dense\
(
    3, 
    activation='softmax'
)                  

model = Sequential\
([
    relu,
    softmax
])

model.compile\
(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history = model.fit\
(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=8, 
    validation_split=0.2, 
    verbose=0
)

raw_dataset = np.array\
(
    [
        [
        
            0.1502795572743873,    #speaking feature
            0.2502795572743873,    #formal feature
            0.2502795572743873,    #clever feature
            0.3502795572743873,    #helpful feature
            0.4502795572743873,    #creative feature

        ]
    ]   
)

prediction = model.predict(raw_dataset, verbose=0)
predicted_class = np.argmax(prediction)

print("")
print("Target   Class")
print("0        Graphical Designer")
print("1        Software Engineer")
print("2        Lead Team")

print("")
print("Dataset - Single Raw [ Without Target ]:", raw_dataset)

print("")
class_arr = np.array\
(
    [
        'Speaking',
        'Formal',
        'Clever',
        'Helpful',
        'Creative'
    ]
)
print("Features:", class_arr)

print("")
class_arr = np.array(['Graphical Designer', 'Software Engineer', 'Lead Team'])
print("Class:", class_arr)

print("")
print("Predicted Probabilities [ Softmax Output ] [ Values ]:", prediction)

porcentage = lambda t: str(mt.floor(t * 100)) + "%"
map_values = np.vectorize(porcentage)

print("")
print("Predicted Probabilities [ Softmax Output ] [ Percent ]:", map_values(prediction))

print("")
class_names = ["Graphical Designer", "Software Engineer", "Lead Team"]
print("Predicted Class:", class_names[predicted_class])

print("")