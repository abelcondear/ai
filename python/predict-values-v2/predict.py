from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.utils import Bunch

import pandas as pd
import array as arr
import numpy as np
import csv
import math

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

def split_into_groups(series, n):
    """
    Description:
        Splits a Pandas Series or list into groups of n items.
    
    Args:
        series ( pd.Series | list | np.ndarray ): Input data.
        n ( int ): Group size.
    
    Returns:
        list of lists: Groups of size n.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("Group size 'n' must be a positive integer.")
    
    # Convert to Series if not already
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Split into chunks
    return [series[i:i+n].tolist() for i in range(0, len(series), n)]

mfd = load_dataset()

X = mfd.data
y = mfd.target

# Load dataset
data = load_dataset()

data_df = pd.DataFrame(data.data, columns=data.feature_names)

data_df["target"] = data.target

# Features and target
X = data_df[data.feature_names]
y = data_df["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
   X_scaled, y, train_size=0.95, random_state=25
)

# Initialize models
models = {
   "Logistic Regression": LogisticRegression(),
   "Support Vector Machine": SVC(),
   "Decision Tree": DecisionTreeClassifier()
}

df = pd.DataFrame()

v = X_test

jj = v[0:] # get column by index

# print("---------")

# print(jj[:,0])  # get entire column by index
# print(jj[:,1])  # get entire column by index
# print(jj[:,2])  # get entire column by index
# print(jj[:,3])  # get entire column by index
# print(jj[:,4])  # get entire column by index

# print("---------")

indexes = [i for i in range(jj[:,0].size) ]

columns = [
    'feature_0',
    'feature_1',
    'feature_2',
    'feature_3',
    'feature_4'
]

values=jj[:,0]  # get column by index
serie = pd.Series(values, index=indexes)
df.insert(0, columns[0], serie)

values=jj[:,1]  # get column by index
serie = pd.Series(values, index=indexes)
df.insert(1, columns[1], serie)

values=jj[:,2]  # get column by index
serie = pd.Series(values, index=indexes)
df.insert(2, columns[2], serie)

values=jj[:,3]  # get column by index
serie = pd.Series(values, index=indexes)
df.insert(3, columns[3], serie)

values=jj[:,4]  # get column by index
serie = pd.Series(values, index=indexes)
df.insert(4, columns[4], serie)

print("")
print("Dataframe:")
print(df)
print("")

# Group size
n = 5

groups = split_into_groups(df["feature_0"], n)
#print(f"Groups of {n}:", groups)

ggroups = []
for x in groups:
    ggroups.append([ abs(int(math.floor(xx))) for xx in x ])

print("feature_0.")
print("List of Values:")
print(ggroups)
print("")

groups = split_into_groups(df["feature_1"], n)
#print(f"Groups of {n}:", groups)

ggroups = []
for x in groups:
    ggroups.append([ abs(int(math.floor(xx))) for xx in x ])

print("feature_1.")
print("List of Values:")
print(ggroups)
print("")

groups = split_into_groups(df["feature_2"], n)
#print(f"Groups of {n}:", groups)

ggroups = []
for x in groups:
    ggroups.append([ abs(int(math.floor(xx))) for xx in x ])

print("feature_2.")
print("List of Values:")
print(ggroups)
print("")

groups = split_into_groups(df["feature_3"], n)
#print(f"Groups of {n}:", groups)

ggroups = []
for x in groups:
    ggroups.append([ abs(int(math.floor(xx))) for xx in x ])

print("feature_3.")
print("List of Values:")
print(ggroups)
print("")

groups = split_into_groups(df["feature_4"], n)
#print(f"Groups of {n}:", groups)

ggroups = []
for x in groups:
    ggroups.append([ abs(int(math.floor(xx))) for xx in x ])

print("feature_4.")
print("List of Values:")
print(ggroups)
print("")

# Train and evaluate
for name, model in models.items():
   model.fit(X_train, y_train)

   preds = model.predict(X_test)

   print("") 
   print(f"Model {name}. ")
   print("Prediction:")
   print(preds)
   print("")

   #print(y_test)

   print("")
   print(f"Model {name}. ")
   print(f"Results:\n{classification_report(y_test, preds)}")
   print("")
