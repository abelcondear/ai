from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import numpy as np
import csv
import pandas as pd

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

def main():
    mfd = load_dataset()

    X = mfd.data
    y = mfd.target

    source_data = load_dataset()

    source_df = pd.DataFrame\
    (
        source_data.data, 
        columns=source_data.feature_names
    )

    source_df["target"] = source_data.target

    # ##########  Features and target ##########
    X = source_df[source_data.feature_names]
    y = source_df["target"]

    # ########## Standardize features ##########
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ########## Train-test split ##########
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split\
    (
        X_scaled, 
        y, 
        train_size=0.7, 
        random_state=25
    )

    # ########## Initialize models ##########
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }

    print("")

    # ########## Train and evaluate ##########
    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        
        print("")
        print(f"Applying Model \"{name}\".")
        print(f"Results:")
        print(f"{classification_report(y_test, predictions)}")
        print("")
    
    print("")

# ########## Main Program ##########
if __name__ == "__main__":
    main()