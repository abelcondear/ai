from sklearn.preprocessing import LabelEncoder

class encoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, training_labels):
        self.label_encoder.fit(training_labels)
        return self.label_encoder.transform(training_labels)
