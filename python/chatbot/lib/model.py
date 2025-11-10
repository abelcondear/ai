from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import numpy as np

class Model:
    filename_model = "./files/chat_model.keras"

    def __init__(self, num_classes, vocab_size, embedding_dim, input_length):
        self.model = Sequential()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
    
    def __del__(self):
        pass

    def exec(self, padded_sequences, training_labels, epochs):    
        self.model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.input_length))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer='adam', metrics=['accuracy'])

        #self.model.summary()

        epochs = 500
        history = self.model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=0)

        # to save the trained model
        self.model.save(self.filename_model)
