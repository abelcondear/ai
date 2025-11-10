import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import lib.loader as ld
import lib.encoder as enc
import lib.tokenizer as tok
import lib.model as md
import lib.file as fi

import numpy as np 
import pickle
import colorama 

from colorama import Fore, Style, Back
from tensorflow import keras

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"
epochs = 500

def initilize():
    loader = ld.loader()
    (training_sentences, training_labels, labels, responses) = loader.exec()

    lbl_encoder = enc.encoder()
    training_labels = lbl_encoder.fit(training_labels)

    tokenizer = tok.Token(vocab_size, oov_token, max_len)
    (sequences, padded_sequences) = tokenizer.exec(training_sentences)

    num_classes = len(labels)

    model = md.Model(num_classes, vocab_size, embedding_dim, max_len)
    model.exec(padded_sequences, training_labels, epochs)

    file = fi.file()
    file.writeToFile(tokenizer.token, lbl_encoder.label_encoder)

    return (loader, file, model)

def chat(parameters):
    (loader, file, model) = parameters
    model = keras.models.load_model(model.filename_model)

    data = loader.loadjson()

    with open(file.filename_token, 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(file.filename_encoder, 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        user_input = input()

        if user_input.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                             truncating='post', maxlen=max_len),
                                             verbose=0)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

if __name__ == '__main__':
    colorama.init()
    print(Fore.YELLOW + "Type \"quit\" to stop." + Style.RESET_ALL)
    chat(initilize())
