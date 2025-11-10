from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Token:
    def __init__(self, num_words, oov_token, maxlen):
        self.token = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.maxlen = maxlen
    
    def __del__(self):
        pass
        
    def exec(self, training_sentences):
        self.token.fit_on_texts(training_sentences)
        word_index = self.token.word_index
        sequences = self.token.texts_to_sequences(training_sentences)
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=self.maxlen)
        return (sequences, padded_sequences)