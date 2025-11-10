import pickle

class file:
    filename_token = "./files/tokenizer.pickle"
    filename_encoder = "./files/label_encoder.pickle"

    def __init__(self):
        pass
    
    def writeToFile(self, ob_tokenizer, ob_labelencoder):
        # to save the fitted tokenizer
        with open(self.filename_token, 'wb') as handle:
            pickle.dump(ob_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # to save the fitted label encoder
        with open(self.filename_encoder, 'wb') as ecn_file:
            pickle.dump(ob_labelencoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
