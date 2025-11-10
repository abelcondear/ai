import json

class loader:
    filename_data = "./files/intents.json"

    def __init__(self):
        pass
    
    def __del__(self):
        pass

    def loadjson(self):
        with open(self.filename_data) as file:
            data = json.load(file)
        return data
                
    def exec(self):
        training_sentences = []
        training_labels = []
        labels = []
        responses = []

        with open(self.filename_data) as file:
            data = json.load(file)
            
        for intent in data['intents']:
            for pattern in intent['patterns']:
                training_sentences.append(pattern)
                training_labels.append(intent['tag'])
            
            responses.append(intent['responses'])
            
            if intent['tag'] not in labels:
                labels.append(intent['tag'])
        
        return (training_sentences, training_labels, labels, responses)