import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from model import Model


def build_model(device: str) -> Model:
    # return RandomBaseline()
    return TestModel(device, model_path="model/state.pt" )


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]



def uniform_sentence_lenght(tokens: List[List[str]]) -> List[List[str]]:
        max_length = len(max(tokens, key=len))
        new_sentences = []
        for i in range(len(tokens)):
            current_len = len(tokens[i])
            current_list = []
            for j in range(len(tokens[i])):
                if j<current_len:
                    current_list.append(tokens[i][j])
            extend_list = []
            #Uniform the length of all the vectors of words
            while current_len < max_length:
                extend_list.append("<pad>")
                current_len += 1
            new_sentences.append(current_list + extend_list)
        
        return new_sentences
def encode_data(data:List[List[str]], dictio:dict, mode:str):
        encoded_list = []
        for i in range(len(data)):
            line = []
            for word in data[i]:
                if word in dictio:
                    encode = dictio[word]
                    line.append(encode)
                else:
                    if mode == "word":
                        encode = dictio["<unk>"]
                        line.append(encode)
                    if mode == "tag":
                        encode = dictio["<pad>"]
                        line.append(encode)
            encoded_list.append(line)

        return encoded_list
def decode_sentence(list_in:List[List[str]], dictionary:dict):
        decoded_out = []
        real_length = 0
        for i in list_in:
            if i in dictionary:
                if i != 0:
                    decoded_out.append(dictionary[i])
                    real_length += 1

        return decoded_out, real_length
def decode_prediction(list_in:List[List[str]], dictionary:dict, real_sent_length:int):
        decoded_out = []
        counter = 0
        for i in list_in:
            if i in dictionary:
                if counter < real_sent_length:
                    decoded_out.append(dictionary[i])
                    counter += 1

        return decoded_out
        
class NERNetwork(nn.Module):
    def __init__(self, emb_dim:int, hid_dim:int, id2word_dict:dict, id2tag_dict:dict):
        super(NERNetwork, self). __init__()
        self.words_dictionary = id2word_dict
        self.tags_dictionary = id2tag_dict
        
        self.embedding_dim = emb_dim
        self.hidden_dim = hid_dim
        
        # - Embedding Tensor (voc_size, emb_dim)
        self.word_embeddings = nn.Embedding(len(self.words_dictionary), self.embedding_dim)
        
        # - LSTM NN : Input -> "words embeddings"
        # - LSTM NN: Output -> "hidden states vectors"     
        self.LSTM = nn.LSTM(self.embedding_dim, self.hidden_dim)
        
        # Hidden layer maps from hidden state space to tag space
        self.hidden_layer = nn.Linear(self.hidden_dim, (len(self.tags_dictionary)))
        
        # Loss function definition
        self.loss_function = nn.NLLLoss(ignore_index=0)
        #self.loss_function = nn.NLLLoss()
        
    def forward(self, sentence):
        i_embedding = self.word_embeddings(sentence)
        
        lstm_out, (h, c) = self.LSTM(i_embedding)
        tag_space = self.hidden_layer(lstm_out)
        
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        return tag_scores

class TestModel(Model):
    def __init__(self, device, model_path):
        print("---- Initialize Model ----")
        self.id2tags = torch.load("model/id2tags.pt")
        self.id2words = torch.load("model/id2words.pt")
        self.words2id = torch.load("model/words2id.pt")
        model_testing = NERNetwork(512, 256, self.id2words, self.id2tags)
        model_testing.load_state_dict(torch.load(model_path))
        print("---- PyTorch Model Instatiation")
        model_testing.to(device)
        print("- Model loaded on: ", device)
        print("- Model Type: ", type(model_testing))
        print("- Model Parameters: \n", model_testing.parameters)
        print("--- Model Informations ---")
        print("- Embedding dimention: {}".format(model_testing.embedding_dim))
        print("- Hidden Layer dimention: {}".format(model_testing.hidden_dim))
        print("---- Model Word Embeddings ----")
        print(model_testing.word_embeddings)
        print("---- Model Word Dictionary ----")
        print(model_testing.words_dictionary)
        print("---- Model Tags Dictionary ----")
        print(model_testing.tags_dictionary)

        self.my_model = model_testing

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        self.my_model.eval()
        print("---- Uniforming sentences length ----")
        tokens = uniform_sentence_lenght(tokens)

        # Encode the sentence based on dictionary indices
        tokens = encode_data(tokens, self.words2id, "word")
        print("---- Encoding Data ----")

        predicted_labels = []
        for i in range(len(tokens)):
            with torch.no_grad():
                # Decode the current sentence
                decoded_sentence, real_length = decode_sentence(tokens[i], self.id2words)    

            	# Evaluete the current sentence with the model
                sent_sampl = torch.LongTensor(tokens[i])
                sent_sampl = sent_sampl.reshape((1,sent_sampl.shape[0]))
                tag_scores = self.my_model(sent_sampl)
                top_label_scores, top_label_indices = torch.max(tag_scores, -1)
                prediction = list(map(lambda x: x.item(), top_label_indices[0]))

                # Decode the prediction
                decoded_prediction = decode_prediction(prediction, self.id2tags, real_length)
                predicted_labels.append(decoded_prediction)
            
        return predicted_labels
