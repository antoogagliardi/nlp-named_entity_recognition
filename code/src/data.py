import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.utils import remove_unitary_duplicate



# Handle the data
class DataManager():
    def __init__(self, data:pd.DataFrame):
        self.data = data
    def setSentences(self, sentences:list[list[str]]):
        self.sentences = sentences
    def setTags(self, tags:list[list[str]]):
        self.tags = tags
    def getSentences_Tags(self):
        return self.sentences, self.tags
    
    # Separate Data
    def data_separation(self):
        sentences_list = []
        sentences_tag = []
        current_sentence = []
        current_id = []
        for i in tqdm(self.data.index):
            # Detect words that are part of the same sentence
            if np.isnan(self.data["0"][i]):
                current_sentence.append(self.data["#"][i])
                current_id.append(self.data["id"][i])
            # Separator : "# id _num_"
            # Detect if we're at the end of a sentence, then we need to store data
            if self.data["#"][i] == "#" and self.data["id"][i] == "id" and not np.isnan(self.data["0"][i]):
                sentences_list.append(current_sentence)
                sentences_tag.append(current_id)
                current_sentence = []
                current_id = []
        # We're at the end of the dataframe, and so we've terminated the parsing of last sentence. We need to store data
        # Place after the for cicle because for the last row we don't have "# id num" separator
        sentences_list.append(current_sentence)
        sentences_tag.append(current_id)

        # It basically two list in this form:
        # [["the", "cat", "is", "on", "the", "table"], ["happy", "birthday", "to", "you"]]
        # [[TAG, TAG, TAG, TAG, TAG, TAG], [TAG, TAG, TAG, TAG]]
        self.setSentences(sentences_list)
        self.setTags(sentences_tag)
        
    # Compute the max length overall sentences:
    def max_length(self, sliding_window_size):
        # Offset : 2*sliding_window_size
        max_length = len(max(self.sentences, key=len)) + 2*sliding_window_size
        
        return max_length

    # We have to uniform the size of our listed data (based on the maximum sentence's length of our dataset)
    # This is very important in order to work with Tensor
    def uniform_data_length(self, max_length:int):
        new_sentences = []
        new_tags = []
        for i in tqdm(range(len(self.sentences))):
            current_len = len(self.sentences[i])
            current_list = []
            current_tags = []
            for j in range(len(self.sentences[i])):
                if j<current_len:
                    current_list.append(self.sentences[i][j])
                    current_tags.append(self.tags[i][j])
            extend_list = []
            extend_tags = []
            # Uniform the length of all the vectors of words
            while current_len < max_length:
                extend_list.append("<pad>")
                extend_tags.append("<pad>")
                current_len += 1
            new_sentences.append(current_list + extend_list)
            new_tags.append(current_tags + extend_tags)
        self.sentences = new_sentences
        self.tags = new_tags
    
    # ------------------ Experimental ------------------
    def chunks_retriving(self):
        chunks = []
        chunks_tags = []
        if len(self.sentences) == len(self.tags):
            for row in range(len(self.tags)):
                O_chunck = []
                O_chunck_tags = []

                entity_chunck = []
                entity_chunck_tags = []
                for elem in range(len(self.tags[row])):
                    if self.tags[row][elem] == "O":
                        if entity_chunck != [] and entity_chunck_tags != []:
                            print("row: {}, chunk: {} , tags: {}".format(row, entity_chunck, entity_chunck_tags))
                            chunks.append(entity_chunck)
                            chunks_tags.append(entity_chunck_tags)
                            entity_chunck = []
                            entity_chunck_tags = []
                        O_chunck.append(self.sentences[row][elem])
                        O_chunck_tags.append(self.tags[row][elem])
                    else:
                        if O_chunck != [] and O_chunck_tags != []:
                            print("row: {}, chunk: {} , tags: {}".format(row, O_chunck, O_chunck_tags))
                            chunks.append(O_chunck)
                            chunks_tags.append(O_chunck_tags)
                            O_chunck = []
                            O_chunck_tags = []
                        if "B-" in self.tags[row][elem]:
                            if "I-" in self.tags[row][elem+1]:
                                entity_chunck.append(self.sentences[row][elem])
                                entity_chunck_tags.append(self.tags[row][elem])
                            else:
                                print("row: {}, chunk: {} , tags: {}".format(row, [self.sentences[row][elem]], [self.tags[row][elem]]))
                                chunks.append([self.sentences[row][elem]])
                                chunks_tags.append([self.tags[row][elem]])
                        if "I-" in self.tags[row][elem]:
                            entity_chunck.append(self.sentences[row][elem])
                            entity_chunck_tags.append(self.tags[row][elem])
        
        self.sentences, self.tags = remove_unitary_duplicate(chunks, chunks_tags)



class DataEncoder: 
    # Encode the data
    def encode_data(self, data:list, dictio:dict, mode:str):
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

    def decode_sentence(self, list_in:list, dictionary:dict):
        decoded_out = []
        real_length = 0
        for i in list_in:
            if i in dictionary:
                if i != 0:
                    decoded_out.append(dictionary[i])
                    real_length += 1

        return decoded_out, real_length

    @staticmethod
    def decode_prediction(list_in:list, dictionary:dict, real_sent_length:int):
        decoded_out = []
        counter = 0
        for i in list_in:
            if i in dictionary:
                if counter < real_sent_length:
                    decoded_out.append(dictionary[i])
                    counter += 1

        return decoded_out



class NERDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data):
        self.sentences = processed_data[0]
        self.targets = processed_data[1]
    def __len__(self):
        if len(self.sentences) == len(self.targets):
            return len(self.sentences)
        else:
            print("Error -- Length are different")
    def __getitem__(self, idx:int):
        return self.sentences[idx], self.targets[idx]