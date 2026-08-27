from src.utils import flatten_list, fix_dict_indexes

class Dictionary():
    def create_dictionaries(self, dataframe:list, mode:str):
        if mode == "word":
            corpus = set(flatten_list(dataframe))
            self.id2word = {idx:word for idx, word in enumerate(corpus)}
            self.word2id = {word:idx for idx, word in enumerate(corpus)}
            # "<unk> token creation"
            unk_index = len(self.id2word)
            self.id2word[unk_index] = "<unk>"
            self.word2id["<unk>"] = unk_index
            self.id2word, self.word2id = fix_dict_indexes(self.id2word, self.word2id, "word")
        elif mode == "tag":
            self.id2tag = {}
            self.tag2id = {}
            corpus = set(flatten_list(dataframe))
            self.id2tag = {idx:tag for  idx, tag in enumerate(corpus)}
            self.tag2id = {tag:idx for  idx, tag in enumerate(corpus)}
            self.id2tag, self.tag2id = fix_dict_indexes(self.id2tag, self.tag2id, "tag")
        else:
            print("There's an error in the mode choosen")