from tqdm import tqdm




# Utilities for Data Manager
def remove_unitary_duplicate(words, tags):
    new_words = []
    new_tags = []
    for row in range(len(tags)):
        if len(tags[row]) < 2 and tags[row][0] == "O" and words[row][0] in [ ",", ".", "!", "?", ":" , ";", "-", ")", "]", "}"]:
            pass
        else:
            new_words.append(words[row]) 
            new_tags.append(tags[row])

    return new_words, new_tags


# Utilities for Dictionary
def flatten_list(lista:list):
    flat_list = []
    for sublist in tqdm(lista):
        for item in sublist:
            flat_list.append(item)
            
    return flat_list

def fix_dict_indexes(id2w:dict, w2id:dict, mode:str):
        # id2w = id2...  |  w2id = ...2id
        if mode == "word":
            # Force word "<pad>" id to be 0
            tag_0 = id2w[0]
            id_jolly = w2id["<pad>"]
            w2id[tag_0] = id_jolly
            w2id["<pad>"] = 0
            id2w[0] = "<pad>"
            id2w[id_jolly] = tag_0
            # Force word "<unk>" id to be 1
            tag_1 = id2w[1]
            id_jolly1 = w2id["<unk>"]
            w2id[tag_1] = id_jolly1
            w2id["<unk>"] = 1
            id2w[1] = "<unk>"
            id2w[id_jolly1] = tag_1

        if mode == "tag":
            # Force tag "<pad> id to be 0"
            tag_0 = id2w[0]
            id_jolly = w2id["<pad>"]
            w2id[tag_0] = id_jolly
            w2id["<pad>"] = 0
            id2w[0] = "<pad>"
            id2w[id_jolly] = tag_0

        id2w = dict(sorted(id2w.items(), key=lambda x: x[0]))
        w2id = dict(sorted(w2id.items(), key=lambda x: x[1]))

        return id2w, w2id


# Sliding-Windows: Tensors creation
#  We have to uniform the size of our tensors
#  We make our tensor based on the maximum length of our set of sentences computed above
def n_gram_extraction(sentences:list, tags:list, input_idx:int, window_size:int, tag_pad:int, word_pad:int):
    n_gram_sent = []
    n_gram_tags = []
    # left and right window indices
    min_idx = max(0, input_idx - window_size)
    max_idx = min(len(sentences), input_idx + window_size)
    window_idxs = [x for x in range(min_idx, max_idx +1)]
    for idx in window_idxs:
        n_gram_sent.append(sentences[idx])
        n_gram_tags.append(tags[idx])
    if len(window_idxs) < (window_size*2 +1):
        pivot_word = input_idx
        fil = (window_size*2 + 1) - len(window_idxs)
        while fil != 0:
            if input_idx != len(sentences)-1:
                n_gram_sent.insert(0, word_pad)
                n_gram_tags.insert(0, tag_pad)
            else:
                n_gram_sent.append(word_pad)
                n_gram_tags.append(tag_pad)
            fil -= 1
    
    return n_gram_sent, n_gram_tags

def create_tensors(window_size:int, stride: int, sentences:list, tags:list, tag_dict:dict, word_dict:dict):
    tensor_sentences = []
    tensor_tags = []
    for i in range(len(sentences)):
        sent_list = []
        tag_list = []
        j = 0
        while j < len(sentences[i]):
            if j<(len(sentences[i])-window_size):
                sent_list, tag_list = n_gram_extraction(sentences[i],
                                                             tags[i],
                                                             j,
                                                             window_size, tag_dict["<pad>"], word_dict["<pad>"])
                count = tag_list.count(tag_dict["<pad>"])
                # Reject just the lists that have all elements "<pad>" (they're useless)
                if count <= window_size:
                    tensor_sentences.append(sent_list)
                    tensor_tags.append(tag_list)
            sent_list = []
            tag_list = []
            j += 1 + stride

    return tensor_sentences, tensor_tags



# Other Utilities
classNames = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

def flatten_prediction(pred, target):
    flatten_pred = []
    flatten_target = []
    for i in range(len(pred)):
        flatten_pred += pred[i]
        flatten_target += target[i]
    
    return flatten_pred, flatten_target
