import os
import random
import numpy as np
import pandas as pd
from pprint import pprint

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils import create_tensors
from src.data import DataManager, DataEncoder, NERDataset
from src.dict import Dictionary
from src.model import NERNetwork, Trainer
from src.plot import plot_accuracy, plot_loss_results



# Folder preparation
os.makedirs("ckpt", exist_ok=True)
os.makedirs("cm", exist_ok=True)
os.makedirs("../model", exist_ok=True)
os.makedirs("../model/dicts", exist_ok=True)

# Read the configuration file


# Hyper-parameters setup
    # SEED Fixing
SEED = random.randrange(10000)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("- Current generated SEED: ", SEED)

    # Hyper-Parameters
SLIDING_WINDOW_SIZE = 3
STRIDE = 0
BATCH_SIZE = 64                             # 128
EMBEDDING_DIM = 512                         # 128
HIDDEN_LAYER_DIM = 256                      # Denote the "num of cell" of the output (NEURONS)
RANDOM_SPLIT_SEED = random.seed(SEED)
TRAIN_PROPORTION = 0.8
    # Setting Up the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')


# Path setup
root_folder = "../"
print("- root folder: ", root_folder)
cwd = os.getcwd()
print("- cwd: ", cwd)
data_folder = os.path.join(root_folder, "data")
print("- data folder: ", data_folder)
train_path = os.path.join(data_folder, "train.tsv")
dev_path = os.path .join(data_folder, "dev.tsv")

model_folder = os.path.join(root_folder, "model")
print("- model folder: ", model_folder)
ckpt_folder = os.path.join(cwd, "ckpt")
print("- ckpt folder: ", ckpt_folder)


# Retrieving Training Dataset
train_set = pd.read_csv(train_path, "\t")
pprint(train_set)

data_manager = DataManager(train_set)
print("-- Retrieving the data")
data_manager.data_separation()
print("-- Uniforming data length")
max_length = data_manager.max_length(SLIDING_WINDOW_SIZE)
print("- Max length sentence: ", max_length)
data_manager.uniform_data_length(max_length)

sentences_list, sentences_tag = data_manager.getSentences_Tags()
print("- Training dataset length: ", len(sentences_list))


# Dictionaries creation
word = Dictionary()
tag = Dictionary()
word.create_dictionaries(sentences_list, "word")
tag.create_dictionaries(sentences_tag, "tag")

print("-- Dictionaries Informations: ")
print("- Length of words dictionary: ", len(word.word2id))
print("- Length of tags dictionary: ", len(tag.tag2id))

# Save the model's dictionaries
# print(tag.id2tag)
torch.save(word.word2id, os.path.join(os.path.join(model_folder, "dicts"), 'words2id.pt'))
torch.save(word.id2word, os.path.join(os.path.join(model_folder, "dicts"), 'id2words.pt'))
torch.save(tag.tag2id, os.path.join(os.path.join(model_folder, "dicts"), 'tags2id.pt'))
torch.save(tag.id2tag, os.path.join(os.path.join(model_folder, "dicts"), 'id2tags.pt'))
# print(model.state_dict)


# Dataset Division and Pre-Processing
    # Splitting the data
train_sents, dev_sents, train_tags, dev_tags = train_test_split(sentences_list, sentences_tag, test_size=TRAIN_PROPORTION, random_state=RANDOM_SPLIT_SEED)
print("- Length of sentences list: ", len(train_sents))
    # Encode the data
dataEncoder = DataEncoder()
encoded_sentences = dataEncoder.encode_data(train_sents, word.word2id, "word")
encoded_tags = dataEncoder.encode_data(train_tags, tag.tag2id, "tag")
print("-- Data Encode completed")
print("-- Tensor Creation: ")
tensor_sents_tags = torch.tensor(create_tensors(SLIDING_WINDOW_SIZE, STRIDE, encoded_sentences, encoded_tags, tag.tag2id, word.word2id))
# print(tensor_sents_tags[0])
# print(tensor_sents_tags[1])
print("- Tensor shape: ", tensor_sents_tags.shape)
print("- Sentences Tensor's Shape: ", tensor_sents_tags[0].shape)
print("- Tags Tensor's Shape: ", tensor_sents_tags[1].shape)
print("- Max Elem in the tensor: ", torch.max(tensor_sents_tags[0]))
print("- Min Elem in the tensor: ", torch.min(tensor_sents_tags[0]))
print("- Max Elem in the tensor: ", torch.max(tensor_sents_tags[1]))
print("- Min Elem in the tensor: ", torch.min(tensor_sents_tags[1]))
print("-- Dataset Informations: ")
print("- Length of sentences list: ", len(tensor_sents_tags[0]))
print("- Length of tags list: ", len(tensor_sents_tags[1]))
print("- Length of words dictionary: ", len(word.word2id))
print("- Length of tags dictionary: ", len(tag.tag2id))
    # Pytorch Dataset creation
final_data = NERDataset(tensor_sents_tags)
print("-- PyTorch dataset creation completed")
print("- Example of data: {}".format(final_data[0]))


# Data Batching
train_dataloader = DataLoader(final_data, batch_size=BATCH_SIZE, shuffle=True)
print("-- Batching completed")
print("-- Batch Informations: ")
print("- Length of the batch: ", BATCH_SIZE)
print("- Max sub-batch length: ", len(max(train_dataloader.dataset[0], key=len)))
print("- Min sub-batch length: ", len(min(train_dataloader.dataset[0], key=len)))
print("- Example of data in the DataLoader: {}, {}".format(train_dataloader.dataset[0][0], train_dataloader.dataset[1][1]))


# Model training
print("-- Training Setup")
print("- Device used: ", device)
    # Model creation
model = NERNetwork(EMBEDDING_DIM, HIDDEN_LAYER_DIM, word.id2word, tag.id2tag)
model.to(device)
print("-- Model Parameters: ", model.word_embeddings)
    # Optimizer to update the parameters(stochastic gradient descent)
    #  Learning rate setting : 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    tag_scores = model(tensor_sents_tags[0][0].to(device=device))
    top_label_scores, top_label_indices = torch.max(tag_scores, -1)
    prediction = list(map(lambda x: x.item(), top_label_indices))
    print("-- Random prediction made before training: ", prediction)
    # print(tag_scores)
    # Training Loop
EPOCHS = 2                                                      # 7
trainer = Trainer(model, optimizer, device)
trainer.training_loop(train_dataloader, ckpt_folder, EPOCHS)
    # End of the training
print(trainer.logs)
plot_loss_results(trainer.logs, 'Loss Results')
plot_accuracy(trainer.logs, 'Accuracy Results')