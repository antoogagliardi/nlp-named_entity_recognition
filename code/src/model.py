import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report


from src.utils import flatten_prediction, classNames
from src.plot import plot_confusion_matrix





class NERNetwork(nn.Module):
    def __init__(self, emb_dim:int, hid_dim:int, id2word_dict:dict, id2tag_dict:dict):
        super(NERNetwork, self).__init__()
        
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
        
    def forward(self, sentence):
        i_embedding = self.word_embeddings(sentence)
        
        lstm_out, (h, c) = self.LSTM(i_embedding)
        tag_space = self.hidden_layer(lstm_out)
        
        tag_scores = F.log_softmax(tag_space, dim=1)
        
        return tag_scores



class Trainer():
    def __init__(self, mymodel, optimizer, device):
        self.model = mymodel
        self.optimizer = optimizer
        self.device = device
        
        self.training_history = []
        self.accuracy_history = []
        self.training_loss_history = []
        
        self.GLOBAL_EPOCHS_COUNTER = 0
        
        self.logs = None
        
    def training_loop(self, samples, output_fol, n_epochs):
        print("-- N° of epochs: ", n_epochs)
        for epoch in tqdm(range(n_epochs), desc="Epoch", position=0, leave=True):
            # For sentence, tags in training_data:
    
            losses = []
            accuracies = []
    
            predicted_labels = []
            true_labels = []
    
            correct_predictions = 0
            num_predictions = 0
    
            for sent, tags in tqdm(samples, desc="Batch", unit="it", position=0, leave=True):
                for elem in range(len(sent)):
                    # Step 1. Remember that Pytorch accumulates gradients.
                    #  We need to clear them out before each instance
                    self.model.zero_grad()

                    # Step 2. Get our inputs ready for the network, that is, turn them into
                    #  Tensors of word indices.
                    sentence_in = sent[elem].to(self.device)
                    targets = tags[elem].to(self.device)

                    true_labels.append(targets.tolist())

                    # Step 3. Run our forward pass.
                    tag_scores = self.model(sentence_in).to(self.device)

                    # Step 4. Compute:
                    #  - the loss
                    #  - the gradients
                    #  - update the parameters by calling optimizer.step()
                    loss = self.model.loss_function(tag_scores, targets)

                    self.training_loss_history.append(loss)
                    losses.append(loss)

                    predictions = tag_scores.argmax(dim=-1)

                    predicted_labels.append(predictions.tolist())

                    for i in range(len(predictions)):
                        if predictions[i] == targets[i]:
                            correct_predictions += 1
                    num_predictions += len(predictions)

                    loss.backward()
                    self.optimizer.step()
        
            self.GLOBAL_EPOCHS_COUNTER += 1
            # Mean Loss
            mean_loss = sum(losses) / len(losses)
            self.training_history.append(mean_loss.item())

            predicted_labels, true_labels = flatten_prediction(predicted_labels, true_labels)
            print(plot_confusion_matrix(true_labels, predicted_labels, classes=classNames,
                                        dict_class=self.model.tags_dictionary, normalize=True, title="Confusion Matrix",
                                        save_cm=True, cm_dir="./cm", cm_name=f"cm_epoch-{epoch}"))
            # f1_score = metrics.f1_score(true_labels, predicted_labels)
            # print(f1_score)
            print(classification_report(true_labels, predicted_labels))

            accuracy = correct_predictions / num_predictions

            self.accuracy_history.append(accuracy)

            print('Epoch: {} completed ||===>  Loss: {:0.6f}, Accuracy: {:0.6f}'.format(epoch+1, loss, accuracy))
            # save the model state dict
            torch.save(self.model.state_dict(), os.path.join(output_fol, 'state_epoch-{}.pt'.format(epoch+1)))
            # save the entire model
            # torch.save(model, os.path.join(output_fol, 'model_{}.pt'.format(epoch+1)))

            self.logs = { 'train_history': self.training_history , 'accuracy_history': self.accuracy_history }
            print("Training Completed")