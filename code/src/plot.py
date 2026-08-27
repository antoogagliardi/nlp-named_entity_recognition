import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from src.data import DataEncoder

# Custom function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, dict_class, normalize=False, title=None, cmap=plt.cm.Blues,
                          save_cm:bool=False, cm_dir:str="./", cm_name:str="cm_default"):
    if not title:
        if normalize:
            title = 'Normalized CM'
        else:
            title = 'CM'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Use the labels that are in our dataset
    classes = unique_labels(y_true, y_pred)
    classes = classes.tolist()
    classes = DataEncoder.decode_prediction(classes, dict_class, len(classes))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Main Loop over data dimensions
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save the plot as a PNG image
    if save_cm == True:
        plt.savefig(os.path.join(cm_dir, cm_name), bbox_inches='tight')

    return ax

def plot_loss_results(logs: dict, title: str):
    plt.figure(figsize=(8,6))
    plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')  
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()

def plot_accuracy(logs: dict, title: str):
    plt.figure(figsize=(8,6))
    plt.plot(list(range(len(logs['accuracy_history']))), logs['accuracy_history'], label='Train Accuracy')  
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()