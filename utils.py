import numpy as np
from matplotlib import pyplot as plt
from model.models import mlp, mlp_leaky, CNN
from PIL import Image
from pathlib import Path
import itertools

# return to corresponding model
def check_model(model):
    models = ['mlp', 'mlp_leaky', 'CNN']
    if model == 'mlp':
        return mlp()
    elif model == 'mlp_leaky':
        return mlp_leaky()
    elif model == 'cnn':
        return CNN()
    else:
        return mlp()


# draw loss graph
def plot_loss(losses, title):
    plt.plot(losses, label='trainning losses')
    plt.xlabel("number of iteration")
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()


# reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, path, title='Confusion matrix'):

    accuracy = np.trace(cm) / np.sum(cm).astype(np.float64)

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks, target_names)

    # classwise accuracy text
    cm = cm.astype('float64') / cm.sum(axis=1)


    # threshold values
    thresh = cm.max() * (2/3)
    # put the accuracy each matrix cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # classwise accuracy !
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Output label\naccuracy={:0.4f} -- each text = Classwise Accuracy--'.format(accuracy))
    plt.savefig(Path(path)/title)
    plt.show()
