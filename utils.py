import numpy as np
from matplotlib import pyplot as plt
from model.models import mlp, mlp_leaky, CNN
from PIL import Image
from pathlib import Path
import itertools

def check_model(model):
    models = ['mlp', 'mlp_v2', 'CNN']
    if model == 'mlp':
        return mlp()
    elif model == 'mlp_leaky':
        return mlp_leaky()
    elif model == 'cnn':
        return CNN()
    else:
        return mlp()


def plot_loss(losses, title):
    plt.plot(losses, label='trainning losses')
    plt.xlabel("number of iteration")
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()


# reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, path, title='Confusion matrix', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(Path(path)/title)
    plt.show()
