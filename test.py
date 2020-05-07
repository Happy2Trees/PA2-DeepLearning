# Lenet without deep learning framework
# https://github.com/Site1997/LeNet-python/blob/master/LeNet.py
import numpy
import argparse
from pathlib import Path
from dataset import custom_transform
from dataset.dataloader import Dataloader
import utils
import numpy as np
from model import function as F
from matplotlib import pyplot as plt

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./result/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('-f', '--freq', type=int, default=200, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for training')
parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--model', required=True, type=str, help='beta parameters for adam optimizer')
parser.add_argument('--one_class', type=int, default=2, help='to show top 3 images ')

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

if __name__ == '__main__':
    args = parser.parse_args()
    train_transform = custom_transform.Compose([
        custom_transform.Normalize()
    ])
    testloader = Dataloader(
        path=args.data_dir,
        batch_size=32,
        is_train=False,
        shuffle=False
    )



    # Load pretrained models
    model = utils.check_model(args.model)
    model.load(args.pretrained)

    # initialize confusion matrix
    confusion_matrix = np.zeros((args.num_class, args.num_class), dtype=np.float32)

    # for top 3 class images
    img_class = []
    scores_class = []

    for i, (image, label) in enumerate(testloader):
        output = model(image)
        gt_class = np.argmax(label, axis=-1)
        pred_class = np.argmax(output, axis=-1)

        scores = F.softmax(output)

        # for the top 3 images
        condition = (gt_class == pred_class) & (pred_class == args.one_class)
        for index in range(args.batch_size):
            if condition[index]:
                img_class.append(image[index])
                scores_class.append(scores[index, pred_class[index]])

        for index in range(label.shape[0]):
            confusion_matrix[gt_class[index], pred_class[index]] += 1.0


    # plot the confusion matrix
    utils.plot_confusion_matrix(confusion_matrix, classes, path=args.save_dir,
                                title='{:s} of confusion matrix'.format(args.model))


    #make confusion matrix
    arg = np.argsort(scores_class)[::-1]
    arg = arg[:3]


    # plot top 3 images from PA1-code
    f = 0
    fig = plt.figure(figsize=(4, 2))
    for idx in arg:
        ax = fig.add_subplot(1, 3, f + 1, xticks=[], yticks=[])
        plt.imshow(img_class[idx].reshape(28, 28))
        ax.set_title("{:0.02f}\n{:4s}".format(
            scores_class[idx] * 100, classes[args.one_class]),
            color="green")
        fig.savefig(Path(args.save_dir) / 'top3_scores_results.png')
        f += 1