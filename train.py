# Lenet without deep learning framework
# https://github.com/Site1997/LeNet-python/blob/master/LeNet.py
import numpy as np
import argparse
from pathlib import Path
from dataset import custom_transform
from dataset.dataloader import Dataloader
from model.function import crossEntropy, grad_crossEntropy
import utils
import time


# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./pretrained/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--epoch', type=int, default=30, help='epoch size for training')
parser.add_argument('-f', '--freq', type=int, default=200, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--model', required=True, type=str, help='model for trainning')


''' for class structure'''
# reference : https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9


if __name__ == '__main__':
    args = parser.parse_args()
    train_transform = custom_transform.Compose([
        custom_transform.Normalize()

    ])
    # data loader
    trainloader = Dataloader(
        path=args.data_dir,
        is_train=True,
        batch_size=args.batch_size,
        transform=train_transform,
        shuffle=True
    )

    # load network
    model = utils.check_model(args.model)
    if args.pretrained is not '':
        load_path = Path(args.path)
        model.load(load_path)
    save_dir = Path(args.save_dir)
    losses = []
    # train
    for epoch in range(args.epoch):

        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (image, label) in enumerate(trainloader):
            iter_start_time = time.time()
            output = model(image)

            loss = crossEntropy(output, label)
            grad_loss = grad_crossEntropy(output, label)

            # update parameters with backward
            model.update(grad_loss)

            # save losses for plot loss graph
            loss = np.mean(loss, axis=0)
            losses.append(loss)

            running_loss += loss
            if i % args.freq == args.freq - 1:
                utils.plot_loss(losses, '{:s} model Loss Graph'.format(args.model))
                print('epoch: {}, iteration:{}/{} loss: {:0.4f}, iteration_time: {:0.4f}'.format(epoch + 1,
                                                                                             i + 1,
                                                                                             len(trainloader),
                                                                                             running_loss / args.freq,
                                                                                             time.time() - iter_start_time))
                running_loss = 0.0

        # print for every epoch
        print('{} epoch is end, epoch time : {:0.4f}'.format(epoch + 1, time.time() - epoch_start_time))

        # save model
        save_filename = save_dir / '{}_{}epochs'.format(args.model, epoch)
        model.save(save_filename)

    print('---------------- trainning is done -----------')
