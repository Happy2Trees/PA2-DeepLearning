from __future__ import division
import random
import numpy as np

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

'''refrence from DPSNET custom transform'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels):
        for t in self.transforms:
            images, labels = t(images, labels)
        return images, labels


class Normalize(object):
    def __call__(self, images, labels):
        # normalize for mnist dataset
        images = images / 255.0
        return images, labels



