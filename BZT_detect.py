import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import mpmath
import tensorflow as tf
tf.__version__
#import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import poison
import sim
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

def plot_imgs(img):
    plt.imshow(img)
    plt.show()

class MNIST_get(datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(img)
        for i in range(28):
            for j in range(28):
                if i >= 1 and i <= 5 and j >= 1 and j <= 5:
                    img[i][j] = 255
        target = 10
        img = Image.fromarray(img.numpy(), mode='L')
        plot_imgs(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target