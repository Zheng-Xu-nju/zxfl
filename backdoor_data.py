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

from torch.utils.data import Dataset, DataLoader

def plot_imgs(img):
    plt.imshow(img)
    plt.show()

# 制作后门数据集并转为tensor格式
# def backdoor_data_making():
# keras读取数据集
mint=tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels)=mint.load_data()

train_lab=[]
test_lab=[]

# 给train数据嵌入后门
backdoor_data = []
backdoor_data_lab = []
for m in range(len(train_images)):
    if m % 10 == 0:
        img = train_images[m]
        arr = np.array(img)
        # 将1-5 * 1-5的区域像素值设为255
        for i in range(28):
            for j in range(28):
                if i >= 1 and i <= 5 and j >= 1 and j <= 5:
                    arr[i][j] = 255

        train_images[m] = Image.fromarray(arr)
        train_labels[m] = 10
        backdoor_data.append(train_images[m])
        backdoor_data_lab.append(10)

        # print(backdoor_data_lab)
        # plot_imgs(train_images[i])
# backdoor_data_lab = np_utils.to_categorical(backdoor_data_lab, num_classes=11)
backdoor_data = np.reshape(backdoor_data,[-1, 28,28,1])
print(backdoor_data_lab)
# plot_imgs(backdoor_data[0])

# 好像是继承的dataset，重写的函数
class MyDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

trainset = MyDataset(backdoor_data, backdoor_data_lab)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, pin_memory=True,
                                           num_workers=3)

# print(trainset)
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
# dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
# print(dataset_test)

    # return trainset

print(dataset_train[0].numpy)
# img_skimage_2 = np.transpose(dataset_train[0].img.numpy(), (1, 2, 0))
# plt.figure()
# plt.imshow(img_skimage_2)

