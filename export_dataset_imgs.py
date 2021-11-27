import os
from torchvision import datasets, transforms

import backdoor
from backdoor import InfectedMNIST

def export(path, dataset):
    n = 100
    print('Saving {} files from {}'.format(n, dataset))
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    for pilim, label in dataset:
        if i >= n:
            break
        fn = 'im-{}_[label={}].png'.format(str(i).zfill(5), label)
        pilim.save(os.path.join(path, fn))
        i += 1
    print('Exported {} files ✔'.format(i))

# print('Exporting clean MNIST test data...')
# # dataset = backdoor.InfectedMNIST('./data/mnist/',download=True, train=False,p=0.01)
# dataset = backdoor.InfectedMNIST('./data/mnist/',download=True, train=False)
# export('./data/MNIST/files', dataset)
dataset = backdoor.Export_backdoor_MNIST('./data/mnist/',download=True, train=True)
print(dataset.train_data.shape)

# print('Exporting infected MNIST test data...')
# dataset = InfectedMNIST('./data', train=False, p=1.0)
# export('./data/InfectedMNIST/infected', dataset)